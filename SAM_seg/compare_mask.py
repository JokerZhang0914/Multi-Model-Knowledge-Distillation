import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)
if PVT_ROOT not in sys.path:
    sys.path.insert(2, PVT_ROOT)

from dataset import load_pairs_from_root, open_l, open_rgb
from utils import dice_ci95, dice_coefficient, load_model_weights_flexible
from lib.pvt import PolypPVT
from CAM_SAM.model.sam import build_medsam_predictor, build_sam_predictor
from CAM_SAM.utils import bbox2sam_mask, ensure_boxes_2d, mask2bbox


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare Polyp-PVT masks with SAM/MedSAM masks prompted by PVT->bbox"
    )
    parser.add_argument(
        "--testdataset_root",
        default="data/seg_data/TestDataset",
        type=str,
        help="root for seg_data/TestDataset",
    )
    parser.add_argument(
        "--val_dataset",
        default="all",
        type=str,
        choices=["CVC-ColonDB", "CVC-300", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir", "all"],
        help="subset under --testdataset_root; use 'all' for all subsets",
    )
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means evaluate all")
    parser.add_argument("--ci_bootstrap", default=2000, type=int, help="bootstrap iterations for Dice 95%%CI")
    parser.add_argument("--ci_seed", default=42, type=int)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--pvt_weights",
        default="runs/checkpoint/zero_shot_Sota_53PolypPVT.pth",
        type=str,
        help="Polyp-PVT checkpoint path",
    )
    parser.add_argument(
        "--pvt_backbone_pretrained",
        default="",
        type=str,
        help="optional pvt_v2_b2 backbone pretrained path for model init",
    )

    parser.add_argument(
        "--sam_ckpt",
        default="/mnt/nas1/disk03/zhaokaizhang/code/sam_vit_h_4b8939.pth",
        type=str,
        help="path to SAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--sam_model_type",
        default="vit_h",
        choices=["vit_b", "vit_l", "vit_h", "default"],
        type=str,
    )
    parser.add_argument(
        "--medsam_ckpt",
        default="/mnt/nas1/disk03/zhaokaizhang/code/medsam_vit_b.pth",
        type=str,
        help="path to MedSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--medsam_model_type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h", "default"],
        type=str,
    )
    parser.add_argument("--sam_multimask", default=True, type=bool)

    parser.add_argument("--bbox_min_area", default=30, type=int)
    parser.add_argument("--bbox_enlarge", default=0.1, type=float)
    parser.add_argument("--bbox_mode", default="all", choices=["largest", "union", "all"])
    parser.add_argument("--bbox_pre_dilate", default=0, type=int)
    parser.add_argument("--bbox_min_size", default=1, type=int)
    parser.add_argument(
        "--no_box_list_path",
        default="",
        type=str,
        help="optional output txt path for no-box samples: image_path<TAB>mask_path",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _resolve_dataset_root(user_root: str, fallback_roots, dataset_name: str) -> str:
    candidates = []
    if user_root:
        candidates.append(resolve_path(user_root))
    for path in fallback_roots:
        p = resolve_path(path)
        if p not in candidates:
            candidates.append(p)
    for path in candidates:
        if os.path.isdir(path):
            return path
    cand_text = "\n".join([f"  - {p}" for p in candidates]) if candidates else "  (none)"
    raise FileNotFoundError(f"{dataset_name} root not found. Checked:\n{cand_text}")


def _load_single_subset(root: str, subset: str):
    subset_root = os.path.join(root, subset)
    if not os.path.isdir(subset_root):
        raise FileNotFoundError(f"Validation subset not found: {subset_root}")
    return load_pairs_from_root(
        subset_root,
        candidates=[("images", "masks"), ("image", "mask"), ("Original", "Ground Truth")],
        name=f"TestDataset/{subset}",
        verbose=True,
        check_readable=True,
    )


def _build_eval_pairs(args):
    test_root = _resolve_dataset_root(
        args.testdataset_root,
        fallback_roots=["data/seg_data/TestDataset"],
        dataset_name="TestDataset",
    )
    val_name = str(args.val_dataset).strip()
    if not val_name:
        raise ValueError("--val_dataset cannot be empty.")

    if val_name.lower() == "all":
        subset_names = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
        if not subset_names:
            raise RuntimeError(f"No subsets found under {test_root}")
        pairs = []
        for subset in subset_names:
            pairs.extend(_load_single_subset(test_root, subset))
        return pairs, "all"

    pairs = _load_single_subset(test_root, val_name)
    return pairs, val_name


def _build_pvt_model(args, device: torch.device):
    if not args.pvt_weights:
        raise ValueError("--pvt_weights is required.")

    backbone_pretrained = args.pvt_backbone_pretrained.strip()
    if backbone_pretrained:
        backbone_pretrained = resolve_path(backbone_pretrained)
        if not os.path.isfile(backbone_pretrained):
            raise FileNotFoundError(f"--pvt_backbone_pretrained not found: {backbone_pretrained}")
        model = PolypPVT(pretrained_path=backbone_pretrained).to(device)
    else:
        model = PolypPVT(pretrained_path=None).to(device)

    ckpt_path = resolve_path(args.pvt_weights)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--pvt_weights not found: {ckpt_path}")
    load_info = load_model_weights_flexible(model, ckpt_path, device)
    model.eval()
    return model, ckpt_path, load_info


def _forward_pvt_logits(model, x: torch.Tensor, target_size):
    p1, p2 = model(x)
    logits = p1 + p2
    if logits.ndim != 4:
        raise RuntimeError(f"Polyp-PVT output must be NCHW, got shape={tuple(logits.shape)}")

    if logits.shape[1] == 1:
        fg_logits = logits
    elif logits.shape[1] >= 2:
        fg_logits = logits[:, 1:2] - logits[:, 0:1]
    else:
        raise RuntimeError("Polyp-PVT returned invalid channel count.")

    if fg_logits.shape[-2:] != target_size:
        fg_logits = F.interpolate(fg_logits, size=target_size, mode="bilinear", align_corners=False)
    return fg_logits


def _build_prompt_predictors(args, device: torch.device):
    sam_ckpt = resolve_path(args.sam_ckpt)
    medsam_ckpt = resolve_path(args.medsam_ckpt)
    if not os.path.isfile(sam_ckpt):
        raise FileNotFoundError(f"--sam_ckpt not found: {sam_ckpt}")
    if not os.path.isfile(medsam_ckpt):
        raise FileNotFoundError(f"--medsam_ckpt not found: {medsam_ckpt}")

    sam_predictor = build_sam_predictor(
        checkpoint=sam_ckpt,
        device=device,
        model_type=args.sam_model_type,
    )
    medsam_predictor = build_medsam_predictor(
        checkpoint=medsam_ckpt,
        device=device,
        model_type=args.medsam_model_type,
    )
    return sam_predictor, medsam_predictor, sam_ckpt, medsam_ckpt


def _summarize_dice(name: str, values, ci_bootstrap: int, ci_seed: int):
    summary = dice_ci95(
        values,
        type="test",
        confidence=0.95,
        n_bootstrap=ci_bootstrap,
        seed=ci_seed,
    )
    low, high = summary["dice_ci95"]
    print(f"{name}: mean={summary['dice_mean']:.6f} | 95%CI=[{low:.6f}, {high:.6f}]")
    return summary["dice_mean"]


def run_compare(args):
    if torch.cuda.is_available():
        gpu_index = int(str(args.gpu).split(",")[0].strip())
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    model, pvt_ckpt_path, load_info = _build_pvt_model(args, device)
    sam_predictor, medsam_predictor, sam_ckpt, medsam_ckpt = _build_prompt_predictors(args, device)
    pairs, eval_name = _build_eval_pairs(args)

    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if len(pairs) == 0:
        raise RuntimeError("No evaluation samples found.")

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.trainsize, args.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dice_pvt = []
    dice_sam = []
    dice_medsam = []
    no_box_count = 0
    no_box_samples = []

    with torch.no_grad():
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"[Compare:{eval_name}]"), start=1):
            image = open_rgb(img_path).resize((args.trainsize, args.trainsize), Image.BILINEAR)
            gt = open_l(mask_path).resize((args.trainsize, args.trainsize), Image.NEAREST)
            gt_mask = (np.array(gt) > 127).astype(np.uint8)

            x = preprocess(image).unsqueeze(0).to(device)
            logits = _forward_pvt_logits(model, x, target_size=(args.trainsize, args.trainsize))
            pvt_mask = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() > args.threshold).astype(np.uint8)

            boxes = mask2bbox(
                pvt_mask,
                min_area=args.bbox_min_area,
                box_enlarge=args.bbox_enlarge,
                mode=args.bbox_mode,
                clip=True,
                pre_dilate=args.bbox_pre_dilate,
                min_box_size=args.bbox_min_size,
            )
            boxes = ensure_boxes_2d(boxes)

            img_rgb = np.asarray(image, dtype=np.uint8)
            if boxes.shape[0] == 0:
                no_box_count += 1
                no_box_samples.append((img_path, mask_path))
                sam_mask = np.zeros_like(pvt_mask, dtype=np.uint8)
                medsam_mask = np.zeros_like(pvt_mask, dtype=np.uint8)
            else:
                sam_mask = bbox2sam_mask(
                    sam_predictor,
                    img_rgb,
                    boxes,
                    type="sam",
                    multimask_output=bool(args.sam_multimask),
                )
                medsam_mask = bbox2sam_mask(
                    medsam_predictor,
                    img_rgb,
                    boxes,
                    type="medsam",
                    multimask_output=False,
                )

            d_pvt = dice_coefficient(pvt_mask, gt_mask)
            d_sam = dice_coefficient(sam_mask, gt_mask)
            d_medsam = dice_coefficient(medsam_mask, gt_mask)
            dice_pvt.append(float(d_pvt))
            dice_sam.append(float(d_sam))
            dice_medsam.append(float(d_medsam))

            if args.verbose:
                print(
                    f"[{idx}/{len(pairs)}] {os.path.basename(img_path)} "
                    f"boxes={boxes.shape[0]} pvt={d_pvt:.4f} sam={d_sam:.4f} medsam={d_medsam:.4f}"
                )

    print("=" * 84)
    print("Model: Polyp-PVT + bbox-prompted SAM/MedSAM")
    print(f"Testset: {eval_name}")
    print(f"Samples: {len(pairs)}")
    print(f"PVT checkpoint: {pvt_ckpt_path}")
    print(f"SAM checkpoint: {sam_ckpt}")
    print(f"MedSAM checkpoint: {medsam_ckpt}")
    print(f"Loaded PVT params: {load_info['loaded']}/{load_info['candidate']} (total={load_info['total']})")
    print(f"No-box samples (PVT mask -> bbox failed): {no_box_count}")
    if no_box_count > 0:
        print("No-box sample list:")
        for i, (img_path, mask_path) in enumerate(no_box_samples, start=1):
            print(f"  [{i}] image={img_path} | mask={mask_path}")

    if args.no_box_list_path:
        out_txt = resolve_path(args.no_box_list_path)
        out_dir = os.path.dirname(out_txt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            for img_path, mask_path in no_box_samples:
                f.write(f"{img_path}\t{mask_path}\n")
        print(f"No-box list saved to: {out_txt}")

    mean_pvt = _summarize_dice("PVT Dice", dice_pvt, ci_bootstrap=args.ci_bootstrap, ci_seed=args.ci_seed)
    mean_sam = _summarize_dice("SAM Dice", dice_sam, ci_bootstrap=args.ci_bootstrap, ci_seed=args.ci_seed)
    mean_medsam = _summarize_dice("MedSAM Dice", dice_medsam, ci_bootstrap=args.ci_bootstrap, ci_seed=args.ci_seed)
    print(f"Delta(SAM - PVT): {mean_sam - mean_pvt:+.6f}")
    print(f"Delta(MedSAM - PVT): {mean_medsam - mean_pvt:+.6f}")


def main():
    args = get_args()
    run_compare(args)


if __name__ == "__main__":
    main()
