import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import load_pairs_from_root, open_l, open_rgb
from utils import (
    decode_pidnet_outputs,
    dice_ci95,
    evaluate_metrics_with_type,
    load_model_weights_flexible,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")
if PVT_ROOT not in sys.path:
    sys.path.insert(0, PVT_ROOT)
UNET_ROOT = os.path.join(THIS_DIR, "model", "UNet")
if UNET_ROOT not in sys.path:
    sys.path.insert(0, UNET_ROOT)
PIDNET_ROOT = os.path.join(THIS_DIR, "model", "PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)

from lib.pvt import PolypPVT
from unet import UNet
from models.pidnet import PIDNet


def get_args():
    parser = argparse.ArgumentParser(description="Segmentation test script")
    parser.add_argument("--model", default="pvt", choices=["pvt", "unet", "pidnet"], type=str)
    parser.add_argument(
        "--testdataset_root",
        default="data/seg_data/TestDataset",
        type=str,
        help="root for seg_data/TestDataset",
    )
    parser.add_argument(
        "--val_dataset",
        default="CVC-ColonDB",
        type=str,
        choices=["CVC-ColonDB", "CVC-300", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir", "all"],
        help="subfolder under --testdataset_root, e.g. CVC-ColonDB/Kvasir/CVC-300; use 'all' for all subsets",
    )
    parser.add_argument("--type", default="test", choices=["test", "val"], type=str, help="eval mode")
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means evaluate all")
    parser.add_argument("--ci_bootstrap", default=2000, type=int, help="bootstrap iterations for DSC 95%%CI")
    parser.add_argument("--ci_seed", default=42, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--weights",
        default="",
        type=str,
        help="model checkpoint path",
    )
    parser.add_argument(
        "--pvt_pretrained",
        default="",
        type=str,
        help="optional pvt_v2_b2 pretrain path for PolypPVT backbone",
    )
    parser.add_argument("--unet_n_channels", default=3, type=int)
    parser.add_argument("--unet_n_classes", default=1, type=int)
    parser.add_argument("--unet_bilinear", default=False, type=str, choices=["True", "False", "true", "false", "1", "0"])
    parser.add_argument("--pidnet_num_classes", default=2, type=int)
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


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


def _load_weights_for_model(model, weights: str, device: torch.device, model_name: str):
    ckpt_path = resolve_path(weights)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--weights not found: {ckpt_path}")
    try:
        load_model_weights_flexible(model, ckpt_path, device)
        return
    except RuntimeError as e:
        if model_name != "unet":
            raise
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise e
        if "model_state_dict" in checkpoint or "state_dict" in checkpoint:
            raise e
        state_dict = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if len(state_dict) == 0:
            raise e
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        if len(matched) == 0:
            raise e
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)


def _build_model(args, device):
    if not args.weights:
        raise ValueError("--weights is required for testing.")

    if args.model == "pvt":
        pvt_pretrained = args.pvt_pretrained.strip() if args.pvt_pretrained else None
        if pvt_pretrained and not os.path.isfile(pvt_pretrained):
            raise FileNotFoundError(f"--pvt_pretrained not found: {pvt_pretrained}")
        model = PolypPVT(pretrained_path=pvt_pretrained).to(device)
        _load_weights_for_model(model, args.weights, device, model_name="pvt")
        model.eval()
        return model

    if args.model == "unet":
        n_classes = int(args.unet_n_classes)
        if n_classes < 1:
            raise ValueError("--unet_n_classes must be >= 1")
        bilinear = str2bool(args.unet_bilinear)
        model = UNet(
            n_channels=int(args.unet_n_channels),
            n_classes=n_classes,
            bilinear=bilinear,
        ).to(device)
        _load_weights_for_model(model, args.weights, device, model_name="unet")
        model.eval()
        return model

    if args.model == "pidnet":
        num_classes = int(args.pidnet_num_classes)
        if num_classes < 2:
            raise ValueError("--pidnet_num_classes must be >= 2 for binary foreground extraction.")
        model = PIDNet(
            m=2,
            n=3,
            num_classes=num_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        ).to(device)
        _load_weights_for_model(model, args.weights, device, model_name="pidnet")
        model.eval()
        return model

    raise ValueError(f"Unsupported model: {args.model}")


def _forward_binary_logits(model, model_name: str, x: torch.Tensor, target_size):
    if model_name == "pvt":
        p1, p2 = model(x)
        logits = p1 + p2
    elif model_name == "unet":
        out = model(x)
        if out.shape[1] == 1:
            logits = out
        elif out.shape[1] >= 2:
            logits = out[:, 1:2] - out[:, 0:1]
        else:
            raise RuntimeError("UNet returned invalid channel count.")
    elif model_name == "pidnet":
        outputs = model(x)
        _, main_logits, _ = decode_pidnet_outputs(outputs)
        if main_logits.shape[1] == 1:
            logits = main_logits
        elif main_logits.shape[1] >= 2:
            logits = main_logits[:, 1:2] - main_logits[:, 0:1]
        else:
            raise RuntimeError("PIDNet returned invalid channel count.")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def run_test(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    model = _build_model(args, device)
    pairs, eval_name = _build_eval_pairs(args)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    preprocess = transforms.Compose([
        transforms.Resize((args.trainsize, args.trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.type == "val":
        metric_names = ["dice"]
    else:
        metric_names = ["dice", "precision", "recall", "hd", "hd95"]
    values = {k: [] for k in metric_names}

    with torch.no_grad():
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"[Test:{eval_name}]"), start=1):
            image = open_rgb(img_path)
            gt = open_l(mask_path).resize((args.trainsize, args.trainsize), Image.NEAREST)
            gt_mask = (np.array(gt) > 127).astype(np.uint8)

            x = preprocess(image).unsqueeze(0).to(device)
            logits = _forward_binary_logits(model, args.model, x, target_size=(args.trainsize, args.trainsize))
            pred_mask = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() > args.threshold).astype(np.uint8)

            m = evaluate_metrics_with_type(pred_mask, gt_mask, type=args.type)
            for k in metric_names:
                values[k].append(float(m[k]))

            if args.verbose:
                if args.type == "val":
                    print(f"[{idx}/{len(pairs)}] {os.path.basename(img_path)} dice={m['dice']:.4f}")
                else:
                    print(
                        f"[{idx}/{len(pairs)}] {os.path.basename(img_path)} "
                        f"dice={m['dice']:.4f} hd95={m['hd95']:.4f} "
                        f"precision={m['precision']:.4f} recall={m['recall']:.4f}"
                    )

    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"checkpoint: {args.weights}")
    print(f"Type: {args.type}")
    print(f"Testset: {eval_name}")
    print(f"Samples: {len(pairs)}")

    dice_summary = dice_ci95(
        values["dice"],
        type=args.type,
        confidence=0.95,
        n_bootstrap=args.ci_bootstrap,
        seed=args.ci_seed,
    )
    if dice_summary["dice_ci95"] is None:
        print(f"dice: mean={dice_summary['dice_mean']:.6f}")
    else:
        low, high = dice_summary["dice_ci95"]
        print(f"dice: mean={dice_summary['dice_mean']:.6f} | 95%CI=[{low:.6f}, {high:.6f}]")

    for k in metric_names:
        if k == "dice":
            continue
        arr = np.asarray(values[k], dtype=np.float64)
        if k in ("hd", "hd95"):
            finite = arr[np.isfinite(arr)]
            inf_count = int((~np.isfinite(arr)).sum())
            mean_val = float(finite.mean()) if finite.size > 0 else float("inf")
            print(f"{k}: mean={mean_val:.6f} (inf_cases={inf_count})")
        else:
            print(f"{k}: mean={float(arr.mean()):.6f}")


def main():
    args = get_args()
    run_test(args)


if __name__ == "__main__":
    main()
