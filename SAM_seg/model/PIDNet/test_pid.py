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
SAM_SEG_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
PIDNET_ROOT = THIS_DIR
if SAM_SEG_ROOT not in sys.path:
    sys.path.insert(0, SAM_SEG_ROOT)
if PIDNET_ROOT not in sys.path:
    sys.path.insert(1, PIDNET_ROOT)

from dataset import load_pairs_from_root, open_l, open_rgb
from utils import (
    decode_pidnet_outputs,
    dice_ci95,
    evaluate_metrics_with_type,
    get_state_dict_from_checkpoint,
    load_model_weights_flexible,
    normalize_state_dict_keys,
)

from models.pidnet import PIDNet


def get_args():
    parser = argparse.ArgumentParser(description="PIDNet test script")
    parser.add_argument(
        "--testdataset_root",
        default="data/seg_data/TestDataset",
        type=str,
        help="root for seg_data/TestDataset",
    )
    parser.add_argument(
        "--val_dataset",
        default="Kvasir",
        type=str,
        choices=["CVC-ColonDB", "CVC-300", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir", "all"],
        help="subfolder under --testdataset_root; use 'all' for all subsets",
    )
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means evaluate all")
    parser.add_argument("--ci_bootstrap", default=2000, type=int, help="bootstrap iterations for Dice 95%%CI")
    parser.add_argument("--ci_seed", default=42, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--weights", 
        default="runs/seg_pids/2026-0325-2037_CVC-ColonDB/checkpoint/best_pidnet_s5109.pth", 
        type=str, help="PIDNet checkpoint path")
    parser.add_argument("--pidnet_num_classes", default=2, type=int, help="fallback class count when ckpt inference fails")
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


def _load_weights_for_model(model, weights: str, device: torch.device):
    ckpt_path = resolve_path(weights)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--weights not found: {ckpt_path}")
    info = load_model_weights_flexible(model, ckpt_path, device)
    return ckpt_path, info


def _infer_pidnet_num_classes_from_ckpt(weights: str, device: torch.device):
    ckpt_path = resolve_path(weights)
    if not os.path.isfile(ckpt_path):
        return None
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = normalize_state_dict_keys(get_state_dict_from_checkpoint(checkpoint))

    for key in ("final_layer.conv2.weight", "seghead_p.conv2.weight"):
        tensor = state_dict.get(key)
        if torch.is_tensor(tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])

    for key in ("final_layer.conv2.bias", "seghead_p.conv2.bias"):
        tensor = state_dict.get(key)
        if torch.is_tensor(tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _build_model(args, device):
    if not args.weights:
        raise ValueError("--weights is required for testing.")

    num_classes = int(args.pidnet_num_classes)
    if num_classes < 1:
        raise ValueError("--pidnet_num_classes must be >= 1.")

    inferred_classes = _infer_pidnet_num_classes_from_ckpt(args.weights, device)
    if inferred_classes is not None and inferred_classes != num_classes:
        print(
            f"[PIDNet] --pidnet_num_classes={num_classes} mismatches checkpoint head channels "
            f"({inferred_classes}); override to {inferred_classes}."
        )
        num_classes = inferred_classes

    model = PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ).to(device)
    ckpt_path, load_info = _load_weights_for_model(model, args.weights, device)
    model.eval()
    return model, ckpt_path, load_info, num_classes


def _forward_binary_logits(model, x: torch.Tensor, target_size):
    outputs = model(x)
    _, main_logits, _ = decode_pidnet_outputs(outputs)
    if main_logits.ndim != 4:
        raise RuntimeError(f"PIDNet main logits must be NCHW, got shape={tuple(main_logits.shape)}")

    if main_logits.shape[1] == 1:
        logits = main_logits
    elif main_logits.shape[1] >= 2:
        logits = main_logits[:, 1:2] - main_logits[:, 0:1]
    else:
        raise RuntimeError("PIDNet returned invalid channel count.")

    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def run_test(args):
    if torch.cuda.is_available():
        gpu_index = int(str(args.gpu).split(",")[0].strip())
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    model, ckpt_path, load_info, num_classes = _build_model(args, device)
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

    metric_names = ["dice", "precision", "recall", "hd", "hd95"]
    values = {k: [] for k in metric_names}

    with torch.no_grad():
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"[Test:{eval_name}]"), start=1):
            image = open_rgb(img_path)
            gt = open_l(mask_path).resize((args.trainsize, args.trainsize), Image.NEAREST)
            gt_mask = (np.array(gt) > 127).astype(np.uint8)

            x = preprocess(image).unsqueeze(0).to(device)
            logits = _forward_binary_logits(model, x, target_size=(args.trainsize, args.trainsize))
            pred_mask = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() > args.threshold).astype(np.uint8)

            m = evaluate_metrics_with_type(pred_mask, gt_mask, type="test")
            for k in metric_names:
                values[k].append(float(m[k]))

            if args.verbose:
                print(
                    f"[{idx}/{len(pairs)}] {os.path.basename(img_path)} "
                    f"dice={m['dice']:.4f} hd={m['hd']:.4f} hd95={m['hd95']:.4f} "
                    f"precision={m['precision']:.4f} recall={m['recall']:.4f}"
                )

    print("=" * 72)
    print("Model: pidnet")
    print(f"checkpoint: {ckpt_path}")
    print(f"Num classes: {num_classes}")
    print(f"Loaded params: {load_info['loaded']}/{load_info['candidate']} (total={load_info['total']})")
    print(f"Testset: {eval_name}")
    print(f"Samples: {len(pairs)}")

    dice_summary = dice_ci95(
        values["dice"],
        type="test",
        confidence=0.95,
        n_bootstrap=args.ci_bootstrap,
        seed=args.ci_seed,
    )
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
