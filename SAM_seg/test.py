import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import get_test_pairs, open_l, open_rgb
from utils import dice_ci95, evaluate_metrics_with_type, load_model_weights_flexible


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")
if PVT_ROOT not in sys.path:
    sys.path.insert(0, PVT_ROOT)

from lib.pvt import PolypPVT


def get_args():
    parser = argparse.ArgumentParser(description="Segmentation test script")
    parser.add_argument("--model", default="polyp-pvt", choices=["polyp-pvt"], type=str)
    parser.add_argument("--testset", default="kvasir", choices=["cvc", "kvasir"], type=str)
    parser.add_argument("--type", default="test", choices=["test", "val"], type=str, help="eval mode")
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means evaluate all")
    parser.add_argument("--ci_bootstrap", default=2000, type=int, help="bootstrap iterations for DSC 95%CI")
    parser.add_argument("--ci_seed", default=42, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--weights",
        default="/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/seg_pvt/2026-0319-1248_polyp_pvt/checkpoint/best_pvt.pth",
        type=str,
        help="model checkpoint path",
    )
    parser.add_argument(
        "--pvt_pretrained",
        default="",
        type=str,
        help="optional pvt_v2_b2 pretrain path for PolypPVT backbone",
    )
    return parser.parse_args()


def _build_model(args, device):
    if args.model != "polyp-pvt":
        raise ValueError(f"Unsupported model: {args.model}")
    pvt_pretrained = args.pvt_pretrained.strip() if args.pvt_pretrained else None
    if pvt_pretrained and not os.path.isfile(pvt_pretrained):
        raise FileNotFoundError(f"--pvt_pretrained not found: {pvt_pretrained}")
    model = PolypPVT(pretrained_path=pvt_pretrained).to(device)
    load_model_weights_flexible(model, args.weights, device)
    model.eval()
    return model


def run_test(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    model = _build_model(args, device)
    pairs = get_test_pairs(args.testset)
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
        for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"[Test:{args.testset}]"), start=1):
            image = open_rgb(img_path)
            gt = open_l(mask_path).resize((args.trainsize, args.trainsize), Image.NEAREST)
            gt_mask = (np.array(gt) > 127).astype(np.uint8)

            x = preprocess(image).unsqueeze(0).to(device)
            p1, p2 = model(x)
            logits = p1 + p2
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
    print(f"Testset: {args.testset}")
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
