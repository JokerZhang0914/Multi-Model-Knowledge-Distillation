import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(1, THIS_DIR)
if PVT_ROOT not in sys.path:
    sys.path.append(PVT_ROOT)

from SAM_seg.dataset import open_rgb
from SAM_seg.utils import load_model_weights_flexible
from lib.pvt import PolypPVT
from CAM_SAM.model.sam import build_medsam_predictor
from CAM_SAM.utils import bbox2sam_mask, ensure_boxes_2d, mask2bbox


def get_args():
    parser = argparse.ArgumentParser(description="Generate MedSAM pseudo masks from PVT-predicted bboxes")
    parser.add_argument(
        "--input_csv",
        default="data/sam_pseudo_mask_pairs.csv",
        type=str,
        help="source csv; first column is image path",
    )
    parser.add_argument(
        "--output_csv",
        default="data/seg1_pseudo_mask_pairs.csv",
        type=str,
        help="output csv with two columns: image_path, pseudo_mask_path",
    )
    parser.add_argument(
        "--output_mask_root",
        default="data/PseudoMask-seg1",
        type=str,
        help="directory to save MedSAM masks",
    )
    parser.add_argument(
        "--path_prefix_to_strip",
        default="",
        type=str,
        help="prefix stripped in output csv; empty means project root",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        type=str,
        help="root used to preserve subfolder structure under output_mask_root",
    )
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means all")

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
        help="optional pvt_v2_b2 pretrained path used when building PVT model",
    )
    parser.add_argument(
        "--medsam_ckpt",
        default="/mnt/nas1/disk03/zhaokaizhang/code/medsam_vit_b.pth",
        type=str,
        help="path to MedSAM checkpoint",
    )
    parser.add_argument(
        "--medsam_model_type",
        default="vit_b",
        type=str,
        choices=["vit_b", "vit_l", "vit_h", "default"],
    )

    parser.add_argument("--bbox_min_area", default=30, type=int)
    parser.add_argument("--bbox_enlarge", default=0.1, type=float)
    parser.add_argument("--bbox_mode", default="all", choices=["largest", "union", "all"])
    parser.add_argument("--bbox_pre_dilate", default=0, type=int)
    parser.add_argument("--bbox_min_size", default=1, type=int)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--cleanup_empty_only",
        action="store_true",
        help="only clean existing csv rows whose pseudo mask is empty; no model inference",
    )
    parser.add_argument(
        "--cleanup_csv_path",
        default="data/seg1_pseudo_mask_pairs.csv",
        type=str,
        help="csv path to clean; empty means --output_csv",
    )
    parser.add_argument(
        "--cleanup_min_foreground_pixels",
        default=0,
        type=int,
        help="remove row if pseudo mask foreground pixels <= this value",
    )
    parser.add_argument(
        "--cleanup_backup_csv",
        action="store_true",
        help="backup original csv to <csv>.bak before overwrite",
    )
    parser.add_argument(
        "--cleanup_dry_run",
        action="store_true",
        help="only report how many rows would be removed; do not overwrite csv",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def read_first_column_image_paths(csv_path: str) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"input csv not found: {csv_path}")

    image_paths: List[str] = []
    seen = set()
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            raw = row[0].strip()
            if not raw:
                continue
            if row_idx == 0 and raw.lower() in ("image_path", "image", "img_path", "path"):
                continue
            img_path = resolve_path(raw)
            if img_path in seen:
                continue
            seen.add(img_path)
            image_paths.append(img_path)
    return image_paths


def scale_box_xyxy(
    box_xyxy: np.ndarray,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    sx = float(dst_w) / float(max(1, src_w))
    sy = float(dst_h) / float(max(1, src_h))
    boxes = ensure_boxes_2d(box_xyxy)
    if boxes.shape[0] == 0:
        return boxes

    out = boxes.copy()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    out[:, 0] = np.clip(out[:, 0], 0, max(0, dst_w - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, dst_h - 1))
    out[:, 2] = np.clip(out[:, 2], 0, dst_w)
    out[:, 3] = np.clip(out[:, 3], 0, dst_h)
    return out.astype(np.float32)


def save_binary_mask(mask: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(mask_u8).save(out_path)


def simplify_path(path: str, prefix_to_strip: str) -> str:
    p = os.path.abspath(path)
    if not prefix_to_strip:
        return p
    pref = os.path.abspath(prefix_to_strip)
    try:
        rel = os.path.relpath(p, pref)
    except ValueError:
        return p
    if rel.startswith(".."):
        return p
    return rel


def count_foreground_pixels(mask_path: str) -> int:
    try:
        arr = np.asarray(Image.open(mask_path).convert("L"))
    except Exception:
        return -1
    return int((arr > 0).sum())


def cleanup_empty_rows_in_csv(
    csv_path: str,
    min_foreground_pixels: int = 0,
    backup_csv: bool = False,
    dry_run: bool = False,
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"cleanup csv not found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if len(fieldnames) < 2:
        raise ValueError(f"cleanup csv must have at least 2 columns, got: {fieldnames}")

    image_col = fieldnames[0]
    mask_col = fieldnames[1]

    kept_rows = []
    total_rows = len(rows)
    removed_empty = 0
    removed_missing_or_bad = 0

    for row in tqdm(rows, desc="[Cleanup CSV]", total=total_rows):
        mask_path_raw = (row.get(mask_col) or "").strip()
        if not mask_path_raw:
            removed_missing_or_bad += 1
            continue

        mask_abs = resolve_path(mask_path_raw)
        if not os.path.isfile(mask_abs):
            removed_missing_or_bad += 1
            continue

        fg_pixels = count_foreground_pixels(mask_abs)
        if fg_pixels < 0:
            removed_missing_or_bad += 1
            continue

        if fg_pixels <= int(min_foreground_pixels):
            removed_empty += 1
            continue

        kept_rows.append(row)

    removed_total = removed_empty + removed_missing_or_bad
    stats = {
        "csv_path": csv_path,
        "image_col": image_col,
        "mask_col": mask_col,
        "total_rows": total_rows,
        "kept_rows": len(kept_rows),
        "removed_total": removed_total,
        "removed_empty": removed_empty,
        "removed_missing_or_bad": removed_missing_or_bad,
        "dry_run": bool(dry_run),
    }

    if dry_run:
        return stats

    if backup_csv:
        backup_path = f"{csv_path}.bak"
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as src, open(
            backup_path, "w", newline="", encoding="utf-8-sig"
        ) as dst:
            dst.write(src.read())
        stats["backup_path"] = backup_path

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return stats


def build_pvt_model(args, device: torch.device):
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


def build_medsam(args, device: torch.device):
    medsam_ckpt = resolve_path(args.medsam_ckpt)
    if not os.path.isfile(medsam_ckpt):
        raise FileNotFoundError(f"--medsam_ckpt not found: {medsam_ckpt}")
    predictor = build_medsam_predictor(
        checkpoint=medsam_ckpt,
        device=device,
        model_type=args.medsam_model_type,
    )
    return predictor, medsam_ckpt


def infer_pvt_binary_mask(model, image_pil: Image.Image, preprocess, threshold: float, trainsize: int):
    x = preprocess(image_pil).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    with torch.no_grad():
        p1, p2 = model(x)
        logits = p1 + p2
        if logits.shape[-2:] != (trainsize, trainsize):
            logits = F.interpolate(logits, size=(trainsize, trainsize), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (prob > float(threshold)).astype(np.uint8)


def build_output_mask_path(img_path: str, output_mask_root: str, data_root: str) -> str:
    img_path = os.path.abspath(img_path)
    data_root = os.path.abspath(data_root)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    try:
        rel = os.path.relpath(img_path, data_root)
    except ValueError:
        rel = os.path.basename(img_path)
    if rel.startswith(".."):
        rel = os.path.basename(img_path)

    rel_dir = os.path.dirname(rel)
    out_dir = os.path.join(output_mask_root, rel_dir) if rel_dir else output_mask_root
    return os.path.join(out_dir, f"{stem}_seg1.png")


def main():
    args = get_args()

    input_csv = resolve_path(args.input_csv)
    output_csv = resolve_path(args.output_csv)
    output_mask_root = resolve_path(args.output_mask_root)
    data_root = resolve_path(args.data_root)
    prefix_to_strip = resolve_path(args.path_prefix_to_strip) if args.path_prefix_to_strip else PROJECT_ROOT

    if args.cleanup_empty_only:
        cleanup_csv_path = resolve_path(args.cleanup_csv_path) if args.cleanup_csv_path else output_csv
        stats = cleanup_empty_rows_in_csv(
            csv_path=cleanup_csv_path,
            min_foreground_pixels=args.cleanup_min_foreground_pixels,
            backup_csv=args.cleanup_backup_csv,
            dry_run=args.cleanup_dry_run,
        )
        print("-" * 72)
        print(f"[Cleanup] csv: {stats['csv_path']}")
        print(f"[Cleanup] columns: image={stats['image_col']} mask={stats['mask_col']}")
        print(f"[Cleanup] total rows: {stats['total_rows']}")
        print(f"[Cleanup] kept rows: {stats['kept_rows']}")
        print(f"[Cleanup] removed total: {stats['removed_total']}")
        print(f"[Cleanup] removed empty: {stats['removed_empty']}")
        print(f"[Cleanup] removed missing/bad mask: {stats['removed_missing_or_bad']}")
        if stats.get("backup_path"):
            print(f"[Cleanup] backup: {stats['backup_path']}")
        if stats["dry_run"]:
            print("[Cleanup] dry-run enabled, csv not overwritten.")
        return

    if torch.cuda.is_available():
        gpu_index = int(str(args.gpu).split(",")[0].strip())
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    print(f"[Info] device: {device}")
    print(f"[Info] input_csv: {input_csv}")
    print(f"[Info] output_csv: {output_csv}")
    print(f"[Info] output_mask_root: {output_mask_root}")
    print(f"[Info] simplify_prefix: {prefix_to_strip}")

    image_paths = read_first_column_image_paths(input_csv)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]
    if len(image_paths) == 0:
        raise RuntimeError("No valid image paths found in first column of input csv.")

    os.makedirs(output_mask_root, exist_ok=True)
    out_csv_dir = os.path.dirname(output_csv)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)

    pvt_model, pvt_ckpt, pvt_load_info = build_pvt_model(args, device)
    medsam_predictor, medsam_ckpt = build_medsam(args, device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.trainsize, args.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    rows = []
    total = 0
    missing = 0
    no_box = 0
    empty_after_medsam = 0
    saved = 0

    for img_path in tqdm(image_paths, desc="[PVT->BBox->MedSAM]"):
        total += 1
        if not os.path.isfile(img_path):
            missing += 1
            if args.verbose:
                print(f"[Skip:missing] {img_path}")
            continue

        image_pil = open_rgb(img_path)
        orig_w, orig_h = image_pil.size

        pvt_mask = infer_pvt_binary_mask(
            pvt_model,
            image_pil=image_pil,
            preprocess=preprocess,
            threshold=args.threshold,
            trainsize=args.trainsize,
        )

        box = mask2bbox(
            pvt_mask,
            min_area=args.bbox_min_area,
            box_enlarge=args.bbox_enlarge,
            mode=args.bbox_mode,
            clip=True,
            pre_dilate=args.bbox_pre_dilate,
            min_box_size=args.bbox_min_size,
        )
        box = ensure_boxes_2d(box)

        if box.shape[0] == 0:
            no_box += 1
            sam_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            box_orig = scale_box_xyxy(
                box,
                src_w=args.trainsize,
                src_h=args.trainsize,
                dst_w=orig_w,
                dst_h=orig_h,
            )
            box_orig = ensure_boxes_2d(box_orig)
            if box_orig.shape[0] == 0:
                no_box += 1
                sam_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            else:
                img_rgb = np.asarray(image_pil, dtype=np.uint8)
                sam_mask = bbox2sam_mask(
                    medsam_predictor,
                    img_rgb,
                    box_orig,
                    type="medsam",
                    multimask_output=False,
                )
                if int((sam_mask > 0).sum()) == 0:
                    empty_after_medsam += 1

        out_mask_path = build_output_mask_path(
            img_path=img_path,
            output_mask_root=output_mask_root,
            data_root=data_root,
        )
        save_binary_mask(sam_mask, out_mask_path)
        saved += 1

        rows.append(
            {
                "image_path": simplify_path(img_path, prefix_to_strip),
                "pseudo_mask_path": simplify_path(out_mask_path, prefix_to_strip),
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pseudo_mask_path"])
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 72)
    print(f"[Done] PVT ckpt: {pvt_ckpt}")
    print(f"[Done] MedSAM ckpt: {medsam_ckpt}")
    print(
        f"[Done] PVT loaded params: {pvt_load_info['loaded']}/"
        f"{pvt_load_info['candidate']} (total={pvt_load_info['total']})"
    )
    print(f"[Done] total in csv(first column): {total}")
    print(f"[Done] missing images skipped: {missing}")
    print(f"[Done] no-box cases (saved as empty mask): {no_box}")
    print(f"[Done] empty MedSAM outputs: {empty_after_medsam}")
    print(f"[Done] saved masks: {saved}")
    print(f"[Done] output csv rows: {len(rows)}")
    print(f"[Done] output csv: {output_csv}")


if __name__ == "__main__":
    main()
