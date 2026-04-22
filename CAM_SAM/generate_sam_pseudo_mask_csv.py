import csv
import os
import re
from typing import List, Tuple
import argparse

import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.resnet import PretrainedResNet18_Encoder, Student_Head, FullResNet
from model.sam import build_medsam_predictor
from utils import (
    bbox2sam_mask,
    build_cls_transform,
    cam2mask,
    ensure_boxes_2d,
    load_checkpoint,
    mask2bbox,
    safe_open_rgb,
)

SOURCE_ROOTS = [
    # "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/0_normal_aligned",
    "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/肿瘤医院_1/dataset_aligned_zhongliu",
    "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集mix",
]

PSEUDO_MASK_ROOT = "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/PseudoMask"

PATIENT_DIR_PATTERN = re.compile(r"^[tf]_patient\d+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def get_args():
    parser = argparse.ArgumentParser(description="Generate MedSAM pseudo masks from GradCAM++ boxes")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--positive_threshold", default=0.9, type=float)
    parser.add_argument("--positive_class_index", default=1, type=int)
    parser.add_argument(
        "--encoder_ckpt",
        default="/mnt/nas1/disk03/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1757_gradcampp/checkpoint/best_resnet_encoder.pth",
        type=str,
    )
    parser.add_argument(
        "--student_ckpt",
        default="/mnt/nas1/disk03/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1757_gradcampp/checkpoint/best_resnet_student.pth",
        type=str,
    )
    parser.add_argument("--medsam_ckpt", default="/mnt/nas1/disk03/zhaokaizhang/code/medsam_vit_b.pth", type=str)
    parser.add_argument("--medsam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--csv_output", default=os.path.join(os.getcwd(), "sam_pseudo_mask_pairs.csv"), type=str)

    parser.add_argument("--cam_low", default=0.5, type=float)
    parser.add_argument("--cam_high", default=0.75, type=float)
    parser.add_argument("--cam_min_area", default=100, type=int)
    parser.add_argument("--cam_sigma", default=1.0, type=float)
    parser.add_argument("--cam_dilate", default=1, type=int)

    parser.add_argument("--bbox_min_area", default=50, type=int)
    parser.add_argument("--bbox_enlarge", default=0.1, type=float)
    parser.add_argument("--bbox_mode", default="all", choices=["largest", "union", "all"])
    parser.add_argument("--bbox_pre_dilate", default=0, type=int)
    parser.add_argument("--bbox_min_size", default=2, type=int)
    return parser.parse_args()

def is_image_file(file_name: str) -> bool:
    return os.path.splitext(file_name)[1].lower() in IMAGE_EXTS


def list_patient_dirs(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"[Warning] Root not found, skipped: {root_dir}")
        return []
    patient_dirs = []
    for name in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path) and PATIENT_DIR_PATTERN.match(name):
            patient_dirs.append(full_path)
    return patient_dirs


def list_image_paths(patient_dir: str) -> List[str]:
    image_paths = []
    for name in sorted(os.listdir(patient_dir)):
        full_path = os.path.join(patient_dir, name)
        if os.path.isfile(full_path) and is_image_file(name):
            image_paths.append(full_path)
    return image_paths


def resize_cam(cam: np.ndarray, out_size: int) -> np.ndarray:
    cam_u8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    cam_resized = Image.fromarray(cam_u8).resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(cam_resized, dtype=np.float32) / 255.0


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
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_u8).save(out_path)


def build_models(device: torch.device, args):
    encoder = PretrainedResNet18_Encoder(freeze=False).to(device)
    student = Student_Head(input_dim=512, num_classes=2).to(device)
    model = FullResNet(encoder, student).to(device)

    load_checkpoint(encoder, args.encoder_ckpt, device)
    load_checkpoint(student, args.student_ckpt, device)
    model.eval()
    encoder.eval()
    student.eval()

    cam_engine = GradCAMPlusPlus(model=model, target_layers=[encoder.features[-2]])
    medsam_predictor = build_medsam_predictor(
        checkpoint=args.medsam_ckpt,
        device=device,
        model_type=args.medsam_model_type,
    )
    preprocess = build_cls_transform(args.crop_size)
    return model, cam_engine, medsam_predictor, preprocess


def collect_all_patient_dirs(source_roots: List[str]) -> List[Tuple[str, str]]:
    out = []
    for root in source_roots:
        patients = list_patient_dirs(root)
        for p in patients:
            out.append((root, p))
    return out


def main():
    args = get_args()
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] CSV output: {args.csv_output}")
    print(f"[Info] Pseudo-mask root: {PSEUDO_MASK_ROOT}")

    os.makedirs(PSEUDO_MASK_ROOT, exist_ok=True)
    model, cam_engine, medsam_predictor, preprocess = build_models(device, args)
    patient_items = collect_all_patient_dirs(SOURCE_ROOTS)

    if not patient_items:
        raise RuntimeError("No valid patient folders found under SOURCE_ROOTS.")

    rows = []
    total_images = 0
    skipped_low_conf = 0
    skipped_no_box = 0
    skipped_empty_mask = 0
    skipped_duplicate = 0
    saved_count = 0
    fieldnames = [
        "image_path",
        "pseudo_mask_path",
        "patient",
        "source_root",
        "pred_class",
        "pred_positive_prob",
        "box_xyxy",
    ]
    existing_image_paths = set()

    if os.path.isfile(args.csv_output):
        with open(args.csv_output, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row.get("image_path", "")
                if image_path:
                    existing_image_paths.add(os.path.normpath(image_path))
        print(f"[Info] Existing CSV entries: {len(existing_image_paths)}")

    for root_dir, patient_dir in patient_items:
        patient_name = os.path.basename(patient_dir)
        image_paths = list_image_paths(patient_dir)
        if not image_paths:
            continue

        print(f"[Info] Processing {patient_name}, images={len(image_paths)}")
        for img_path in image_paths:
            total_images += 1
            norm_img_path = os.path.normpath(img_path)
            if norm_img_path in existing_image_paths:
                skipped_duplicate += 1
                continue

            img_pil = safe_open_rgb(img_path)
            orig_w, orig_h = img_pil.size

            with torch.no_grad():
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_cls = int(torch.argmax(probs, dim=1).item())
                pos_prob = float(probs[0, args.positive_class_index].item())

            if pos_prob <= args.positive_threshold:
                skipped_low_conf += 1
                continue

            cam = cam_engine(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(args.positive_class_index)],
            )[0]
            cam = resize_cam(cam, args.crop_size)

            cam_mask = cam2mask(
                cam,
                low_thresh=args.cam_low,
                high_thresh=args.cam_high,
                min_area=args.cam_min_area,
                smooth_sigma=args.cam_sigma,
                dilate_iter=args.cam_dilate,
                test=False,
            ).astype(np.uint8)

            box = mask2bbox(
                cam_mask,
                min_area=args.bbox_min_area,
                box_enlarge=args.bbox_enlarge,
                mode=args.bbox_mode,
                clip=True,
                pre_dilate=args.bbox_pre_dilate,
                min_box_size=args.bbox_min_size,
            )
            box = ensure_boxes_2d(box)
            if box.shape[0] == 0:
                skipped_no_box += 1
                continue

            box_orig = scale_box_xyxy(
                box,
                src_w=args.crop_size,
                src_h=args.crop_size,
                dst_w=orig_w,
                dst_h=orig_h,
            )
            box_orig = ensure_boxes_2d(box_orig)
            if box_orig.shape[0] == 0:
                skipped_no_box += 1
                continue

            img_rgb = np.asarray(img_pil, dtype=np.uint8)
            pseudo_mask = bbox2sam_mask(
                medsam_predictor,
                img_rgb,
                box_orig,
                type="medsam",
                multimask_output=False,
            )

            if int((pseudo_mask > 0).sum()) == 0:
                skipped_empty_mask += 1
                continue

            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            out_mask_path = os.path.join(PSEUDO_MASK_ROOT, patient_name, f"{img_stem}_pseudo.png")
            save_binary_mask(pseudo_mask, out_mask_path)

            rows.append(
                {
                    "image_path": img_path,
                    "pseudo_mask_path": out_mask_path,
                    "patient": patient_name,
                    "source_root": root_dir,
                    "pred_class": pred_cls,
                    "pred_positive_prob": f"{pos_prob:.6f}",
                    "box_xyxy": ";".join(
                        [" ".join([str(int(round(v))) for v in b.tolist()]) for b in box_orig]
                    ),
                }
            )
            saved_count += 1
            existing_image_paths.add(norm_img_path)

            if saved_count % 50 == 0:
                print(f"[Info] Saved {saved_count} pseudo masks so far...")

    csv_exists = os.path.isfile(args.csv_output)
    write_header = (not csv_exists) or os.path.getsize(args.csv_output) == 0
    with open(args.csv_output, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print("-" * 72)
    print(f"[Done] Total images scanned: {total_images}")
    print(f"[Done] Saved pseudo masks: {saved_count}")
    print(f"[Done] Skipped by low confidence (<= {args.positive_threshold}): {skipped_low_conf}")
    print(f"[Done] Skipped by invalid/no bbox: {skipped_no_box}")
    print(f"[Done] Skipped by empty MedSAM mask: {skipped_empty_mask}")
    print(f"[Done] Skipped by duplicate CSV entries: {skipped_duplicate}")
    print(f"[Done] Appended CSV rows: {len(rows)}")
    print(f"[Done] CSV file: {args.csv_output}")


if __name__ == "__main__":
    main()
