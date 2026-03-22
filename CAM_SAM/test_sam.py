import os
import argparse
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset.load_data import load_CVC_img_mask, load_kvasir_img_mask
from model.resnet import PretrainedResNet18_Encoder, Student_Head, FullResNet
from model.sam import build_sam_predictor, build_medsam_predictor, build_sammed2d_predictor
from utils import (
    add_label,
    bbox2sam_mask,
    blend_overlay,
    build_cls_transform,
    cam2mask,
    cam_to_heatmap,
    dice_score,
    draw_box,
    ensure_box_1d,
    load_checkpoint,
    load_image_pair,
    mask2bbox,
    mask_overlay,
)


def get_args():
    parser = argparse.ArgumentParser(description="ResNet Grad-CAM++ -> bbox -> SAM on CVC")
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
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
    parser.add_argument("--sam_model_type", default="vit_h", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--sam_ckpt", default="/mnt/nas1/disk03/zhaokaizhang/code/sam_vit_h_4b8939.pth", type=str, help="path to SAM checkpoint (.pth)")
    parser.add_argument("--medsam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--medsam_ckpt", default="/mnt/nas1/disk03/zhaokaizhang/code/medsam_vit_b.pth", type=str, help="path to MedSAM checkpoint (.pth)")
    parser.add_argument("--sammed2d_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--sammed2d_ckpt", default="/mnt/nas1/disk03/zhaokaizhang/code/sam-med2d_b.pth", type=str, help="path to SAM-Med2D checkpoint (.pth)")
    parser.add_argument("--sammed2d_image_size", default=256, type=int, help="SAM-Med2D image_size")
    parser.add_argument("--sammed2d_encoder_adapter", action="store_true", default=True, help="SAM-Med2D encoder_adapter")
    parser.add_argument("--min_id", default=1, type=int)
    parser.add_argument("--max_id", default=612, type=int)
    parser.add_argument("--cam_target", default=1, type=int, help="-1 means use predicted class")
    parser.add_argument("--out_dir", default="runs/cam_sam/resnet_gradcampp", type=str)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_ext", default="png", choices=["png", "jpg", "jpeg"])

    parser.add_argument("--cam_low", default=0.5, type=float)
    parser.add_argument("--cam_high", default=0.75, type=float)
    parser.add_argument("--cam_min_area", default=100, type=int)
    parser.add_argument("--cam_sigma", default=1.0, type=float)
    parser.add_argument("--cam_dilate", default=1, type=int)

    parser.add_argument("--bbox_min_area", default=50, type=int)
    parser.add_argument("--bbox_enlarge", default=0.05, type=float)
    parser.add_argument("--bbox_mode", default="largest", choices=["largest", "union", "all"])
    parser.add_argument("--bbox_pre_dilate", default=0, type=int)
    parser.add_argument("--bbox_min_size", default=2, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    if not args.sam_ckpt:
        raise ValueError("Please provide --sam_ckpt for SAM model.")
    if not os.path.isfile(args.sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_ckpt}")
    if not args.medsam_ckpt:
        raise ValueError("Please provide --medsam_ckpt for MedSAM model.")
    if not os.path.isfile(args.medsam_ckpt):
        raise FileNotFoundError(f"MedSAM checkpoint not found: {args.medsam_ckpt}")
    if not args.sammed2d_ckpt:
        raise ValueError("Please provide --sammed2d_ckpt for SAM-Med2D model.")
    if not os.path.isfile(args.sammed2d_ckpt):
        raise FileNotFoundError(f"SAM-Med2D checkpoint not found: {args.sammed2d_ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    encoder = PretrainedResNet18_Encoder(freeze=False).to(device)
    student = Student_Head(input_dim=512, num_classes=2).to(device)
    model = FullResNet(encoder, student).to(device)
    if args.encoder_ckpt:
        load_checkpoint(encoder, args.encoder_ckpt, device)
    if args.student_ckpt:
        load_checkpoint(student, args.student_ckpt, device)
    model.eval()
    encoder.eval()
    student.eval()

    target_layers = [encoder.features[-2]]
    cam_engine = GradCAMPlusPlus(model=model, target_layers=target_layers)
    preprocess = build_cls_transform(args.crop_size)

    sam_predictor = build_sam_predictor(
        checkpoint=args.sam_ckpt,
        device=device,
        model_type=args.sam_model_type,
    )
    medsam_predictor = build_medsam_predictor(
        checkpoint=args.medsam_ckpt,
        device=device,
        model_type=args.medsam_model_type,
    )
    sammed2d_predictor = build_sammed2d_predictor(
        checkpoint=args.sammed2d_ckpt,
        device=device,
        model_type=args.sammed2d_model_type,
        image_size=args.sammed2d_image_size,
        encoder_adapter=args.sammed2d_encoder_adapter,
    )

    img_paths, mask_paths = load_CVC_img_mask(min_id=args.min_id, max_id=args.max_id)
    # img_paths, mask_paths = load_kvasir_img_mask()
    if len(img_paths) == 0:
        print("No CVC image/mask pairs found.")
        return

    dice_cam_all = []
    dice_sam_all = []
    dice_medsam_all = []
    dice_sammed2d_all = []
    no_box_count = 0

    for idx, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths), start=1):
        img_pil, gt_pil = load_image_pair(img_path, mask_path, args.crop_size)
        img_rgb = np.array(img_pil)
        gt_mask = (np.array(gt_pil) > 127).astype(np.uint8)

        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_cls = int(torch.argmax(probs, dim=1).item())

        target_cls = pred_cls if args.cam_target < 0 else int(args.cam_target)
        cam = cam_engine(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target_cls)])[0]
        cam = np.clip(cam, 0, 1)
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((args.crop_size, args.crop_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

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
        box = ensure_box_1d(box)

        if box is None:
            no_box_count += 1
            sam_mask = np.zeros_like(cam_mask, dtype=np.uint8)
            medsam_mask = np.zeros_like(cam_mask, dtype=np.uint8)
            sammed2d_mask = np.zeros_like(cam_mask, dtype=np.uint8)
        else:
            sam_mask = bbox2sam_mask(
                sam_predictor,
                img_rgb,
                box,
                type="sam",
                multimask_output=True,
            )
            medsam_mask = bbox2sam_mask(
                medsam_predictor,
                img_rgb,
                box,
                type="medsam",
                multimask_output=False,
            )
            sammed2d_mask = bbox2sam_mask(
                sammed2d_predictor,
                img_rgb,
                box,
                type="sammed2d",
                multimask_output=True,
            )

        d_cam = dice_score(cam_mask > 0, gt_mask > 0)
        d_sam = dice_score(sam_mask > 0, gt_mask > 0)
        d_medsam = dice_score(medsam_mask > 0, gt_mask > 0)
        d_sammed2d = dice_score(sammed2d_mask > 0, gt_mask > 0)
        dice_cam_all.append(d_cam)
        dice_sam_all.append(d_sam)
        dice_medsam_all.append(d_medsam)
        dice_sammed2d_all.append(d_sammed2d)

        img_id = os.path.splitext(os.path.basename(img_path))[0]
        print(
            f"[{idx}/{len(img_paths)}] id={img_id} pred={pred_cls} "
            f"p1={probs[0, 1].item():.4f} dice_cam={d_cam:.4f} "
            f"dice_sam={d_sam:.4f} dice_medsam={d_medsam:.4f} dice_sammed2d={d_sammed2d:.4f} "
            f"box={'None' if box is None else [int(v) for v in box.tolist()]}"
        )

        if args.save_vis:
            gt_overlay = mask_overlay(img_rgb, gt_mask, color=(255, 0, 0), alpha=0.4)
            cam_overlay = blend_overlay(img_rgb, cam_to_heatmap(cam), alpha=0.5)
            cam_box_overlay = mask_overlay(img_rgb, cam_mask, color=(0, 255, 0), alpha=0.4)
            if box is not None:
                cam_box_overlay = draw_box(cam_box_overlay, box, color=(0, 0, 255), width=2)
            sam_overlay = mask_overlay(img_rgb, sam_mask, color=(255, 255, 0), alpha=0.4)
            medsam_overlay = mask_overlay(img_rgb, medsam_mask, color=(0, 255, 255), alpha=0.4)
            sammed2d_overlay = mask_overlay(img_rgb, sammed2d_mask, color=(255, 128, 0), alpha=0.4)

            p1 = add_label(img_rgb, "Original")
            p2 = add_label(gt_overlay, "GT")
            p3 = add_label(cam_overlay, "Grad-CAM++")
            p4 = add_label(cam_box_overlay, f"CAM mask + box | Dice={d_cam:.3f}")
            p5 = add_label(sam_overlay, f"SAM pred | Dice={d_sam:.3f}")
            p6 = add_label(medsam_overlay, f"MedSAM pred | Dice={d_medsam:.3f}")
            p7 = add_label(sammed2d_overlay, f"SAM-Med2D pred | Dice={d_sammed2d:.3f}")
            canvas = np.concatenate([p1, p2, p3, p4, p5, p6, p7], axis=1)
            out_path = os.path.join(args.out_dir, f"{img_id}_sam_compare.{args.save_ext}")
            Image.fromarray(canvas).save(out_path)

    mean_cam = float(np.mean(dice_cam_all)) if dice_cam_all else 0.0
    mean_sam = float(np.mean(dice_sam_all)) if dice_sam_all else 0.0
    mean_medsam = float(np.mean(dice_medsam_all)) if dice_medsam_all else 0.0
    mean_sammed2d = float(np.mean(dice_sammed2d_all)) if dice_sammed2d_all else 0.0
    print("-" * 60)
    print(f"Total samples: {len(img_paths)}")
    print(f"No-box samples: {no_box_count}")
    print(f"CAM-mask Dice mean: {mean_cam:.4f}")
    print(f"SAM-box Dice mean: {mean_sam:.4f}")
    print(f"MedSAM-box Dice mean: {mean_medsam:.4f}")
    print(f"SAM-Med2D-box Dice mean: {mean_sammed2d:.4f}")


if __name__ == "__main__":
    main()
