import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import torch
from torchvision import transforms

from dataset.load_data import load_CVC_img_mask
from utils import cam2mask, load_checkpoint
from model.swin_transformer import SwinTransformer
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser(description="Swin Transformer Grad-CAM on test set")
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--embed_dim", default=96, type=int)
    parser.add_argument("--depths", default="2,2,6,2", type=str)
    parser.add_argument("--num_heads", default="3,6,12,24", type=str)
    parser.add_argument("--window_size", default=7, type=int)
    parser.add_argument("--mlp_ratio", default=4.0, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--attn_drop_rate", default=0.0, type=float)
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    parser.add_argument("--ape", action="store_true")
    parser.add_argument("--patch_norm", action="store_true", default=True)
    parser.add_argument("--use_checkpoint", action="store_true")

    parser.add_argument("--ckpt", default="/mnt/nas1/disk03/zhaokaizhang/code/test_code/runs/cam_swin/2026-0311-1805_gradcampp/checkpoint/best_swin.pth", type=str)
    parser.add_argument("--min_id", default=140, type=int)
    parser.add_argument("--max_id", default=145, type=int)
    parser.add_argument("--out_dir", default="runs/cam/swin", type=str)
    parser.add_argument("--cam_low", default=0.5, type=float)
    parser.add_argument("--cam_high", default=0.75, type=float)
    parser.add_argument("--cam_min_area", default=100, type=int)
    parser.add_argument("--cam_sigma", default=1.0, type=float)
    parser.add_argument("--cam_dilate", default=1, type=int)
    parser.add_argument("--no_cam_test", action="store_true", help="disable area labels")
    parser.add_argument("--save_ext", default="png", choices=["png", "jpg", "jpeg"])
    return parser.parse_args()


def build_transforms(crop_size):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _safe_open_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def _safe_open_l(path):
    try:
        return Image.open(path).convert("L")
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return Image.fromarray(arr.astype(np.uint8)).convert("L")


def load_image_pair(img_path, mask_path, crop_size):
    img = _safe_open_rgb(img_path)
    mask = _safe_open_l(mask_path)
    img = img.resize((crop_size, crop_size), Image.BILINEAR)
    mask = mask.resize((crop_size, crop_size), Image.NEAREST)
    return img, mask


def cam_to_heatmap(cam):
    cam_u8 = (cam * 255).astype(np.uint8)
    heat = np.zeros((cam_u8.shape[0], cam_u8.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = cam_u8
    heat[..., 1] = np.clip(255 - np.abs(cam_u8 - 128) * 2, 0, 255)
    heat[..., 2] = 255 - cam_u8
    return heat


def blend_overlay(img_rgb, overlay, alpha=0.5):
    return (img_rgb.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)


def add_label(img_rgb, text):
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box = (0, 0, text_w + pad * 2, text_h + pad * 2)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def generate_cam(engine, input_tensor, pred):
    targets = [ClassifierOutputTarget(pred)]
    cam = engine(input_tensor=input_tensor, targets=targets)[0]
    cam = np.clip(cam, 0, 1)
    return cam


def mask_overlay(img_rgb, mask, color=(0, 255, 0), alpha=0.4):
    overlay = img_rgb.copy()
    color_arr = np.zeros_like(overlay)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    overlay = blend_overlay(overlay, color_arr, alpha=alpha)
    overlay[mask == 0] = img_rgb[mask == 0]
    return overlay


def make_colorbar(height, width=48):
    vals = np.linspace(1.0, 0.0, height).reshape(height, 1)
    bar = cam_to_heatmap(vals)
    bar = np.array(Image.fromarray(bar).resize((width, height), Image.NEAREST))
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((2, 2), "1.0", fill=(255, 255, 255), font=font)
    draw.text((2, height // 2 - 6), "0.5", fill=(255, 255, 255), font=font)
    draw.text((2, height - 12), "0.0", fill=(255, 255, 255), font=font)
    return np.array(img)


def cam_panel(cam, title, bar_width=48):
    heat = cam_to_heatmap(cam)
    heat = add_label(heat, title)
    bar = make_colorbar(heat.shape[0], width=bar_width)
    return np.concatenate([heat, bar], axis=1)


def build_model(args, device):
    depths = tuple(int(x) for x in args.depths.split(','))
    num_heads = tuple(int(x) for x in args.num_heads.split(','))
    model = SwinTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=2,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        ape=args.ape,
        patch_norm=args.patch_norm,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    if args.ckpt:
        load_checkpoint(model, args.ckpt, device)
    model.eval()
    return model


def get_cam_hw(model):
    h, w = model.layers[-1].input_resolution
    return int(h), int(w)


def reshape_transform(tensor, model):
    h, w = get_cam_hw(model)
    if tensor.ndim == 3:
        b, l, c = tensor.shape
        if l == h * w:
            tensor = tensor.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return tensor


def main():
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.img_size is None:
        args.img_size = args.crop_size
    if args.img_size != args.crop_size:
        args.img_size = args.crop_size
    if args.img_size % args.patch_size != 0:
        raise ValueError(f"img_size ({args.img_size}) must be divisible by patch_size ({args.patch_size}).")
    patches_res = args.img_size // args.patch_size
    if patches_res % args.window_size != 0:
        new_w = None
        for w in range(args.window_size, 0, -1):
            if patches_res % w == 0:
                new_w = w
                break
        if new_w is None:
            new_w = 1
        args.window_size = new_w

    model = build_model(args, device)

    target_layers = [model.layers[-1]]
    gradcampp_engine = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers,
        reshape_transform=lambda t: reshape_transform(t, model)
    )
    layercam_engine = LayerCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=lambda t: reshape_transform(t, model)
    )

    img_paths, mask_paths = load_CVC_img_mask(min_id=args.min_id, max_id=args.max_id)
    if len(img_paths) == 0:
        print("No test images found.")
        return

    preprocess = build_transforms(args.crop_size)
    total = len(img_paths)
    correct = 0
    cam_test = not args.no_cam_test

    for img_path, mask_path in zip(img_paths, mask_paths):
        img_pil, mask_pil = load_image_pair(img_path, mask_path, args.crop_size)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        cam_gcpp = generate_cam(gradcampp_engine, img_tensor, pred)
        cam_layer = generate_cam(layercam_engine, img_tensor, pred)

        def resize_cam(cam):
            return np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize((args.crop_size, args.crop_size), Image.BILINEAR)
            ) / 255.0

        cam_gcpp = resize_cam(cam_gcpp)
        cam_layer = resize_cam(cam_layer)

        img_rgb = np.array(img_pil)
        mask = np.array(mask_pil) > 127
        gt_overlay = img_rgb.copy()
        red_mask = np.zeros_like(img_rgb)
        red_mask[..., 0] = 255
        gt_overlay = blend_overlay(gt_overlay, red_mask, alpha=0.4)
        gt_overlay[~mask] = img_rgb[~mask]

        cam_panel_gcpp = cam_panel(cam_gcpp, "Grad-CAM++ CAM")
        cam_panel_layer = cam_panel(cam_layer, "Layer-CAM CAM")

        if cam_test:
            mask_gcpp, vis_gcpp, areas_gcpp = cam2mask(
                cam_gcpp,
                low_thresh=args.cam_low,
                high_thresh=args.cam_high,
                min_area=args.cam_min_area,
                smooth_sigma=args.cam_sigma,
                dilate_iter=args.cam_dilate,
                test=True
            )
            mask_layer, vis_layer, areas_layer = cam2mask(
                cam_layer,
                low_thresh=args.cam_low,
                high_thresh=args.cam_high,
                min_area=args.cam_min_area,
                smooth_sigma=args.cam_sigma,
                dilate_iter=args.cam_dilate,
                test=True
            )
            panel5 = add_label(vis_gcpp, f"Grad-CAM++ Mask (n={len(areas_gcpp)})")
            panel6 = add_label(vis_layer, f"Layer-CAM Mask (n={len(areas_layer)})")
        else:
            mask_gcpp = cam2mask(
                cam_gcpp,
                low_thresh=args.cam_low,
                high_thresh=args.cam_high,
                min_area=args.cam_min_area,
                smooth_sigma=args.cam_sigma,
                dilate_iter=args.cam_dilate,
                test=False
            )
            mask_layer = cam2mask(
                cam_layer,
                low_thresh=args.cam_low,
                high_thresh=args.cam_high,
                min_area=args.cam_min_area,
                smooth_sigma=args.cam_sigma,
                dilate_iter=args.cam_dilate,
                test=False
            )
            panel5 = add_label(mask_overlay(img_rgb, mask_gcpp), "Grad-CAM++ Mask")
            panel6 = add_label(mask_overlay(img_rgb, mask_layer), "Layer-CAM Mask")

        panel1 = add_label(img_rgb, "Original")
        panel2 = add_label(gt_overlay, "GroundTruth")

        row1 = np.concatenate([panel1, panel2, cam_panel_gcpp], axis=1)
        row2 = np.concatenate([cam_panel_layer, panel5, panel6], axis=1)
        concat = np.concatenate([row1, row2], axis=0)

        img_id = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, f"{img_id}_compare.{args.save_ext}")
        Image.fromarray(concat).save(out_path)

        if pred == 1:
            correct += 1

        print(f"{img_id} pred={pred} prob={probs[0, pred].item():.4f}")

    acc = correct / total
    print(f"ACC: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
