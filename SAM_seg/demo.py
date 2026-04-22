import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")
UNET_ROOT = os.path.join(THIS_DIR, "model", "UNet")
PIDNET_ROOT = os.path.join(THIS_DIR, "model", "PIDNet")
FCB_ROOT = os.path.join(THIS_DIR, "model", "FCBFormer")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

# Keep SAM_seg paths ahead of model subpaths so `from utils import ...`
# resolves to SAM_seg/utils.py instead of model-specific `utils` packages.
for p in [PVT_ROOT, UNET_ROOT, PIDNET_ROOT, FCB_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

from dataset import open_l, open_rgb
from SAM_seg.utils import (
    decode_pidnet_outputs,
    dice_coefficient,
    get_state_dict_from_checkpoint,
    load_model_weights_flexible,
    normalize_state_dict_keys,
)

from lib.pvt import PolypPVT
from unet import UNet
from models.pidnet import PIDNet
from Models import models as fcb_models


def get_args():
    parser = argparse.ArgumentParser(description="Single-image segmentation demo")
    parser.add_argument(
        "--image_path", 
        default="data/seg_data/TestDataset/ETIS-LaribPolypDB/images/25.png",
        type=str, help="input image path")
    parser.add_argument(
        "--mask_path", 
        default="data/seg_data/TestDataset/ETIS-LaribPolypDB/masks/25.png",
        type=str, help="ground-truth mask path")
    parser.add_argument(
        "--model",
        default="fcbformer",
        type=str,
        choices=["unet", "pidnet", "fcbformer", "polyp-pvt"],
        help="choose one segmentation model",
    )
    parser.add_argument(
        "--weights", 
        default="/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/checkpoint/FCB_Train_on_KvasirandDB30_Best.pt", 
        type=str, 
        help="checkpoint path; empty means model-specific default")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--alpha", default=0.45, type=float, help="overlay alpha in [0,1]")
    parser.add_argument("--save_path", default="runs/compare/", type=str, help="optional output image path")
    parser.add_argument("--no_show", action="store_true", help="disable matplotlib window")

    parser.add_argument("--unet_n_channels", default=3, type=int)
    parser.add_argument("--unet_n_classes", default=1, type=int)
    parser.add_argument(
        "--unet_bilinear",
        default="False",
        type=str,
        choices=["True", "False", "true", "false", "1", "0"],
    )
    parser.add_argument("--pidnet_num_classes", default=2, type=int)
    parser.add_argument("--pvt_backbone_pretrained", default="", type=str)
    parser.add_argument(
        "--fcb_pvt_b3_weights",
        default="SAM_seg/model/FCBFormer/Models/pvt_v2_b3.pth",
        type=str,
        help="path to pvt_v2_b3.pth for FCBFormer backbone init",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _default_weights_for_model(model_name: str) -> str:
    mapping = {
        "unet": "runs/seg_unet/2026-0326-2156_public_CVC-ColonDB_unet/checkpoint/best_unet5093.pth",
        "pidnet": "runs/seg_pids/2026-0326-1224_public_CVC-ColonDB/checkpoint/best_pidnet_s6943.pth",
        "fcbformer": "runs/seg_fcb/2026-0408-1605_csv_all_fcb/checkpoint/best_fcb.pth",
        "polyp-pvt": "runs/checkpoint/zero_shot_Sota_53PolypPVT.pth",
    }
    return mapping[model_name]


def _resolve_weights_arg(model_name: str, user_weights: str) -> str:
    if user_weights and user_weights.strip():
        ckpt = resolve_path(user_weights.strip())
    else:
        ckpt = resolve_path(_default_weights_for_model(model_name))
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def _load_weights_for_model(model, ckpt_path: str, device: torch.device, model_name: str):
    try:
        info = load_model_weights_flexible(model, ckpt_path, device)
        return info
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
        return {"loaded": len(matched), "candidate": len(state_dict), "total": len(model_dict)}


def _infer_pidnet_num_classes_from_ckpt(ckpt_path: str, device: torch.device):
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


def _resolve_fcb_pvt_weight_path(user_path: str) -> str:
    candidates = []
    if user_path and user_path.strip():
        candidates.append(resolve_path(user_path.strip()))
    candidates.extend(
        [
            os.path.join(FCB_ROOT, "pvt_v2_b3.pth"),
            os.path.join(os.getcwd(), "pvt_v2_b3.pth"),
            os.path.join(PROJECT_ROOT, "pvt_v2_b3.pth"),
        ]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    checked = "\n".join([f"  - {os.path.abspath(x)}" for x in candidates])
    raise FileNotFoundError(
        "PVT-B3 pretrained weight not found for FCBFormer.\n"
        "Please provide --fcb_pvt_b3_weights /path/to/pvt_v2_b3.pth.\n"
        f"Checked:\n{checked}"
    )


def _build_fcbformer_with_pvt(trainsize: int, pvt_weight_path: str):
    pvt_weight_path = os.path.abspath(pvt_weight_path)
    base = os.path.basename(pvt_weight_path)
    old_cwd = os.getcwd()
    temp_dir = None
    build_cwd = os.path.dirname(pvt_weight_path)
    try:
        if base != "pvt_v2_b3.pth":
            temp_dir = tempfile.TemporaryDirectory()
            link_path = os.path.join(temp_dir.name, "pvt_v2_b3.pth")
            try:
                os.symlink(pvt_weight_path, link_path)
            except OSError:
                shutil.copy2(pvt_weight_path, link_path)
            build_cwd = temp_dir.name
        os.chdir(build_cwd)
        model = fcb_models.FCBFormer(size=trainsize)
    finally:
        os.chdir(old_cwd)
        if temp_dir is not None:
            temp_dir.cleanup()
    return model


def _build_model(args, device: torch.device):
    ckpt_path = _resolve_weights_arg(args.model, args.weights)

    if args.model == "unet":
        model = UNet(
            n_channels=int(args.unet_n_channels),
            n_classes=int(args.unet_n_classes),
            bilinear=str2bool(args.unet_bilinear),
        ).to(device)
        load_info = _load_weights_for_model(model, ckpt_path, device, model_name="unet")
        model.eval()
        return model, ckpt_path, load_info, {"num_classes": int(args.unet_n_classes)}

    if args.model == "pidnet":
        num_classes = int(args.pidnet_num_classes)
        inferred = _infer_pidnet_num_classes_from_ckpt(ckpt_path, device)
        if inferred is not None and inferred != num_classes:
            print(
                f"[PIDNet] override num_classes: {num_classes} -> {inferred} "
                f"(inferred from checkpoint)"
            )
            num_classes = inferred
        model = PIDNet(
            m=2,
            n=3,
            num_classes=num_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        ).to(device)
        load_info = _load_weights_for_model(model, ckpt_path, device, model_name="pidnet")
        model.eval()
        return model, ckpt_path, load_info, {"num_classes": num_classes}

    if args.model == "fcbformer":
        pvt_path = _resolve_fcb_pvt_weight_path(args.fcb_pvt_b3_weights)
        model = _build_fcbformer_with_pvt(args.trainsize, pvt_path).to(device)
        load_info = _load_weights_for_model(model, ckpt_path, device, model_name="fcbformer")
        model.eval()
        return model, ckpt_path, load_info, {"pvt_b3": pvt_path}

    if args.model == "polyp-pvt":
        backbone_pretrained = args.pvt_backbone_pretrained.strip()
        if backbone_pretrained:
            backbone_pretrained = resolve_path(backbone_pretrained)
            if not os.path.isfile(backbone_pretrained):
                raise FileNotFoundError(f"--pvt_backbone_pretrained not found: {backbone_pretrained}")
            model = PolypPVT(pretrained_path=backbone_pretrained).to(device)
        else:
            model = PolypPVT(pretrained_path=None).to(device)
        load_info = _load_weights_for_model(model, ckpt_path, device, model_name="polyp-pvt")
        model.eval()
        return model, ckpt_path, load_info, {"backbone_pretrained": backbone_pretrained or None}

    raise ValueError(f"Unsupported model: {args.model}")


def _forward_binary_logits(model, model_name: str, x: torch.Tensor, target_size):
    if model_name == "unet":
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
    elif model_name == "fcbformer":
        out = model(x)
        if out.shape[1] == 1:
            logits = out
        elif out.shape[1] >= 2:
            logits = out[:, 1:2] - out[:, 0:1]
        else:
            raise RuntimeError("FCBFormer returned invalid channel count.")
    elif model_name == "polyp-pvt":
        p1, p2 = model(x)
        out = p1 + p2
        if out.shape[1] == 1:
            logits = out
        elif out.shape[1] >= 2:
            logits = out[:, 1:2] - out[:, 0:1]
        else:
            raise RuntimeError("Polyp-PVT returned invalid channel count.")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def _render_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, dice: float, model_name: str, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = np.asarray(image_rgb, dtype=np.uint8)
    gt = np.asarray(gt_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    gt_only = np.logical_and(gt, np.logical_not(pred))
    pred_only = np.logical_and(pred, np.logical_not(gt))
    overlap = np.logical_and(gt, pred)

    out = base.astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    yellow = np.array([255.0, 255.0, 0.0], dtype=np.float32)

    out[gt_only] = (1.0 - alpha) * out[gt_only] + alpha * red
    out[pred_only] = (1.0 - alpha) * out[pred_only] + alpha * green
    out[overlap] = (1.0 - alpha) * out[overlap] + alpha * yellow
    out = np.clip(out, 0, 255).astype(np.uint8)

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    text = f"model={model_name} | dice={dice:.4f} | red=gt green=pred yellow=overlap"
    pad = 6
    text_h = 22
    draw.rectangle((0, 0, pil.width, text_h + pad), fill=(0, 0, 0))
    draw.text((pad, 4), text, fill=(255, 255, 255))
    return np.array(pil)


def _show_with_matplotlib(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    model_name: str,
    dice: float,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warning] matplotlib unavailable, skip show: {e}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay_rgb)
    axes[3].set_title(f"Overlay | {model_name} | Dice={dice:.4f}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def _resolve_output_image_path(save_path: str, image_path: str, model_name: str) -> str:
    out_path = resolve_path(save_path)

    # Directory-like input: save into that directory with an auto file name.
    if out_path.endswith(os.sep) or os.path.isdir(out_path):
        img_stem = os.path.splitext(os.path.basename(image_path))[0] or "demo"
        out_path = os.path.join(out_path, f"{img_stem}_{model_name}_overlay.png")

    # No extension: default to PNG to avoid PIL save format errors.
    root, ext = os.path.splitext(out_path)
    if ext == "":
        out_path = f"{root}.png"
    return out_path


def run_demo(args):
    image_path = resolve_path(args.image_path)
    mask_path = resolve_path(args.mask_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"--image_path not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"--mask_path not found: {mask_path}")

    if torch.cuda.is_available():
        gpu_index = int(str(args.gpu).split(",")[0].strip())
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    model, ckpt_path, load_info, extra = _build_model(args, device)

    image_pil = open_rgb(image_path)
    gt_pil = open_l(mask_path).resize(image_pil.size, Image.NEAREST)
    gt_mask = (np.array(gt_pil) > 127).astype(np.uint8)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.trainsize, args.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        x = preprocess(image_pil).unsqueeze(0).to(device)
        logits = _forward_binary_logits(
            model,
            args.model,
            x,
            target_size=(image_pil.height, image_pil.width),
        )
        pred_mask = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() > float(args.threshold)).astype(np.uint8)

    dice = float(dice_coefficient(pred_mask, gt_mask))
    vis = _render_overlay(np.asarray(image_pil, dtype=np.uint8), gt_mask, pred_mask, dice=dice, model_name=args.model, alpha=args.alpha)

    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"checkpoint: {ckpt_path}")
    print(f"image: {image_path}")
    print(f"mask: {mask_path}")
    print(f"Loaded params: {load_info['loaded']}/{load_info['candidate']} (total={load_info['total']})")
    if args.model == "fcbformer":
        print(f"PVT-B3: {extra['pvt_b3']}")
    if args.model == "pidnet":
        print(f"Num classes: {extra['num_classes']}")
    print(f"Dice: {dice:.6f}")

    if args.save_path:
        out_path = _resolve_output_image_path(args.save_path, image_path=image_path, model_name=args.model)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(vis).save(out_path)
        print(f"Saved overlay: {out_path}")

    _show_with_matplotlib(
        image_rgb=np.asarray(image_pil, dtype=np.uint8),
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        overlay_rgb=vis,
        model_name=args.model,
        dice=dice,
    )


def main():
    args = get_args()
    run_demo(args)


if __name__ == "__main__":
    main()
