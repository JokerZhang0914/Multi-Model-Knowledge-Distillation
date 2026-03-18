import logging
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from torchvision import transforms
import cv2  
from scipy import ndimage as ndi  


# =========================
# Checkpoint / Training
# =========================
def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logging.info(f"[*] Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    is_model_parallel = isinstance(model, nn.DataParallel)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        if name.startswith('model.'):
            name = name[6:]
        if is_model_parallel:
            name = 'module.' + name
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        logging.info("    -> Loaded successfully (Strict).")
    except RuntimeError as e:
        logging.warning(f"    [Warning] Strict loading failed, trying non-strict. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)


# =========================
# Metrics
# =========================
def dice_loss(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def dice_score(pred_mask, gt_mask, eps=1e-6):
    inter = (pred_mask & gt_mask).sum()
    return float((2 * inter) / (pred_mask.sum() + gt_mask.sum() + eps))


# =========================
# Image IO / Transform
# =========================
def build_cls_transform(crop_size):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def safe_open_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def safe_open_l(path):
    try:
        return Image.open(path).convert("L")
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return Image.fromarray(arr.astype(np.uint8)).convert("L")


def load_image_pair(img_path, mask_path, crop_size):
    img = safe_open_rgb(img_path).resize((crop_size, crop_size), Image.BILINEAR)
    mask = safe_open_l(mask_path).resize((crop_size, crop_size), Image.NEAREST)
    return img, mask


# =========================
# Visualization
# =========================
def add_label(img_rgb, text):
    out = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    pad = 5
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rectangle((0, 0, w + 2 * pad, h + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(out)


def blend_overlay(img_rgb, color_overlay, alpha=0.4):
    return (img_rgb.astype(np.float32) * (1 - alpha) + color_overlay.astype(np.float32) * alpha).astype(np.uint8)


def mask_overlay(img_rgb, mask, color=(0, 255, 0), alpha=0.4):
    overlay = np.zeros_like(img_rgb)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    out = blend_overlay(img_rgb, overlay, alpha=alpha)
    out[mask == 0] = img_rgb[mask == 0]
    return out


def draw_box(img_rgb, box, color=(0, 0, 255), width=2):
    out = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = [int(v) for v in box]
    for i in range(width):
        draw.rectangle((x0 - i, y0 - i, x1 + i, y1 + i), outline=color)
    return np.array(out)


def cam_to_heatmap(cam):
    cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat = np.zeros((cam_u8.shape[0], cam_u8.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = cam_u8
    heat[..., 1] = np.clip(255 - np.abs(cam_u8 - 128) * 2, 0, 255)
    heat[..., 2] = 255 - cam_u8
    return heat


# =========================
# CAM / Mask Post-process
# =========================
def cam2mask(cam, low_thresh=0.5, high_thresh=0.7, min_area=300, smooth_sigma=1.0, dilate_iter=1, test=False):
    """
    将 CAM 热图转为二值伪标签。
    思路：平滑 -> 双阈值滞后(高阈值种子 + 低阈值扩展) -> 小连通域去除 -> 形态学细化
    cam: np.ndarray, 值域可为 [0,1] 或 [0,255]
    test: True 时返回 (mask, vis, areas)，其中 vis 标注各连通域面积
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.max() > 1.0:
        cam = cam / 255.0
    cam = np.clip(cam, 0.0, 1.0)

    # 平滑抑制零星噪点
    if smooth_sigma and smooth_sigma > 0:
        if ndi is not None:
            cam = ndi.gaussian_filter(cam, smooth_sigma)
        elif cv2 is not None:
            k = int(max(3, smooth_sigma * 6) // 2 * 2 + 1)
            cam = cv2.GaussianBlur(cam, (k, k), smooth_sigma)

    high_mask = cam >= high_thresh
    low_mask = cam >= low_thresh

    # 低阈值区域中仅保留与高阈值连通的部分
    if ndi is not None:
        labeled, num = ndi.label(low_mask)
        if num > 0:
            high_labels = np.unique(labeled[high_mask])
            keep = np.isin(labeled, high_labels)
        else:
            keep = low_mask
    elif cv2 is not None:
        num, labeled = cv2.connectedComponents(low_mask.astype(np.uint8))
        if num > 1:
            high_labels = np.unique(labeled[high_mask.astype(bool)])
            keep = np.isin(labeled, high_labels)
        else:
            keep = low_mask
    else:
        keep = low_mask

    # 去除小区域
    if ndi is not None:
        labeled, num = ndi.label(keep)
        if num > 0:
            sizes = ndi.sum(keep, labeled, index=np.arange(1, num + 1))
            keep_ids = {i + 1 for i, s in enumerate(sizes) if s >= min_area}
            keep = np.isin(labeled, list(keep_ids))
    elif cv2 is not None:
        num, labeled = cv2.connectedComponents(keep.astype(np.uint8))
        if num > 1:
            counts = np.bincount(labeled.reshape(-1))
            keep_ids = set(i for i, c in enumerate(counts) if c >= min_area and i != 0)
            keep = np.isin(labeled, list(keep_ids))

    # 形态学闭运算 + 膨胀，补全外层黄色区域
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
        if dilate_iter > 0:
            keep = cv2.dilate(keep.astype(np.uint8), kernel, iterations=dilate_iter).astype(bool)
    elif ndi is not None:
        keep = ndi.binary_closing(keep, iterations=1)
        if dilate_iter > 0:
            keep = ndi.binary_dilation(keep, iterations=dilate_iter)

    mask = keep.astype(np.uint8)

    if not test:
        return mask

    # 生成可视化并标注面积
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask > 0] = (255, 255, 255)
    areas = []

    if ndi is not None:
        labeled, num = ndi.label(mask)
        for idx in range(1, num + 1):
            ys, xs = np.where(labeled == idx)
            if ys.size == 0:
                continue
            area = int(ys.size)
            areas.append(area)
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            vis[cy, cx] = (255, 0, 0)
    elif cv2 is not None:
        num, labeled = cv2.connectedComponents(mask.astype(np.uint8))
        for idx in range(1, num):
            ys, xs = np.where(labeled == idx)
            if ys.size == 0:
                continue
            area = int(ys.size)
            areas.append(area)
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            vis[cy, cx] = (255, 0, 0)
    else:
        if mask.sum() > 0:
            areas.append(int(mask.sum()))

    # 在 vis 上写面积文本
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(vis)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        if ndi is not None or cv2 is not None:
            if ndi is not None:
                labeled, num = ndi.label(mask)
                ids = range(1, num + 1)
            else:
                num, labeled = cv2.connectedComponents(mask.astype(np.uint8))
                ids = range(1, num)

            for idx in ids:
                ys, xs = np.where(labeled == idx)
                if ys.size == 0:
                    continue
                area = int(ys.size)
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                text = f"{area}"
                draw.text((cx, cy), text, fill=(255, 0, 0), font=font)
        else:
            if mask.sum() > 0:
                draw.text((2, 2), f"{int(mask.sum())}", fill=(255, 0, 0), font=font)
        vis = np.array(img)
    except Exception:
        pass

    return mask, vis, areas


# =========================
# BBox / SAM
# =========================
def ensure_box_1d(box):
    box = np.asarray(box)
    if box.size < 4:
        return None
    if box.ndim == 2:
        box = box[0]
    box = box.astype(np.float32).reshape(-1)
    if box.shape[0] < 4:
        return None
    x0, y0, x1, y1 = box[:4]
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask2bbox(
    mask,
    min_area=50,
    box_enlarge=0.1,
    mode="largest",
    clip=True,
    pre_dilate=0,
    min_box_size=2,
    return_areas=False,
):
    """
    从二值 mask 提取 bbox，支持多连通域处理与盒子放大。
    返回格式: [x0, y0, x1, y1] (xyxy, 右下为开区间)
    mode:
      - "largest": 返回面积最大的单个 box
      - "union": 返回所有 box 的并集
      - "all": 返回所有 box (按面积降序)
    """
    if mask is None:
        return (np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.int64)) if return_areas else np.zeros((0, 4), dtype=np.int32)

    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    mask = mask.astype(bool)

    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return (np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.int64)) if return_areas else np.zeros((0, 4), dtype=np.int32)

    # Optional pre-dilate to avoid overly tight boxes
    if pre_dilate and pre_dilate > 0:
        if cv2 is not None:
            k = int(pre_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        elif ndi is not None:
            mask = ndi.binary_dilation(mask, iterations=int(pre_dilate))

    boxes = []
    areas = []

    if ndi is not None:
        labeled, num = ndi.label(mask)
        if num > 0:
            slices = ndi.find_objects(labeled)
            for idx, slc in enumerate(slices, start=1):
                if slc is None:
                    continue
                ys, xs = slc[0], slc[1]
                area = int((labeled[ys, xs] == idx).sum())
                if area < min_area:
                    continue
                y0, y1 = ys.start, ys.stop
                x0, x1 = xs.start, xs.stop
                if (x1 - x0) < min_box_size or (y1 - y0) < min_box_size:
                    continue
                boxes.append([x0, y0, x1, y1])
                areas.append(area)
    elif cv2 is not None:
        num, labeled = cv2.connectedComponents(mask.astype(np.uint8))
        for idx in range(1, num):
            ys, xs = np.where(labeled == idx)
            if ys.size == 0:
                continue
            area = int(ys.size)
            if area < min_area:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            if (x1 - x0) < min_box_size or (y1 - y0) < min_box_size:
                continue
            boxes.append([x0, y0, x1, y1])
            areas.append(area)
    else:
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            area = int(ys.size)
            if area >= min_area and (x1 - x0) >= min_box_size and (y1 - y0) >= min_box_size:
                boxes.append([x0, y0, x1, y1])
                areas.append(area)

    if not boxes:
        empty_boxes = np.zeros((0, 4), dtype=np.int32)
        empty_areas = np.zeros((0,), dtype=np.int64)
        return (empty_boxes, empty_areas) if return_areas else empty_boxes

    boxes = np.asarray(boxes, dtype=np.float32)
    areas = np.asarray(areas, dtype=np.int64)
    order = np.argsort(areas)[::-1]
    boxes = boxes[order]
    areas = areas[order]

    def _enlarge_box(box):
        x0, y0, x1, y1 = box.tolist()
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        if isinstance(box_enlarge, (int, np.integer)):
            pad_w = float(box_enlarge)
            pad_h = float(box_enlarge)
        else:
            pad_w = float(box_enlarge) * bw
            pad_h = float(box_enlarge) * bh
        x0 -= pad_w
        y0 -= pad_h
        x1 += pad_w
        y1 += pad_h
        if clip:
            x0 = max(0.0, x0)
            y0 = max(0.0, y0)
            x1 = min(float(w), x1)
            y1 = min(float(h), y1)
        if x1 <= x0:
            x1 = min(float(w), x0 + 1.0)
        if y1 <= y0:
            y1 = min(float(h), y0 + 1.0)
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    if mode == "all":
        out_boxes = np.stack([_enlarge_box(b) for b in boxes], axis=0)
    elif mode == "union":
        x0 = boxes[:, 0].min()
        y0 = boxes[:, 1].min()
        x1 = boxes[:, 2].max()
        y1 = boxes[:, 3].max()
        out_boxes = _enlarge_box(np.array([x0, y0, x1, y1], dtype=np.float32))
    else:  # "largest"
        out_boxes = _enlarge_box(boxes[0])

    if return_areas:
        return out_boxes.astype(np.int32), areas
    return out_boxes.astype(np.int32)


def bbox2sam_mask(
    predictor,
    image,
    box,
    multimask_output=None,
    return_score=False,
    type="sam",
):
    """
    使用 SAM predictor 将 bbox (xyxy) 转为分割 mask。
    输入:
      predictor: segment_anything.SamPredictor
      image: HxWxC, uint8/rgb
      box: [x0, y0, x1, y1] 或 (N,4) (会取第一个)
      type: "sam" / "medsam" / "sammed2d"
    返回:
      mask: uint8, HxW, {0,1}
      可选返回 best_score
    """
    prompt_type = str(type).lower()
    if prompt_type not in ("sam", "medsam", "sammed2d"):
        raise ValueError(f"Unsupported type: {type}. Expected 'sam'/'medsam'/'sammed2d'.")
    if multimask_output is None:
        # SAM / SAM-Med2D 默认多候选，MedSAM 通常使用单候选
        multimask_output = (prompt_type in ("sam", "sammed2d"))

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    if predictor is None or box is None:
        return (empty, 0.0) if return_score else empty

    box = np.asarray(box)
    if box.size < 4:
        return (empty, 0.0) if return_score else empty
    if box.ndim == 2:
        box = box[0]
    box = box.astype(np.float32).reshape(-1)
    if box.shape[0] < 4:
        return (empty, 0.0) if return_score else empty

    x0, y0, x1, y1 = box[:4]
    x0 = float(np.clip(x0, 0, max(0, w - 1)))
    y0 = float(np.clip(y0, 0, max(0, h - 1)))
    x1 = float(np.clip(x1, 0, w))
    y1 = float(np.clip(y1, 0, h))
    if x1 <= x0 or y1 <= y0:
        return (empty, 0.0) if return_score else empty
    box_xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)

    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_xyxy,
        multimask_output=bool(multimask_output),
        return_logits=False,
    )

    if masks is None:
        return (empty, 0.0) if return_score else empty

    masks = np.asarray(masks)
    if masks.ndim == 2:
        best_mask = masks
        best_score = float(scores[0]) if scores is not None and len(scores) > 0 else 0.0
    elif masks.ndim == 3 and masks.shape[0] > 0:
        best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx]) if scores is not None and len(scores) > 0 else 0.0
    else:
        return (empty, 0.0) if return_score else empty

    best_mask = (best_mask > 0).astype(np.uint8)
    if return_score:
        return best_mask, best_score
    return best_mask
