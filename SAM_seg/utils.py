import logging
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(work_dir: str, log_filename: str):
    log_file = os.path.join(work_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_p = logits.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = target_t == 1
    neg_index = target_t == 0

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    if sum_num.item() == 0:
        return F.binary_cross_entropy_with_logits(log_p, target_t, reduction="mean")

    weight = torch.zeros_like(log_p)
    weight[pos_index] = neg_num.float() / sum_num.float()
    weight[neg_index] = pos_num.float() / sum_num.float()

    return F.binary_cross_entropy_with_logits(log_p, target_t, weight=weight, reduction="mean")


def make_boundary_target(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).float()


def clip_gradient(optimizer, grad_clip: float):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def dice_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = (masks > 0.5).float()

    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.mean().item()), float(iou.mean().item())


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("model."):
            nk = nk[6:]
        cleaned[nk] = v
    return cleaned


def get_state_dict_from_checkpoint(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format.")


def load_model_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
    match_shape: bool = False,
):
    if not ckpt_path:
        return {"loaded": 0, "candidate": 0, "total": len(model.state_dict())}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"weights file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = normalize_state_dict_keys(get_state_dict_from_checkpoint(checkpoint))

    if match_shape:
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        if len(matched) == 0:
            raise RuntimeError(f"No matched keys found when loading weights: {ckpt_path}")
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        return {"loaded": len(matched), "candidate": len(state_dict), "total": len(model_dict)}

    model.load_state_dict(state_dict, strict=strict)
    return {"loaded": len(state_dict), "candidate": len(state_dict), "total": len(model.state_dict())}


def load_model_weights_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not ckpt_path:
        return {"loaded": 0, "candidate": 0, "total": len(model.state_dict())}
    try:
        return load_model_weights(model, ckpt_path, device, strict=True, match_shape=False)
    except RuntimeError:
        return load_model_weights(model, ckpt_path, device, strict=False, match_shape=False)


def decode_pidnet_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 3:
            aux_logits, main_logits, boundary_logits = outputs[0], outputs[1], outputs[2]
        elif len(outputs) == 2:
            aux_logits, main_logits = outputs[0], outputs[1]
            boundary_logits = None
        elif len(outputs) == 1:
            aux_logits, main_logits, boundary_logits = None, outputs[0], None
        else:
            aux_logits, main_logits, boundary_logits = None, None, None
    else:
        aux_logits, main_logits, boundary_logits = None, outputs, None

    if main_logits is None:
        raise RuntimeError("PIDNet returned no segmentation logits.")

    return aux_logits, main_logits, boundary_logits


def _to_bool_mask(mask, threshold=0.5):
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.dtype == np.bool_:
        return arr
    return arr > threshold


def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    pred = _to_bool_mask(pred_mask)
    gt = _to_bool_mask(gt_mask)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def precision_score(pred_mask, gt_mask, eps=1e-6):
    pred = _to_bool_mask(pred_mask)
    gt = _to_bool_mask(gt_mask)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score(pred_mask, gt_mask, eps=1e-6):
    pred = _to_bool_mask(pred_mask)
    gt = _to_bool_mask(gt_mask)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return float((tp + eps) / (tp + fn + eps))


def _surface_distances(src_mask, ref_mask, spacing=None):
    src = _to_bool_mask(src_mask)
    ref = _to_bool_mask(ref_mask)
    if src.shape != ref.shape:
        raise ValueError(f"Mask shape mismatch: {src.shape} vs {ref.shape}")

    if src.sum() == 0 or ref.sum() == 0:
        return np.array([], dtype=np.float64)

    footprint = generate_binary_structure(src.ndim, 1)
    src_border = np.logical_xor(src, binary_erosion(src, structure=footprint, border_value=0))
    ref_border = np.logical_xor(ref, binary_erosion(ref, structure=footprint, border_value=0))

    if not np.any(src_border) or not np.any(ref_border):
        return np.array([], dtype=np.float64)

    dt = distance_transform_edt(~ref_border, sampling=spacing)
    return dt[src_border]


def hausdorff_distance(pred_mask, gt_mask, spacing=None):
    pred = _to_bool_mask(pred_mask)
    gt = _to_bool_mask(gt_mask)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    d1 = _surface_distances(pred, gt, spacing=spacing)
    d2 = _surface_distances(gt, pred, spacing=spacing)
    if d1.size == 0 or d2.size == 0:
        return float("inf")
    return float(max(d1.max(), d2.max()))


def hausdorff_distance_95(pred_mask, gt_mask, spacing=None):
    pred = _to_bool_mask(pred_mask)
    gt = _to_bool_mask(gt_mask)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    d1 = _surface_distances(pred, gt, spacing=spacing)
    d2 = _surface_distances(gt, pred, spacing=spacing)
    all_d = np.concatenate([d1, d2], axis=0) if d1.size and d2.size else np.array([], dtype=np.float64)
    if all_d.size == 0:
        return float("inf")
    return float(np.percentile(all_d, 95))


def evaluate_binary_metrics(pred_mask, gt_mask, spacing=None):
    return {
        "dice": dice_coefficient(pred_mask, gt_mask),
        "precision": precision_score(pred_mask, gt_mask),
        "recall": recall_score(pred_mask, gt_mask),
        "hd": hausdorff_distance(pred_mask, gt_mask, spacing=spacing),
        "hd95": hausdorff_distance_95(pred_mask, gt_mask, spacing=spacing),
    }


def evaluate_metrics_with_type(pred_mask, gt_mask, type="test", spacing=None):
    mode = str(type).lower()
    if mode == "val":
        return {"dice": dice_coefficient(pred_mask, gt_mask)}
    return evaluate_binary_metrics(pred_mask, gt_mask, spacing=spacing)


def dice_ci95(dice_values, type="test", confidence=0.95, n_bootstrap=2000, seed=42):
    arr = np.asarray(dice_values, dtype=np.float64)
    if arr.size == 0:
        return {"dice_mean": float("nan"), "dice_ci95": None}

    mean_val = float(arr.mean())
    mode = str(type).lower()
    if mode == "val":
        return {"dice_mean": mean_val, "dice_ci95": None}

    if arr.size == 1:
        v = float(arr[0])
        return {"dice_mean": mean_val, "dice_ci95": (v, v)}

    confidence = float(confidence)
    confidence = min(max(confidence, 1e-6), 1.0 - 1e-6)
    n_bootstrap = max(100, int(n_bootstrap))

    rng = np.random.default_rng(int(seed))
    n = arr.size
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()

    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {"dice_mean": mean_val, "dice_ci95": (low, high)}
