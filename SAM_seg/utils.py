import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


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
    """
    type:
      - "val": 只计算 dice
      - "test": 计算 dice/precision/recall/hd/hd95
    """
    mode = str(type).lower()
    if mode == "val":
        return {"dice": dice_coefficient(pred_mask, gt_mask)}
    return evaluate_binary_metrics(pred_mask, gt_mask, spacing=spacing)


def dice_ci95(dice_values, type="test", confidence=0.95, n_bootstrap=2000, seed=42):
    """
    计算 DSC 均值及 95%CI (bootstrap percentile)。
    type:
      - "val": 只返回 dice_mean，不计算 CI
      - "test": 返回 dice_mean 和 dice_ci95
    """
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
