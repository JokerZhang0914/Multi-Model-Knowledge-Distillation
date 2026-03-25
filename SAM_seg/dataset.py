import csv
import os
import random
from glob import glob
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def is_image_file(filename):
    img_extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    return filename.lower().endswith(img_extensions)


def natural_sort_key(text: str):
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - vmin) / (vmax - vmin)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def open_tif_fallback(path: str, mode="RGB"):
    arr = None

    try:
        import tifffile

        arr = tifffile.imread(path)
    except Exception:
        arr = None

    if arr is None:
        try:
            import imageio.v2 as imageio

            arr = imageio.imread(path)
        except Exception as e:
            raise RuntimeError(f"Failed to decode TIFF file: {path}") from e

    arr = np.asarray(arr)
    if arr.ndim == 2:
        pil = Image.fromarray(to_uint8(arr), mode="L")
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            pil = Image.fromarray(to_uint8(arr[..., 0]), mode="L")
        else:
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            pil = Image.fromarray(to_uint8(arr), mode="RGB")
    else:
        raise RuntimeError(f"Unsupported TIFF ndim={arr.ndim}: {path}")

    return pil.convert(mode)


def open_rgb(path: str):
    try:
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")
    except Exception as pil_error:
        try:
            return open_tif_fallback(path, mode="RGB")
        except Exception:
            raise pil_error


def open_l(path: str):
    try:
        with open(path, "rb") as f:
            return Image.open(f).convert("L")
    except Exception as pil_error:
        try:
            return open_tif_fallback(path, mode="L")
        except Exception:
            raise pil_error


def pair_image_mask_from_dirs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    img_files = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if is_image_file(name) and os.path.isfile(os.path.join(image_dir, name))
    ]
    mask_files = [
        os.path.join(mask_dir, name)
        for name in os.listdir(mask_dir)
        if is_image_file(name) and os.path.isfile(os.path.join(mask_dir, name))
    ]

    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_files}
    common = sorted(set(img_map.keys()) & set(mask_map.keys()), key=natural_sort_key)

    return [(img_map[k], mask_map[k]) for k in common]


def load_pairs_from_root(
    root_dir: str,
    candidates: Sequence[Tuple[str, str]],
    name: str,
    verbose: bool = True,
    check_readable: bool = False,
) -> List[Tuple[str, str]]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{name} root does not exist: {root_dir}")

    used_dirs: Optional[Tuple[str, str]] = None
    pairs: List[Tuple[str, str]] = []
    for image_subdir, mask_subdir in candidates:
        image_dir = os.path.join(root_dir, image_subdir)
        mask_dir = os.path.join(root_dir, mask_subdir)
        if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
            pairs = pair_image_mask_from_dirs(image_dir, mask_dir)
            used_dirs = (image_dir, mask_dir)
            if pairs:
                break

    if used_dirs is None:
        cand_desc = ", ".join([f"({i},{m})" for i, m in candidates])
        raise RuntimeError(f"{name}: no valid image/mask dirs in {root_dir}, candidates={cand_desc}")

    if len(pairs) == 0:
        raise RuntimeError(f"{name}: found dirs but no valid paired samples in {used_dirs[0]} and {used_dirs[1]}")

    if check_readable:
        valid_pairs: List[Tuple[str, str]] = []
        skipped = 0
        for img_path, mask_path in pairs:
            try:
                _ = open_rgb(img_path)
                _ = open_l(mask_path)
                valid_pairs.append((img_path, mask_path))
            except Exception:
                skipped += 1
        if skipped > 0 and verbose:
            print(f"[{name}] skipped unreadable pairs: {skipped}")
        pairs = valid_pairs
        if len(pairs) == 0:
            raise RuntimeError(f"{name}: no readable image/mask pairs after validation.")

    if verbose:
        print(f"[*] {name}: {len(pairs)} pairs from {used_dirs[0]} and {used_dirs[1]}")
    return pairs


def read_csv_pairs(csv_path: str) -> List[Tuple[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    pairs = []
    missing = 0
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames or "pseudo_mask_path" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: image_path, pseudo_mask_path")
        for row in reader:
            image_path = row["image_path"].strip()
            mask_path = row["pseudo_mask_path"].strip()
            if os.path.isfile(image_path) and os.path.isfile(mask_path):
                pairs.append((image_path, mask_path))
            else:
                missing += 1

    if missing > 0:
        print(f"[CSV] skipped {missing} missing file pairs.")
    if len(pairs) == 0:
        raise RuntimeError("No valid image/mask pairs found in CSV.")
    return pairs


def prepare_train_val_pairs(
    csv_path: str,
    split: float,
    seed: int,
    downsample: float = 1.0,
):
    if split <= 0 or split >= 1:
        raise ValueError("--split must be in (0, 1).")
    if downsample <= 0 or downsample > 1:
        raise ValueError("--downsample must be in (0, 1].")

    pairs = read_csv_pairs(csv_path)
    original_total = len(pairs)

    rng_downsample = np.random.RandomState(seed)
    if downsample < 1.0:
        keep = int(round(original_total * downsample))
        keep = max(2, keep)
        keep = min(original_total, keep)
        indices = rng_downsample.choice(original_total, size=keep, replace=False)
        pairs = [pairs[i] for i in indices]

    rng_split = np.random.RandomState(seed + 1)
    perm = rng_split.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    val_size = max(1, int(round(len(pairs) * split)))
    if len(pairs) - val_size < 1:
        raise RuntimeError("Training set is empty after split/downsample, please increase data or adjust ratios.")

    train_pairs = pairs[:-val_size]
    val_pairs = pairs[-val_size:]
    stats = {
        "original_total": original_total,
        "after_downsample": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
    }
    return train_pairs, val_pairs, stats


def prepare_public_train_val_pairs(
    kvasir_root: str,
    cvc_root: str,
    split: float,
    seed: int,
    downsample: float,
):
    if split <= 0 or split >= 1:
        raise ValueError("--split must be in (0, 1).")
    if downsample <= 0 or downsample > 1:
        raise ValueError("--downsample must be in (0, 1].")

    kvasir_pairs = load_pairs_from_root(
        kvasir_root,
        candidates=[("images", "masks")],
        name="kvasir-seg",
        verbose=False,
        check_readable=True,
    )
    cvc_pairs = load_pairs_from_root(
        cvc_root,
        candidates=[("Original", "Ground Truth"), ("images", "masks")],
        name="CVC-ClinicDB",
        verbose=False,
        check_readable=True,
    )

    pairs = kvasir_pairs + cvc_pairs
    original_total = len(pairs)

    rng_downsample = np.random.RandomState(seed)
    if downsample < 1.0:
        keep = int(round(original_total * downsample))
        keep = max(2, keep)
        keep = min(original_total, keep)
        indices = rng_downsample.choice(original_total, size=keep, replace=False)
        pairs = [pairs[i] for i in indices]

    rng_split = np.random.RandomState(seed + 1)
    perm = rng_split.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    val_size = max(1, int(round(len(pairs) * split)))
    if len(pairs) - val_size < 1:
        raise RuntimeError("Training set is empty after split/downsample, please adjust ratios.")

    train_pairs = pairs[:-val_size]
    val_pairs = pairs[-val_size:]
    stats = {
        "original_total": original_total,
        "after_downsample": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "val_source": "public_split",
    }
    return train_pairs, val_pairs, stats


def load_CVC_img_mask(min_id=1, max_id=612):
    """
    加载测试集：原图和 Mask
    """
    val_root_dir = "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/CVC-ClinicDB"

    root_dir = val_root_dir
    if not os.path.exists(root_dir):
        print(f"错误: 测试集根目录不存在: {root_dir}")
        return [], []

    img_dir = os.path.join(root_dir, "Original")
    mask_dir = os.path.join(root_dir, "Ground Truth")

    if not os.path.exists(img_dir):
        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

    if not os.path.exists(img_dir):
        print(f"错误: 在 {root_dir} 下找不到 Original/images 文件夹")
        return [], []

    img_path_list = sorted(glob(os.path.join(img_dir, "*")))
    label_path_list = sorted(glob(os.path.join(mask_dir, "*")))

    img_path_list = [x for x in img_path_list if is_image_file(x)]
    label_path_list = [x for x in label_path_list if is_image_file(x)]

    def _path_id(p):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(name)
        except Exception:
            return None

    img_map = {}
    for p in img_path_list:
        pid = _path_id(p)
        if pid is not None:
            img_map[pid] = p

    mask_map = {}
    for p in label_path_list:
        pid = _path_id(p)
        if pid is not None:
            mask_map[pid] = p

    common_ids = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if min_id is not None:
        common_ids = [i for i in common_ids if i >= min_id]
    if max_id is not None:
        common_ids = [i for i in common_ids if i <= max_id]

    img_path_list = [img_map[i] for i in common_ids]
    label_path_list = [mask_map[i] for i in common_ids]

    return img_path_list, label_path_list


def load_kvasir_img_mask():
    """
    加载 Kvasir-SEG：支持多个根目录，每个目录下包含 images/ 与 masks/ 子文件夹
    """
    root_dirs = [
        "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/kvasir-seg",
    ]

    img_path_list = []
    mask_path_list = []

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"警告: 路径不存在，已跳过: {root_dir}")
            continue

        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"警告: 缺少 images 或 masks 文件夹，已跳过: {root_dir}")
            continue

        imgs = [p for p in sorted(glob(os.path.join(img_dir, "*"))) if is_image_file(p)]
        masks = [p for p in sorted(glob(os.path.join(mask_dir, "*"))) if is_image_file(p)]

        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
        common_keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

        img_path_list.extend([img_map[k] for k in common_keys])
        mask_path_list.extend([mask_map[k] for k in common_keys])

        print(f"[Kvasir] {root_dir} -> {len(common_keys)} pairs")

    return img_path_list, mask_path_list


def get_test_pairs(testset: str) -> List[Tuple[str, str]]:
    mode = str(testset).lower()
    if mode == "kvasir":
        img_paths, mask_paths = load_kvasir_img_mask()
    elif mode == "cvc":
        img_paths, mask_paths = load_CVC_img_mask()
    else:
        raise ValueError(f"Unsupported testset: {testset}")

    pairs = list(zip(img_paths, mask_paths))
    if len(pairs) == 0:
        raise RuntimeError(f"No valid image/mask pairs found for testset={testset}")
    return pairs


class CsvPolypDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], trainsize: int, augmentation: bool):
        self.pairs = pairs
        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _augment_pair(self, image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        image = open_rgb(img_path)
        mask = open_l(mask_path)

        if self.augmentation:
            image, mask = self._augment_pair(image, mask)

        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        mask = mask.resize((self.trainsize, self.trainsize), Image.NEAREST)

        image = self.to_tensor(image)
        image = self.normalize(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask

    def __len__(self):
        return len(self.pairs)
