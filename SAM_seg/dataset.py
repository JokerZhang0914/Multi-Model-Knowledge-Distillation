import csv
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob

def is_image_file(filename):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(IMG_EXTENSIONS)

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


def load_CVC_img_mask(min_id=1, max_id=612):
    """
    加载测试集：原图和 Mask
    """
    VAL_ROOT_DIR = "/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/CVC-ClinicDB"

    root_dir = VAL_ROOT_DIR
    if not os.path.exists(root_dir):
        print(f"错误: 测试集根目录不存在: {root_dir}")
        return [], []

    img_dir = os.path.join(root_dir, 'Original')
    mask_dir = os.path.join(root_dir, 'Ground Truth')

    if not os.path.exists(img_dir):
        img_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')

    if not os.path.exists(img_dir):
        print(f"错误: 在 {root_dir} 下找不到 Original/images 文件夹")
        return [], []

    img_path_list = sorted(glob(os.path.join(img_dir, '*')))
    label_path_list = sorted(glob(os.path.join(mask_dir, '*')))

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
    ROOT_DIRS = [
        '/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/kvasir-seg'#,
        # '/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/kvasir-seg/sessile-main-Kvasir-SEG'
    ]

    img_path_list = []
    mask_path_list = []

    for root_dir in ROOT_DIRS:
        if not os.path.exists(root_dir):
            print(f"警告: 路径不存在，已跳过: {root_dir}")
            continue

        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"警告: 缺少 images 或 masks 文件夹，已跳过: {root_dir}")
            continue

        imgs = [p for p in sorted(glob(os.path.join(img_dir, '*'))) if is_image_file(p)]
        masks = [p for p in sorted(glob(os.path.join(mask_dir, '*'))) if is_image_file(p)]

        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
        common_keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

        img_path_list.extend([img_map[k] for k in common_keys])
        mask_path_list.extend([mask_map[k] for k in common_keys])

        print(f"[Kvasir] {root_dir} -> {len(common_keys)} pairs")

    return img_path_list, mask_path_list


class CsvPolypDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], trainsize: int, augmentation: bool):
        self.pairs = pairs
        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = (arr - vmin) / (vmax - vmin)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return arr

    @classmethod
    def _open_tif_with_tifffile(cls, path: str, to_mode: str):
        try:
            import tifffile
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode TIFF with PIL and tifffile is unavailable: {path}"
            ) from e

        arr = tifffile.imread(path)
        arr = np.asarray(arr)

        if arr.ndim == 2:
            arr_u8 = cls._to_uint8(arr)
            pil = Image.fromarray(arr_u8, mode="L")
        elif arr.ndim == 3:
            # Convert CHW -> HWC when needed
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
                arr_u8 = cls._to_uint8(arr)
                pil = Image.fromarray(arr_u8, mode="L")
            else:
                if arr.shape[-1] > 3:
                    arr = arr[..., :3]
                arr_u8 = cls._to_uint8(arr)
                pil = Image.fromarray(arr_u8, mode="RGB")
        else:
            raise RuntimeError(f"Unsupported TIFF ndim={arr.ndim}: {path}")

        return pil.convert(to_mode)

    @classmethod
    def _open_rgb(cls, path: str):
        try:
            with open(path, "rb") as f:
                return Image.open(f).convert("RGB")
        except Exception:
            if path.lower().endswith((".tif", ".tiff")):
                return cls._open_tif_with_tifffile(path, "RGB")
            raise

    @classmethod
    def _open_l(cls, path: str):
        try:
            with open(path, "rb") as f:
                return Image.open(f).convert("L")
        except Exception:
            if path.lower().endswith((".tif", ".tiff")):
                return cls._open_tif_with_tifffile(path, "L")
            raise

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
        image = self._open_rgb(img_path)
        mask = self._open_l(mask_path)

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
