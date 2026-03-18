import csv
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


class CsvPolypDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], trainsize: int, augmentation: bool):
        self.pairs = pairs
        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def _open_rgb(path: str):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    @staticmethod
    def _open_l(path: str):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

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
