import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import imageio.v2 as imageio


def _safe_open_rgb(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def _safe_open_l(path):
    try:
        return Image.open(path).convert('L')
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return Image.fromarray(arr.astype(np.uint8)).convert('L')


class ResNetDataset(Dataset):
    """适配 load_dataset_from_folders 且使用 finetune 预处理逻辑的数据集"""
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        if isinstance(label, (list, np.ndarray)) and len(label) > 1:
            label = int(np.argmax(label))
        else:
            if isinstance(label, (list, np.ndarray)):
                label = int(np.array(label).reshape(-1)[0])
            else:
                label = int(label)

        img = _safe_open_rgb(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label


class SwinDataset(Dataset):
    """沿用 train_res 的 Dataset 逻辑"""
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        if isinstance(label, (list, np.ndarray)) and len(label) > 1:
            label = int(np.argmax(label))
        else:
            if isinstance(label, (list, np.ndarray)):
                label = int(np.array(label).reshape(-1)[0])
            else:
                label = int(label)

        img = _safe_open_rgb(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label


class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, mask_size=512):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = _safe_open_rgb(img_path)
        mask = _safe_open_l(mask_path)

        if self.transform:
            img = self.transform(img)

        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask = np.array(mask) > 127
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return img, mask
