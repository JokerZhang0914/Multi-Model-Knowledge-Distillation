import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import os
from torchvision import transforms as T
import imageio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transforms



# ==========================================
#               路径配置区域 (硬编码)
# ==========================================

# 1. 阳性训练样本文件夹列表 (包含息肉的图片)
TRAIN_POS_DIRS = [
    "D:/公开数据集/2_adenomatous",
    "D:/公开数据集/1_hyperplastic"
]

# 2. 阴性训练样本文件夹列表 (包含正常/背景的图片)
TRAIN_NEG_DIRS = [
    "D:/公开数据集/0_normal"
]

# 3. 测试集根目录
#    该目录下必须包含 'Original' (或 'images') 和 'Ground Truth' (或 'masks') 两个子文件夹
VAL_ROOT_DIR = "E:/AAA_joker/本科毕设/CVC-ClinicDB"
# ==========================================

# 支持的图片扩展名
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def load_dataset_from_folders():
    """
    读取硬编码的文件夹路径加载数据集。
    直接使用文件顶部的 TRAIN_POS_DIRS 和 TRAIN_NEG_DIRS。
    """
    img_path_list = []
    label_list = []
    
    # --- 1. 处理阳性文件夹 (Label = 1.0) ---
    print(f"--- 正在加载阳性样本 (Label=1.0) ---")
    for p_dir in TRAIN_POS_DIRS:
        if not os.path.exists(p_dir):
            print(f"警告: 路径不存在，已跳过: {p_dir}")
            continue
            
        files = glob(os.path.join(p_dir, '*'))
        images = [f for f in files if is_image_file(f)]
        
        if len(images) == 0:
            print(f"提示: {p_dir} 中没有找到图片。")
            continue
            
        print(f"  [+] {len(images)} 张 -> {p_dir}")
        
        img_path_list.extend(images)
        label_list.extend([np.array([1.0])] * len(images))

    # --- 2. 处理阴性文件夹 (Label = 0.0) ---
    if TRAIN_NEG_DIRS:
        print(f"--- 正在加载阴性样本 (Label=0.0) ---")
        for n_dir in TRAIN_NEG_DIRS:
            if not os.path.exists(n_dir):
                print(f"警告: 路径不存在，已跳过: {n_dir}")
                continue
                
            files = glob(os.path.join(n_dir, '*'))
            images = [f for f in files if is_image_file(f)]
            
            if len(images) == 0:
                print(f"提示: {n_dir} 中没有找到图片。")
                continue

            print(f"  [-] {len(images)} 张 -> {n_dir}")
            
            img_path_list.extend(images)
            label_list.extend([np.array([0.0])] * len(images))
    else:
        print("未配置阴性样本文件夹 (TRAIN_NEG_DIRS 为空)")

    return np.array(img_path_list), np.array(label_list)


def load_test_img_mask():
    """
    加载测试集：原图和 Mask
    直接使用 VAL_ROOT_DIR
    """
    root_dir = VAL_ROOT_DIR
    if not os.path.exists(root_dir):
        print(f"错误: 测试集根目录不存在: {root_dir}")
        return [], []

    # 尝试匹配常见的文件夹名称
    img_dir = os.path.join(root_dir, 'Original')
    mask_dir = os.path.join(root_dir, 'Ground Truth')
    
    # 兼容另一种命名
    if not os.path.exists(img_dir):
        img_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')

    if not os.path.exists(img_dir):
        print(f"错误: 在 {root_dir} 下找不到 Original/images 文件夹")
        return [], []

    img_path_list = sorted(glob(os.path.join(img_dir, '*')))
    label_path_list = sorted(glob(os.path.join(mask_dir, '*')))
    
    # 过滤非图片文件
    img_path_list = [x for x in img_path_list if is_image_file(x)]
    label_path_list = [x for x in label_path_list if is_image_file(x)]

    return img_path_list, label_path_list


class Dataset_CAM(Dataset):
    def __init__(self,
                img_path,
                label_path,
                type='train',
                resize_range=[512, 640],
                rescale_range=[0.6, 1.0],
                crop_size=448,
                img_fliplr=True,
                ignore_index=255,
                num_classes=2,
                aug=False,
                **kwargs):
        super().__init__()

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.type = type
        self.label_path = label_path
        self.img_path = img_path

        # transforms
        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)
        
        # 归一化
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 颜色翻转增强
        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])
        
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        
        self.local_view = T.Compose([
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.img_path)

    def __transforms(self, image):
        img_box = None
        local_image = None
        
        if self.aug:
            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image, 
                    crop_size=self.crop_size, 
                    mean_rgb=[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index
                )
            
            local_image = self.local_view(Image.fromarray(image)).float()
    
        image = np.array(image)
        image = transforms.normalize_img2(image)
        return image, local_image, img_box
    
    def __getitem__(self, index):
        img_item_path = self.img_path[index]
        
        if self.type == 'val':
            img_name = os.path.basename(img_item_path).split('.')[0]

        try:
            image = np.array(imageio.imread(img_item_path))
            if len(image.shape) == 3 and image.shape[2] == 4:
                 image = image[:, :, :3]
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
        except Exception as e:
            print(f"Error loading image {img_item_path}: {e}")
            image = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        pil_image = Image.fromarray(image)

        image, local_image, img_box = self.__transforms(image=image)
        image = np.transpose(image, (2, 0, 1))
        
        # === Train ===
        if self.type == 'train':
            cls_label = self.label_path[index]
            
            if self.aug:
                crops = []
                crops.append(image.astype(np.float64))
                crops.append(self.global_view2(pil_image).float())
                crops.append(local_image)

                return image, cls_label, img_box, crops
            else:
                return image, cls_label

        # === Val ===
        elif self.type == 'val':
            label_mask_path = self.label_path[index]
            try:
                seg_mask = np.asarray(Image.open(label_mask_path).convert('L'))
                seg_mask = (seg_mask >= 127)
            except Exception as e:
                print(f"Error loading mask {label_mask_path}: {e}")
                seg_mask = np.zeros((image.shape[1], image.shape[2]), dtype=bool)

            if np.any(seg_mask):
                cls_label = np.array([1.0], dtype=np.float32)
            else:
                cls_label = np.array([0.0], dtype=np.float32)
            
            return img_name, image, seg_mask, cls_label

        
if __name__ == "__main__":
    
    print("\n========= 正在初始化数据集 =========")
    
    # 1. 加载训练数据 (从顶部 TRAIN_POS_DIRS / TRAIN_NEG_DIRS)
    img_path_list_train, label_train = load_dataset_from_folders()
    print(f"训练集加载完成: 总计 {len(img_path_list_train)} 张图片")

    # 2. 加载测试数据 (从顶部 VAL_ROOT_DIR)
    img_path_list_test, mask_test_path = load_test_img_mask()
    print(f"测试集加载完成: 总计 {len(img_path_list_test)} 张图片")
    print("===================================\n")

    # 3. 创建 DataLoader 示例
    if len(img_path_list_train) > 0:
        train_ds = Dataset_CAM(img_path_list_train, label_train, aug=True, type='train')
        train_loader = DataLoader(
            train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        print("正在检查 Batch 输出结构...")
        for i, data in enumerate(train_loader):
            image, cls_label, img_box, crops = data
            print(f"  [Batch {i}]")
            print(f"  - Image Shape: {image.shape}")
            print(f"  - Label Shape: {cls_label.shape}") 
            print(f"  - Crop Views : {len(crops)}")
            break