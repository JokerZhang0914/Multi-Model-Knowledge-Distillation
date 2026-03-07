import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import os
from torchvision import transforms as T
import imageio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from . import transforms


def is_image_file(filename):
    # 支持的图片扩展名
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(IMG_EXTENSIONS)

def load_dataset_from_folders():
    """
    读取硬编码的文件夹路径加载数据集。
    """
    # 1. 阳性训练样本文件夹列表 (包含息肉的图片)
    TRAIN_POS_DIRS = [
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/1_hyperplastic_aligned",
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/2_adenomatous_aligned"
    ]

    # 2. 阴性训练样本文件夹列表 (包含正常/背景的图片)
    TRAIN_NEG_DIRS = [
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/0_normal_aligned"
    ]

    img_path_list = []
    label_list = []
    
    # --- 1. 处理阳性文件夹 (Label = 1.0) ---
    print(f"--- 正在加载阳性样本 (Label=1.0) ---")
    for p_dir in TRAIN_POS_DIRS:
        if not os.path.exists(p_dir):
            print(f"警告: 路径不存在，已跳过: {p_dir}")
            continue
            
        sub_dirs = [os.path.join(p_dir, d) for d in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, d))]
        
        all_images = []
        for sub_dir in sub_dirs:
            files = glob(os.path.join(sub_dir, '*'))
            images = [f for f in files if is_image_file(f)]
            all_images.extend(images)
        
        if len(all_images) == 0:
            print(f"提示: {p_dir} 中没有找到图片。")
            continue
            
        print(f"  [+] {len(all_images)} 张 -> {p_dir}")
        
        img_path_list.extend(all_images)
        label_list.extend([np.array([1.0])] * len(all_images))

    # --- 2. 处理阴性文件夹 (Label = 0.0) ---
    if TRAIN_NEG_DIRS:
        print(f"--- 正在加载阴性样本 (Label=0.0) ---")
        for n_dir in TRAIN_NEG_DIRS:
            if not os.path.exists(n_dir):
                print(f"警告: 路径不存在，已跳过: {n_dir}")
                continue
                
            sub_dirs = [os.path.join(n_dir, d) for d in os.listdir(n_dir) if os.path.isdir(os.path.join(n_dir, d))]
            
            all_images = []
            for sub_dir in sub_dirs:
                files = glob(os.path.join(sub_dir, '*'))
                images = [f for f in files if is_image_file(f)]
                all_images.extend(images)
            
            if len(all_images) == 0:
                print(f"提示: {n_dir} 中没有找到图片。")
                continue

            print(f"  [-] {len(all_images)} 张 -> {n_dir}")
            
            img_path_list.extend(all_images)
            label_list.extend([np.array([0.0])] * len(all_images))
    else:
        print("未配置阴性样本文件夹 (TRAIN_NEG_DIRS 为空)")

    return np.array(img_path_list), np.array(label_list)

def load_dataset_from_LD(folder_ids=None):
    """
    从 LDPolypVideo 数据集加载数据
    Args:
        folder_ids (list): 需要加载的文件夹 ID 列表。如果为 None，则加载所有可用文件夹。
    Returns:
        np.array(img_path_list): 所有图片的绝对路径 (numpy array)
        np.array(label_list): 对应的分类标签 (numpy array of np.array([1.0]) or np.array([0.0]))
    """
    img_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/TrainValid/Images'
    anno_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/TrainValid/Annotations'
    
    # 如果没指定 ID，则获取 Images 下所有目录名
    if folder_ids is None:
        folder_ids = [d for d in os.listdir(img_root_base) if os.path.isdir(os.path.join(img_root_base, d))]
    
    img_path_list = []
    label_list = []
    
    print(f"--- 正在加载 LDPolypVideo 数据集 ({len(folder_ids)} 个序列) ---")
    
    for folder_id in folder_ids:
        folder_name = str(folder_id)
        img_dir = os.path.join(img_root_base, folder_name)
        anno_dir = os.path.join(anno_root_base, folder_name)
        
        if not os.path.exists(img_dir) or not os.path.exists(anno_dir):
            print(f"警告: 路径不存在，已跳过序列 {folder_name}")
            continue
        
        # 遍历文件夹内的图片
        all_images = []
        for img_name in sorted(os.listdir(img_dir)):
            if is_image_file(img_name):  # 使用全局的 is_image_file 函数过滤
                img_path = os.path.join(img_dir, img_name)
                
                txt_name = os.path.splitext(img_name)[0] + '.txt'
                txt_path = os.path.join(anno_dir, txt_name)
                
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                # 解析逻辑：首个数字 > 0 为阳性(1.0)，否则为阴性(0.0)
                                num_polyps = int(content.split()[0])
                                cls_label = 1.0 if num_polyps > 0 else 0.0
                                img_path_list.append(img_path)
                                label_list.append(np.array([cls_label]))
                    except Exception as e:
                        print(f"警告: 解析 {txt_path} 失败 ({e})，跳过图像 {img_name}")
                else:
                    print(f"警告: 缺少注解 {txt_path}，跳过图像 {img_name}")
        
        if len(all_images) > 0:  # 注意：all_images 未使用，改为统计实际添加的
            added_count = sum(1 for path in img_path_list if folder_name in path)  # 粗略统计本序列添加数量
            print(f"  [{folder_name}] {added_count} 张 -> {img_dir}")
    
    if len(img_path_list) == 0:
        print("提示: LDPolypVideo 中没有找到有效图片。")
    
    return np.array(img_path_list), np.array(label_list)


def load_test_img_mask():
    """
    加载测试集：原图和 Mask
    """

    # 测试集根目录
    # 该目录下必须包含 'Original' (或 'images') 和 'Ground Truth' (或 'masks') 两个子文件夹
    VAL_ROOT_DIR = "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/CVC-ClinicDB"

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

def load_dataset_from_LD_test(start_id=101, max_id=160):
    """
    加载 LDPolypVideo 测试集 (用于单纯的分类能力验证 val2)
    仿照 test.py 的硬编码路径和正负例判断逻辑。
    """
    img_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/Test/Images'
    anno_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/Test/Annotations'
    
    img_path_list = []
    label_list = []
    
    print(f"--- 正在加载 LDPolypVideo 测试集 (Folders {start_id} to {max_id}) ---")
    
    for folder_id in range(start_id, max_id + 1):
        folder_name = str(folder_id)
        img_dir = os.path.join(img_root_base, folder_name)
        anno_dir = os.path.join(anno_root_base, folder_name)

        # 跳过不存在的文件夹
        if not os.path.exists(img_dir) or not os.path.exists(anno_dir):
            continue

        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, filename)
                txt_name = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(anno_dir, txt_name)

                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                # 获取息肉数量
                                num_polyps = int(content.split()[0])
                                # 二值化逻辑：>0 为正例 (1.0)，否则为负例 (0.0)
                                label = 1.0 if num_polyps > 0 else 0.0
                                
                                img_path_list.append(img_path)
                                label_list.append(np.array([label], dtype=np.float32))
                    except Exception as e:
                        print(f"[Error] Failed to read label for {filename}: {e}")
                        
    print(f"[*] Total test images loaded: {len(img_path_list)}")
    return np.array(img_path_list), np.array(label_list)

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
        
        if self.type in ['val', 'val2']:
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
        
        elif self.type == 'val2':
            cls_label = self.label_path[index]
            
            # 为了适配 validate2 函数中期望获取 cls_label[:, 1] 的多类别结构
            # 若 num_classes=2 且 cls_label 只有1维 [0.0] 或 [1.0]，则将其转换为 one-hot 形式
            if len(cls_label) == 1 and self.num_classes == 2:
                new_label = np.zeros(2, dtype=np.float32)
                if cls_label[0] > 0.5:
                    new_label[1] = 1.0  # 阳性
                else:
                    new_label[0] = 1.0  # 阴性
                cls_label = new_label

            # 因为仅验证分类，缺少真实的 Mask，所以返回全0的 dummy mask 占位
            # 保证网络能通过 (img_name, inputs, labels, cls_label) 格式正常解包
            dummy_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
            
            return img_name, image, dummy_mask, cls_label

        
if __name__ == "__main__":
    
    print("\n========= 正在初始化数据集 =========")
    
    # 加载原有的公开数据集
    img_paths, labels = load_dataset_from_folders()
    
    # 加载 LDPolypVideo 数据集 (例如文件夹 1 到 100)
    ld_ids = list(range(1, 101))
    ld_paths, ld_labels = load_dataset_from_LD(folder_ids=ld_ids)
    
    # 合并数据
    img_path_list_train = np.concatenate([img_paths, ld_paths], axis=0)
    label_train = np.concatenate([labels, ld_labels], axis=0)

    # 验证拼接结果
    print(f"拼接后总数据量: 图片路径数 {img_path_list_train.shape[0]}, 标签数 {label_train.shape[0]}")

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