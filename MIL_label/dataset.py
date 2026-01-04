import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

import tqdm
import os
from skimage import io

import torch
from torchvision import transforms

"""
A. 文件目录结构 (Directory Structure)
硬盘上的图像数据应该像这样组织：
Plaintext
Dataset_Root/
├── Patient_ZhangSan/      <-- 文件夹名需与临床信息表匹配
│   ├── image_001.JPG      <-- 必须是 .JPG (大写)
│   ├── image_002.JPG
│   └── ...
├── Patient_LiSi/
│   ├── img1.JPG
│   └── ...
└── ...
B. 标签 CSV 文件 (raw_label 读取的文件)
这应该是一个包含标签矩阵的 CSV 文件（无表头或代码需跳过表头，pd.read_csv默认读表头，代码中直接转array，建议确认是否含表头）。
第 0 列 (Column 0): 关键 ID。例如病人的住院号、编号等。
后续列: 具体的分类标签（例如 0 或 1 代表阴性/阳性）。
示例 labels.csv:
代码段
1001, 0
1002, 1
1003, 0
C. 临床信息 Excel 文件 (clinical_info 读取的文件)
这是一个用于“翻译” ID 到文件夹名的查找表。

第 1 列 (Column 1, 索引为1): 关键 ID。必须与 CSV 文件的第 0 列对应。
第 2 列 (Column 2, 索引为2): 文件夹名称。必须严格匹配硬盘上的文件夹名（字符串完全匹配）。
(第 0 列代码未用到，通常是序号)
示例 clinical_info.xlsx: 
| Index (Col 0) | Patient ID (Col 1) | Folder Name (Col 2) | ... | | :--- | :--- | :--- | :--- | 
| 1 | 1001 | Patient_ZhangSan | ... | 
| 2 | 1002 | Patient_LiSi | ... | 
| 3 | 1003 | Patient_WangWu | ... |

总结：数据流向图
代码读 labels.csv 第0列 -> 拿到 1001
代码去 clinical_info.xlsx 第1列找 1001 -> 发现对应第2列是 Patient_ZhangSan
代码去 Dataset_Root 找名为 Patient_ZhangSan 的文件夹
代码进入该文件夹，确认里面有 .JPG 图片
成功 -> 加入数据集

3. 代码中需要填空的路径
在使用前，您需要在 dataset_Endo.py 中填入以下路径：

pd.read_csv('这里填标签CSV路径')
pd.read_excel("这里填临床信息Excel路径")
glob("这里填包含所有病人文件夹的根目录路径/*") (例如: /data/endo_images/*)

"""

def gather_align_EndoImg(root_dir='', split=0.7):
    """
    数据准备函数：读取CSV/Excel表格中的标签信息，并与实际文件路径进行匹配对齐。
    最后按比例划分为训练集和测试集。
    """
    raw_label = np.concatenate([
        np.array(pd.read_csv('')),
        np.array(pd.read_csv('')),
        # ...
    ], axis=1)
    clinical_info = pd.read_excel("").to_numpy()

    endo_patient_fanlin = glob("")
    endo_patient_all = endo_patient_fanlin
    endo_patient_all = np.array(endo_patient_all)

    # match img and label
    clip_label = []
    clip_path = []
    not_found_list = []
    overlap_found_list = []
    oversize_list = []
    for i in tqdm(range(raw_label.shape[0]), desc='Matching'):
        check_idx = raw_label[i, 0]
        search_idx = np.where(clinical_info[:, 1] == check_idx)[0]
        if len(search_idx) != 1:
            raise
        patient_name = clinical_info[search_idx[0], 2]

        find_flag = 0
        patient_dir = 0
        for patient_i in endo_patient_all:
            if patient_name == patient_i.split('/')[-1]:
                patient_dir = patient_i
                find_flag = find_flag + 1
        if find_flag == 0:
            not_found_list.append(patient_name)
        elif find_flag > 1:
            overlap_found_list.append(patient_name)
        else:
            sample_image = io.imread(glob(os.path.join(patient_dir, "*.jpg"))[0])
            clip_path.append(patient_dir)
            clip_label.append(raw_label[i])
    clip_path = np.array(clip_path)
    clip_label = np.array(clip_label)
    clip_data_all = np.concatenate([clip_path[:, None], clip_label], axis=1)

    num_patient = clip_data_all.shape[0]
    idx_train_test = np.random.choice(num_patient, num_patient, replace=False)
    idx_train = idx_train_test[: int(split * num_patient)]
    idx_test = idx_train_test[int(split * num_patient):]
    return clip_data_all[idx_train], clip_data_all[idx_test]

class DataSet_MIL(torch.utils.data.Dataset):

    # @profile
    def __init__(self, ds, downsample=0.1, transform=None, return_bag=False, preload=False):
        """
        初始化函数
        :param ds: 数据列表，由 gather_align_EndoImg 返回 (路径+标签)
        :param downsample: 下采样比例，仅使用部分病人数据进行调试或快速实验
        :param transform: 图像预处理/增强操作
        :param return_bag: True表示返回一个包(整个病人的所有图,用于教师)，False表示返回单张图(用于学生)
        :param preload: 是否将所有图片预加载到内存
        """
        self.root_dir = ds
        self.transform = transform
        self.downsample = downsample
        self.return_bag = return_bag
        self.preload = preload

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.31512642, 0.20470396, 0.13602576],
                                     std=[0.30684662, 0.21589851, 0.15230405])
            ])

        all_slides = ds

        # 1.1 数据下采样 (随机抽取部分病人)
        print("================ Down sample Slide {} ================".format(downsample))
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        self.num_patches = 0
        for i in all_slides:
            self.num_patches = self.num_patches + len(glob(os.path.join(i[0], "*.JPG")))
        
        # 2. 提取所有可用的 patch 并构建对应的标签索引
        if self.preload:
            self.all_patches = np.zeros([self.num_patches, 576, 720, 3], dtype=np.uint8)
        else:
            self.all_patches = []

        self.patch_label = [] # patch对应的标签
        self.patch_corresponding_slide_label = [] # patch所属Slide(病人)的标签
        self.patch_corresponding_slide_index = [] # patch所属Slide的索引(0~N)
        self.patch_corresponding_slide_name = []  # patch所属Slide的名字

        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            patient_path = i[0]
            for j, file_j in enumerate(glob(os.path.join(patient_path, "*.JPG"))):
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(file_j)
                else:
                    self.all_patches.append(file_j)
                self.patch_label.append(0)
                self.patch_corresponding_slide_label.append(int(i[2]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(patient_path.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

    def __len__(self):
        if self.return_bag:
            # 如果是Bag模式，长度为Slide(病人)的数量
            return self.patch_corresponding_slide_index.max() + 1
        else:
            # 如果是Instance模式，长度为所有Patch(图片)的总数
            return self.num_patches

    def __getitem__(self, index):
        if self.return_bag:
            # --- MIL 模式 (Bag) ---
            # index 代表 Slide (病人) 的索引
            # 找出所有属于该 Slide 的 patch 索引
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]

            # 硬编码限制：每个Bag最多取100张图
            if len(idx_patch_from_slide_i) > 100:
                idx_patch_from_slide_i = idx_patch_from_slide_i[:100]

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 512, 512], dtype=np.float32)
            for i in range(bag.shape[0]):
                if self.preload:
                    instance_img = bag[i]
                else:
                    instance_img = io.imread(bag[i])
                bag_normed[i, :, :, :] = self.transform(Image.fromarray(np.uint8(instance_img), 'RGB'))
            bag = bag_normed
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i].max()
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        
        else:
            # --- 普通模式 (Instance) ---
            # index 代表 Patch (图片) 的索引
            if self.preload:
                patch_image = self.all_patches[index]
            else:
                patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            # patch_image = patch_image[:, 35:35+512, 165:165+512]
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

def cal_img_mean_std():
    """
    计算整个训练集的均值和方差，用于Normalize参数设置。
    """
    ds_train, ds_test = gather_align_EndoImg()
    train_ds = DataSet_MIL(ds=ds_train, transform=None, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                               shuffle=False, num_workers=6, drop_last=True, pin_memory=True)
    print("Length of dataset: {}".format(len(train_ds)))
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in tqdm(train_loader, desc="Calculating Mean and Std"):
        img = data[0]
        for d in range(3):
            mean[d] += img[:, d, :, :].mean()
            std[d] += img[:, d, :, :].std()
    mean.div_(len(train_ds))
    std.div_(len(train_ds))
    mean = list(mean.numpy()*128)
    std = list(std.numpy()*128)
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))
    return mean, std

if __name__ == '__main__':
    mean, std = cal_img_mean_std()