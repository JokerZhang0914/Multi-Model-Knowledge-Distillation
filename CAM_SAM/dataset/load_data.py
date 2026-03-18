import os
from glob import glob
import numpy as np


def is_image_file(filename):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(IMG_EXTENSIONS)


def load_dataset_from_folders():
    """
    读取硬编码的文件夹路径加载数据集。
    """
    TRAIN_POS_DIRS = [
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/1_hyperplastic_aligned",
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/2_adenomatous_aligned"
    ]

    TRAIN_NEG_DIRS = [
        "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/0_normal_aligned"
    ]

    img_path_list = []
    label_list = []

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


def load_CVC_img_mask(min_id=1, max_id=612):
    """
    加载测试集：原图和 Mask
    """
    VAL_ROOT_DIR = "/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/CVC-ClinicDB"

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
        '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/kvasir-seg'#,
        # '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/kvasir-seg/sessile-main-Kvasir-SEG'
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


def load_report(patient_prefixes=("f_", "t_"), recursive=False):
    """
    加载报告数据：若干根目录下包含病人文件夹（如 f_patient1 / t_patient543）。
    仅返回图片路径，不包含标签。
    """
    REPORT_ROOT_DIRS = [
        '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集/0_normal_aligned',
        '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/公开数据集mix',
        '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/肿瘤医院_1/dataset_aligned_zhongliu'
    ]

    img_paths = []

    for root_dir in REPORT_ROOT_DIRS:
        if not os.path.exists(root_dir):
            print(f"警告: 路径不存在，已跳过: {root_dir}")
            continue

        sub_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        patient_dirs = []
        for d in sub_dirs:
            name = os.path.basename(d)
            if any(name.startswith(p) for p in patient_prefixes):
                patient_dirs.append(d)

        if not patient_dirs:
            print(f"提示: {root_dir} 下未找到匹配的病人文件夹。")
            continue

        root_count = 0
        for p_dir in patient_dirs:
            if recursive:
                files = []
                for base, _, filenames in os.walk(p_dir):
                    for fn in filenames:
                        files.append(os.path.join(base, fn))
            else:
                files = glob(os.path.join(p_dir, '*'))

            images = [f for f in files if is_image_file(f)]
            if images:
                images = sorted(images)
                img_paths.extend(images)
                root_count += len(images)

        print(f"[Report] {root_dir} -> {root_count} images")

    return np.array(img_paths)
