import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.model import PretrainedResNet18_Encoder, get_student
import json
from datetime import datetime

class SingleImageDataset(Dataset):
    def __init__(self, max_id, start_id=101, transform=None):
        """
        遍历从 start_id (默认101) 到 max_id 的所有文件夹，读取图片和对应标签
        """
        # 硬编码你的路径
        self.img_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/Test/Images'
        self.anno_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/Test/Annotations'
        self.transform = transform
        self.samples = [] # 存储 (path, label)

        print(f"[*] Scanning folders from {start_id} to {max_id}...")
        
        for folder_id in range(start_id, max_id + 1):
            folder_name = str(folder_id)
            img_dir = os.path.join(self.img_root_base, folder_name)
            anno_dir = os.path.join(self.anno_root_base, folder_name)

            # 跳过不存在的文件夹
            if not os.path.exists(img_dir) or not os.path.exists(anno_dir):
                # print(f"    [Info] Folder {folder_name} skipped (not found).")
                continue

            # 遍历图片文件
            file_count = 0
            for filename in os.listdir(img_dir):
                if filename.endswith('.jpg'): # 根据你的截图，图片是 jpg 格式
                    img_path = os.path.join(img_dir, filename)
                    
                    # 寻找对应的 txt 文件
                    txt_name = filename.replace('.jpg', '.txt')
                    txt_path = os.path.join(anno_dir, txt_name)

                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r') as f:
                                content = f.read().strip()
                                
                                if content:
                                    # 1. 获取第一个数（代表息肉个数）
                                    num_polyps = int(content.split()[0])
                                    
                                    # 2. 【核心修改】二值化逻辑
                                    # 如果个数 > 0，标签设为 1 (正例)
                                    # 如果个数 == 0，标签设为 0 (负例)
                                    label = 1 if num_polyps > 0 else 0
                                    
                                    self.samples.append((img_path, label))
                                    file_count += 1
                        except Exception as e:
                            print(f"[Error] Failed to read label for {filename}: {e}")
            
            # print(f"    -> Folder {folder_name}: Loaded {file_count} images.")
        
        print(f"[*] Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 打开图片
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_test_loader(max_id, start_id=101, batch_size=32):
    # 定义预处理 (需与训练时 Encoder 输入一致)
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # ResNet 输入通常是 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if start_id >= 101 and max_id <= 160 and start_id <= max_id:
        dataset = SingleImageDataset(start_id=start_id, max_id=max_id, transform=transform)
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        return loader
    else:
        print(f'[Error] please check id: start_id={start_id} max_id={max_id}')
        return

# 引入模型定义 (假设 model.py 在同一目录下)


# ==========================================
# 1. 权重加载辅助函数 (复用 train.py 逻辑)
# ==========================================
def load_checkpoint(model, ckpt_path, device):
    """
    智能加载权重，自动处理 DataParallel 的 'module.' 前缀差异
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"[*] Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 1. 尝试获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # 假设本身就是 state_dict

    # 2. 处理 'module.' 前缀
    if isinstance(model, nn.DataParallel):
        is_model_parallel = True
    else:
        is_model_parallel = False
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # 如果权重有 module. 但模型没有 -> 去掉
        if k.startswith('module.') and not is_model_parallel:
            name = k[7:]
        # 如果权重没有 module. 但模型有 -> 加上
        elif not k.startswith('module.') and is_model_parallel:
            name = 'module.' + k
        else:
            name = k
        new_state_dict[name] = v
    
    # 3. 加载
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("    -> Loaded successfully (Strict).")
    except RuntimeError as e:
        print(f"    [Warning] Strict loading failed, trying non-strict. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)


def save_test_results(
    args,
    metrics: dict,
    extra_metrics: dict = None
):
    """
    将测试结果追加写入 test_result/test_log.txt
    """
    # 1. 结果目录
    result_dir = args.result_dir or "test_result"
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, "test_log.txt")

    # 2. 时间戳
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. 配置（args -> dict，避免不可序列化对象）
    args_dict = vars(args).copy()

    # 4. 写入
    with open(result_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Test Time: {now}\n")
        f.write("-" * 80 + "\n")

        f.write("[Configuration]\n")
        for k, v in args_dict.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Metrics]\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        if extra_metrics is not None:
            f.write("\n[Extra Metrics]\n")
            for k, v in extra_metrics.items():
                f.write(f"{k}: {v}\n")

        f.write("=" * 80 + "\n")


# ==========================================
# 2. 测试逻辑
# ==========================================
def test(args):
    # --- 设备设置 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 模型初始化 ---
    print("Initializing models...")
    # 1. Encoder
    encoder = PretrainedResNet18_Encoder(freeze=False).to(device)
    # 2. Student
    student = get_student().to(device)

    # --- 加载权重 ---
    load_checkpoint(encoder, args.encoder_ckpt, device)
    load_checkpoint(student, args.student_ckpt, device)

    # 设置为评估模式
    encoder.eval()
    student.eval()

    # --- 数据集加载 (等待下一条指令填充) ---
    
    print(f"\nLoading Test Data (Folders {args.start_id}-{args.max_id})...")
    
    # 直接调用刚才定义的 helper 函数
    test_loader = get_test_loader(start_id=args.start_id, max_id=args.max_id, batch_size=args.batch_size)
    
    if  (not test_loader) or len(test_loader.dataset) == 0:
        print("[Error] No images found! Check path or start_id/max_id.")
        return

    # --- 推理循环 ---
    print("Starting Inference...")
    y_true = []
    y_scores = [] # 预测概率 (Class 1)
    y_preds = []  # 预测类别 (0 或 1)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            imgs = imgs.to(device)
            # 确保 label 是 tensor 并且维度正确
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels = labels.to(device)

            # 1. Encoder 提取特征
            feats = encoder(imgs) # [B, 512]

            # 2. Student 分类
            logits = student(feats) # [B, 2]
            
            # 3. 获取概率
            probs = torch.softmax(logits, dim=1)[:, 1] # 取阳性概率
            preds = (probs > 0.5).long() # 阈值 0.5 转为类别

            # 4. 收集结果
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())

    # --- 计算指标 ---
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    print("\n" + "="*30)
    print("       Test Results")
    print("="*30)

    # 1. AUC
    if len(np.unique(y_true)) < 2:
        print("[Warning] 测试数据仅包含单个类别（全为0或全为1）。")
        print(f"当前包含类别: {np.unique(y_true)}")
        print("无法计算 AUC 和 AP，将设为 0.0。")
        auc = 0.0
        ap = 0.0
    else:
        # 1. 计算 AUC
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0
        
        # 2. 计算 AP (Average Precision)
        try:
            ap = average_precision_score(y_true, y_scores)
        except Exception as e:
            print(f"[Warning] AP calculation error: {e}")
            ap = 0.0

    # 3. ACC
    acc = accuracy_score(y_true, y_preds)

    # 4. F1-Score
    f1 = f1_score(y_true, y_preds)
    
    # 混淆矩阵 (Optional)
    cm = confusion_matrix(y_true, y_preds)

    print(f"AUC      : {auc:.4f}")
    print(f"AP       : {ap:.4f}")
    print(f"ACC      : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-" * 30)
    print(f"Confusion Matrix:\n{cm}")
    print("\n" + "="*40)
    print("       Metrics at Fixed Recall = 0.8")
    print("="*40)

    metrics = {
    "AUC": float(auc),
    "AP": float(ap),
    "ACC": float(acc),
    "F1": float(f1)
    }


    # 1. 确保数据不为空且包含正例 (否则 recall 无意义)
    if np.sum(y_true) == 0:
        print("[Warning] No positive samples found. Cannot calculate Recall-based metrics.")
    else:
        from sklearn.metrics import precision_recall_curve

        # 2. 获取 P-R 曲线上的所有点
        # precisions, recalls, thresholds 都是数组
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # 3. 找到 Recall 最接近 0.8 的索引
        # 注意：recalls 数组通常是随阈值增加而递减的
        target_recall = 0.8
        # 使用 abs 找差值最小的点
        idx = np.argmin(np.abs(recalls - target_recall))
        
        # 处理边界情况：thresholds 的长度比 recalls 少 1
        if idx < len(thresholds):
            fixed_threshold = thresholds[idx]
        else:
            fixed_threshold = thresholds[-1] # 取最后一个阈值

        # 4. 使用该阈值生成二值化预测
        y_pred_at_80 = (y_scores >= fixed_threshold).astype(int)

        # 5. 计算混淆矩阵元素
        # confusion_matrix 返回顺序是: tn, fp, fn, tp
        tn_80, fp_80, fn_80, tp_80 = confusion_matrix(y_true, y_pred_at_80).ravel()

        # 6. 计算指标
        # Precision = TP / (TP + FP)
        precision_at_80 = tp_80 / (tp_80 + fp_80) if (tp_80 + fp_80) > 0 else 0.0
        
        # Real Recall (校验一下是否接近 0.8)
        real_recall = tp_80 / (tp_80 + fn_80) if (tp_80 + fn_80) > 0 else 0.0
        
        # FPR = FP / (FP + TN)
        fpr_at_80 = fp_80 / (fp_80 + tn_80) if (fp_80 + tn_80) > 0 else 0.0

        # 7. 打印结果
        print(f"[*] Threshold Selected: {fixed_threshold:.4f}")
        print(f"[*] Actual Recall     : {real_recall:.4f} (Target: 0.8)")
        print("-" * 20)
        print(f"TP  : {tp_80}")
        print(f"FN  : {fn_80}")
        print(f"FP  : {fp_80}")
        print(f"TN  : {tn_80}")
        print("-" * 20)
        print(f"Precision : {precision_at_80:.4f}")
        print(f"FPR       : {fpr_at_80:.4f}")

        extra_metrics = {
        "Recall_target": 0.8,
        "Recall_actual": float(real_recall),
        "Precision_at_Recall_0.8": float(precision_at_80),
        "FPR_at_Recall_0.8": float(fpr_at_80),
        "Threshold_at_Recall_0.8": float(fixed_threshold),
        "TP": int(tp_80),
        "FP": int(fp_80),
        "FN": int(fn_80),
        "TN": int(tn_80)
        }

    
    print("="*40)
    save_test_results(
    args=args,
    metrics=metrics,
    extra_metrics=extra_metrics
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Encoder + Student on Single Images"
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/MIL_label/config/test_config.yaml',
        help='Path to config yaml file'
    )

    # 权重路径
    parser.add_argument('--encoder_ckpt', type=str, help='Path to encoder checkpoint')
    parser.add_argument('--student_ckpt', type=str, help='Path to student checkpoint')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start_id', type=int, default=101)
    parser.add_argument('--max_id', type=int, default=110)
    parser.add_argument('--result_dir', type=str, default='./MIL_label/result')

    args = parser.parse_args()

    # 如果提供了 config，则用 config 覆盖默认参数
    if args.config is not None:
        assert os.path.exists(args.config), f"Config file not found: {args.config}"
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # 必要参数检查
    assert args.encoder_ckpt is not None, "encoder_ckpt is required"
    assert args.student_ckpt is not None, "student_ckpt is required"

    return args

if __name__ == "__main__":

    args = parse_args()
    
    test(args)