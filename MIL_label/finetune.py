import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import yaml
import datetime

# 引入你的模型定义和Dataset
from models.model import PretrainedResNet18_Encoder, get_student

from PIL import Image

class FinetuneDataset(Dataset):
    def __init__(self, folder_ids, transform=None):
        """
        Args:
            folder_ids (list): 文件夹 ID 列表 (e.g. [1, 2, 5, ...])
            transform: 图片预处理
        """
        self.img_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/TrainValid/Images'
        self.anno_root_base = '/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/LDPolypVideo/TrainValid/Annotations'
        self.transform = transform
        self.samples = [] 

        # 遍历指定的文件夹列表
        for folder_id in tqdm(folder_ids):
            folder_name = str(folder_id)
            img_dir = os.path.join(self.img_root_base, folder_name)
            anno_dir = os.path.join(self.anno_root_base, folder_name)

            if not os.path.exists(img_dir) or not os.path.exists(anno_dir):
                continue

            for filename in os.listdir(img_dir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(img_dir, filename)
                    txt_name = os.path.splitext(filename)[0] + '.txt'
                    txt_path = os.path.join(anno_dir, txt_name)

                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    # 解析逻辑：首个数字 > 0 为正例(1)，否则为负例(0)
                                    num_polyps = int(content.split()[0])
                                    label = 1 if num_polyps > 0 else 0
                                    self.samples.append((img_path, label))
                        except Exception:
                            pass
        
        print(f"[*] Dataset initialized with folders {min(folder_ids)}-{max(folder_ids)} (Total: {len(folder_ids)} folders). Found {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
# ==========================================
# 辅助函数: 加载权重
# ==========================================
def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Info] No checkpoint found at '{ckpt_path}'. Training from scratch/ImageNet.")
        return

    print(f"[*] Loading weights from: {ckpt_path}")
    # 添加 weights_only=False 解决 PyTorch 2.6+ 报错
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 处理 DataParallel 的 module. 前缀
    is_model_parallel = isinstance(model, nn.DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.') and not is_model_parallel:
            name = k[7:]
        elif not k.startswith('module.') and is_model_parallel:
            name = 'module.' + k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("    -> Loaded successfully.")

# ==========================================
# 训练主逻辑
# ==========================================
def main(args):
    # 1. 环境设置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"FT_{current_time}_seed{args.seed}_split{args.split}_lr{args.lr}_batchsize{args.batch_size}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)


    # 2. 数据准备 (按文件夹 ID 划分)
    all_ids = list(range(1, 101)) # 1 到 100
    train_ids, val_ids = train_test_split(all_ids, train_size=args.split, random_state=args.seed, shuffle=True)
    
    print(f"Split: {len(train_ids)} Train Folders, {len(val_ids)} Valid Folders")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FinetuneDataset(folder_ids=train_ids, transform=train_transform)
    val_dataset = FinetuneDataset(folder_ids=val_ids, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Computing class weights...")
    num_pos = sum([s[1] for s in train_dataset.samples])
    num_neg = len(train_dataset) - num_pos
    print(f"[*] Train Set: Positive={num_pos}, Negative={num_neg}")
    
    # 2. 设置权重：负样本多，权重低；正样本少，权重高
    # 这里的 weight 只需要传给 CrossEntropyLoss
    # 通常 weight = total_samples / (num_classes * count_class_i)
    # 或者简单粗暴一点：weight_pos = num_neg / num_pos
    
    if num_pos > 0:
        pos_weight = num_pos / num_neg
        # weights 对应类别 [0, 1] -> [Background, Polyp]
        class_weights = torch.tensor([1.0, float(pos_weight)]).to(device)
        print(f"[*] Using Class Weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("[!] Warning: No positive samples in train set.")
        criterion = nn.CrossEntropyLoss()
    # 3. 模型初始化与预训练权重加载
    print("Initializing models...")
    encoder = PretrainedResNet18_Encoder(freeze=False).to(device) # 微调时Encoder通常不冻结
    student = get_student().to(device)

    # 加载之前的 MIL 预训练权重
    load_checkpoint(encoder, args.pretrained_encoder, device)
    load_checkpoint(student, args.pretrained_student, device)

    # 4. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    
    # Encoder 使用较小的学习率，Student 使用正常的学习率
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': float(args.lr) * 0.05}, # 从 0.1 改为 0.01
        {'params': student.parameters(), 'lr': args.lr}
    ], weight_decay=1e-3) # 增加 weight_decay 防止过拟合 (从 1e-4 改为 1e-3)

    best_val_auc = 0.0

    # 5. 训练循环
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # --- Training ---
        encoder.train()
        student.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            feats = encoder(imgs)
            logits = student(feats)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 记录用于计算Epoch指标
            probs = torch.softmax(logits, dim=1)[:, 1]
            train_preds.extend(probs.detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, np.array(train_preds) > 0.5)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)

        # --- Validation ---
        encoder.eval()
        student.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(device), labels.to(device)
                
                feats = encoder(imgs)
                logits = student(feats)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, np.array(val_preds) > 0.5)
        
        # 计算AUC (防止单类别报错)
        if len(np.unique(val_targets)) > 1:
            val_auc = roc_auc_score(val_targets, val_preds)
        else:
            val_auc = 0.0

        print(f"  Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        writer.add_scalar('Val/AUC', val_auc, epoch)

        # --- Save Best ---
        if val_auc > best_val_auc and val_acc > 0.85:
            best_val_auc = val_auc
            print(f"  [*] New Best Auc: {best_val_auc:.4f}. Saving models...")
            
            # 保存微调后的 Encoder
            torch.save({
                'epoch': epoch,
                'state_dict': encoder.state_dict(),
                'best_auc': best_val_auc
            }, os.path.join(save_dir, 'finetuned_encoder_best.pth'))
            
            # 保存微调后的 Student
            torch.save({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_auc': best_val_auc
            }, os.path.join(save_dir, 'finetuned_student_best.pth'))

    writer.close()
    print("Finetuning Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/MIL_label/config/finetune_config.yaml',
        help='Path to config yaml file'
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=127)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--split', type=float, default=0.7)
    
    # 路径配置
    parser.add_argument('--pretrained_encoder', type=str, help='Path to MIL pretrained encoder')
    parser.add_argument('--pretrained_student', type=str, help='Path to MIL pretrained student')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/finetune')
    parser.add_argument('--log_dir', type=str, default='./runs/finetune')
    
    args = parser.parse_args()

    # 如果提供了 config，则用 config 覆盖默认参数
    if args.config is not None:
        assert os.path.exists(args.config), f"Config file not found: {args.config}"
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    main(args)