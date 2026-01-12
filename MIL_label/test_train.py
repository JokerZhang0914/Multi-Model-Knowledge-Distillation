import argparse
import os
import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 引入自定义模块
import dataset
from model import PretrainedResNet18_Encoder, get_transmil_teacher, get_student

# ==========================================
# 1. 辅助功能函数封装 (Utils)
# ==========================================

def calculate_metrics(y_true, y_pred_prob):
    """
    计算基础指标: AUC, ACC (阈值0.5)
    """
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = 0.0
    
    y_pred_bin = (np.array(y_pred_prob) > 0.5).astype(int)
    acc = (y_true == y_pred_bin).mean()
    return roc_auc, acc

def get_roc_figure(y_true, y_pred_prob):
    """
    绘制 ROC 曲线并返回 matplotlib figure 对象
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return fig

def get_pr_figure(y_true, y_pred_prob):
    """
    绘制 PR 曲线并返回 matplotlib figure 对象
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    
    fig = plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    return fig

def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"[*] Model saved to {path}")

# ==========================================
# 2. 优化器与训练流程类 (Optimizer)
# ==========================================

class Optimizer:
    def __init__(self, 
                 model_encoder, model_teacher, model_student,
                 opt_encoder, opt_teacher, opt_student,
                 train_bag_loader, train_inst_loader,
                 test_bag_loader, test_inst_loader,
                 writer, args):
        
        self.encoder = model_encoder
        self.teacher = model_teacher
        self.student = model_student
        
        self.opt_encoder = opt_encoder
        self.opt_teacher = opt_teacher
        self.opt_student = opt_student
        
        self.train_bag_loader = train_bag_loader
        self.train_inst_loader = train_inst_loader
        self.test_bag_loader = test_bag_loader
        self.test_inst_loader = test_inst_loader
        
        self.writer = writer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 统计 Teacher Attention 分数的极值，用于归一化
        # 初始化为较大的反向值
        self.attn_min = 0.0
        self.attn_max = 1.0 
        
        self.best_auc = 0.0

    def optimize(self):
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            print(f"\n[Epoch {epoch+1}/{self.args.epochs}]")
            
            # --- 1. Train Teacher (Bag Level) ---
            loss_teacher, avg_attn_min, avg_attn_max = self.optimize_teacher(epoch)
            teacher_auc, teacher_ap = self.evaluate_teacher(epoch)
            print(f"  > Teacher Test AUC: {teacher_auc:.4f} | AP: {teacher_ap:.4f}")
            
            # 更新全局统计量 (简单的移动平均或直接使用本Epoch统计)
            # 这里为了简化，直接使用本 Epoch 的统计值作为下一阶段 Student 的参考
            self.attn_min = avg_attn_min
            self.attn_max = avg_attn_max
            
            # --- 2. Train Student (Instance Level via Distillation) ---
            loss_student = self.optimize_student(epoch)
            
            # --- 3. Evaluate Student (Bag Level Aggregation) ---
            test_auc, test_ap, test_acc = self.evaluate_student(epoch)
            
            print(f"  > Teacher Loss: {loss_teacher:.4f} | Attn Range: [{self.attn_min:.4f}, {self.attn_max:.4f}]")
            print(f"  > Student Loss: {loss_student:.4f}")
            print(f"  > Student Test AUC: {test_auc:.4f} | AP: {test_ap:.4f} | ACC: {test_acc:.4f}")
            
            # Save Best Model
            if test_auc > self.best_auc:
                self.best_auc = test_auc
                save_checkpoint({
                    'epoch': epoch,
                    'encoder': self.encoder.state_dict(),
                    'teacher': self.teacher.state_dict(),
                    'student': self.student.state_dict(),
                    'best_auc': self.best_auc
                }, self.args.save_dir, 'best_model.pth')
            
            # Save Latest
            save_checkpoint({
                'epoch': epoch,
                'encoder': self.encoder.state_dict(),
                'teacher': self.teacher.state_dict(),
                'student': self.student.state_dict(),
            }, self.args.save_dir, 'latest_model.pth')

    def optimize_teacher(self, epoch):
        """
        训练 Teacher (TransMIL) 并收集 Attention 分数统计信息
        """
        self.encoder.train()
        self.teacher.train()
        # Student 不在此时更新
        self.student.eval() 
        
        total_loss = 0.
        all_attn_mins = []
        all_attn_maxs = []
        
        # Bag Loader: batch_size=1, returns [1, N, 3, H, W]
        pbar = tqdm(self.train_bag_loader, desc="Train Teacher", leave=False)
        
        for batch_idx, (data, label_info, _) in enumerate(pbar):
            # data: [1, N, 3, 512, 512] -> squeeze -> [N, 3, 512, 512]
            imgs = data.squeeze(0).to(self.device)
            label = label_info[1].long().to(self.device) # Slide Label
            
            if len(imgs) == 0: continue

            self.opt_encoder.zero_grad()
            self.opt_teacher.zero_grad()
            
            # 1. Feature Extraction (DataParallel supports [N, C, H, W])
            feats = self.encoder(imgs) # [N, 512]
            
            # 2. TransMIL Forward
            # TransMIL needs [B, N, Dim], here B=1.
            feats_seq = feats.unsqueeze(0) # [1, N, 512]
            
            # 由于 TransMIL 内部很难并行(batch=1)，我们通常只把模型放在主卡或单卡
            # 如果 teacher 被 DataParallel 包装且 batch=1，可能会报错或无效，
            # 建议在初始化时 Teacher 不使用 DataParallel，或者在这里取出 module
            if isinstance(self.teacher, nn.DataParallel):
                logits, attn_scores = self.teacher.module(feats_seq)
            else:
                logits, attn_scores = self.teacher(feats_seq)
                
            # 3. Loss Calculation
            loss = self.teacher.get_loss(logits, label)
            
            loss.backward()
            
            # Gradient Clipping
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            nn.utils.clip_grad_norm_(self.teacher.parameters(), 5.0)
            
            self.opt_encoder.step()
            self.opt_teacher.step()
            
            total_loss += loss.item()
            
            # 收集统计信息 (detach)
            with torch.no_grad():
                all_attn_mins.append(attn_scores.min().item())
                all_attn_maxs.append(attn_scores.max().item())
                
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss_Teacher', loss.item(), epoch * len(self.train_bag_loader) + batch_idx)

        avg_loss = total_loss / len(self.train_bag_loader)
        
        # 计算整个 Epoch 的 Attention 范围，用于下一个阶段的归一化
        # 也可以使用 np.percentile 排除异常值
        epoch_attn_min = np.min(all_attn_mins)
        epoch_attn_max = np.max(all_attn_maxs)
        
        return avg_loss, epoch_attn_min, epoch_attn_max

    # [Add this method to Optimizer class]
    def evaluate_teacher(self, epoch):
        """
        评估 Teacher 在测试集上的性能 (Bag Level)
        """
        self.encoder.eval()
        self.teacher.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for data, label_info, _ in self.test_bag_loader: # 使用 Test Bag Loader
                imgs = data.squeeze(0).to(self.device)
                label = label_info[1].item()
                
                if len(imgs) == 0: continue
                
                # 1. 提取特征
                feats = self.encoder(imgs) # [N, 512]
                feats_seq = feats.unsqueeze(0) # [1, N, 512]
                
                # 2. Teacher 前向传播
                if isinstance(self.teacher, nn.DataParallel):
                    logits, _ = self.teacher.module(feats_seq)
                else:
                    logits, _ = self.teacher(feats_seq)
                
                # 3. 获取 Bag 概率 (Class 1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                bag_score = probs.item()
                
                all_labels.append(label)
                all_preds.append(bag_score)
        
        # 计算指标
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        roc_auc, acc = calculate_metrics(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        # 记录到 TensorBoard
        self.writer.add_scalar('Test/Teacher_AUC', roc_auc, epoch)
        self.writer.add_scalar('Test/Teacher_AP', ap, epoch)
        
        return roc_auc, ap
    
    def optimize_student(self, epoch):
        """
        训练 Student (MLP) 使用 Teacher 的 Attention 作为伪标签
        """
        self.encoder.train()
        self.student.train()
        self.teacher.eval() # Teacher 固定用于生成标签
        
        total_loss = 0.
        
        # Instance Loader: batch_size=64 (or user defined), returns [B, 3, H, W]
        pbar = tqdm(self.train_inst_loader, desc="Train Student", leave=False)
        
        for batch_idx, (imgs, label_info, _) in enumerate(pbar):
            imgs = imgs.to(self.device)
            # label_info[0] is Patch Label (usually 0), label_info[1] is Bag Label
            bag_labels = label_info[1].float().to(self.device)
            
            self.opt_encoder.zero_grad()
            self.opt_student.zero_grad()
            
            # 1. Feature Extraction
            feats = self.encoder(imgs) # [B, 512]
            
            # 2. Get Pseudo Labels from Teacher
            # Teacher needs Sequence Input: [1, B, 512] (treating the batch as a mini-bag)
            # 注意：这里我们将一个随机 batch 当作一个 bag 输入给 Teacher 来获取相对分数
            # 这种做法是合理的，因为我们希望 Student 学习的是"在这个集合中谁更重要"
            with torch.no_grad():
                feats_seq = feats.unsqueeze(0)
                if isinstance(self.teacher, nn.DataParallel):
                    _, attn_scores = self.teacher.module(feats_seq)
                else:
                    _, attn_scores = self.teacher(feats_seq)
                
                # attn_scores: [1, B] -> [B]
                attn_scores = attn_scores.squeeze(0)
                
                # Normalize to [0, 1] using Teacher's statistics from previous step
                pseudo_labels = (attn_scores - self.attn_min) / (self.attn_max - self.attn_min + 1e-8)
                pseudo_labels = torch.clamp(pseudo_labels, 0, 1)
                
                # Negative Bag Constraint: 
                # 如果该 Batch 属于阴性 Bag，则所有 Patch 标签强制为 0
                # label_info[1] 是 Tensor [B]
                is_neg_bag = (bag_labels == 0)
                pseudo_labels[is_neg_bag] = 0.0
                
            # 3. Student Forward
            student_logits = self.student(feats) # [B, 2]
            student_probs = torch.softmax(student_logits, dim=1)
            
            # 4. Distillation Loss
            # Student 学习拟合 pseudo_labels (即 Class 1 的概率)
            if isinstance(self.student, nn.DataParallel):
                loss = self.student.module.get_loss(student_probs, pseudo_labels, neg_weight=0.5)
            else:
                loss = self.student.get_loss(student_probs, pseudo_labels, neg_weight=0.5)
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            
            self.opt_encoder.step()
            self.opt_student.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss_Student', loss.item(), epoch * len(self.train_inst_loader) + batch_idx)
                
        return total_loss / len(self.train_inst_loader)

    def evaluate_student(self, epoch):
        """
        评估 Student 性能
        逻辑：Student 对 Test Bag 中的所有 Patch 进行预测，取 Max Score 作为 Bag 分数
        """
        self.encoder.eval()
        self.student.eval()
        
        all_labels = []
        all_preds = []
        
        # 使用 Bag Loader 进行测试，方便聚合
        with torch.no_grad():
            for data, label_info, _ in tqdm(self.test_bag_loader, desc="Eval Student", leave=False):
                imgs = data.squeeze(0).to(self.device)
                label = label_info[1].item()
                
                if len(imgs) == 0: continue
                
                # Forward
                feats = self.encoder(imgs)
                logits = self.student(feats)
                probs = torch.softmax(logits, dim=1)[:, 1] # 取 Class 1 概率
                
                # Aggregation: Max Pooling
                bag_score = probs.max().item()
                
                all_labels.append(label)
                all_preds.append(bag_score)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Metrics
        roc_auc, acc = calculate_metrics(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        # TensorBoard Logging
        self.writer.add_scalar('Test/AUC', roc_auc, epoch)
        self.writer.add_scalar('Test/ACC', acc, epoch)
        self.writer.add_scalar('Test/AP', ap, epoch)
        
        # Figures
        roc_fig = get_roc_figure(all_labels, all_preds)
        pr_fig = get_pr_figure(all_labels, all_preds)
        self.writer.add_figure('Test/ROC_Curve', roc_fig, epoch)
        self.writer.add_figure('Test/PR_Curve', pr_fig, epoch)
        plt.close(roc_fig)
        plt.close(pr_fig)
        
        return roc_auc, ap, acc

# ==========================================
# 3. 主程序入口
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='EndoKED-MIL Simplified Training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Student training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device_ids', type=str, default='0,1', help='GPU IDs, e.g., 0,1')
    parser.add_argument('--downsample', type=float, default=0.1, help='Use subset of data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mmked_mil')
    parser.add_argument('--log_dir', type=str, default='./runs/mmked_mil')
    parser.add_argument('--teacher_ckpt', type=str, default='/home/zhaokaizhang/code/test_code/runs/runs_transmil/TRANSMIL_20260109-015353_seed206_split0.75_lr5e-05_ds0.9/best_model.pth', 
                        help='Path to pretrained Teacher (TransMIL) checkpoint. If None, train from scratch.')
    parser.add_argument('--split', default=0.7, type=float, help='训练集占比')
    parser.add_argument('--datasetnum',type=int, default=2, help='使用的数据集数量')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Setup Environment
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 解析 GPU 列表
    device_ids = [int(x) for x in args.device_ids.split(',')]
    main_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"MILKED_{current_time}_seed{args.seed}_split{args.split}_lr{args.lr}_ds{args.downsample}dataset{args.datasetnum}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    print(f"Using Devices: {device_ids}")
    
    # 2. Dataset Setup
    print("Loading Data...")
    ds_train_raw, ds_test_raw = dataset.gather_align_Img(split=0.7)
    
    # Teacher Loaders (Bag Mode, Batch=1)
    train_ds_bag = dataset.DataSet_MIL(ds_train_raw, downsample=args.downsample, return_bag=True)
    test_ds_bag = dataset.DataSet_MIL(ds_test_raw, downsample=args.downsample, return_bag=True)
    
    train_bag_loader = DataLoader(train_ds_bag, batch_size=1, shuffle=True, num_workers=4)
    test_bag_loader = DataLoader(test_ds_bag, batch_size=1, shuffle=False, num_workers=4)
    
    # Student Loaders (Instance Mode, Batch=N)
    train_ds_inst = dataset.DataSet_MIL(ds_train_raw, downsample=args.downsample, return_bag=False)
    # Student Testing uses Bag Loader for aggregation, so we don't strictly need a test_inst_loader unless evaluating per-patch
    test_inst_loader = DataLoader(dataset.DataSet_MIL(ds_test_raw, downsample=args.downsample, return_bag=False), 
                                  batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    train_inst_loader = DataLoader(train_ds_inst, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Train Bags: {len(train_ds_bag)}, Test Bags: {len(test_ds_bag)}")
    
    # 3. Model Setup
    print("Initializing Models...")
    
    # Encoder (DataParallel)
    encoder = PretrainedResNet18_Encoder(freeze=False).to(main_device)
    if len(device_ids) > 1:
        encoder = nn.DataParallel(encoder, device_ids=device_ids)
        
    # Teacher (Single GPU recommended for Bag processing, or handled carefully)
    # TransMIL 很难在 Batch=1 的情况下并行，所以通常不加 DataParallel
    teacher = get_transmil_teacher().to(main_device)

    # [新增] 加载预训练 Teacher 权重
    if args.teacher_ckpt is not None:
        if os.path.isfile(args.teacher_ckpt):
            print(f"Loading Pretrained Teacher from: {args.teacher_ckpt}")
            checkpoint = torch.load(args.teacher_ckpt, map_location=main_device)
            
            # 处理 checkpoint 字典结构
            # 情况 A: 这是一个完整的 checkpoint (包含 epoch, model_state_dict 等)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'teacher' in checkpoint: # 兼容我们之前定义的 save_checkpoint 格式
                state_dict = checkpoint['teacher']
            else:
                # 情况 B: 这是一个纯 state_dict
                state_dict = checkpoint

            # 处理 'module.' 前缀 (以防预训练时用了 DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # 加载权重 (使用 strict=False 以防 wrapper 层命名微小差异，但建议先尝试 True)
            try:
                teacher.model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                print(f"[Warning] Strict loading failed, trying strict=False. Error: {e}")
                teacher.model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"[Error] Teacher checkpoint not found at {args.teacher_ckpt}")
            return
    
    # Student (DataParallel)
    student = get_student().to(main_device)
    if len(device_ids) > 1:
        student = nn.DataParallel(student, device_ids=device_ids)
        
    # 4. Optimizers
    # 通常 Encoder 学习率较低或冻结
    opt_encoder = optim.Adam(encoder.parameters(), lr=args.lr * 0.1) 
    teacher_lr = args.lr * 0.1 if args.teacher_ckpt is not None else args.lr
    opt_teacher = optim.Adam(teacher.parameters(), lr=teacher_lr, weight_decay=1e-5)
    # opt_teacher = optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_student = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 5. Start Training
    optimizer = Optimizer(
        model_encoder=encoder,
        model_teacher=teacher,
        model_student=student,
        opt_encoder=opt_encoder,
        opt_teacher=opt_teacher,
        opt_student=opt_student,
        train_bag_loader=train_bag_loader,
        train_inst_loader=train_inst_loader,
        test_bag_loader=test_bag_loader,
        test_inst_loader=test_inst_loader,
        writer=writer,
        args=args
    )
    
    optimizer.optimize()
    writer.close()
    print("Training Complete.")

if __name__ == '__main__':
    main()