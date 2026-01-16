import argparse
import os
import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 引入自定义模块
import dataset
from model import PretrainedResNet18_Encoder, get_transmil_teacher, get_student
from utils import save_checkpoint, get_features_chunked, calculate_metrics, get_pr_figure, get_roc_figure


# ==========================================
# 1. 优化器与训练流程类 (Optimizer)
# ==========================================

class Optimizer:
    def __init__(self, 
                 model_encoder, model_teacher, model_student,
                 opt_encoder, opt_teacher, opt_student,
                 train_bag_loader, train_inst_loader,
                 test_bag_loader, test_inst_loader,
                 writer, warmup, save_dir, args):
        
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
        self.save_dir = save_dir
        self.args = args
        self.warmup = warmup
        self.afwarmup = self.args.afwarmup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        self.best_auc = 0.0 # 保留原有
        self.best_teacher_ap = 0.0  # 记录 Teacher 最佳 AP
        self.best_student_ap = 0.0  # 记录 Student 最佳 AP
        self.recent_checkpoints = [] # 用于管理最新的3个权重文件

        total_instances = len(train_inst_loader.dataset)
        self.pseudo_label_bank = torch.zeros(total_instances).float().to(self.device)

        # 获取数据集的总包数
        total_train_bags = len(self.train_bag_loader.dataset)
        total_test_bags = len(self.test_bag_loader.dataset)

        # 随机锁定 3 个“绝对 ID”用于训练观察 (ID 范围是 0 到 total_train_bags-1)
        if total_train_bags > 3:
            self.fixed_train_ids = np.random.choice(total_train_bags, 3, replace=False)
        else:
            self.fixed_train_ids = np.arange(total_train_bags)
            
        # 随机锁定 3 个“绝对 ID”用于测试观察 (虽然测试集不shuffle，但用ID更保险)
        if total_test_bags > 3:
            # 注意：测试集的 ID 也是从 0 开始计数的，是相对于测试集 dataset 的索引
            self.fixed_test_ids = np.random.choice(total_test_bags, 3, replace=False)
        else:
            self.fixed_test_ids = np.arange(total_test_bags)

    def optimize(self):
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            print(f"\n[Epoch {epoch+1}/{self.args.epochs}]")
            
            # ==========================================
            # 阶段 1: Teacher Warmup (只训练 Teacher)
            # ==========================================
            if epoch < self.warmup:
                print(f"  [Phase: Teacher Warmup] Training Teacher ({epoch+1}/{self.warmup})...")
                
                # 1. Train Teacher
                loss_teacher, avg_attn_min, avg_attn_max = self.optimize_teacher(epoch)
                
                # 2. Eval Teacher
                teacher_auc, teacher_ap = self.evaluate_teacher(epoch)
                print(f"  > Teacher Loss: {loss_teacher:.4f} | Test AUC: {teacher_auc:.4f} | AP: {teacher_ap:.4f}")
                
                if teacher_ap > self.best_teacher_ap:
                    self.best_teacher_ap = teacher_ap
                    save_checkpoint({
                        'epoch': epoch,
                        'teacher': self.teacher.state_dict(),
                        'best_ap': self.best_teacher_ap
                    }, self.save_dir, 'teacher_best.pth')
            # ==========================================
            # 阶段 2: Student Distillation (只训练 Student)
            # ==========================================
            else:
                # 【关键修改】：仅在刚进入 Distill 阶段的第一轮 (epoch == warmup) 生成一次伪标签
                # 之后 Epoch 都不再更新，保持伪标签固定 (Static Pseudo Labels)
                if epoch == self.warmup:
                    print("  [Phase: Distillation Start] Generating Static Pseudo Labels (ONCE)...")
                    self.update_pseudo_labels()
                else:
                    print("  [Phase: Distillation] Using Static Pseudo Labels (No Update)...")
                
                # 1. Train Student (Teacher 不再训练)
                loss_student = self.optimize_student(epoch)
                
                # 2. Evaluate Student
                test_auc, test_ap, test_acc = self.evaluate_student(epoch)
                print(f"  > Student Loss: {loss_student:.4f} | Test AUC: {test_auc:.4f}")
                
                # 保存最佳 Student
                if test_ap > self.best_student_ap and test_acc > 0.9 and epoch >= self.warmup + self.afwarmup:
                    self.best_student_ap = test_ap
                    # 保存 Student
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.student.state_dict(),
                        'best_ap': self.best_student_ap,
                        'acc': test_acc
                    }, self.save_dir, 'student_best.pth')
                    
                    # 保存 Encoder
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': self.encoder.state_dict(),
                        'best_ap': self.best_student_ap,
                        'acc': test_acc
                    }, self.save_dir, 'encoder_best.pth')  
                # 每个 epoch 结束都覆盖保存，保证是 "latest"
            
            # 1. Save Encoder Latest
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.encoder.state_dict(),
                'optimizer': self.opt_encoder.state_dict()
            }, self.save_dir, 'encoder_latest.pth')

            # 2. Save Teacher Latest
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.teacher.state_dict(),
                'optimizer': self.opt_teacher.state_dict()
            }, self.save_dir, 'teacher_latest.pth')

            # 3. Save Student Latest
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.student.state_dict(),
                'optimizer': self.opt_student.state_dict()
            }, self.save_dir, 'student_latest.pth')
            
            print(f"[*] Saved latest checkpoints for Epoch {epoch+1}")

    def update_pseudo_labels(self):
        """
        更新伪标签库。
        适配新版 TransMIL：直接获取 Instance Probabilities，无需归一化。
        """
        self.encoder.train() # 保持 Train 模式以稳定 BN
        self.teacher.eval()
        
        # 调试统计
        all_probs_mean = []
        
        with torch.no_grad():
            for data, label_info, selected_indices in tqdm(self.train_bag_loader, desc="Update Pseudo Labels", leave=False):
                imgs = data.squeeze(0).to(self.device)
                bag_label = label_info[1].item()
                global_indices = selected_indices.squeeze(0).long().to(self.device)
                
                if len(imgs) == 0: continue

                # 1. 提取特征
                feats = get_features_chunked(self.encoder, imgs, batch_size=64)
                
                # 2. Teacher 推理 (适配新 model.py)
                # 返回值: (bag_logits, instance_probs)
                _, instance_probs = self.teacher(feats.unsqueeze(0))
                
                # instance_probs 已经是 [1, N] 的概率值 (0~1)，直接使用
                probs = instance_probs.squeeze(0) # [N]

                # 3. 阴性 Bag 约束 (Hard Negative Mining)
                if bag_label == 0:
                    probs = torch.zeros_like(probs)
                
                # 4. 存入 Bank
                self.pseudo_label_bank[global_indices] = probs.float()
                
                all_probs_mean.append(probs.mean().item())

        print(f"  [Debug] Pseudo Labels Updated: Global Mean = {np.mean(all_probs_mean):.4f}")     

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
        训练 Student。
        使用 Bag-level Loading，直接从 Bank 读取伪标签。
        包含 TensorBoard Debug 绘图：对比 Student 预测与 Teacher 伪标签。
        """
        self.encoder.eval()
        self.student.train()
        self.teacher.eval() 
        
        total_loss = 0.
        
        debug_indices = self.fixed_train_ids
        
        # 使用 Bag Loader (Batch Size = 1)
        pbar = tqdm(self.train_bag_loader, desc="Train Student (Bag)", leave=False)
        
        for batch_idx, (data, label_info, selected_indices) in enumerate(pbar):
            batch_idx = label_info[2].item()
            # data: [1, N, 3, H, W] -> [N, 3, H, W]
            imgs = data.squeeze(0).to(self.device)
            # indices: [1, N] -> [N]
            indices = selected_indices.squeeze(0).long().to(self.device)
            
            if len(imgs) == 0: continue

            # self.opt_encoder.zero_grad()
            self.opt_student.zero_grad()
            
            # 1. 获取伪标签 (直接从 Bank 读取，已经是概率值)
            pseudo_labels = self.pseudo_label_bank[indices]
            
            # 2. Student 前向
            with torch.no_grad():
                feats = self.encoder(imgs)
            student_logits = self.student(feats) # [N, 2]
            student_probs = torch.softmax(student_logits, dim=1) # [N, 2]
            
            # 3. 计算 Loss
            if isinstance(self.student, nn.DataParallel):
                loss = self.student.module.get_loss(student_probs, pseudo_labels, neg_weight=0.5)
            else:
                loss = self.student.get_loss(student_probs, pseudo_labels, neg_weight=0.5)
            
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            
            # self.opt_encoder.step()
            self.opt_student.step()
            
            total_loss += loss.item()
            
            # 4. 日志记录
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss_Student', loss.item(), epoch * len(self.train_bag_loader) + batch_idx)
            
            # 5. TensorBoard Debug 绘图 (画出同一个 Bag 内的所有 Instance)
            if batch_idx in debug_indices:
                try:
                    slot_id = np.where(debug_indices == batch_idx)[0][0]
                    
                    # 准备绘图数据
                    s_probs = student_probs.detach().cpu().numpy()[:, 1] # Student 预测 (Class 1)
                    p_labels = pseudo_labels.detach().cpu().numpy()      # Teacher 伪标签
                    
                    bag_gt = label_info[1]
                    if isinstance(bag_gt, torch.Tensor):
                        bag_gt = bag_gt.item()

                    fig = plt.figure(figsize=(8, 4))
                    x_axis = np.arange(len(s_probs))
                    
                    # 绘制曲线对比
                    plt.plot(x_axis, s_probs, 'b.-', label='Student Pred', alpha=0.6, linewidth=1)
                    plt.plot(x_axis, p_labels, 'r.--', label='Pseudo Label (Teacher)', alpha=0.6, linewidth=1)
                    
                    plt.title(f'Ep {epoch} | Bag {batch_idx} | GT: {bag_gt} | Loss: {loss.item():.4f}')
                    plt.xlabel('Instance Index')
                    plt.ylabel('Probability')
                    plt.ylim(-0.05, 1.05)
                    plt.legend(loc='upper right')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    
                    self.writer.add_figure(f'Train_Debug/Sample_{slot_id}', fig, epoch)
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"[Warning] Plotting failed: {e}")

        return total_loss / len(self.train_bag_loader)

    def evaluate_student(self, epoch):
        """
        评估 Student 性能
        逻辑：Student 对 Test Bag 中的所有 Patch 进行预测，取 Max Score 作为 Bag 分数
        """
        # 【重要】：保持 Encoder 为 train() 模式，
        # 防止 Batch Size=1 时 BN 层使用错误的全局统计量导致模型坍塌。
        self.encoder.eval() 
        self.student.eval()
        
        all_labels = []
        all_preds = []
        
        sample_indices = self.fixed_test_ids
        
        # 使用 Bag Loader 进行测试
        with torch.no_grad():
            for batch_idx, (data, label_info, _) in enumerate(tqdm(self.test_bag_loader, desc="Eval Student", leave=False)):
                imgs = data.squeeze(0).to(self.device)
                label = label_info[1].item()
                
                if len(imgs) == 0: continue
                
                # Forward
                feats = self.encoder(imgs)
                logits = self.student(feats)
                probs = torch.softmax(logits, dim=1)[:, 1] # 取 Class 1 (阳性) 概率
                
                # Aggregation: Max Pooling
                bag_score = probs.max().item()
                
                all_labels.append(label)
                all_preds.append(bag_score)
                
                # --- 【新增】记录采样 Bag 的 Instance Probs ---
                if batch_idx in sample_indices:
                    # 确定这是第几个采样 (0~4)，用于 TensorBoard Tag 固定
                    sample_rank = np.where(sample_indices == batch_idx)[0][0]
                    
                    _, teacher_instance_probs = self.teacher(feats.unsqueeze(0))
                    teacher_probs_np = teacher_instance_probs.squeeze(0).cpu().numpy()
                    probs_np = probs.cpu().numpy()
                    logits_np = logits.cpu().numpy()
                    
                    # 绘制柱状图
                    fig = plt.figure(figsize=(10, 5)) # 稍微调宽一点方便看
                    
                    # 1. 绘制 Student 预测 (柱状图)
                    plt.bar(range(len(probs_np)), probs_np, color='skyblue', alpha=0.6, label='Student Pred')
                    
                    # 2. 绘制 Teacher 预测 (折线图，叠加在柱状图上)
                    plt.plot(range(len(teacher_probs_np)), teacher_probs_np, 
                             color='red', marker='.', linestyle='--', linewidth=1.5, alpha=0.8, label='Teacher (TransMIL)')
                    
                    # 辅助线和标签
                    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
                    plt.ylim(-0.05, 1.1) # Y轴范围
                    plt.xlabel('Instance Index')
                    plt.ylabel('Predicted Probability (Pos)')
                    plt.title(f'Eval Bag {batch_idx} | GT: {label} | Student BagScore: {bag_score:.4f}')
                    plt.legend(loc='upper right')
                    plt.grid(True, axis='y', alpha=0.3)
                    
                    # 记录到 TensorBoard
                    # Tag 例如: 'Eval_Debug/Sample_0', 'Eval_Debug/Sample_1'...
                    self.writer.add_figure(f'Eval_Debug/Sample_{sample_rank}', fig, epoch)
                    plt.close(fig)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Metrics
        roc_auc, acc = calculate_metrics(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        # TensorBoard Logging (Scalars)
        self.writer.add_scalar('Test/AUC', roc_auc, epoch)
        self.writer.add_scalar('Test/ACC', acc, epoch)
        self.writer.add_scalar('Test/AP', ap, epoch)
        
        # Figures (ROC/PR Curves)
        roc_fig = get_roc_figure(all_labels, all_preds)
        pr_fig = get_pr_figure(all_labels, all_preds)
        self.writer.add_figure('Test/ROC_Curve', roc_fig, epoch)
        self.writer.add_figure('Test/PR_Curve', pr_fig, epoch)
        plt.close(roc_fig)
        plt.close(pr_fig)
        
        return roc_auc, ap, acc

def load_pretrained_wrapper(model, ckpt_path, model_name, main_device):
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"[*] Loading {model_name} from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=main_device)
        
        # 1. 尝试获取 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'teacher' in checkpoint: # 兼容旧 Teacher 格式
            state_dict = checkpoint['teacher']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # 假设本身就是 state_dict

        # 2. 处理 'module.' 前缀匹配问题
        # 获取当前模型的 key 格式（是否被 DataParallel 包裹）
        if isinstance(model, nn.DataParallel):
            model_keys = model.module.state_dict().keys()
            is_model_parallel = True
        else:
            model_keys = model.state_dict().keys()
            is_model_parallel = False
        
        # 处理加载权重的 keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            # 如果权重有 module. 但模型没有 -> 去掉
            if k.startswith('module.') and not is_model_parallel:
                name = k[7:]
            # 如果权重没有 module. 但模型有 -> 加上 (通常 PyTorch DataParallel 会自动处理，但手动更稳)
            elif not k.startswith('module.') and is_model_parallel:
                # 注意：如果这里不加，load_state_dict 时 DataParallel 容器可能无法直接匹配
                # 这里简单的做法是先去掉权重的 module. 前缀，然后让 model.module 去加载
                # 或者保留现状，依赖 load_state_dict 的 strict=False
                name = k 
            
            new_state_dict[name] = v
        
        # 3. 执行加载
        try:
            # 如果模型是 DataParallel，建议对 model.module 加载，这样可以忽略前缀差异
            target_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # 再次清洗 key，确保去除 'module.' 以便通过 target_model 加载
            clean_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}
            
            target_model.load_state_dict(clean_state_dict, strict=True)
            print(f"    -> {model_name} loaded successfully (Strict).")
        except RuntimeError as e:
            print(f"    [Warning] Strict loading failed for {model_name}, trying non-strict. Error: {e}")
            target_model.load_state_dict(clean_state_dict, strict=False)
    else:
        if ckpt_path:
            print(f"[Error] Checkpoint not found: {ckpt_path}")
# ==========================================
# 2. 主程序入口
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='EndoKED-MIL Simplified Training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Student training')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1140)
    parser.add_argument('--device_ids', type=str, default='0,1', help='GPU IDs, e.g., 0,1')
    parser.add_argument('--downsample', type=float, default=1, help='Use subset of data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mmked_mil')
    parser.add_argument('--log_dir', type=str, default='./runs/mmked_mil')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='Path to Teacher checkpoint')
    parser.add_argument('--student_ckpt', type=str, default=None, help='Path to Student checkpoint')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to Encoder checkpoint')
    parser.add_argument('--split', default=0.70, type=float, help='训练集占比')
    parser.add_argument('--datasetnum',type=int, default=4, help='使用的数据集数量')
    parser.add_argument('--warmup', type=int, default=5, help='先训练教师')
    parser.add_argument('--afwarmup', type=int, default=3,help='warmup后训练几个epoch再开始保存best model')
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
    run_name = f"MILKED_{current_time}_seed{args.seed}_split{args.split}_lr{args.lr}_ds{args.downsample}_warmup{args.warmup}_dataset{args.datasetnum}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)


    print(f"Using Devices: {device_ids}")
    
    # 2. Dataset Setup
    print("Loading Data...")
    ds_train_raw, ds_test_raw = dataset.gather_align_Img(split=args.split)
    
    # 1. 实例化 Bag 模式的数据集 (作为基准)
    train_ds_bag = dataset.DataSet_MIL(ds_train_raw, downsample=args.downsample, return_bag=True)
    
    # 2. 【关键修改】通过深拷贝创建 Instance 数据集，确保底层数据完全一致
    train_ds_inst = copy.deepcopy(train_ds_bag)
    train_ds_inst.return_bag = False # 修改模式为 Instance
    
    # 3. 同理处理测试集
    test_ds_bag = dataset.DataSet_MIL(ds_test_raw, downsample=args.downsample, return_bag=True)
    # 如果需要 test_inst，也用 copy
    test_ds_inst = copy.deepcopy(test_ds_bag)
    test_ds_inst.return_bag = False
    
    # 创建 Loaders
    train_bag_loader = DataLoader(train_ds_bag, batch_size=1, shuffle=True, num_workers=4)
    train_inst_loader = DataLoader(train_ds_inst, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_bag_loader = DataLoader(test_ds_bag, batch_size=1, shuffle=False, num_workers=4)
    test_inst_loader = DataLoader(test_ds_inst, batch_size=args.batch_size, shuffle=True,num_workers=4)
    
    # 验证一致性 (可选)
    print(f"Train Bags: {len(train_ds_bag)} | Train Instances: {len(train_ds_inst)}")
    # 如果 dataset.py 实现正确，len(train_ds_inst) 应该等于 train_ds_bag 中所有包的 patches 总和
    
    # 3. Model Setup
    print("Initializing Models...")
    
    # Encoder (DataParallel)
    encoder = PretrainedResNet18_Encoder(freeze=False).to(main_device)
    if len(device_ids) > 1:
        encoder = nn.DataParallel(encoder, device_ids=device_ids)
        
    # Teacher (Single GPU recommended for Bag processing, or handled carefully)
    # TransMIL 很难在 Batch=1 的情况下并行，所以通常不加 DataParallel
    teacher = get_transmil_teacher().to(main_device)
    
    # Student (DataParallel)
    student = get_student().to(main_device)
        
    encoder_ckpt = args.encoder_ckpt
    teacher_ckpt = args.teacher_ckpt
    student_ckpt = args.student_ckpt
    load_pretrained_wrapper(encoder, encoder_ckpt, "Encoder", main_device)
    load_pretrained_wrapper(teacher, teacher_ckpt, "Teacher", main_device)
    load_pretrained_wrapper(student, student_ckpt, "Student", main_device)
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
        warmup=args.warmup,
        save_dir = save_dir,
        args=args
    )
    
    optimizer.optimize()
    writer.close()
    print("Training Complete.")

if __name__ == '__main__':
    main()