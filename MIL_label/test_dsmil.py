import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F

# 引入 TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 导入现有模块
from dataset import DataSet_MIL, gather_align_Img
from model_head import PretrainedResNet18_Encoder, Bag_Classifier_DSMIL_Head
import utils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='Debug DSMIL Training')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0002, type=float, help='DSMIL通常可以用比ABMIL稍大的LR')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--downsample', default=0.05, type=float, help='使用多少比例的数据进行调试')
    parser.add_argument('--seed', default=42, type=int)
    # 新增 log_dir 参数
    parser.add_argument('--log_dir', default='./runs/runs_dsmil', type=str, help='Tensorboard 日志目录')
    return parser.parse_args()

def train_one_epoch(encoder, head, loader, optimizer, device, epoch, writer):
    encoder.train()
    head.train()
    
    total_loss = 0.0
    all_probs = []
    all_labels = []

    print(f"\n[Epoch {epoch}] Training Started...")
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ascii=True)
    criterion_bag = nn.CrossEntropyLoss()
    
    # 获取总步数用于 Tensorboard x轴
    steps_per_epoch = len(loader)

    for batch_idx, (data, label_info, _) in enumerate(pbar):
        global_step = epoch * steps_per_epoch + batch_idx

        bag_imgs = data.squeeze(0).to(device) 
        bag_label = label_info[1].long().to(device) 
        
        optimizer.zero_grad()
        
        # --- 前向传播 ---
        feats = encoder(bag_imgs)
        instance_scores, bag_score, U, b_embed = head(feats)
        
        # --- 计算 Loss ---
        # Part A: Bag Loss
        loss_bag = criterion_bag(bag_score, bag_label)
        
        # Part B: Max Instance Loss
        max_id = torch.argmax(instance_scores[:, 1])
        bag_pred_byMax = instance_scores[max_id, 1] 
        bag_label_float = bag_label.float()
        p_max = torch.sigmoid(bag_pred_byMax)
        
        loss_max = -1. * (bag_label_float * torch.log(p_max + 1e-6) + (1. - bag_label_float) * torch.log(1. - p_max + 1e-6))
        
        loss = 0.5 * loss_bag + 0.5 * loss_max
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # --- 收集结果 ---
        probs = torch.softmax(bag_score, dim=1) 
        prob_pos = probs[0, 1]
        all_probs.append(prob_pos.item())
        all_labels.append(bag_label.item())
        
        # ================= TensorBoard Logging (Step Level) =================
        # 1. 记录 Loss 详情 (帮助判断哪一部分 Loss 占主导)
        writer.add_scalar('Train_Step/Loss_Total', loss.item(), global_step)
        writer.add_scalar('Train_Step/Loss_Bag', loss_bag.item(), global_step)
        writer.add_scalar('Train_Step/Loss_MaxInstance', loss_max.item(), global_step)
        
        # 2. 记录 DSMIL 内部动态 (非常重要)
        # Attention Mean/Max: 如果 Max 长期接近 Mean，说明 Attention 坍塌(失效)
        writer.add_scalar('Debug/Attention_Max', U.max().item(), global_step)
        writer.add_scalar('Debug/Attention_Mean', U.mean().item(), global_step)
        
        # 3. 记录预测概率对比
        # 观察 Bag 分类器和 Max Instance 分类器是否一致
        writer.add_scalar('Debug/Prob_Bag', prob_pos.item(), global_step)
        writer.add_scalar('Debug/Prob_MaxInstance', p_max.item(), global_step)
        
        # 4. 记录学习率
        writer.add_scalar('Train_Step/LR', optimizer.param_groups[0]['lr'], global_step)

        # 5. 直方图 (不要每个 step 都记，太占空间，每 50 个 step 记一次)
        if batch_idx % 50 == 0:
            writer.add_histogram('Debug_Hist/Attention_Weights', U, global_step)
            writer.add_histogram('Debug_Hist/Instance_Scores_Class1', instance_scores[:, 1], global_step)
        # ====================================================================

        # 打印日志
        if batch_idx % 130 == 0:
            u_min = U.min().item()
            u_max = U.max().item()
            u_mean = U.mean().item()
            tqdm.write(
                f"\n[Batch {batch_idx}] Label: {bag_label.item()} | Loss: {loss.item():.4f}\n"
                f"    Attn -> Min: {u_min:.4f} | Max: {u_max:.4f} | Mean: {u_mean:.4f}"
            )

    # --- Epoch 结束计算指标 ---
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    epoch_auc = 0.0
    if len(np.unique(all_labels)) > 1:
        epoch_auc = metrics.roc_auc_score(all_labels, all_probs)
        
    avg_loss = total_loss / len(loader)
    
    # ================= TensorBoard Logging (Epoch Level) =================
    writer.add_scalar('Train_Epoch/Average_Loss', avg_loss, epoch)
    writer.add_scalar('Train_Epoch/AUC', epoch_auc, epoch)
    # =====================================================================
    
    print(f"Epoch {epoch} Result: Loss={avg_loss:.4f}, AUC={epoch_auc:.4f}")
    return avg_loss, epoch_auc

def validate(encoder, head, loader, device, epoch, writer):
    encoder.eval()
    head.eval()
    
    all_probs = []
    all_labels = []
    
    print("Validating...")
    with torch.no_grad():
        for batch_idx, (data, label_info, _) in enumerate(tqdm(loader, ascii=True)):
            bag_imgs = data.squeeze(0).to(device)
            bag_label = label_info[1].long().to(device)
            
            feats = encoder(bag_imgs)
            _, bag_score, _, _ = head(feats)
            
            prob_pos = torch.softmax(bag_score, dim=1)[0, 1]
            
            all_probs.append(prob_pos.item())
            all_labels.append(bag_label.item())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    val_auc = 0.0
    if len(np.unique(all_labels)) > 1:
        val_auc = metrics.roc_auc_score(all_labels, all_probs)
        
    print(f"Validation AUC: {val_auc:.4f}")
    
    # ================= TensorBoard Logging (Validation) =================
    writer.add_scalar('Val/AUC', val_auc, epoch)
    # ====================================================================
    
    return val_auc

def main():
    args = get_args()
    setup_seed(args.seed)
    
    # 初始化 TensorBoard Writer
    # 日志将保存在 runs/dsmil_experiment/ 下
    # 建议加上时间戳或 seed 以区分不同实验
    log_path = os.path.join(args.log_dir, f"seed_{args.seed}_lr_{args.lr}")
    writer = SummaryWriter(log_dir=log_path)
    print(f"Tensorboard log directory: {log_path}")
    
    print("Loading Data...")
    ds_train, ds_test = gather_align_Img()
    
    train_ds = DataSet_MIL(ds_train, downsample=args.downsample, transform=None, return_bag=True)
    val_ds = DataSet_MIL(ds_test, downsample=args.downsample, transform=None, return_bag=True)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Train Bags: {len(train_ds)}, Val Bags: {len(val_ds)}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Initializing Models...")
    
    encoder = PretrainedResNet18_Encoder().to(device)
    
    head = Bag_Classifier_DSMIL_Head(
        features=None, 
        num_classes=[2], 
        init=True, 
        input_feat_dim=512
    ).to(device)
    
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': head.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    best_auc = 0.0
    for epoch in range(args.epochs):
        # 传入 writer
        loss, auc = train_one_epoch(encoder, head, train_loader, optimizer, device, epoch, writer)
        
        if (epoch + 1) % 1 == 0:
            # 传入 writer
            val_auc = validate(encoder, head, val_loader, device, epoch, writer)
            if val_auc > best_auc:
                best_auc = val_auc
                print(f"*** New Best AUC: {best_auc:.4f} ***")
                
                # 可选：保存最佳模型
                # torch.save(head.state_dict(), os.path.join(log_path, 'best_head.pth'))

    # 训练结束后关闭 writer
    writer.close()

if __name__ == '__main__':
    main()