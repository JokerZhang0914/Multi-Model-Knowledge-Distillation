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

# 导入你现有的模块
from dataset import DataSet_MIL, gather_align_Img
from model_head import PretrainedResNet18_Encoder, Bag_Classifier_Attention_Head

# 设置随机种子以保证可复现性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='Debug ABMIL Training')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate') # 降低了默认LR，MIL对此敏感
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--downsample', default=0.2, type=float, help='使用多少比例的数据进行调试')
    parser.add_argument('--seed', default=42, type=int)
    # 新增 log_dir 参数
    parser.add_argument('--log_dir', default='./runs/runs_abmil', type=str, help='Tensorboard 日志目录')
    return parser.parse_args()

def train_one_epoch(encoder, head, loader, optimizer, device, epoch, writer):
    encoder.train()
    head.train()
    
    total_loss = 0.0
    all_probs = []
    all_labels = []

    print(f"\n[Epoch {epoch}] Training Started...")
    
    # 进度条
    pbar = tqdm(loader, desc=f'Epoch {epoch} Train', ascii=True)
    
    # 获取总步数用于 Tensorboard x轴
    steps_per_epoch = len(loader)
    
    for batch_idx, (data, label_info, _) in enumerate(pbar):
        global_step = epoch * steps_per_epoch + batch_idx
        
        # 1. 解包数据
        # data: [1, N_patches, 3, 512, 512] -> [N, 3, 512, 512]
        bag_imgs = data.squeeze(0).to(device) 
        
        # label_info: [patch_labels, slide_label, slide_index, slide_name]
        # 我们只需要 slide_label (bag label)
        bag_label = label_info[1].float().to(device) # [1]
        
        # 2. 前向传播
        optimizer.zero_grad()
        
        # 2.1 特征提取
        # feats: [N, 512]
        feats = encoder(bag_imgs)
        
        # --- DEBUG: 检查特征是否正常 ---
        feat_norm = feats.norm(p=2, dim=1).mean().item()
        if torch.isnan(feats).any():
            print(f"[Error] NaN detected in Features at Batch {batch_idx}")
            continue

        # 2.2 MIL Head 前向传播
        # Bag_Classifier_Attention_Head 返回: logits, 0, Attention_Weights
        logits, _, attention_weights = head(feats)
        
        # logits: [1, num_classes] (Logits before Softmax)
        # attention_weights: [1, N] (Softmaxed weights, sum=1)
        
        # 3. 计算损失
        probs = torch.softmax(logits, dim=1) # [1, 2]
        prob_pos = probs[0, 1]
        
        # 计算 BCE Loss
        loss = -1. * (bag_label * torch.log(prob_pos + 1e-6) + (1. - bag_label) * torch.log(1. - prob_pos + 1e-6))
        
        loss.backward()
        
        # --- DEBUG: 梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集用于计算 Epoch AUC
        all_probs.append(prob_pos.item())
        all_labels.append(bag_label.item())
        
        # ================= TensorBoard Logging (Step Level) =================
        # 1. 基础 Loss 和 LR
        writer.add_scalar('Train_Step/Loss', loss.item(), global_step)
        writer.add_scalar('Train_Step/LR', optimizer.param_groups[0]['lr'], global_step)
        
        # 2. Attention 统计 (ABMIL 的核心)
        # 如果 Max 很小，说明 Attention 没学好
        attn_min = attention_weights.min().item()
        attn_max = attention_weights.max().item()
        attn_mean = attention_weights.mean().item()
        
        writer.add_scalar('Debug/Attention_Max', attn_max, global_step)
        writer.add_scalar('Debug/Attention_Mean', attn_mean, global_step)
        writer.add_scalar('Debug/Attention_Min', attn_min, global_step)
        
        # 3. 特征统计 (监控 ResNet Encoder 是否稳定)
        writer.add_scalar('Debug/Feature_Norm', feat_norm, global_step)
        
        # 4. 预测值监控 (看是否一直输出0或1)
        writer.add_scalar('Debug/Prediction_Prob', prob_pos.item(), global_step)

        # 5. 直方图 (每 50 个 batch 记录一次，太频繁会使日志很大)
        if batch_idx % 50 == 0:
            writer.add_histogram('Debug_Hist/Attention_Weights', attention_weights, global_step)
            # 也可以看特征分布
            # writer.add_histogram('Debug_Hist/Features', feats, global_step)
        # ====================================================================
        
        # --- DEBUG: 详细打印 (每10个Batch打印一次) ---
        if batch_idx % 10 == 0:
            num_patches = bag_imgs.size(0)
            uniform_val = 1.0 / num_patches
            
            tqdm.write(
                f"\n[Batch {batch_idx}] "
                f"Label: {int(bag_label.item())} | Pred: {prob_pos.item():.4f} | Loss: {loss.item():.4f} | "
                f"N_Patch: {num_patches} | FeatNorm: {feat_norm:.2f}\n"
                f"    Attn Stats -> Min: {attn_min:.4f} | Max: {attn_max:.4f} (Uniform would be {uniform_val:.4f})"
            )
            
            if attn_max < uniform_val * 1.5:
                tqdm.write("    [Warning] Attention is very flat! (Model might be guessing)")

    # 计算 Epoch 指标
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 防止只有一个类别导致 AUC 报错
    if len(np.unique(all_labels)) > 1:
        epoch_auc = metrics.roc_auc_score(all_labels, all_probs)
    else:
        epoch_auc = 0.0
        
    epoch_acc = ((all_probs > 0.5) == all_labels).mean()
    avg_loss = total_loss / len(loader)
    
    # ================= TensorBoard Logging (Epoch Level) =================
    writer.add_scalar('Train_Epoch/Average_Loss', avg_loss, epoch)
    writer.add_scalar('Train_Epoch/AUC', epoch_auc, epoch)
    writer.add_scalar('Train_Epoch/Accuracy', epoch_acc, epoch)
    # =====================================================================
    
    print(f"Epoch {epoch} Result: Loss={avg_loss:.4f}, AUC={epoch_auc:.4f}, Acc={epoch_acc:.4f}")
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
            bag_label = label_info[1].float().to(device)
            
            feats = encoder(bag_imgs)
            logits, _, _ = head(feats)
            
            prob_pos = torch.softmax(logits, dim=1)[0, 1]
            
            all_probs.append(prob_pos.item())
            all_labels.append(bag_label.item())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if len(np.unique(all_labels)) > 1:
        val_auc = metrics.roc_auc_score(all_labels, all_probs)
    else:
        val_auc = 0.0
        
    print(f"Validation AUC: {val_auc:.4f}")
    
    # ================= TensorBoard Logging (Validation) =================
    writer.add_scalar('Val/AUC', val_auc, epoch)
    # ====================================================================
    
    return val_auc

def main():
    args = get_args()
    setup_seed(args.seed)
    
    # 初始化 TensorBoard Writer
    # 日志将保存在 ./runs_abmil/seed_42_lr_0.0001/ 下
    log_path = os.path.join(args.log_dir, f"seed_{args.seed}_lr_{args.lr}")
    writer = SummaryWriter(log_dir=log_path)
    print(f"Tensorboard log directory: {log_path}")
    
    # 1. 准备数据
    print("Loading Data...")
    ds_train, ds_test = gather_align_Img() # 使用原来的数据读取函数
    
    # 构造 Dataset, 强制 return_bag=True
    train_ds = DataSet_MIL(ds_train, downsample=args.downsample, transform=None, return_bag=True)
    val_ds = DataSet_MIL(ds_test, downsample=args.downsample, transform=None, return_bag=True)
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Train Bags: {len(train_ds)}, Val Bags: {len(val_ds)}")
    
    # 2. 准备模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("Initializing Models...")
    encoder = PretrainedResNet18_Encoder().to(device)
    
    head = Bag_Classifier_Attention_Head(
        num_classes=[2], 
        init=True, 
        input_feat_dim=512, 
        withoutAtten=False 
    ).to(device)
    
    # 3. 优化器
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.lr * 0.1}, 
        {'params': head.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # 4. 训练循环
    best_auc = 0.0
    for epoch in range(args.epochs):
        # 传入 writer
        loss, auc = train_one_epoch(encoder, head, train_loader, optimizer, device, epoch, writer)
        
        # 验证
        if (epoch + 1) % 1 == 0:
            # 传入 writer
            val_auc = validate(encoder, head, val_loader, device, epoch, writer)
            if val_auc > best_auc:
                best_auc = val_auc
                print(f"*** New Best AUC: {best_auc:.4f} ***")
    
    # 关闭 writer
    writer.close()

if __name__ == '__main__':
    main()