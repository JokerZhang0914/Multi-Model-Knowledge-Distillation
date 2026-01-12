import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# 复用您提供的模块
import utils
from dataset import DataSet_MIL, gather_align_Img
import torchvision.models as models

# ==========================================
# 1. 重新定义清晰的 DSMIL 模型
# ==========================================

class FeatureExtractor(nn.Module):
    """
    ResNet18 特征提取器
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # [N, 3, H, W] -> [N, 512]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class CleanDSMIL(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256):
        super(CleanDSMIL, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        # 1. 降维与非线性映射
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 2. Instance Classifier (Stream 1)
        # 用于给每个 Patch 打分，找出 Critical Instance
        self.instance_classifier = nn.Linear(hidden_dim, 1)
        
        # 3. Attention Projections (Stream 2)
        # Query: 映射特征以计算距离
        self.q_proj = nn.Linear(hidden_dim, 128)
        # Value: 映射特征以包含信息
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5) # 加上Dropout增加鲁棒性
        )
        
        # 4. Bag Classifier
        # 输入聚合后的特征，输出 Bag Logit
        self.bag_classifier = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: [B, N, C, H, W] -> B=1
        """
        x = x.squeeze(0) # [N, C, H, W]
        
        # --- 特征提取 ---
        f = self.feature_extractor(x) # [N, 512]
        h = self.fc(f)                # [N, 256] -> Embedding
        
        # --- Stream 1: Instance Classification ---
        # 计算所有实例的分数
        inst_logits = self.instance_classifier(h) # [N, 1]
        
        # 找出分数最高的实例索引 (Critical Instance)
        max_inst_idx = torch.argmax(inst_logits)
        max_inst_logit = inst_logits[max_inst_idx] # 用于 Loss 计算
        
        # 获取 Critical Instance 的 Embedding 用于 Query
        h_critical = h[max_inst_idx] # [256]
        
        # --- Stream 2: Bag Aggregation ---
        # 计算 Q (所有实例) 和 Q_critical (关键实例)
        Q = self.q_proj(h)           # [N, 128]
        Q_critical = self.q_proj(h_critical).unsqueeze(1) # [128, 1]
        
        # 计算距离/注意力分数: Q * Q_critical
        # 这里的物理意义是：寻找那些和“最像肿瘤的那个Patch”长得像的其他Patch
        scores = torch.mm(Q, Q_critical) # [N, 1]
        
        # 缩放并归一化 (Scaled Dot-Product Attention)
        scale = torch.sqrt(torch.tensor(128.0)).to(x.device)
        A = F.softmax(scores / scale, dim=0) # [N, 1] -> Attention Weights
        
        # 加权聚合 Value
        V = self.v_proj(h) # [N, 128]
        M = torch.mm(A.transpose(0, 1), V) # [1, N] * [N, 128] -> [1, 128]
        
        # --- Final Bag Prediction ---
        bag_logit = self.bag_classifier(M) # [1, 1]
        
        # 返回:
        # bag_logit: 用于 Bag Loss
        # max_inst_logit: 用于 Max-Instance Loss
        # A: 注意力分数 (可视化用)
        return bag_logit, max_inst_logit, A

# ==========================================
# 2. 训练与验证逻辑
# ==========================================

def calculate_metric(y_true, y_pred_prob):
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = 0.5
    return auc

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_bag_probs = []
    
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)
    
    for i, (data, label_info, _) in enumerate(pbar):
        bag_label = label_info[1].float().to(device).view(-1, 1) # [1, 1]
        data = data.to(device)
        
        # 前向传播
        bag_logit, max_inst_logit, _ = model(data)
        
        # --- DSMIL Loss 计算 ---
        # 1. Bag Loss: 整个包预测对了没？
        loss_bag = criterion(bag_logit, bag_label)
        
        # 2. Instance Loss: 那个分数最高的 Patch，它的预测结果应该趋向于 Bag 标签
        # (如果 Bag 是阳性，最像阳性的 Patch 应该是阳性；如果 Bag 是阴性，最像阳性的 Patch 也得是阴性)
        loss_inst = criterion(max_inst_logit.view(1, 1), bag_label)
        
        # 联合 Loss (0.5 是论文推荐权重)
        loss = 0.5 * loss_bag + 0.5 * loss_inst
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录用于计算 AUC
        bag_prob = torch.sigmoid(bag_logit).item()
        all_labels.append(bag_label.item())
        all_bag_probs.append(bag_prob)
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = total_loss / len(loader)
    epoch_auc = calculate_metric(all_labels, all_bag_probs)
    
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/AUC', epoch_auc, epoch)
    print(f"Epoch {epoch} [Train] Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}")
    return epoch_loss, epoch_auc

def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_bag_probs = []
    
    with torch.no_grad():
        for data, label_info, _ in tqdm(loader, desc=f'Val Epoch {epoch}', leave=False):
            bag_label = label_info[1].float().to(device).view(-1, 1)
            data = data.to(device)
            
            bag_logit, max_inst_logit, _ = model(data)
            
            loss_bag = criterion(bag_logit, bag_label)
            loss_inst = criterion(max_inst_logit.view(1, 1), bag_label)
            loss = 0.5 * loss_bag + 0.5 * loss_inst
            
            total_loss += loss.item()
            
            bag_prob = torch.sigmoid(bag_logit).item()
            all_labels.append(bag_label.item())
            all_bag_probs.append(bag_prob)
            
    epoch_loss = total_loss / len(loader)
    epoch_auc = calculate_metric(all_labels, all_bag_probs)
    
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/AUC', epoch_auc, epoch)
    print(f"Epoch {epoch} [Val]   Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}")
    return epoch_auc

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Debug DSMIL Teacher')
    parser.add_argument('--device', default='2', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=2e-4, type=float) # DSMIL 可能需要比 ABMIL 稍大一点的 LR
    parser.add_argument('--dataset_downsample', default=1.0, type=float)
    parser.add_argument('--log_dir', default='./runs/runs_dsmil/', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.setup_runtime(seed=42)

    print(f"Device: {device}")
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + \
           "lr{}_Downsample{}".format(
               args.lr, args.dataset_downsample,
           )
    writer = SummaryWriter(args.log_dir + name)

    # 加载数据 (Return Bag Mode)
    print("Loading Data...")
    ds_train_raw, ds_test_raw = gather_align_Img()
    
    train_ds = DataSet_MIL(ds=ds_train_raw, downsample=args.dataset_downsample, return_bag=True)
    val_ds = DataSet_MIL(ds=ds_test_raw, downsample=args.dataset_downsample, return_bag=True)
    
    # Batch Size 必须为 1
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Train Bags: {len(train_ds)}, Val Bags: {len(val_ds)}")

    # 模型与优化器
    model = CleanDSMIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用 BCEWithLogitsLoss，因为它内部集成了 Sigmoid，数值更稳定
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_auc = validate(model, val_loader, criterion, device, epoch, writer)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_dsmil.pth'))
            print(f"*** Best AUC Saved: {best_auc:.4f} ***")

    writer.close()

if __name__ == '__main__':
    main()