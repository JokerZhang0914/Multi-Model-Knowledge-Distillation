import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import datetime

# 复用您提供的模块
import utils
from dataset import DataSet_MIL, gather_align_Img
import torchvision.models as models

# ==========================================
# 1. 重新定义清晰的 ABMIL 模型
# ==========================================

class FeatureExtractor(nn.Module):
    """
    使用 ResNet18 提取特征，去除最后的 FC 层。
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # 去掉最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # x shape: [N, 3, H, W]
        x = self.features(x)
        # x shape: [N, 512, 1, 1] -> [N, 512]
        x = x.view(x.size(0), -1)
        return x

class CleanABMIL(nn.Module):
    """
    标准的 Attention-based MIL 模型。
    结构: ResNet -> (Projection) -> Attention -> Weighted Sum -> Classifier
    """
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.5):
        super(CleanABMIL, self).__init__()
        
        self.feature_extractor = FeatureExtractor()
        
        # 1. 特征投影层 (可选，用于降低维度或增加非线性)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 注意力网络 (V.T * tanh(W * H))
        # 这里使用经典的 Gated Attention 的简化版 (Tanh Attention)
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh()
        )
        self.attention_W = nn.Linear(128, 1) # 输出分数
        
        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1) # 二分类输出一个 logit
        )

    def forward(self, x):
        """
        x: [B, N, C, H, W] -> 由于是 Bag 模式，B 通常为 1
        """
        x = x.squeeze(0) # [N, C, H, W]
        
        # 1. 提取特征
        H = self.feature_extractor(x) # [N, 512]
        
        # 2. 投影
        H_proj = self.projector(H) # [N, 256]
        
        # 3. 计算注意力分数
        A_V = self.attention_V(H_proj) # [N, 128]
        A = self.attention_W(A_V)      # [N, 1]
        A = torch.transpose(A, 1, 0)   # [1, N]
        A = F.softmax(A, dim=1)        # 归一化注意力分数
        
        # 4. 特征聚合 (Bag Embedding)
        M = torch.mm(A, H_proj)        # [1, N] * [N, 256] -> [1, 256]
        
        # 5. 分类
        logits = self.classifier(M)    # [1, 1]
        
        # 返回 Logits 和 Attention Weights (用于可视化)
        return logits, A

# ==========================================
# 2. 辅助函数
# ==========================================

def calculate_metrics(y_true, y_pred_prob):
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5 # 防止只有一个类别时报错
    return auc

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    # 进度条
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)
    
    for i, (data, label_info, _) in enumerate(pbar):
        # data: [1, N, 3, 512, 512]
        # label_info[1]: slide_label [1]
        
        bag_label = label_info[1].float().to(device)
        data = data.to(device)
        
        # 前向传播
        logits, _ = model(data) # logits: [1, 1]
        
        # 既然是二分类，使用 sigmoid 将 logit 转为概率
        prob = torch.sigmoid(logits)
        
        # 计算 Loss
        loss = criterion(prob.view(-1), bag_label.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集用于计算 Epoch AUC
        all_labels.append(bag_label.item())
        all_probs.append(prob.item())
        
        # 实时更新进度条
        pbar.set_postfix({'loss': loss.item()})

    # 计算 Epoch 级指标
    epoch_loss = total_loss / len(loader)
    epoch_auc = calculate_metrics(all_labels, all_probs)
    
    # TensorBoard
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/AUC', epoch_auc, epoch)
    
    print(f"Epoch {epoch} [Train] Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}")
    return epoch_loss, epoch_auc

def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, label_info, _ in tqdm(loader, desc=f'Val Epoch {epoch}', leave=False):
            bag_label = label_info[1].float().to(device)
            data = data.to(device)
            
            logits, _ = model(data)
            prob = torch.sigmoid(logits)
            
            loss = criterion(prob.view(-1), bag_label.view(-1))
            total_loss += loss.item()
            
            all_labels.append(bag_label.item())
            all_probs.append(prob.item())
            
    epoch_loss = total_loss / len(loader)
    epoch_auc = calculate_metrics(all_labels, all_probs)
    
    # TensorBoard
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/AUC', epoch_auc, epoch)
    
    print(f"Epoch {epoch} [Val]   Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}")
    return epoch_auc

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Debug ABMIL Teacher')
    parser.add_argument('--device', default='2', type=str, help='CUDA device index')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float) # 降低学习率，MIL通常需要较低LR
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dataset_downsample', default=0.1, type=float, help='使用多少数据进行调试')
    parser.add_argument('--log_dir', default='./runs/runs_abmil/', type=str)
    args = parser.parse_args()

    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.setup_runtime(seed=42) # 固定随机种子

    print(f"使用设备: {device}")
    print("初始化 TensorBoard...")
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + \
           "lr{}_Downsample{}".format(
               args.lr, args.dataset_downsample,
           )
    writer = SummaryWriter(args.log_dir + name)

    # 数据加载 (复用 dataset.py)
    print("准备数据...")
    ds_train_raw, ds_test_raw = gather_align_Img() # 使用原来的函数
    
    # 构建 DataSet
    # 注意：return_bag=True，因为我们要测 MIL 性能
    train_ds = DataSet_MIL(ds=ds_train_raw, downsample=args.dataset_downsample, return_bag=True)
    val_ds = DataSet_MIL(ds=ds_test_raw, downsample=args.dataset_downsample, return_bag=True)
    
    # 构建 DataLoader
    # 注意：batch_size=1 是必须的，因为 Bag 大小不固定，不能堆叠
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"训练集大小: {len(train_ds)} Bags")
    print(f"验证集大小: {len(val_ds)} Bags")

    # 模型初始化
    model = CleanABMIL().to(device)
    
    # 优化器
    # 建议使用 Adam 调试 MIL，它比 SGD 对学习率不那么敏感
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 损失函数 (二分类交叉熵)
    criterion = nn.BCELoss()
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_auc = validate(model, val_loader, criterion, device, epoch, writer)
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_abmil.pth'))
            print(f"*** 保存最佳模型 (AUC: {best_auc:.4f}) ***")

    writer.close()
    print("测试结束。")

if __name__ == '__main__':
    main()