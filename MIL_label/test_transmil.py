import os
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, roc_curve, auc
from math import ceil
from einops import rearrange, reduce

# 引入您的 dataset 模块
try:
    import dataset
except ImportError:
    raise ImportError("请确保 dataset.py 在当前目录下，并且已正确配置其中的数据路径。")

# ==========================================
# Part 1: NystromAttention & TransMIL Model
# (直接内嵌，确保脚本独立性)
# ==========================================

def get_roc_figure(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    return fig

def get_pr_figure(labels, preds):
    precision, recall, _ = precision_recall_curve(labels, preds)
    avg_precision = average_precision_score(labels, preds)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    return fig

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-8)
    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z

class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, num_landmarks = 256, pinv_iterations = 6, residual = True, residual_conv_kernel = 33, eps = 1e-8, dropout = 0.):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout1d(dropout))
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)
        
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        
        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out += self.res_conv(v)
        
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            # 提取 CLS Token (Index 0) 对所有 Key 的注意力
            # attn1: [b, h, n, m] -> 取 query 0 -> [b, h, 1, m]
            attn1_cls = attn1[:, :, 0:1, :] 
            # 乘逆矩阵: [b, h, 1, m]
            cls_mid = attn1_cls @ attn2_inv
            # 乘 Key 矩阵: [b, h, 1, n_padded]
            cls_attn = cls_mid @ attn3
            
            # 切片回原始长度 (去除 Nystrom 算法内部填充)
            # 注意：这里的 n 包含了 TransMIL forward 中为了 PPEG 做的 padding，
            # 具体的有效长度需要在 TransMIL 中进一步处理
            cls_attn = cls_attn[..., :n]
            
            # 对多头取平均: [b, h, 1, n] -> [b, 1, n] -> [b, n]
            return out, cls_attn.mean(dim=1).squeeze(1)
        return out

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim, dim_head = dim//8, heads = 8,
            num_landmarks = dim//2, pinv_iterations = 6,
            residual = True, dropout=0.4
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            # 提取 Attention
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, input_size, n_classes, mDim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU(), nn.Dropout(0.2))
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=mDim)
        self.layer2 = TransLayer(dim=mDim)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Linear(mDim, self.n_classes)

    def forward(self, **kwargs):
        # 兼容 kwargs 调用，也可以直接传 data
        h = kwargs.get('data')
        if h is None: h = kwargs.get('x') # fallback
        
        h = h.float() #[B, n, input_size]
        
        # FC1 mapping
        # 考虑到 batch_size=1 但内部 patch 数 n 变化，直接处理
        h = self._fc1(h) #[B, n, mDim]
        
        H_original = h.shape[1]

        # Pad to Square for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, mDim]

        # Add CLS token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # TransLayer 1
        h = self.layer1(h)
        
        # PPEG
        h = self.pos_layer(h, _H, _W)
        
        # TransLayer 2
        h, attn_scores = self.layer2(h, return_attn=True) # <--- 获取 Attn
        
        # === 处理 Attention ===
        # attn_scores shape: [B, 1 + N_padded] (CLS + Patches + Padding)
        # 1. 去除 CLS (Index 0)
        # 2. 去除 Padding (保留 1 到 1+H_original)
        valid_attn = attn_scores[:, 1 : (1 + H_original)]
        
        # 3. 关键修正：重新归一化 (Re-Softmax)
        # 解决 Padding 导致的分数稀释问题，确保有效实例上的分数和为 1
        valid_attn = torch.softmax(valid_attn, dim=-1)
        # =====================

        # Norm & Prediction (Take CLS token)
        h = self.norm(h)[:,0]
        logits = self._fc2(h)
        
        return logits, h

# ==========================================
# Part 2: Feature Extractor (ResNet18)
# ==========================================

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, freeze=False):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # 去掉最后的 FC 层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 根据参数决定是否冻结参数
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # x: [N, 3, 512, 512]
        # output: [N, 512, 1, 1] -> squeeze -> [N, 512]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# ==========================================
# Part 3: Main Training Script
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='Train TransMIL with ResNet18')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--downsample', type=float, default=0.9, help='数据集下采样比例 (调试用)')
    parser.add_argument('--seed', type=int, default=2023, help='随机种子')
    parser.add_argument('--freeze_resnet', action='store_true', default=False, help='是否冻结ResNet参数')
    parser.add_argument('--log_dir', type=str, default='./runs/runs_transmil', help='Tensorboard日志目录')
    parser.add_argument('--pth_dir', type=str, default='./checkpoints/abmil')
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--split', default=0.7, type=float, help='训练集占比')
    parser.add_argument('--datasetnum',type=int, default=2, help='使用的数据集数量')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train_one_epoch(model, feature_extractor, loader, criterion, optimizer, device, epoch, args):
    model.train()
    # 根据参数决定是否冻结 ResNet
    if not args.freeze_resnet:
        feature_extractor.train()
    else:
        feature_extractor.eval()

    total_loss = 0.
    train_labels = []
    train_preds = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for batch_idx, (bag, label_info, _) in enumerate(pbar):
        bag = bag.to(device)
        label = label_info[1].long().to(device)
        
        bag_imgs = bag.squeeze(0)
        
        # 检查空包
        if bag_imgs.size(0) == 0:
            print(f"[Warning] Batch {batch_idx} has 0 patches. Skipping.")
            continue

        # 特征提取
        if args.freeze_resnet:
            with torch.no_grad():
                features = feature_extractor(bag_imgs)
        else:
            features = feature_extractor(bag_imgs)

        features = features.unsqueeze(0)
        logits, _ = model(data=features)
        
        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)
        train_labels.append(label.item())
        train_preds.append(probs[0, 1].item())

        if batch_idx % 50 == 0:
            pbar.set_postfix({'Loss': loss.item(), 'N_Patches': bag_imgs.size(0)})

    avg_loss = total_loss / len(loader)
    
    try:
        auc = roc_auc_score(train_labels, train_preds)
        acc = accuracy_score(train_labels, np.array(train_preds) > 0.5)
    except ValueError:
        auc = 0.0
        acc = 0.0
        
    return avg_loss, auc, acc

def validate(model, feature_extractor, loader, criterion, device, epoch, args, writer=None):
    model.eval()
    feature_extractor.eval()
    
    total_loss = 0.
    val_labels = []
    val_preds = []

    # 用于记录详细信息的列表
    instance_records = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        for batch_idx, (bag, label_info, _) in enumerate(pbar):
            bag = bag.to(device)
            label = label_info[1].long().to(device)
            slide_id = label_info[3][0] # 假设 label_info[3] 是 slide name
            
            bag_imgs = bag.squeeze(0)
            if bag_imgs.size(0) == 0: continue

            features = feature_extractor(bag_imgs)
            features = features.unsqueeze(0)

            logits, attn_scores = model(data=features)
            # logits: [1, 2]
            # attn_scores: [1, N]
            loss = criterion(logits, label)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            bag_score = probs[0, 1].item()
            val_labels.append(label.item())
            val_preds.append(probs[0, 1].item())

            # === 记录每张图的信息 ===
            # TransMIL 没有 Instance Logits，我们记录 Instance Attention
            # 将 Tensor 转为 List
            attn_list = attn_scores.squeeze(0).cpu().numpy().tolist()
            
            # 记录格式: (SlideID, BagLabel, BagProb, InstanceAttns)
            instance_records.append({
                'slide_id': slide_id,
                'bag_label': label.item(),
                'bag_prob': bag_score,
                'instance_attns': attn_list,
                'num_patches': len(attn_list)
            })

            # 打印前几个包的 Attention 统计 (Debug用)
            if batch_idx < 3: 
                 print(f"\n[Debug] Bag {slide_id}: Label={label.item()}, Pred={bag_score:.4f}")
                 print(f"       Attn Min: {min(attn_list):.4f}, Max: {max(attn_list):.4f}")

    avg_loss = total_loss / len(loader)
    
    # 计算指标
    try:
        auc = roc_auc_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
    except ValueError:
        auc = 0.0
        acc = 0.0
    
    ap = average_precision_score(val_labels, val_preds)

    # TensorBoard 记录 (AP 和 曲线)
    if writer is not None:
        writer.add_scalar('Metric/AP_Val', ap, epoch)
        
        # 绘制并记录曲线 (假设 get_roc_figure 和 get_pr_figure 已定义)
        try:
            roc_fig = get_roc_figure(val_labels, val_preds)
            writer.add_figure('Curve/ROC', roc_fig, epoch)
            plt.close(roc_fig)
            
            pr_fig = get_pr_figure(val_labels, val_preds)
            writer.add_figure('Curve/PR', pr_fig, epoch)
            plt.close(pr_fig)
            
            writer.add_pr_curve('PR_Curve_Interactive', 
                                torch.tensor(val_labels), 
                                torch.tensor(val_preds), 
                                epoch)
        except Exception as e:
            print(f"Plotting Error: {e}")

    return avg_loss, auc, acc, ap

def main(): 
    args = get_args()
    set_seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. 数据准备 ---
    print("Preparing Data...")
    try:
        train_data, test_data = dataset.gather_align_Img(split=args.split)
    except Exception as e:
        print(f"[Error] 数据加载失败: {e}")
        return

    train_ds = dataset.DataSet_MIL(train_data, downsample=args.downsample, return_bag=True, preload=False)
    test_ds = dataset.DataSet_MIL(test_data, downsample=args.downsample, return_bag=True, preload=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    print(f"Train Bags: {len(train_ds)}, Test Bags: {len(test_ds)}")

    # --- 2. 模型准备 ---
    print("Initializing Models...")
    feature_extractor = ResNetFeatureExtractor(freeze=args.freeze_resnet).to(device)
    model = TransMIL(input_size=512, n_classes=2, mDim=512).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- 3. Tensorboard ---
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"TRANSMIL_{current_time}_seed{args.seed}_split{args.split}_lr{args.lr}_ds{args.downsample}dataset{args.datasetnum}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logging to: {log_dir}")

    # --- 4. 训练循环 ---
    best_auc = 0.0

    for epoch in range(args.epochs):
        # 调用训练函数
        train_loss, train_auc, train_acc = train_one_epoch(
            model, feature_extractor, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # 调用验证函数
        val_loss, val_auc, val_acc, val_ap = validate(
            model, feature_extractor, test_loader, criterion, device, epoch, args, writer
        )

        # 记录主要标量
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            writer.add_scalar('AUC/Val', val_auc, epoch)
            writer.add_scalar('ACC/Train', train_acc, epoch)
            writer.add_scalar('ACC/Val', val_acc, epoch)

        # 打印日志
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, ACC: {train_acc:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, ACC: {val_acc:.4f}")
        print(f"  Val   -> ... AP: {val_ap:.4f}")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best AUC achieved! Model saved to {save_path}")

    writer.close()
    print("Training Complete.")

if __name__ == '__main__':
    main()