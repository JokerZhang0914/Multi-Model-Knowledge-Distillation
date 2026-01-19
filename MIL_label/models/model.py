import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import numpy as np
from math import ceil
from einops import rearrange, reduce

# ==========================================
# 1. Feature Extractor (Shared)
# ==========================================

class PretrainedResNet18_Encoder(nn.Module):
    def __init__(self, freeze=False):
        super(PretrainedResNet18_Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optional: Freeze weights to save memory/compute or strictly follow "Feature Extractor" paradigm
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [Batch*N, 3, H, W]
        out = self.features(x)
        # out: [Batch*N, 512, 1, 1] -> [Batch*N, 512]
        out = out.view(out.size(0), -1)
        return out

# ==========================================
# 2. TransMIL Teacher (Modified for Attn)
# ==========================================

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-8)
    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z

class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, residual_conv_kernel=33, eps=1e-8, dropout=0.):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout1d(dropout))
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            # mask_landmarks = mask_landmarks_sum > 0 # unused in current simplified version
        
        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        
        # Standard Output Calculation
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out += self.res_conv(v)
        
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        # --- Distillation Feature: Return CLS Attention ---
        # If requested, calculate how much the [CLS] token (index 0) attended to everyone else
        # A_cls = (attn1_cls @ attn2_inv) @ attn3
        # attn1 shape: [b, h, n, m]. We take slice [:, :, 0:1, :] for CLS query
        if return_attn:
            # We want the attention row corresponding to CLS (index 0)
            # attn1: [b, h, n, m] -> take query 0 -> [b, h, 1, m]
            attn1_cls = attn1[:, :, 0:1, :] 
            
            # product with inverse: [b, h, 1, m]
            cls_mid = attn1_cls @ attn2_inv
            
            # product with keys: [b, h, 1, n_padded]
            cls_attn = cls_mid @ attn3
            
            # Slice back to original length n
            cls_attn = cls_attn[..., :n]
            
            # Average over heads: [b, 1, n] -> [b, n]
            return out, cls_attn.mean(dim=1).squeeze(1)

        return out

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim, dim_head=dim//8, heads=8,
            num_landmarks=dim//2, pinv_iterations=6,
            residual=True, dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            # We assume return_attn is only called for the layer where we want to extract weights
            # Note: NystromAttention adds residual inside itself for 'out', 
            # but 'attn' return is purely the attention matrix.
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
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
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL_With_Attn(nn.Module):
    def __init__(self, input_size=512, n_classes=2, mDim=512):
        super(TransMIL_With_Attn, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=mDim)
        self.layer2 = TransLayer(dim=mDim)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Linear(mDim, self.n_classes)

    def forward(self, h):
        # h: [B, n, input_size] (Features)
        
        # 1. Project Features
        h = self._fc1(h) # [B, n, mDim]
        
        # 2. Pad to Square for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1) # [B, N_padded, mDim]

        # 3. Add CLS Token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # 4. TransLayer 1
        h = self.layer1(h)
        
        # 5. PPEG
        h = self.pos_layer(h, _H, _W)
        
        # 6. TransLayer 2 (Extract Attention Here!)
        # We capture the attention of the CLS token attending to all instances
        h = self.layer2(h) 
        
        # 对所有 Token 进行 LayerNorm
        h_norm = self.norm(h) 
        
        # --- A. Bag Level Prediction (基于 CLS Token) ---
        h_cls = h_norm[:, 0]
        bag_logits = self._fc2(h_cls) # [B, n_classes]

        # --- B. Instance Level Prediction (基于 Patch Tokens) ---
        # 还原：去掉第0个 CLS token，并截取原始长度 H (去掉 padding)
        h_patches = h_norm[:, 1:(1+H)] # [B, N_original, mDim]
        
        # 关键操作：将同一个分类器 _fc2 应用于每个 Patch 的特征上
        # 逻辑：TransMIL 的 Encoder 已经混入了上下文信息，Patch Token 此时包含了“该 Patch 是否致病”的语义
        instance_logits = self._fc2(h_patches) # [B, N, n_classes]
        
        # Softmax 归一化：在类别维度 (dim=-1) 进行，保证 P(Pos) + P(Neg) = 1
        # 取出正类 (Class 1) 的概率
        instance_probs = torch.softmax(instance_logits, dim=-1)[:, :, 1] # [B, N]
        
        # 返回：包预测 logits, 实例预测概率 (0~1)
        return bag_logits, instance_probs

# ==========================================
# 3. Student Head (Simple MLP)
# ==========================================

class Student_Head(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super(Student_Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# ==========================================
# 4. Model Wrappers (For Clean Training Loop)
# ==========================================

class map_transmil(nn.Module):
    """
    Wrapper for TransMIL Teacher.
    Handles loss calculation and output formatting.
    """
    def __init__(self, model):
        super(map_transmil, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: [1, N, 512]
        # output: 
        #   bag_logits: [1, n_classes]
        #   instance_probs: [1, N] (范围 0~1 的概率)
        bag_logits, instance_probs = self.model(x)
        return bag_logits, instance_probs

    def get_loss(self, bag_logits, bag_label):
        return self.criterion(bag_logits, bag_label)

class map_student(nn.Module):
    """
    Wrapper for Student MLP.
    Handles KD (pseudo-label) loss calculation.
    """
    def __init__(self, model):
        super(map_student, self).__init__()
        self.model = model

    def forward(self, x):
        # x: [Batch, 512] (Individual patch features)
        return self.model(x)

    def get_loss(self, prediction, target, neg_weight=0.1):
        """
        Weighted Binary Cross Entropy for Distillation.
        :param prediction: Softmax probabilities [Batch, 2]
        :param target: Pseudo-label (prob of class 1) [Batch]
        :param neg_weight: Weight for negative samples (often lower in MIL)
        """
        if len(target.shape) == 1:
            target = target.unsqueeze(1)
            
        # prediction[:, 0] -> Prob of Class 0 (Negative)
        # prediction[:, 1] -> Prob of Class 1 (Positive)
        
        # Loss formula:
        # - [ w_neg * (1-y) * log(p_0) + (1-w_neg) * y * log(p_1) ]
        # If target (y) is high, we want p_1 high.
        
        loss = -1. * torch.mean(
            neg_weight * (1 - target) * torch.log(prediction[:, 0] + 1e-6) + 
            (1 - neg_weight) * target * torch.log(prediction[:, 1] + 1e-6)
        )
        return loss

# ==========================================
# 5. Factory Functions
# ==========================================

def get_transmil_teacher():
    model = TransMIL_With_Attn(input_size=512, n_classes=2, mDim=512)
    return map_transmil(model)

def get_student():
    model = Student_Head(input_dim=512, num_classes=2)
    return map_student(model)