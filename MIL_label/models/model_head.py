import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math


__all__ = ['']
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            nn.AvgPool2d(7, stride=1),
        ])
        if len(num_classes) == 1:
            self.top_layer = nn.Sequential(nn.Linear(512*4, num_classes[0]))
        else:
            for a, i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(512*4, i))
            self.top_layer = None

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.headcount == 1:
            if self.top_layer:
                out = self.top_layer(out)
            return out
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(out))
            return outp

class ResNet_224x224_Encoder(nn.Module):
    """
    [Backbone] 针对 224x224 输入图像的 ResNet 编码器。
    作用：仅作为特征提取器，去掉了最后的分类全连接层。
    用于为 Teacher 和 Student 提供共享的图像特征。
    """
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet_224x224_Encoder, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            nn.AvgPool2d(7, stride=1), # 224经过32倍下采样变为7，7x7 avgpool后变为1x1
        ])


        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return feat


class ResNet_512x512_Encoder(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1, num_classes=[1000]):
        self.inplanes = 64
        super(ResNet_512x512_Encoder, self).__init__()
        self.headcount = len(num_classes)
        self.base = int(64 * width)
        self.features = nn.Sequential(*[
                            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            self._make_layer(block, self.base, layers[0]),
                            self._make_layer(block, self.base * 2, layers[1], stride=2),
                            self._make_layer(block, self.base * 4, layers[2], stride=2),
                            self._make_layer(block, self.base * 8, layers[3], stride=2),
                            # nn.AvgPool2d(7, stride=1),
                            nn.AvgPool2d(16, stride=1), # 512经过32倍下采样变为16，此处使用16x16 avgpool
        ])

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False, return_feat_out=False):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return feat
    

class Bag_Classifier_Attention_Head(nn.Module):
    def __init__(self, num_classes, init=True, withoutAtten=False, input_feat_dim=512):
        super(Bag_Classifier_Attention_Head, self).__init__()
        self.withoutAtten=withoutAtten
        # 1. 特征变换层：将 Backbone 输出的特征映射到注意力层的输入维度
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(inplace=True))
        
        self.L = 1024 # 注意力层输入维度
        self.D = 512  # 注意力层内部维度
        self.K = 1    # 注意力分数维度 (通常为1)

        # 2. 注意力机制网络 
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        # 3. 最终分类层：将聚合后的特征映射到类别 
        self.top_layer = nn.Linear(1024, num_classes[0])
        if init:
            self._initialize_weights()

    def forward(self, x, returnBeforeSoftMaxA=False, scores_replaceAS=None):
        x = self.classifier(x)

        # Attention module
        A_ = self.attention(x)  # 计算每个 Instance 的未归一化分数 -> NxK
        A_ = torch.transpose(A_, 1, 0)  # -> KxN
        A = F.softmax(A_, dim=1)  # 对 N 维度进行 Softmax，得到归一化的注意力分数 -> KxN

        # 允许外部传入分数替换内部计算 (用于调试或特殊推理)
        if scores_replaceAS is not None:
            A_ = scores_replaceAS
            A = F.softmax(A_, dim=1)  # softmax over N

        if self.withoutAtten:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            # 使用注意力分数对 Instance 特征进行加权求和 (矩阵乘法)
            # A (1xN) * x (Nx1024) -> (1x1024)
            x = torch.mm(A, x)  # 得到 Bag Representation,KxL

        if self.return_features: # switch only used for CIFAR-experiments
            return x

        x = self.top_layer(x)
        # 返回: 
        # x: Bag 的预测 Logits
        # 0: 占位符
        # A: 注意力分数 (Teacher 给 Student 的重要指导信息)
        if returnBeforeSoftMaxA:
            return x, torch.zeros_like(x), A, A_.squeeze(0)
        return x, 0, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # def _initialize_weights(self):
    #     """
    #     使用 Xavier 初始化代替原来的微小正态分布。
    #     这对打破 Attention 的对称性至关重要。
    #     """
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # 使用 Xavier Normal 初始化权重
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm1d):
    #             # BN 层初始化为 1
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Conv2d):
    #             # 尽管这个 Head 里没用到 Conv2d，保留以防万一
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #     #nn.init.xavier_normal_(self.attention[2].weight, gain=10.0)

# class Bag_Classifier_Attention_Head(nn.Module):
#     def __init__(self, features=None, num_classes=[2], init=True, input_feat_dim=512):
#         super(Bag_Classifier_Attention_Head, self).__init__()
#         self.features = features
#         self.num_classes = num_classes[0]
#         self.L = input_feat_dim  # 输入特征维度
#         self.D = 128  # 注意力网络隐藏层维度
#         self.K = 1    # 注意力头数
        
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(self.L * self.K, self.num_classes)
#         )
        
#         if init:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 # 使用 Kaiming 初始化（适合非对称激活如 ReLU/Tanh）
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x, returnBeforeSoftMaxA=False, scores_replaceAS=None):
#         if self.features is not None:
#             x = x.squeeze(0)
#             x = self.features(x)
#         H = x.view(x.shape[0], -1)  # 实例嵌入 H = {h_0, ..., h_N-1}
        
#         A = self.attention(H)  # N x K
#         A = torch.transpose(A, 1, 0)  # K x N
#         if scores_replaceAS is not None:
#             A = scores_replaceAS
        
#         if returnBeforeSoftMaxA:
#             A_ = A.clone()  # 保存 pre-softmax 值
        
#         A = F.softmax(A, dim=1)  # softmax 归一化
        
#         M = torch.mm(A, H)  # K x L (加权求和)
        
#         logits = self.classifier(M.view(1, -1))
        
#         if returnBeforeSoftMaxA:
#             return logits, M, A_, A
        
#         return logits, M, A


class Bag_Classifier_DSMIL_Head(nn.Module):
    """
    基于 DSMIL (Dual-stream Multiple Instance Learning) 的分类头。
    
    - Stream 1 (Instance): Max-pooling identifying Critical Instance (h_m)
    - Stream 2 (Bag): Attention (U) based on distance to h_m, aggregated into Bag Embedding (b)
    """
    def __init__(self, features, num_classes, init=True, input_feat_dim=512):
        super(Bag_Classifier_DSMIL_Head, self).__init__()
        self.features = features 
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True))

        # ------------------------------------------------------------------
        # 1. 实例流 (Stream 1: Instance Classifier)
        # ------------------------------------------------------------------
        self.W_0 = nn.Linear(1024, 2) # W_0 (用于计算实例分数)

        # ------------------------------------------------------------------
        # 2. 包流 (Stream 2: Bag Aggregator)
        # ------------------------------------------------------------------
        self.W_q = nn.Linear(1024, 1024) # W_q (Query 投影矩阵)
        
        self.W_v = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(1024, 1024) 
        ) # W_v (Value/Information 投影矩阵)
        
        # 将聚合后的包嵌入映射为包分数
        self.W_b = nn.Conv1d(2, 2, kernel_size=1024) # W_b (包级分类器权重)

        self.headcount = len(num_classes)
        self.return_features = False
        self.top_layer = nn.Linear(1024, num_classes[0])
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # Xavier 初始化
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.Conv1d):
    #             # 处理 DSMIL 中的 W_b 层
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #     #nn.init.xavier_normal_(self.W_0.weight, gain=5.0)

    def forward(self, x, returnBeforeSoftMaxA=False, scores_replaceAS=None):
        # 0. 特征提取 f(x) -> H
        if self.features is not None:
            x = x.squeeze(0)
            x = self.features(x)
        x = x.view(x.shape[0], -1)
        H = self.classifier(x) # H = {h_0, ..., h_N-1}, 实例嵌入集合
        
        device = H.device

        # ------------------------------------------------------------------
        # Stream 1: Max-pooling (Instance Stream)
        # ------------------------------------------------------------------
        # 计算每个实例的分数
        instance_scores = self.W_0(H) 

        # 寻找“关键实例” (Critical Instance), 即具有最大分数的实例
        _, sorted_indices = torch.sort(instance_scores, 0, descending=True)
        # h_m: 关键实例的嵌入向量 (Critical Instance Embedding)
        m_indices = sorted_indices[0, :] 
        h_m = torch.index_select(H, dim=0, index=m_indices) 

        # ------------------------------------------------------------------
        # Stream 2: Attention Aggregation (Bag Stream)
        # ------------------------------------------------------------------
        # 1. 投影变换
        Q = self.W_q(H).view(H.shape[0], -1)       # 所有实例的 Query 向量集合
        V = self.W_v(H)                            # 所有实例的 Value (Information) 向量集合
        
        # 关键实例的 Query 向量
        q_m = self.W_q(h_m) 

        # 2. 距离度量 (Distance Measurement / Attention)
        # 计算所有实例 Q 与关键实例 q_m 的内积相似度
        raw_scores = torch.mm(Q, q_m.transpose(0, 1)) 
        
        # Scale factor (Transformer standard practice, optional but recommended)
        scale = torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))
        
        # Softmax 归一化得到注意力权重 U
        U = F.softmax(raw_scores / scale, dim=0) # U represents attention weights

        # 3. 生成包嵌入 (Bag Embedding)
        # 利用计算出的距离权重 U 对信息向量 V 进行加权求和
        b_embedding = torch.mm(U.transpose(0, 1), V) # Shape: C x Dim
        
        # Reshape for Conv1d classifier (Batch, Channel, Length) -> (1, C, Dim)
        b_embedding = b_embedding.view(1, b_embedding.shape[0], b_embedding.shape[1])

        # 4. 包级评分 (Bag Score)
        bag_score = self.W_b(b_embedding) # Output of bag classifier
        bag_score = bag_score.view(1, -1)

        # ------------------------------------------------------------------
        # 返回结果
        # ------------------------------------------------------------------
        # instance_scores: 用于计算 max-pooling loss (Stream 1)
        # bag_score: 用于计算 bag loss (Stream 2)
        # U: 注意力权重 (用于可视化/可解释性)
        # b_embedding: 聚合后的包特征
        return instance_scores, bag_score, U, b_embedding
    
class Instance_Classifier_Head(nn.Module):
    """
    [Student Head] 实例级分类器。
    这是双流网络中的 'Student' 部分。
    """
    def __init__(self, num_classes, init=True, input_feat_dim=512):
        super(Instance_Classifier_Head, self).__init__()
        # 简单的 MLP 结构：Linear -> ReLU -> Dropout -> Linear -> ReLU
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_feat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        # 最终分类层
        self.top_layer = nn.Linear(4096, num_classes[0])
        if init:
            self._initialize_weights()

    def forward(self, x, also_return_last_feat=False):
        x = self.classifier(x)
        if self.return_features: # switch only used for CIFAR-experiments
            return x
        if self.top_layer: # this way headcount can act as switch.
            x_ = self.top_layer(x)
            if also_return_last_feat:
                return x_, x
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # Xavier Normal 初始化
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm1d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

class PretrainedResNet18_Encoder(nn.Module):
    def __init__(self):
        super(PretrainedResNet18_Encoder, self).__init__()
        model_raw = models.resnet18(pretrained=True)
        self.pretrained_model = nn.Sequential(*list(model_raw.children())[:-1])

    def forward(self, x):
        return self.pretrained_model(x).squeeze(-1).squeeze(-1)

def teacher_Attention_head(bn=True, num_classes=[2], init=True, input_feat_dim=512):
    model = Bag_Classifier_Attention_Head(num_classes=num_classes, init=init, input_feat_dim=input_feat_dim)
    return model


def teacher_DSMIL_head(bn=True, num_classes=[2], init=True, input_feat_dim=512):
    model = Bag_Classifier_DSMIL_Head(features=None, num_classes=num_classes, init=init, input_feat_dim=input_feat_dim)
    return model


def student_head(num_classes=[2], init=True, input_feat_dim=512):
    model = Instance_Classifier_Head(num_classes, init, input_feat_dim=input_feat_dim)
    return model

class map_abmil(nn.Module):
    """
    Teacher Head 包装器 1：Attention-based MIL
    包装了模型前向传播和 Loss 计算，简化主循环代码。
    """
    def __init__(self, model):
        super(map_abmil, self).__init__()
        self.model = model

    def forward(self, x):
        bag_prediction, _, _, instance_attn_score = self.model(x, returnBeforeSoftMaxA=True, scores_replaceAS=None)
        # instance_attn_score = torch.sigmoid(instance_attn_score)
        # if instance_attn_score.dim() == 1:
        #     instance_attn_score = instance_attn_score.unsqueeze(0) # 修正为 [1, 2]
        # instance_attn_score = torch.softmax(instance_attn_score, dim=1)[:, 1]  # 取正类概率
        if len(instance_attn_score.shape)==1:
            instance_attn_score = instance_attn_score.unsqueeze(0)
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output shape: 1
        # label shape: 1
        # 标准的二值交叉熵损失 (BCE Loss)
        output_bag = output_bag[0, 1]
        bag_label = bag_label.squeeze()
        loss = -1. * (bag_label * torch.log(output_bag+1e-5) 
                      + (1. - bag_label) * torch.log(1. - output_bag+1e-5))
        return loss


class map_dsmil(nn.Module):
    """
    Teacher Head 包装器 2：DSMIL
    DSMIL 使用特殊的 Loss 组合。
    """
    def __init__(self, model):
        super(map_dsmil, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        instance_attn_score, bag_prediction, _, _ = self.model(x)
        instance_attn_score = torch.softmax(instance_attn_score, dim=1)
        instance_attn_score = instance_attn_score[:, 1].unsqueeze(0)
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output shape: 1
        # label shape: 1
        # DSMIL Loss = 0.5 * Bag_Loss + 0.5 * Max_Instance_Loss
        # 即使只有 Bag 标签，DSMIL 也会尝试约束分数最高的 Instance 预测结果趋向于 Bag 标签
        max_id = torch.argmax(output_inst.squeeze(0))
        bag_pred_byMax = output_inst.squeeze(0)[max_id]
        bag_loss = self.criterion(output_bag, bag_label)

        bag_label = bag_label.squeeze()
        bag_loss_byMax = -1. * (bag_label * torch.log(bag_pred_byMax+1e-5) 
                                + (1. - bag_label) * torch.log(1. - bag_pred_byMax+1e-5))
        loss = 0.5 * bag_loss + 0.5 * bag_loss_byMax
        return loss
    

class map_student(nn.Module):
    """
    Student Head 包装器
    封装了 Student 的前向传播和加权 Loss 计算逻辑。
    """
    def __init__(self, model):
        super(map_student, self).__init__()
        self.model = model

    def forward(self, x):
        # 直接调用内部模型的 forward
        # 返回 logits: [Batch_Size, Num_Classes]
        return self.model(x)

    def get_loss(self, prediction, target, neg_weight=0.1):
        """
        计算 Student 的加权二值交叉熵损失。
        :param prediction: 经过 Softmax 后的概率值 (Batch_Size, 2)
        :param target: 伪标签 (Pseudo Label), 对应 class 1 的概率 (Batch_Size, )
        :param neg_weight: 负样本的权重系数
        """
        # 确保 target 维度匹配
        if len(target.shape) == 1:
            target = target.unsqueeze(1)
        
        # 计算加权 Loss
        # target 是 label=1 的概率 (伪标签)
        # prediction[:, 0] 是预测为 label=0 的概率
        # prediction[:, 1] 是预测为 label=1 的概率
        loss = -1. * torch.mean(
            neg_weight * (1 - target) * torch.log(prediction[:, 0] + 1e-5) + 
            (1 - neg_weight) * target * torch.log(prediction[:, 1] + 1e-5)
        )
        return loss

class map_abmil_test(nn.Module):
    """
    Teacher Head 包装器 1：Attention-based MIL
    """
    def __init__(self, model):
        super(map_abmil_test, self).__init__()
        self.model = model

    def forward(self, x):
        # 这里的 instance_attn_score 对应 A_ (Raw Attention Scores, Before Softmax)
        # 这对于 Min-Max 归一化统计非常重要，因为 Raw Score 的分布范围更广，更容易统计出 min/max
        bag_prediction, _, _, instance_attn_score = self.model(x, returnBeforeSoftMaxA=True, scores_replaceAS=None)
        
        # 确保维度是 [1, N]
        if len(instance_attn_score.shape) == 1:
            instance_attn_score = instance_attn_score.unsqueeze(0)
            
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output_bag: [1, 2] 经过 Softmax 后的概率
        # bag_label: [1] 真实标签 (0 或 1)
        
        # 取出正类概率 (Class 1)
        prob_pos = output_bag[0, 1]
        bag_label = bag_label.float().squeeze()
        
        # 手动计算 Binary Cross Entropy
        # 加上 1e-6 防止 log(0)
        loss = -1. * (bag_label * torch.log(prob_pos + 1e-6) 
                      + (1. - bag_label) * torch.log(1. - prob_pos + 1e-6))
        return loss


class map_dsmil_test(nn.Module):
    """
    Teacher Head 包装器 2：DSMIL
    【重要修改】：移除了 CrossEntropyLoss，改为手动计算 BCE，适配 Optimizer 传入的 Probabilities。
    """
    def __init__(self, model):
        super(map_dsmil_test, self).__init__()
        self.model = model
        # 移除 self.criterion = torch.nn.CrossEntropyLoss()，因为它不能处理概率输入

    def forward(self, x):
        # DSMIL Head 返回: instance_scores(Raw), bag_score(Logits), U, b_embedding
        instance_scores, bag_prediction, _, _ = self.model(x)
        
        # 将 Instance Raw Scores 转为概率，用于后续计算 Max-Instance Loss
        # 同时也作为伪标签的基础返回给 Optimizer
        instance_probs = torch.softmax(instance_scores, dim=1)
        
        # 取出正类概率，并调整形状为 [1, N]
        instance_prob_pos = instance_probs[:, 1].unsqueeze(0)
        
        return bag_prediction, instance_prob_pos

    def get_loss(self, output_bag, output_inst, bag_label):
        # output_bag: [1, 2] (Optimizer 传入的 Bag 概率)
        # output_inst: [1, N] (Forward 返回的 Instance 概率)
        # bag_label: [1]
        
        bag_label = bag_label.float().squeeze()
        
        # 1. Bag Loss (Stream 2)
        prob_bag_pos = output_bag[0, 1]
        bag_loss = -1. * (bag_label * torch.log(prob_bag_pos + 1e-6) 
                          + (1. - bag_label) * torch.log(1. - prob_bag_pos + 1e-6))

        # 2. Max Instance Loss (Stream 1)
        # 找出分数最高的 Instance
        max_id = torch.argmax(output_inst.squeeze(0))
        bag_prob_byMax = output_inst.squeeze(0)[max_id] # 这是概率
        
        bag_loss_byMax = -1. * (bag_label * torch.log(bag_prob_byMax + 1e-6) 
                                + (1. - bag_label) * torch.log(1. - bag_prob_byMax + 1e-6))
        
        # 联合 Loss
        loss = 0.5 * bag_loss + 0.5 * bag_loss_byMax
        return loss