"""
/home/zhaokaizhang/.conda/envs/mmked/bin/python /home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/MIL_label/test_train.py --device 0,1 --epochs 20 --dataset_downsampling 1
tensorboard --logdir /home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs_mil_label --port 6006"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import copy

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm
import datetime
from tensorboardX import SummaryWriter

import utils
from sklearn import metrics
from dataset import DataSet_MIL, gather_align_Img
from model_head import PretrainedResNet18_Encoder, teacher_Attention_head, teacher_DSMIL_head, student_head
from model_head import map_abmil, map_dsmil, map_student, map_abmil_test,map_dsmil_test


class Optimizer:
    def __init__(self, model_encoder, model_teacherHead, model_studentHead,
                 optimizer_encoder, optimizer_teacherHead, optimizer_studentHead,
                 train_bagloader, train_instanceloader, test_bagloader, test_instanceloader,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 PLPostProcessMethod='NegGuide', StuFilterType='ReplaceAS', smoothE=100,
                 stu_loss_weight_neg=0.1, stuOptPeriod=1,
                 teacher_loss_weight=[1.0, 1.0],
                 teacher_pseudo_label_merge_weight=[0.5, 0.5]):
        self.model_encoder = model_encoder
        self.model_teacherHead = model_teacherHead
        self.model_studentHead = model_studentHead
        
        # 【修改点 1】强制使用 Adam 优化器 (MIL 对 SGD 非常敏感，Adam 更容易收敛)
        # 即使外部传入了 SGD，这里也会覆盖为 Adam
        print("[Info] Forcing Optimizer to Adam for better MIL convergence.")
        self.optimizer_encoder = torch.optim.Adam(
            self.model_encoder.parameters(), lr=1e-4, weight_decay=1e-4)
        
        self.optimizer_teacherHead = []
        for th in self.model_teacherHead:
            # 如果是 DataParallel，需要取 .module
            params = th.module.parameters() if isinstance(th, nn.DataParallel) else th.parameters()
            self.optimizer_teacherHead.append(
                torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
            )
            
        # Student 也建议用 Adam
        self.optimizer_studentHead = torch.optim.Adam(
            self.model_studentHead.parameters(), lr=1e-4, weight_decay=1e-4)
        
        self.train_bagloader = train_bagloader
        self.train_instanceloader = train_instanceloader
        self.test_bagloader = test_bagloader
        self.test_instanceloader = test_instanceloader
        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10
        self.PLPostProcessMethod = PLPostProcessMethod
        self.StuFilterType = StuFilterType
        self.smoothE = smoothE
        self.stu_loss_weight_neg = stu_loss_weight_neg
        self.stuOptPeriod = stuOptPeriod
        self.num_teacher = len(model_teacherHead)
        self.teacher_loss_weight = teacher_loss_weight
        self.teacher_pseudo_label_merge_weight = teacher_pseudo_label_merge_weight
        self.best_bag_auc = 0.0
        self.save_dir = None
        self.scaler = torch.amp.GradScaler('cuda')

        # 初始化统计参数，防止第一次 valid 报错
        self.estimated_AttnScore_norm_para_min = [0.0] * self.num_teacher
        self.estimated_AttnScore_norm_para_max = [1.0] * self.num_teacher

    def optimize(self, save_path_root):
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        self.best_bag_auc = 0.0

        for epoch in range(self.num_epoch):
            print(f"\n[Epoch {epoch+1}/{self.num_epoch}]:")
            
            # --- 训练 Teacher ---
            loss_teacher = self.optimize_teacher(epoch)
            
            # --- 评估 Teacher ---
            # 只有当模型稍微收敛一点后再频繁评估，或者每轮都评估
            aucs_teacher = self.evaluate_teacher(epoch)

            loss_student = 0.0
            student_test_auc = 0.0

            # --- 训练 Student (按周期) ---
            if (epoch+1) % self.stuOptPeriod == 0:
                loss_student = self.optimize_student(epoch)
                student_test_auc = self.evaluate_student(epoch)

            print(f"  > [Summary] Teacher Loss: {loss_teacher:.4f} | Student Loss: {loss_student:.4f}")
            print(f"  > [Summary] Student AUC: {student_test_auc:.4f}")
            for idx, auc_t in enumerate(aucs_teacher):
                print(f"  > [Summary] Teacher {idx} AUC: {auc_t:.4f}")
            print("-" * 50)
            
            # 保存模型逻辑 (保持不变)
            if save_path_root and (epoch+1) % self.stuOptPeriod == 0:
                checkpoint = {
                    'epoch': epoch,
                    'best_bag_auc': self.best_bag_auc,
                    'model_encoder': self.model_encoder.state_dict(),
                    'model_studentHead': self.model_studentHead.state_dict(),
                    'optimizer_encoder': self.optimizer_encoder.state_dict(),
                    'optimizer_studentHead': self.optimizer_studentHead.state_dict(),
                }
                for i in range(self.num_teacher):
                    checkpoint[f'model_teacherHead_{i}'] = self.model_teacherHead[i].state_dict()
                
                torch.save(checkpoint, os.path.join(save_path_root, 'checkpoint_latest.pth'))

                if student_test_auc > self.best_bag_auc:
                    self.best_bag_auc = student_test_auc
                    torch.save(checkpoint, os.path.join(save_path_root, 'checkpoint_best.pth'))
                    print(f"  *** New Best Student AUC {self.best_bag_auc:.4f} Saved! ***")
        
        return 0
    
    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        for model_t in self.model_teacherHead:
            model_t.train()
        
        # 定义一个简单的二分类 Log，用于 Debug
        debug_bag_preds = [[] for _ in range(self.num_teacher)]
        debug_bag_labels = []
        
        epoch_loss_sum = 0.0
        loader = self.train_bagloader
        
        pbar = tqdm(loader, desc='Teacher Training', ascii=True)
        
        for iter, (data, label, selected) in enumerate(pbar):
            # Label 处理
            bag_label = label[1].to(self.dev).float() # 确保是 float
            data = data.to(self.dev)

            # --- 前向传播 ---
            # data: [1, N_patches, C, H, W] -> squeeze -> [N, C, H, W]
            feat = self.model_encoder(data.squeeze(0))

            loss_teacher = 0
            
            # 临时存储每个 Teacher 的 Attn Score，用于后续统计 Max/Min
            current_attn_scores = [] 

            for i in range(self.num_teacher):
                # Teacher Forward
                # bag_pred: [1, 2] logits
                # attn_score: [1, N]
                bag_pred_logits, attn_score = self.model_teacherHead[i](feat)
                
                # 收集原始 Attn Score 用于归一化参数统计
                current_attn_scores.append(attn_score.detach().cpu())

                # Softmax 得到概率
                bag_prob = torch.softmax(bag_pred_logits, dim=1)
                
                # 计算 Loss
                loss_t = self.model_teacherHead[i].get_loss(
                    output_bag=bag_prob, 
                    output_inst=attn_score, 
                    bag_label=bag_label
                )
                loss_teacher += loss_t * self.teacher_loss_weight[i]
                
                # Debug: 收集正类概率
                debug_bag_preds[i].append(bag_prob[0, 1].item())

            debug_bag_labels.append(bag_label.item())

            # --- 反向传播 ---
            self.optimizer_encoder.zero_grad()
            for opt_t in self.optimizer_teacherHead:
                opt_t.zero_grad()
            
            loss_teacher.backward()

            # 【修改点 2】 增加梯度裁剪，防止 Teacher 训练初期梯度爆炸导致 AUC 为 0
            torch.nn.utils.clip_grad_norm_(self.model_encoder.parameters(), max_norm=5.0)
            for model_t in self.model_teacherHead:
                torch.nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=5.0)

            self.optimizer_encoder.step()
            for opt_t in self.optimizer_teacherHead:
                opt_t.step()

            epoch_loss_sum += loss_teacher.item()

            # --- 统计 Attn Score 分布 (用于 Student 训练) ---
            # 这里简单地用当前 Batch 更新全局统计 (EMA 或 直接覆盖)
            # 原代码是 epoch 结束统计所有，这里为了简单，我们动态维护或在 epoch 结束时统一处理
            # 为了保持原逻辑兼容，我们这里先暂存所有 Attn
            if iter == 0:
                self.epoch_attn_collection = [[] for _ in range(self.num_teacher)]
            for i in range(self.num_teacher):
                self.epoch_attn_collection[i].append(current_attn_scores[i].squeeze(0))

            # --- 【修改点 3】 详细 Debug 打印 ---
            if iter % 20 == 0:
                desc_str = f"Loss: {loss_teacher.item():.4f}"
                for i in range(self.num_teacher):
                    avg_pred = np.mean(debug_bag_preds[i][-20:]) if len(debug_bag_preds[i]) > 0 else 0
                    desc_str += f" | T{i}_Prob: {avg_pred:.2f}"
                pbar.set_description(desc_str)

        # Epoch 结束：统计 Attention Score 的 Min/Max (供 Student 使用)
        self.estimated_AttnScore_norm_para_min = []
        self.estimated_AttnScore_norm_para_max = []
        for i in range(self.num_teacher):
             # 拼接所有 patch 的 score
             all_scores = torch.cat(self.epoch_attn_collection[i])
             self.estimated_AttnScore_norm_para_min.append(all_scores.min())
             self.estimated_AttnScore_norm_para_max.append(all_scores.max())
        
        return epoch_loss_sum / len(loader)

    def evaluate_teacher(self, epoch):
        self.model_encoder.eval()
        for model_t in self.model_teacherHead:
            model_t.eval()
            
        preds = [[] for _ in range(self.num_teacher)]
        gts = []
        
        with torch.no_grad():
            for data, label, _ in tqdm(self.test_bagloader, desc='Evaluating Teacher', ascii=True):
                bag_label = label[1].float()
                data = data.to(self.dev)
                
                feat = self.model_encoder(data.squeeze(0))
                
                for i in range(self.num_teacher):
                    bag_logits, _ = self.model_teacherHead[i](feat)
                    bag_prob = torch.softmax(bag_logits, dim=1)
                    # 收集正类概率
                    preds[i].append(bag_prob[0, 1].item())
                
                gts.append(bag_label.item())
        
        aucs = []
        for i in range(self.num_teacher):
            # 只有当标签包含 0 和 1 时才计算 AUC
            if len(np.unique(gts)) > 1:
                try:
                    auc = metrics.roc_auc_score(gts, preds[i])
                except ValueError:
                    auc = 0.0
            else:
                auc = 0.0
            aucs.append(auc)
            
        return aucs

    def optimize_student(self, epoch):
        self.model_encoder.train()
        for model_t in self.model_teacherHead:
            model_t.eval() # Teacher 必须 Eval
        self.model_studentHead.train()
        
        epoch_loss = 0.0
        loader = self.train_instanceloader
        
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student Training', ascii=True)):
            data = data.to(self.dev)
            bag_label_gt = label[1].to(self.dev) # [Batch_Size]

            # 1. 提取特征
            feat = self.model_encoder(data) # [B, 512]
            
            # 2. Teacher 生成伪标签
            pseudo_labels = torch.zeros(data.size(0)).float().to(self.dev) # [B]
            
            with torch.no_grad():
                for i in range(self.num_teacher):
                    _, attn_score = self.model_teacherHead[i](feat)
                    
                    # --- 【关键修复】 强制展平为 [B] ---
                    # 无论之前是 [B, 1] 还是 [1, B] 还是 [B]，全部转为 [B]
                    attn_score = attn_score.view(-1)
                        
                    # 归一化 Attention Score
                    _min = self.estimated_AttnScore_norm_para_min[i].to(self.dev)
                    _max = self.estimated_AttnScore_norm_para_max[i].to(self.dev)
                    
                    # 避免除以 0
                    denom = _max - _min
                    if denom < 1e-8:
                        denom = 1.0
                    
                    prob = (attn_score - _min) / (denom + 1e-8)
                    prob = prob.clamp(0.01, 0.99) # 截断
                    
                    # 形状检查 (Debug 用，确保不再发生广播)
                    if prob.shape != pseudo_labels.shape:
                        print(f"Error: Shape Mismatch! Prob: {prob.shape}, Pseudo: {pseudo_labels.shape}")
                        # 强制对齐
                        prob = prob.view_as(pseudo_labels)
                    
                    pseudo_labels += self.teacher_pseudo_label_merge_weight[i] * prob

            # Hard Negative Mining: 阴性包里的所有 Patch 伪标签强制为 0
            # [B] 的掩码操作
            pseudo_labels[bag_label_gt == 0] = 0.0

            # 3. Student 预测
            student_logits = self.model_studentHead(feat)
            student_probs = torch.softmax(student_logits, dim=1)
            
            # 4. Loss
            loss = self.model_studentHead.get_loss(student_probs, pseudo_labels, neg_weight=self.stu_loss_weight_neg)
            
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model_encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.model_studentHead.parameters(), max_norm=5.0)
            
            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(loader)

    def evaluate_student(self, epoch):
        self.model_encoder.eval()
        self.model_studentHead.eval()
        
        # Student 评估通常是基于 Bag Level 的 Max Pooling 结果
        # 因为 Test Loader 是 Instance Loader，我们需要手动聚合
        
        bag_preds = {}
        bag_gts = {}
        
        with torch.no_grad():
            for data, label, _ in tqdm(self.test_instanceloader, desc='Eval Student', ascii=True):
                data = data.to(self.dev)
                
                # 1. 批量推理
                feat = self.model_encoder(data)
                logits = self.model_studentHead(feat)
                probs = torch.softmax(logits, dim=1)[:, 1] # [Batch_Size]
                
                # 2. 将 Tensor 转为 CPU List 以便遍历
                # label[2] 是 slide_index, label[1] 是 slide_label
                batch_slide_indices = label[2].cpu().numpy()
                batch_slide_labels = label[1].cpu().numpy()
                batch_probs = probs.cpu().numpy()
                
                # 3. 遍历当前 Batch 中的每一张图
                for i in range(len(batch_slide_indices)):
                    s_idx = batch_slide_indices[i]
                    s_label = batch_slide_labels[i]
                    s_prob = batch_probs[i]
                    
                    # 记录每个 Slide 的所有 patch 预测值
                    if s_idx not in bag_preds:
                        bag_preds[s_idx] = []
                        bag_gts[s_idx] = s_label
                    
                    bag_preds[s_idx].append(s_prob)
        
        # 聚合：Max Pooling
        final_preds = []
        final_gts = []
        for idx in bag_preds:
            final_preds.append(np.max(bag_preds[idx]))
            final_gts.append(bag_gts[idx])
            
        if len(np.unique(final_gts)) > 1:
            return metrics.roc_auc_score(final_gts, final_preds)
        else:
            return 0.0

    def norm_AttnScore2Prob(self, attn_score, idx_teacher):
        """
        辅助函数：将 Teacher 的 Raw Attention Score 归一化为 0-1 的概率值
        自动处理 1D (Global Log) 和 2D (Student Training) 两种情况
        """
        # --- 情况 A: 输入是 1D Tensor ---
        # 来源: optimize_teacher 函数末尾，所有 Patch 拼接后的大长条
        # 目的: 仅用于计算 AUC 或 Tensorboard 记录，不需要复杂的 Bag-wise 归一化
        if attn_score.dim() == 1:
            _min = attn_score.min()
            _max = attn_score.max()
            # 简单的全局归一化，防止除零
            prob = (attn_score - _min) / (_max - _min + 1e-8)
            return prob

        # --- 情况 B: 输入是 2D Tensor [Batch, N_patches] ---
        # 来源: optimize_student 训练循环中，处理单个 Bag
        # 目的: 生成伪标签 (Pseudo Label) 指导学生学习，必须精确
        elif attn_score.dim() == 2:
            if idx_teacher == 0: 
                # Teacher 0 (ABMIL): 使用【Bag 内部】Min-Max 归一化
                # 解释: ABMIL 的 Softmax 导致分数依赖于 Bag 大小，
                # 必须在当前 Bag 内部拉伸分数，凸显相对重要的 Instance
                _min = attn_score.min(dim=1, keepdim=True)[0]
                _max = attn_score.max(dim=1, keepdim=True)[0]
                prob = (attn_score - _min) / (_max - _min + 1e-8)
                return prob
            else:
                # Teacher 1 (DSMIL): 维持原有逻辑 (使用全局统计参数)
                # DSMIL 的输出通常不受 Bag 大小剧烈影响，可以使用全局参数
                global_min = self.estimated_AttnScore_norm_para_min[idx_teacher]
                global_max = self.estimated_AttnScore_norm_para_max[idx_teacher]
                prob = (attn_score - global_min) / (global_max - global_min + 1e-8)
                return prob
        
        return attn_score
    
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=1000, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # SK algo
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')

    # architecture
    # parser.add_argument('--arch', default='alexnet_MNIST', type=str, help='alexnet or resnet (default: alexnet)')

    # housekeeping
    parser.add_argument('--device', default='0,1', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='DEBUG_MultiTeacher_newPHM', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--dataset_downsampling', default=0.05, type=float, help='sampling the dataset for Debug')

    parser.add_argument('--PLPostProcessMethod', default='NegGuide', type=str,
                        help='Post-processing method of Attention Scores to build Pseudo Lables',
                        choices=['NegGuide', 'NegGuide_TopK', 'NegGuide_Similarity'])
    parser.add_argument('--StuFilterType', default='PseudoBag_85_15_2', type=str,
                        help='Type of using Student Prediction to imporve Teacher '
                             '[ReplaceAS, FilterNegInstance_Top95, FilterNegInstance_ThreProb95, PseudoBag_88_20]')
    parser.add_argument('--smoothE', default=100, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=0.5, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')
    # parser.add_argument('--TeacherLossWeight', nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0', required=True)
    # parser.add_argument('--PLMergeWeight', nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5', required=True)
    parser.add_argument('--teacher_loss_weight', default=[0, 1.0], nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0')
    parser.add_argument('--PLMergeWeight', default=[0, 1], nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5')
    parser.add_argument('--save_dir', default='/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--resume_mode', default='none', choices=['none', 'best', 'latest'], 
                        help='Resume training from: "none" (start fresh), "best" (best AUC), or "latest" (last epoch)')
    parser.add_argument('--resume_path', default='', type=str, 
                        help='Path to the folder containing checkpoints (required if resume_mode is not none)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    
    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_lr{}_Downsample{}_PLPostProcessBy{}_StuFilterType{}_smoothE{}_weightN{}_StuOptP{}_TeacherLossW{}_MergeW{}".format(
               args.seed, args.batch_size, args.lr, args.dataset_downsampling,
               args.PLPostProcessMethod, args.StuFilterType, args.smoothE, args.stu_loss_weight_neg, args.stuOptPeriod,
               str(args.teacher_loss_weight), str(args.PLMergeWeight),
           )

    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device

    utils.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))
    print(name)

    writer = SummaryWriter('/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs_mil_label/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # model
    cuda_num = 0
    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')

    # model
    model_encoder = PretrainedResNet18_Encoder().to(device)
    model_abmil_teacher_head = map_abmil_test(teacher_Attention_head(input_feat_dim=512)).to(device)
    model_dsmil_teacher_head = map_dsmil_test(teacher_DSMIL_head(input_feat_dim=512)).to(device)
    model_student_head = map_student(student_head(input_feat_dim=512)).to(device)

    # optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    # optimizer_abmil_teacher_head = torch.optim.SGD(model_abmil_teacher_head.model.parameters(), lr=args.lr)
    # optimizer_dsmil_teacher_head = torch.optim.SGD(model_dsmil_teacher_head.model.parameters(), lr=args.lr)
    # optimizer_student_head = torch.optim.SGD(model_student_head.parameters(), lr=args.lr)

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr * 0.1, weight_decay=1e-4, momentum=0.9)
    optimizer_abmil_teacher_head = torch.optim.SGD(model_abmil_teacher_head.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    optimizer_dsmil_teacher_head = torch.optim.SGD(model_dsmil_teacher_head.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    optimizer_student_head = torch.optim.SGD(model_student_head.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)

    ds_train, ds_test = gather_align_Img()

    train_ds_return_instance = DataSet_MIL(ds=ds_train, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=False)
    train_ds_return_bag = copy.deepcopy(train_ds_return_instance)
    train_ds_return_bag.return_bag = True
    val_ds_return_instance = DataSet_MIL(ds=ds_test, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=False)
    val_ds_return_bag = DataSet_MIL(ds=ds_test, downsample=args.dataset_downsampling, transform=None, preload=False, return_bag=True)

    train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    print("[Data] {} training samples".format(len(train_loader_instance.dataset)))
    print("[Data] {} evaluating samples".format(len(val_loader_instance.dataset)))

    if torch.cuda.device_count() > 1:
        print("Let's use", len(args.modeldevice), "GPUs for the model")
        if len(args.modeldevice) == 1:
            print('single GPU model', flush=True)
        else:
            model_encoder = nn.DataParallel(model_encoder, device_ids=list(range(len(args.modeldevice))))
            # model_abmil_teacher_head = nn.DataParallel(model_abmil_teacher_head, device_ids=list(range(len(args.modeldevice))))
            # model_dsmil_teacher_head = nn.DataParallel(model_dsmil_teacher_head, device_ids=list(range(len(args.modeldevice))))
            # model_student_head = nn.DataParallel(model_student_head, device_ids=list(range(len(args.modeldevice))))

    start_epoch = 0
    if args.resume_mode != 'none':
        if not args.resume_path or not os.path.exists(args.resume_path):
            print(f"[Warning] Resume path '{args.resume_path}' does not exist! Starting from scratch.")
        else:
            # 确定文件名
            ckpt_name = f'checkpoint_{args.resume_mode}.pth' # checkpoint_best.pth 或 checkpoint_latest.pth
            ckpt_full_path = os.path.join(args.resume_path, ckpt_name)
            
            if os.path.exists(ckpt_full_path):
                print(f"Loading {args.resume_mode} checkpoint from {ckpt_full_path}...")
                checkpoint = torch.load(ckpt_full_path, map_location=f'cuda:{args.device[0]}')
                
                # 加载模型参数
                model_encoder.load_state_dict(checkpoint['model_encoder'])
                model_student_head.load_state_dict(checkpoint['model_studentHead'])
                # 如果使用了 DataParallel，加载时可能需要注意 key 是否包含 'module.'
                
                model_abmil_teacher_head[0].load_state_dict(checkpoint['model_teacherHead_0'])
                model_dsmil_teacher_head[1].load_state_dict(checkpoint['model_teacherHead_1'])
                
                # 加载优化器参数 (如果是 latest 模式，通常需要恢复优化器状态)
                if args.resume_mode == 'latest':
                    optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder'])
                    optimizer_student_head.load_state_dict(checkpoint['optimizer_studentHead'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming form Epoch {start_epoch}")
                else:
                    print("Loaded Best weights. Starting fine-tuning/inference.")
            else:
                print(f"[Error] Checkpoint file {ckpt_full_path} not found!")

    optimizer = Optimizer(model_encoder=model_encoder, 
                          model_teacherHead=[model_abmil_teacher_head, model_dsmil_teacher_head],
                          model_studentHead=model_student_head,
                          optimizer_encoder=optimizer_encoder,
                          optimizer_teacherHead=[optimizer_abmil_teacher_head, optimizer_dsmil_teacher_head],
                          optimizer_studentHead=optimizer_student_head,
                          train_bagloader=train_loader_bag,
                          train_instanceloader=train_loader_instance,
                          test_bagloader=val_loader_bag,
                          test_instanceloader=val_loader_instance,
                          writer=writer,
                          num_epoch=args.epochs,
                          dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          PLPostProcessMethod=args.PLPostProcessMethod,
                          StuFilterType=args.StuFilterType,
                          stu_loss_weight_neg=args.stu_loss_weight_neg,
                          stuOptPeriod=args.stuOptPeriod,
                          teacher_loss_weight=args.teacher_loss_weight,
                          teacher_pseudo_label_merge_weight=args.PLMergeWeight,
                          smoothE=args.smoothE

    )

    save_path = os.path.join(args.save_dir)
    optimizer.optimize(save_path_root=save_path)