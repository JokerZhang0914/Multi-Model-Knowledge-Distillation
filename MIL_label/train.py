"""
/home/zhaokaizhang/.conda/envs/mmked/bin/python /home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/MIL_label/train.py --device 0,1 --epochs 20 --dataset_downsampling 1
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

from dataset import DataSet_MIL, gather_align_Img
from model_head import PretrainedResNet18_Encoder, teacher_Attention_head, teacher_DSMIL_head, student_head
from model_head import map_abmil, map_dsmil, map_student


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
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_teacherHead = optimizer_teacherHead
        self.optimizer_studentHead = optimizer_studentHead
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
        self.stuOptPeriod = stuOptPeriod # Student 优化的频率 (每多少个Epoch优化一次)
        self.num_teacher = len(model_teacherHead)
        self.teacher_loss_weight = teacher_loss_weight
        self.teacher_pseudo_label_merge_weight = teacher_pseudo_label_merge_weight
        self.best_bag_auc = 0.0
        self.save_dir = None # 将在 optimize 中赋值或通过参数传入
        self.scaler = torch.amp.GradScaler('cuda')


    def optimize(self, save_path_root):
        self.Bank_all_Bags_label = None
        self.Bank_all_instances_pred_byTeacher = None
        self.Bank_all_instances_feat_byTeacher = None
        self.Bank_all_instances_pred_processed = None
        self.Bank_all_instances_pred_byStudent = None

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        self.best_bag_auc = 0.0

        for epoch in range(self.num_epoch):
            print(f"\n[Epoch {epoch+1}/{self.num_epoch}]:")
            
            loss_teacher = self.optimize_teacher(epoch)
            torch.cuda.empty_cache()  # <--- 清理碎片 
            aucs_teacher = self.evaluate_teacher(epoch)
            torch.cuda.empty_cache()  # <--- 清理碎片

            loss_student = 0.0 # 默认为 0，如果本 Epoch 不训练 Student
            student_test_auc = 0.0

            if epoch % self.stuOptPeriod == 0:
                loss_student = self.optimize_student(epoch)
                torch.cuda.empty_cache()  # <--- 清理碎片
                student_test_auc = self.evaluate_student(epoch)
                torch.cuda.empty_cache()  # <--- 清理碎片

            if (epoch + 1) % 1 == 0:
                print(f"  > Train Loss (Teacher): {loss_teacher:.4f}")
                print(f"  > Train Loss (Student): {loss_student:.4f}")
                print(f"  > Test AUC (Student)  : {student_test_auc:.4f}")
                for idx, auc_t in enumerate(aucs_teacher):
                    print(f"  > Test AUC (Teacher {idx}): {auc_t:.4f}")
                print("-" * 30)
            
            if save_path_root and epoch % self.stuOptPeriod == 0:
                # 1. 准备要保存的状态字典 (包含所有模型和优化器)
                checkpoint = {
                    'epoch': epoch,
                    'best_bag_auc': self.best_bag_auc,
                    'model_encoder': self.model_encoder.state_dict(),
                    'model_studentHead': self.model_studentHead.state_dict(),
                    'optimizer_encoder': self.optimizer_encoder.state_dict(),
                    'optimizer_studentHead': self.optimizer_studentHead.state_dict(),
                    # 保存 Teacher 列表 (假设有两个 Teacher)
                    'model_teacherHead_0': self.model_teacherHead[0].state_dict(),
                    'model_teacherHead_1': self.model_teacherHead[1].state_dict(),
                    'optimizer_teacherHead_0': self.optimizer_teacherHead[0].state_dict(),
                    'optimizer_teacherHead_1': self.optimizer_teacherHead[1].state_dict(),
                }

                # 2. 保存 Latest (最新) 权重
                torch.save(checkpoint, os.path.join(save_path_root, 'checkpoint_latest.pth'))

                # 3. 保存 Best (最佳) 权重 (基于 Student Bag AUC)
                self.current_student_auc = student_test_auc
                if hasattr(self, 'current_student_auc') and self.current_student_auc > self.best_bag_auc:
                    self.best_bag_auc = self.current_student_auc
                    torch.save(checkpoint, os.path.join(save_path_root, 'checkpoint_best.pth'))
                    print(f"Epoch {epoch}: New Best AUC {self.best_bag_auc:.4f} Saved!")
        
        return 0
    
    def optimize_teacher(self, epoch):
        """
        Teacher 训练：
        使用 Bag Loader，输入一组图片，预测 Bag 标签。
        同时收集 Attention Score 用于后续归一化。
        """
        self.model_encoder.train()
        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.train()
        self.model_studentHead.eval() 

        # data shape: [1, N_patches, C, H, W]
        loader = self.train_bagloader

        patch_label_gt = []
        bag_label_gt = []
        patch_label_pred = [[] for i in range(self.num_teacher)]
        bag_label_pred = [[] for i in range(self.num_teacher)]
        patch_corresponding_bag_label = []

        epoch_loss_sum = 0.0

        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher training')):
            # 移动标签到 GPU
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # --- 前向传播：特征提取 ---
            # data.squeeze(0) 去掉 batch 维度，变为 [N_patches, C, H, W]
            # feat: [N_patches, Feature_Dim]
            feat = self.model_encoder(data.squeeze(0))

            if epoch > self.smoothE and label[1] == 1:
                pass
            else:
                loss_teacher = 0
                # --- 多教师前向传播与 Loss 计算 ---
                for i in range(self.num_teacher):
                    # 输入特征，得到 Bag 预测 logits 和 Instance 注意力分数
                    # bag_prediction_teacher_i: [1, Num_Classes]
                    # instance_attn_score_teacher_i: [1, N_patches] (Raw Logits or Scores)
                    bag_prediction_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead[i](feat)
                    
                    bag_prediction_teacher_i = torch.softmax(bag_prediction_teacher_i, dim=1)

                    # 计算loss
                    loss_teacher_i = self.model_teacherHead[i].get_loss(
                        output_bag = bag_prediction_teacher_i, 
                        output_inst = instance_attn_score_teacher_i, 
                        bag_label = label[1])
                    loss_teacher += loss_teacher_i * self.teacher_loss_weight[i]

                    # 收集预测结果 (detach 切断梯度，节省显存)
                    patch_label_pred[i].append(instance_attn_score_teacher_i.detach().squeeze(0))
                    bag_label_pred[i].append(bag_prediction_teacher_i.detach()[0, 1])

                # --- 反向传播与优化 ---
                self.optimizer_encoder.zero_grad()
                for optimizer_teacherHead_i in self.optimizer_teacherHead:
                    optimizer_teacherHead_i.zero_grad()
                loss_teacher.backward()

                if isinstance(loss_teacher, torch.Tensor):
                    epoch_loss_sum += loss_teacher.item()

                self.optimizer_encoder.step()
                for optimizer_teacherHead_i in self.optimizer_teacherHead:
                    optimizer_teacherHead_i.step()
            
            # 收集 Ground Truth
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_gt.append(label[1])
            patch_corresponding_bag_label.append(torch.ones([data.shape[1]]).to(self.dev)*label[1])
            
            # 记录 TensorBoard
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Teacher', loss_teacher.item(), niter)

        # 4. 数据整理与统计
        # 将列表拼接成大 Tensor
        patch_label_pred = [torch.cat(i) for i in patch_label_pred]
        bag_label_pred = [torch.tensor(i) for i in bag_label_pred]
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_gt = torch.cat(bag_label_gt)
        patch_corresponding_bag_label = torch.cat(patch_corresponding_bag_label)

        # *** 关键步骤：统计 Attention Score 的分布范围 ***
        # 计算整个训练集中所有切片分数的最小值和最大值
        # 这将在 optimize_student 中用于将 Teacher 的分数归一化为 [0, 1] 的伪标签
        self.estimated_AttnScore_norm_para_min = [i.min() for i in patch_label_pred]
        self.estimated_AttnScore_norm_para_max = [i.max() for i in patch_label_pred]

        # # 5. 计算指标并记录AUC
        for i in range(self.num_teacher):
            # 将分数归一化到 [0, 1] (仅用于评估 AUC，不影响 min/max 参数)
            patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred[i], idx_teacher=i)
            # 计算 Bag AUC
            bag_auc_ByTeacher = utils.cal_auc(bag_label_gt.reshape(-1), bag_label_pred[i].reshape(-1))

            self.writer.add_scalar('train_bag_AUC_byTeacher{}'.format(i), bag_auc_ByTeacher, epoch)

        # print("Epoch:{} train_bag_AUC_byTeacher:{}".format(epoch, bag_auc_ByTeacher))
        epoch_avg_loss = epoch_loss_sum / len(loader)

        return epoch_avg_loss
    
    
    def optimize_student(self, epoch):
        """
        Student训练
        利用 Teacher 生成的 Attention Score 作为伪标签 (Pseudo Label)，训练 Student 对单张切片进行分类。
        """

        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.eval() # TODO: 源码为train()，感觉学生训练时应是eval()
            # model_teacherHead_i.train()
        self.model_encoder.train()
        self.model_studentHead.train()

        # data shape: [Batch_Size, C, H, W]
        loader = self.train_instanceloader

        # 初始化记录张量
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)

        epoch_loss_sum = 0.0
        
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc="Student Training")):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # --- teacher 生成伪标签 ---
            # 提取特征
            feat = self.model_encoder(data)
            # 生成伪标签
            pseudo_instance_label = torch.zeros_like(label[0]).float()
            with torch.no_grad():
                for i in range(self.num_teacher):
                    _, instance_attn_score = self.model_teacherHead[i](feat)

                    # 归一化并融合：
                    # 1. norm_AttnScore2Prob: 利用 optimize_teacher 中统计的 min/max 将分数映射到 [0, 1]
                    # 2. clamp: 截断数值防止 log(0)
                    # 3. 加权平均多个 Teacher 的结果
                    pseudo_instance_label += self.teacher_pseudo_label_merge_weight[i] * \
                        self.norm_AttnScore2Prob(instance_attn_score, idx_teacher=i).clamp(min=1e-5, max=1-1e-5).squeeze(0)

            # --- 关键策略：Hard Negative Mining (硬负样本修正) ---
            # 利用先验知识：如果一个包是阴性(0)，那么它里面所有的切片一定都是阴性(0)。
            # 强制将这些样本的伪标签设为 0，纠正 Teacher 可能的误判 (False Positive)。
            pseudo_instance_label[label[1]==0] = 0

            # --- Student 训练 ---
            # 输入同样的特征，输出预测标签
            patch_prediction = self.model_studentHead(feat)
            patch_prediction = torch.softmax(patch_prediction, dim=1)

            loss_student = self.model_studentHead.get_loss(
                prediction=patch_prediction, 
                target=pseudo_instance_label, 
                neg_weight=self.stu_loss_weight_neg
            )

            # --- 反向传播与参数更新 ---
            self.optimizer_encoder.zero_grad()
            self.optimizer_studentHead.zero_grad()

            loss_student.backward()

            epoch_loss_sum += loss_student.item()

            self.optimizer_encoder.step()
            self.optimizer_studentHead.step()

            # 记录数据用于评估
            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)

        # 4. 计算 Student 的 Bag-Level AUC
        # 一个 Bag 的得分为其包含的所有切片中得分最高的那一个。
        bag_label_gt_coarse = []
        bag_label_prediction = []

        # 获取所有唯一的 Bag ID
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            # 找到属于当前 Bag 的所有切片索引
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            # 记录 Bag 真实标签
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            # 记录 Bag 预测分 (取最大值)
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())
        
        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)
        
        # 计算并记录 AUC
        bag_auc_ByStudent = utils.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('train_bag_AUC_byStudent', bag_auc_ByStudent, epoch)

        epoch_avg_loss = epoch_loss_sum / len(loader)
        return epoch_avg_loss

    def evaluate_teacher(self, epoch):
        """评估 Teacher 的性能"""
        self.model_encoder.eval()
        for model_teacherHead_i in self.model_teacherHead:
            model_teacherHead_i.eval()
        self.model_studentHead.eval()
        
        loader = self.test_bagloader

        # 初始化记录列表
        patch_label_gt = []
        bag_label_gt = []
        patch_label_pred = [[] for i in range(self.num_teacher)]
        bag_label_prediction_withAttnScore = [[] for i in range(self.num_teacher)]

        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0)) # 提取特征
                for i in range(self.num_teacher):
                    # 获取预测结果
                    bag_prediction_withAttnScore_i, instance_attn_score = self.model_teacherHead[i](feat)
                    bag_prediction_withAttnScore_i = torch.softmax(bag_prediction_withAttnScore_i, 1)

                    # 收集结果
                    patch_label_pred[i].append(instance_attn_score.detach().squeeze(0))
                    bag_label_prediction_withAttnScore[i].append(bag_prediction_withAttnScore_i.detach()[0, 1])
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_gt.append(label[1])

        # 整理结果并计算 AUC
        patch_label_pred = [torch.cat(i) for i in patch_label_pred]
        bag_label_prediction_withAttnScore = [torch.tensor(i) for i in bag_label_prediction_withAttnScore]
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_gt = torch.cat(bag_label_gt)

        teacher_aucs = []
        for i in range(self.num_teacher):
            patch_label_pred_normed = (patch_label_pred[i] - patch_label_pred[i].min()) / (patch_label_pred[i].max() - patch_label_pred[i].min())
            # 计算 Bag AUC
            bag_auc_ByTeacher_withAttnScore = utils.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore[i].reshape(-1))
            # 记录到 TensorBoard
            self.writer.add_scalar('test_bag_AUC_byTeacher{}'.format(i), bag_auc_ByTeacher_withAttnScore, epoch)

            teacher_aucs.append(bag_auc_ByTeacher_withAttnScore)
        return teacher_aucs
    

    def evaluate_student(self, epoch):
        """评估 Student 的性能"""
        self.model_encoder.eval()
        self.model_studentHead.eval()
        
        loader = self.test_instanceloader

        # 初始化记录张量
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).float().to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)

        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # --- Student 推理 ---
            with torch.no_grad():
                feat = self.model_encoder(data)
                patch_prediction = self.model_studentHead(feat)
                patch_prediction = torch.softmax(patch_prediction, dim=1)

            # 记录结果 (通过 selected 索引填入大张量的对应位置)
            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction.detach()[:, 1]
            patch_label_gt[selected, 0] = label[0]
            bag_label_gt[selected, 0] = label[1]
        
        # 计算 Bag-Level AUC
        bag_label_gt_coarse = []
        bag_label_prediction = []
        available_bag_idx = patch_corresponding_slide_idx.unique()

        # Max Pooling 聚合策略
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            # 校验数据一致性
            if bag_label_gt[idx_same_bag_i].max() != bag_label_gt[idx_same_bag_i].max():
                raise
            bag_label_gt_coarse.append(bag_label_gt[idx_same_bag_i].max())
            bag_label_prediction.append(patch_label_pred[idx_same_bag_i].max())

        bag_label_gt_coarse = torch.tensor(bag_label_gt_coarse)
        bag_label_prediction = torch.tensor(bag_label_prediction)

        # 计算并记录 AUC
        bag_auc_ByStudent = utils.cal_auc(bag_label_gt_coarse.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_AUC_byStudent', bag_auc_ByStudent, epoch)

        return bag_auc_ByStudent

    def norm_AttnScore2Prob(self, attn_score, idx_teacher):
        """辅助函数：将 Teacher 的 Raw Attention Score 归一化为 0-1 的概率值"""
        prob = (attn_score - self.estimated_AttnScore_norm_para_min[idx_teacher]) / \
               (self.estimated_AttnScore_norm_para_max[idx_teacher] - self.estimated_AttnScore_norm_para_min[idx_teacher])
        return prob
    
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
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='DEBUG_MultiTeacher_newPHM', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--dataset_downsampling', default=0.1, type=float, help='sampling the dataset for Debug')

    parser.add_argument('--PLPostProcessMethod', default='NegGuide', type=str,
                        help='Post-processing method of Attention Scores to build Pseudo Lables',
                        choices=['NegGuide', 'NegGuide_TopK', 'NegGuide_Similarity'])
    parser.add_argument('--StuFilterType', default='PseudoBag_85_15_2', type=str,
                        help='Type of using Student Prediction to imporve Teacher '
                             '[ReplaceAS, FilterNegInstance_Top95, FilterNegInstance_ThreProb95, PseudoBag_88_20]')
    parser.add_argument('--smoothE', default=100, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=1.0, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')
    # parser.add_argument('--TeacherLossWeight', nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0', required=True)
    # parser.add_argument('--PLMergeWeight', nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5', required=True)
    parser.add_argument('--teacher_loss_weight', default=[1.0, 1.0], nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0')
    parser.add_argument('--PLMergeWeight', default=[0.5, 0.5], nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5')
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
    model_abmil_teacher_head = map_abmil(teacher_Attention_head(input_feat_dim=512)).to(device)
    model_dsmil_teacher_head = map_dsmil(teacher_DSMIL_head(input_feat_dim=512)).to(device)
    model_student_head = map_student(student_head(input_feat_dim=512)).to(device)

    optimizer_encoder = torch.optim.SGD(model_encoder.parameters(), lr=args.lr)
    optimizer_abmil_teacher_head = torch.optim.SGD(model_abmil_teacher_head.parameters(), lr=args.lr)
    optimizer_dsmil_teacher_head = torch.optim.SGD(model_dsmil_teacher_head.parameters(), lr=args.lr)
    optimizer_student_head = torch.optim.SGD(model_student_head.parameters(), lr=args.lr)

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