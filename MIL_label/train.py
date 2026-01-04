import argparse
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

import utils

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

    def optimize(self):
        self.Bank_all_Bags_label = None
        self.Bank_all_instances_pred_byTeacher = None
        self.Bank_all_instances_feat_byTeacher = None
        self.Bank_all_instances_pred_processed = None
        self.Bank_all_instances_pred_byStudent = None

        for epoch in range(self.num_epoch):
            self.optimize_teacher(epoch)
            self.evaluate_teacher(epoch)
            if epoch % self.stuOptPeriod == 0:
                self.optimize_student(epoch)
                self.evaluate_student(epoch)
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

        return 0
    
    
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
            pseudo_instance_label = torch.zeros_like(label[0])
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

        return 0

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

        for i in range(self.num_teacher):
            patch_label_pred_normed = (patch_label_pred[i] - patch_label_pred[i].min()) / (patch_label_pred[i].max() - patch_label_pred[i].min())
            # 计算 Bag AUC
            bag_auc_ByTeacher_withAttnScore = utils.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore[i].reshape(-1))
            # 记录到 TensorBoard
            self.writer.add_scalar('test_bag_AUC_byTeacher{}'.format(i), bag_auc_ByTeacher_withAttnScore, epoch)
        return 0
    

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

        return 0

    def norm_AttnScore2Prob(self, attn_score, idx_teacher):
        """辅助函数：将 Teacher 的 Raw Attention Score 归一化为 0-1 的概率值"""
        prob = (attn_score - self.estimated_AttnScore_norm_para_min[idx_teacher]) / \
               (self.estimated_AttnScore_norm_para_max[idx_teacher] - self.estimated_AttnScore_norm_para_min[idx_teacher])
        return prob
    
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size (default: 256)')
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
    parser.add_argument('--smoothE', default=0, type=int, help='num of epoch to apply StuFilter')
    parser.add_argument('--stu_loss_weight_neg', default=1.0, type=float, help='weight of neg instances in stu training')
    parser.add_argument('--stuOptPeriod', default=1, type=int, help='period of stu optimization')
    parser.add_argument('--TeacherLossWeight', nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0', required=True)
    parser.add_argument('--PLMergeWeight', nargs='+', type=float, help='weight of merge teachers pseudo label, like: 0.5 0.5', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device