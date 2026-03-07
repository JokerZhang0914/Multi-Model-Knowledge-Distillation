import random
import datetime
import argparse
import numpy as np
import torch
import logging
import os
import pyparsing
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm

from model.model_cam import network
from dataset.dataloader_public import load_dataset_from_folders, load_test_img_mask, load_dataset_from_LD, load_dataset_from_LD_test, Dataset_CAM
from utils import evaluate, imutils, optimizer,imutils2
from utils.pyutils import AverageMeter, format_tabs, setup_logger, cal_eta
from utils.camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from model.losses import CTCLoss_neg, DenseEnergyLoss, DiceLoss, get_masked_ptc_loss, get_seg_loss_update, get_energy_loss
from model.PAR import PAR



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description='CAM Training')
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
    parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
    parser.add_argument("--local_crop_size", default=112, type=int, help="crop_size for local view")
    parser.add_argument("--ignore_index", default=255, type=int, help="random index")
    parser.add_argument("--crop_nums", default=10, type=int)

    parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

    parser.add_argument("--work_dir", default="runs/cam", type=str, help="work_dir")
    parser.add_argument("--spg", default=2, type=int, help="samples_per_gpu")
    parser.add_argument("--scales", default=(0.5, 0.7), help="random rescale in training")

    parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
    parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
    parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
    parser.add_argument("--power", default=0.9, type=float, help="power factor for poly scheduler")

    parser.add_argument("--max_iters", default=2000, type=int, help="max training iters")
    parser.add_argument("--log_iters", default=20, type=int, help=" logging iters")
    parser.add_argument("--eval_iters", default=200, type=int, help="validation iters")
    parser.add_argument("--warmup_iters", default=150, type=int, help="warmup_iters")
    parser.add_argument("--save_iters", default=2500, type=int)

    parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score")
    parser.add_argument("--low_thre", default=0.15, type=float, help="low_bkg_score")
    parser.add_argument("--bkg_thre", default=0.25, type=float, help="bkg_score")
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75,1.25, 1.5), help="multi_scales for cam")

    parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
    parser.add_argument("--w_ctc", default=0.0, type=float, help="w_ctc")
    parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
    parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
    parser.add_argument("--w_dice", default=0.1, type=float, help="w_dice")

    parser.add_argument("--temp", default=0.5, type=float, help="temp")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

    parser.add_argument("--seed", default=32, type=int, help="fix random seed")
    parser.add_argument("--save_ckpt", default=True, help="fix random seed")

    parser.add_argument("--local_rank", default=int(os.environ.get('LOCAL_RANK', 0)), type=int, help="local_rank")
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
    parser.add_argument('--backend', default='nccl')

    # 添加本地权重路径参数
    parser.add_argument("--weights", default=None, type=str, help="Path to local weights (.pth)")

    return parser.parse_args()

def validate(model=None, data_loader=None, args=None, n_iter=1, writer=None):
    preds, gts, cams, cams_aux = [], [], [], []
    dice_all = 0.0
    hd95_all = 0.0
    idx = 0

    visual_idx = torch.randint(0, len(data_loader), (1,))

    model.eval()
    avg_meter = AverageMeter()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="validate", leave=True):
            idx += 1
            # 解包数据：图片名、输入图像、像素级标签(Mask)、图像级分类标签
            img_name, inputs, labels, cls_label = data
            
            # 数据移至 GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            # 将输入图像缩放到统一的 crop_size (如 448x448) 进行推理
            # 注意：验证时通常不进行随机裁剪，但这里为了保持输入一致性进行了缩放
            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            # 模型前向传播
            # cls: 图像级分类 Logits
            # segs: 分割分支输出的 Logits
            cls, segs, _, _ = model(inputs,)

            # --- 1. 计算图像级分类指标 ---
            # 将 Logits 转换为二值预测 (阈值 0)
            cls_pred = (cls > 0).type(torch.int16)
            # 计算多标签分类分数 (如 F1-score)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy(), cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            # --- 2. 生成并评估 CAM (类激活图) ---
            # 使用多尺度策略生成 CAM，增强其鲁棒性
            # _cams: 主分类头的 CAM, _cams_aux: 辅助分类头的 CAM
            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            if idx == 1:  # 只打印第一个 batch 看看
                print(f"--- Debug: Raw CAM Max: {_cams.max().item():.4f}, Mean: {_cams.mean().item():.4f} ---")
                logging.info(f"--- Debug: Raw CAM Max: {_cams.max().item():.4f}, Mean: {_cams.mean().item():.4f} ---")
            
            # 处理主 CAM：插值回原始标签尺寸
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            # 将连续的 CAM 热力图转换为离散的伪标签 (背景 vs 前景)
            # 使用 args 中的阈值 (high_thre, low_thre) 确定前景和背景
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            # 处理辅助 CAM：同上
            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            # --- 3. 处理分割分支预测 ---
            # 将分割预测插值回原始标签尺寸
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            # --- 4. 收集数据用于全局评估 ---
            # 将本 batch 的结果添加到列表中 (转为 numpy int16 节省内存)
            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16)) # 分割预测
            cams += list(cam_label.cpu().numpy().astype(np.int16))      # CAM 伪标签
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))  # 辅助 CAM 伪标签
            gts += list(labels.cpu().numpy().astype(np.int16))          # 真实标签

            # --- 5. 计算当前 Batch 的分割指标 ---
            # cal_dice 函数计算 Dice 系数和 HD95 距离
            dice, hd95 = evaluate.cal_dice(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16), labels.cpu().numpy().astype(np.int16))
            dice_all += dice
            hd95_all += hd95

            # --- 6. TensorBoard 可视化 (仅针对选定的那一个 batch) ---
            if idx == visual_idx:
                # 准备数据：取 argmax 得到类别索引图
                preds_ = torch.argmax(resized_segs, dim=1,).cpu().numpy().astype(np.int16)
                pseudo_gts = cam_label.cpu().numpy().astype(np.int16)
                pseudo_gts_aux = cam_label_aux.cpu().numpy().astype(np.int16)
                true_gts = labels.cpu().numpy().astype(np.int16)
                
                # 生成可视化的 Grid 图片 (原图 + 热力图覆盖)
                # imutils2.tensorboard_image_solo 用于处理单张图片的显示
                grid_imgs, grid_cam = imutils2.tensorboard_image_solo(imgs=inputs.clone(), cam=resized_cam)
                _, grid_cam_aux = imutils2.tensorboard_image_solo(imgs=inputs.clone(), cam=resized_cam_aux)

                # 生成标签 Mask 的 Grid 图片 (黑白/彩色 Mask)
                grid_preds = imutils2.tensorboard_label_solo(labels=preds_)
                grid_pseudo_gt = imutils2.tensorboard_label_solo(labels=pseudo_gts)
                grid_true_gt = imutils2.tensorboard_label_solo(labels=true_gts)
                grid_pseudo_gt_aux = imutils2.tensorboard_label_solo(labels=pseudo_gts_aux)
                
                # 写入 TensorBoard
                if writer is not None:
                    writer.add_image("val/images", grid_imgs, global_step=n_iter)           # 输入图像
                    writer.add_image("val/gts", grid_true_gt, global_step=n_iter)           # 真实 Mask
                    writer.add_image("val/preds", grid_preds, global_step=n_iter)           # 分割预测 Mask
                    writer.add_image("val/pseudo_pseudo_gts", grid_pseudo_gt, global_step=n_iter)       # 主 CAM 生成的伪标签
                    writer.add_image("val/pseudo_pseudo_gts_aux", grid_pseudo_gt_aux, global_step=n_iter) # 辅助 CAM 生成的伪标签

            # (可选) 保存特定的 CAM 结果到本地文件
            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})
    
    # --- 7. 计算并返回全局指标 ---
    # 计算平均 Dice 和 HD95
    dice_mean = dice_all / idx
    hd95_mean = hd95_all / idx
    
    # 获取平均分类分数
    cls_score = avg_meter.pop('cls_score')
    
    # 使用 evaluate.scores 计算详细的分割指标 (IoU, Precision, Recall 等)
    # 分别评估：分割预测、主 CAM、辅助 CAM
    seg_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, num_classes=args.num_classes)

    # 记录标量指标到 TensorBoard
    if writer is not None:
        writer.add_scalar('val/cls_score', cls_score, global_step=n_iter)
        writer.add_scalar('val/dice', dice_mean, global_step=n_iter)
        writer.add_scalar('val/hd95', hd95_mean, global_step=n_iter)

    # 恢复模型到训练模式
    model.train()

    # 格式化输出表格字符串
    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=["Background","Endo"])

    # 返回指标供日志打印
    return cls_score, tab_results, dice_mean, hd95_mean

def validate2(model=None, data_loader=None, args=None, n_iter=1, writer=None):
    """
    专门验证模型分类能力的函数，仿照 test.py 的逻辑
    只保留 ACC, AP, AUC 指标
    """
    model.eval()
    
    y_true = []
    y_scores = [] 
    y_preds = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="validate2 (Classification)", leave=True):
            img_name, inputs, labels, cls_label = data
            
            inputs = inputs.cuda()
            cls_label = cls_label.cuda()

            # 保持与验证时一致的输入尺寸
            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            # 模型前向传播，仅使用分类头输出 cls
            cls, segs, _, _ = model(inputs,)

            # 提取阳性概率与预测类别 (兼容 1 维和 2 维输出)
            if cls.shape[1] == 1:
                probs = torch.sigmoid(cls)[:, 0]
            else:
                probs = torch.sigmoid(cls)[:, 1]
                
            preds = (probs > 0.5).long()

            # 提取真实标签 (兼容 1 维和 2 维标签)
            if cls_label.shape[1] == 1:
                labels_true = cls_label[:, 0]
            else:
                labels_true = cls_label[:, 1]
            
            y_true.extend(labels_true.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    # 仿照 test.py 计算指标
    if len(np.unique(y_true)) < 2:
        # print("[Warning] 测试数据仅包含单个类别（全为0或全为1），无法计算 AUC 和 AP，将设为 0.0。")
        auc = 0.0
        ap = 0.0
    else:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0
        
        try:
            ap = average_precision_score(y_true, y_scores)
        except Exception as e:
            # print(f"[Warning] AP calculation error: {e}")
            ap = 0.0

    # 计算 ACC
    acc = accuracy_score(y_true, y_preds)

    # 记录到 TensorBoard
    if writer is not None:
        writer.add_scalar('val_classification/AUC', auc, global_step=n_iter)
        writer.add_scalar('val_classification/AP', ap, global_step=n_iter)
        writer.add_scalar('val_classification/ACC', acc, global_step=n_iter)
    print("val2 classification -> ACC: %.4f | AP: %.4f | AUC: %.4f" % (acc, ap, auc))

    # 恢复模型到训练模式
    model.train()

    return acc, ap, auc

def train(args=None):
    # 设置当前进程使用的 GPU 设备 ID
    torch.cuda.set_device(args.local_rank)
    # 初始化分布式进程组，backend 通常是 'nccl' (用于 NVIDIA GPU)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    # 记录起始时间
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # 加载原有的公开数据集
    img_paths, labels = load_dataset_from_folders()
    
    # 加载 LDPolypVideo 数据集 (例如文件夹 1 到 100)
    ld_paths, ld_labels = load_dataset_from_LD()
    
    # 合并数据
    img_path_list_train = np.concatenate([img_paths, ld_paths], axis=0)
    label_train = np.concatenate([labels, ld_labels], axis=0)
    
    img_path_list_test, mask_test_path = load_test_img_mask()

    val2_img_paths, val2_labels = load_dataset_from_LD_test(start_id=101, max_id=110)

    # train_dataset = Dataset_CAM(
    #     img_path_list_train,
    #     label_train,
    #     type='train',           # 训练模式
    #     resize_range=[512, 640],# 随机缩放范围
    #     rescale_range=args.scales,
    #     crop_size=args.crop_size, # 裁剪尺寸 (如 448)
    #     img_fliplr=True,        # 开启水平翻转增强
    #     ignore_index=255,       # 忽略的标签索引
    #     num_classes=args.num_classes,
    #     aug=True,               # 开启数据增强
    # )

    train_dataset = Dataset_CAM(
        ld_paths,
        ld_labels,
        type='train',           # 训练模式
        resize_range=[512, 640],# 随机缩放范围
        rescale_range=args.scales,
        crop_size=args.crop_size, # 裁剪尺寸 (如 448)
        img_fliplr=True,        # 开启水平翻转增强
        ignore_index=255,       # 忽略的标签索引
        num_classes=args.num_classes,
        aug=True,               # 开启数据增强
    )

    val_dataset = Dataset_CAM(
        img_path_list_test,
        mask_test_path,
        type='val',             # 验证模式
        aug=False,              # 关闭数据增强
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    # 2. 传入 Dataset_CAM，指定 type 为 'val2'
    val2_dataset = Dataset_CAM(
        val2_img_paths,
        val2_labels,         # 此处传入的是标签数组而不是路径
        type='val2', 
        aug=False,
        num_classes=args.num_classes
    )

    # 创建分布式采样器，确保不同 GPU 读取不同的数据子集
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,      # 每个 GPU 的 batch size
        num_workers=args.num_workers, # 数据加载线程数
        pin_memory=False,         # 是否锁页内存
        drop_last=True,           # 丢弃最后一个不完整的 batch
        sampler=train_sampler,    # 使用分布式采样器
        prefetch_factor=4)        # 预取因子

    # 创建验证数据加载器 (Batch size 通常为 1 用于评估)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
    
    val2_loader = DataLoader(val2_dataset, 
                             batch_size=8, 
                             shuffle=False, 
                             num_workers=args.num_workers)

    model = network(
        backbone=args.backbone,       # 如 vit_base_patch16_224
        num_classes=args.num_classes, # 类别数
        pretrained=args.pretrained,   # 是否使用预训练权重
        init_momentum=args.momentum,
        aux_layer=args.aux_layer      # 辅助层的层级索引
    )

    if args.weights is not None:
        if os.path.exists(args.weights):
            logging.info(f"Loading local weights from {args.weights}")
            # map_location 确保在分布式训练时正确映射到对应的 GPU
            state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(args.local_rank))
            
            # 如果权重文件是由 DistributedDataParallel 保存的，需要去除 'module.' 前缀
            if 'model' in state_dict: # 兼容某些保存了整个 dict 的情况
                state_dict = state_dict['model']
            
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # strict=False 可以在微调时允许部分层不匹配（如分类头）
            msg = model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"Checkpoint loaded with message: {msg}")
        else:
            logging.error(f"No weights file found at {args.weights}")

    device = torch.device(args.local_rank)
    param_groups = model.get_param_groups()
    model.to(device) # 将模型移动到 GPU
    
    writer = None
    if args.local_rank == 0:
        writer = SummaryWriter(args.tb_dir)
        raw_dict = vars(args).copy()
        hparam_dict = {}

        for k, v in raw_dict.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                hparam_dict[k] = v
            else:
                # tuple / list / None 等全部转为字符串
                hparam_dict[k] = str(v)

        metric_dict = {"hparam/init": 0.0}

        writer.add_hparams(hparam_dict, metric_dict)

    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters, # 预热迭代次数
        max_iter=args.max_iters,       # 最大迭代次数
        warmup_ratio=args.warmup_lr,   # 预热起始学习率比例
        power=args.power)              # 多项式衰减的幂次
    
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    # 密集能量损失 (Regularization Loss)，用于边界约束
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    
    # 裁剪数量配置
    ncrops = args.crop_nums
    # 对比标记对比损失 (Contrastive Token Contrast Loss)
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=args.temp).cuda()
    # Dice 损失 (用于分割)
    DICE_loss = DiceLoss(args.num_classes).to(device)

    # PAR (Pixel Adaptive Refinement): 基于像素相似度细化伪标签
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in tqdm(range(args.max_iters), desc="train", leave=True):
        # --- 数据获取 (处理 Iterator 耗尽的情况) ---
        try:
            # inputs: 图片张量, cls_label: 图像级标签, img_box: 边界框(若有), crops: 预裁剪的局部图
            inputs, cls_label, img_box, crops = next(train_loader_iter)
        except:
            # 若遍历完一个 epoch，重置 sampler 并重新创建 iterator
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            inputs, cls_label, img_box, crops = next(train_loader_iter)

        # 数据移至 GPU
        inputs = inputs.to(device, non_blocking=True)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs_denorm = imutils.denormalize_img2(inputs.clone()) # 反归一化，用于 PAR 颜色参考
        cls_label = cls_label.to(device, non_blocking=True)

        # --- 步骤 1: 生成 CAM 并提取 ROI (Region of Interest) ---
        # 使用当前模型生成多尺度 CAM (Class Activation Maps)
        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        
        # 基于 CAM 生成 ROI Mask，用于确定不确定区域或前景区域
        roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)

        # --- 步骤 2: 动态裁剪 (Local Crops) ---
        # 根据 ROI mask 在原图上裁剪出局部视角 (Local Crops)，用于对比学习 (CTC)
        # 这里的目的是让模型“看清”那些 CAM 激活较弱或模糊的区域
        local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        
        # 组合全局视图 (crops[:2]) 和 局部视图 (local_crops)
        roi_crops = crops[:2] + local_crops

        # --- 步骤 3: 前向传播 ---
        # cls: 分类 logits
        # segs: 分割 logits
        # fmap: 特征图 (用于 PTC Loss)
        # out_t, out_s: 教师/学生网络的输出 (用于 CTC Loss)
        cls, segs, fmap, cls_aux, out_t, out_s = model(inputs, crops=roi_crops, n_iter=n_iter)

        # ==============================================================
        # 6. 损失计算 (核心部分)
        # ==============================================================
        
        # A. 分类损失 (Classification Loss)
        # 确保模型能识别图像中是否有息肉 (多标签软间隔损失)
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # B. 对比损失 (CTC Loss)
        # 约束局部视图 (Local) 和全局视图 (Global) 的特征一致性
        ctc_loss = CTC_loss(out_s, out_t, flags)

        # C. 补丁标记对比损失 (PTC Loss - Patch Token Contrast)
        # 生成辅助伪标签
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=0.35, high_thre=0.5, low_thre=0.3, ignore_index=args.ignore_index)
        # 将伪标签转换为亲和力矩阵 (Affinity Mask)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        # 约束特征图 (fmap) 符合亲和力矩阵的结构 (同一类别的 patch 特征应相似)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # D. 分割损失 (Segmentation Loss) & 正则化损失 (Reg Loss)
        # 生成高质量伪标签用于监督分割分支
        valid_cam_aux,_ = cam_to_label(cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        
        # 使用 PAR 模块利用图像颜色信息细化 CAM 边界
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        refined_pseudo_label_aux = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        
        # 课程学习策略 (Curriculum Learning):
        # 训练后期 (>=12000 iters) 使用主 CAM 生成的标签，前期使用辅助 CAM 标签
        if n_iter >= 12000:
            supervison_mask = refined_pseudo_label
        else:
            supervison_mask = refined_pseudo_label_aux
            
        segs = F.interpolate(segs, size=supervison_mask.shape[1:], mode='bilinear', align_corners=False)
        
        # 计算交叉熵分割损失
        # seg_loss = get_seg_loss_update(segs, supervison_mask.type(torch.long), ignore_index=args.ignore_index)
        ce_loss = CrossEntropyLoss(ignore_index=args.ignore_index, reduction='mean')
        # seg_loss = get_seg_loss_update(segs, supervison_mask.type(torch.long), ignore_index=args.ignore_index)
        seg_loss = ce_loss(segs, supervison_mask.type(torch.long))
        # 计算能量损失 (Reg Loss)，利用 RGB 相似性约束分割边界
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=supervison_mask, img_box=img_box, loss_layer=loss_layer)

        # 计算 Dice 损失
        dice_loss = DICE_loss(segs, supervison_mask.type(torch.long), softmax=True)

        # E. 总损失加权求和 (Total Loss)
        # Warmup 策略: 前 1500 iter 不计算分割和正则损失，先让分类器收敛
        # if n_iter <= 4500:
        #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        # else:
        #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + args.w_seg * seg_loss + args.w_reg * reg_loss + args.w_dice * dice_loss

        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 6000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss + 0.0 * dice_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + args.w_seg * seg_loss + 0.0 * reg_loss + args.w_dice * dice_loss
        # ==============================================================
        # 7. 反向传播与日志记录
        # ==============================================================
        # 计算当前分类得分 (用于显示)
        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        # 更新统计器
        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'ctc_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'dice_loss': dice_loss.item(),
            'cls_score': cls_score,
        })

        # 梯度清零、反向传播、参数更新
        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪 (此处被注释)
        optim.step()
        # 100 record tensorboard
        if (n_iter + 1) % 100 == 0:
            preds = torch.argmax(segs,dim=1,).cpu().numpy().astype(np.int16)
            refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            refined_gts_aux = refined_pseudo_label_aux.cpu().numpy().astype(np.int16)
            grid_imgs, grid_cam = imutils2.tensorboard_image(imgs=inputs.clone(), cam=valid_cam,nrow=4)
            _, grid_cam_aux = imutils2.tensorboard_image(imgs=inputs.clone(), cam=valid_cam_aux,nrow=4)

            grid_preds = imutils2.tensorboard_label(labels=preds,nrow=4)
            grid_refined_gt = imutils2.tensorboard_label(labels=refined_gts,nrow=4)
            grid_refined_gt_aux = imutils2.tensorboard_label(labels=refined_gts_aux,nrow=4)

            if writer is not None:
                writer.add_image("train/images", grid_imgs, global_step=n_iter)
                writer.add_image("train/preds", grid_preds, global_step=n_iter)
                writer.add_image("train/pseudo_pseudo_gts", grid_refined_gt, global_step=n_iter)
                writer.add_image("train/pseudo_pseudo_gts_aux", grid_refined_gt_aux, global_step=n_iter)

                #writer.add_image("train/pseudo_irn_gts", grid_irn_gt, global_step=n_iter)
                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
                writer.add_image("cam/aux_cams", grid_cam_aux, global_step=n_iter)

        if writer is not None:
            writer.add_scalar('train/seg_loss', seg_loss.item(), global_step=n_iter)
            writer.add_scalar('train/dice_loss', dice_loss.item(), global_step=n_iter)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_step=n_iter)
            writer.add_scalar('train/total_loss', loss.item(), global_step=n_iter)
            writer.add_scalar('train/reg_loss', reg_loss.item(), global_step=n_iter)
            writer.add_scalar('train/ptc_loss', ptc_loss.item(), global_step=n_iter)
            writer.add_scalar('train/ctc_loss', ctc_loss.item(), global_step=n_iter)

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, ctc_loss: %.4f, seg_loss: %.4f,dice_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('ctc_loss'), avg_meter.pop('seg_loss'), avg_meter.pop('dice_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            val2_acc, val2_ap, val2_auc = validate2(model=model, data_loader=val2_loader, n_iter=n_iter, args=args, writer=writer)
            val_cls_score, tab_results, dice_mean, hd95_mean = validate(model=model, data_loader=val_loader,n_iter=n_iter, args=args,writer=writer)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("val dice: %.6f" % (dice_mean))
                logging.info("val hd95 %.6f" % (hd95_mean))
                logging.info("val2 classification -> ACC: %.4f | AP: %.4f | AUC: %.4f" % (val2_acc, val2_ap, val2_auc))
                logging.info("\n"+tab_results)
                
        if (n_iter + 1)% args.save_iters == 0 or (n_iter + 1) == args.max_iters:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
    return True


if __name__ == "__main__":
    
    args = get_args()

    timestamp = "{0:%Y-%m%d-%H%M-%S}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tb_dir = os.path.join(args.work_dir, "tensorboard_log")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)