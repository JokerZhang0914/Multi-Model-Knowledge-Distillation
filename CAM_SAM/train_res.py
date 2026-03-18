import os
import argparse
import datetime
import logging
import random
import signal
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.resnet import PretrainedResNet18_Encoder, Student_Head, FullResNet
from dataset.load_data import load_dataset_from_folders, load_kvasir_img_mask, load_CVC_img_mask
from dataset.dataset_cam import ResNetDataset, SegDataset
from utils import cam2mask, load_checkpoint, dice_loss, dice_score
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 辅助函数与参数配置
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description='ResNet18 (Encoder+Student) 训练脚本')
    parser.add_argument("--seed", default=311, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--split", default=0.15, type=float)
    parser.add_argument("--cam_method", default="gradcampp", choices=["gradcampp", "layercam"])
    parser.add_argument("--cam_weight", default=0.1, type=float)
    parser.add_argument("--cam_warmup_epochs", default=15, type=int)
    parser.add_argument("--cls_downsample", default=0.1, type=float, help="classification data ratio (0,1]")
        
    parser.add_argument("--work_dir", default="runs/cam_res", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    
    # 预训练权重加载 (可选)
    # parser.add_argument("--encoder_ckpt", default='/home/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1445_layercam/checkpoint/best_resnet_encoder.pth', type=str, help="初始 Encoder 权重路径")
    # parser.add_argument("--student_ckpt", default='/home/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1445_layercam/checkpoint/best_resnet_student.pth', type=str, help="初始 Student 权重路径")

    parser.add_argument("--encoder_ckpt", default='/home/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1757_gradcampp/checkpoint/best_resnet_encoder.pth', type=str, help="初始 Encoder 权重路径")
    parser.add_argument("--student_ckpt", default='/home/zhaokaizhang/code/test_code/runs/cam_res/2026-0311-1757_gradcampp/checkpoint/best_resnet_student.pth', type=str, help="初始 Student 权重路径")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(work_dir):
    log_file = os.path.join(work_dir, 'train_resnet.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def _log_runtime_state(tag):
    try:
        pid = os.getpid()
        logging.info(f"[{tag}] pid={pid}")
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logging.info(f"[{tag}] max_rss_kb={rss_kb}")
        except Exception as e:
            logging.info(f"[{tag}] resource_unavailable: {e}")

        if torch.cuda.is_available():
            try:
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                logging.info(f"[{tag}] cuda_mem_alloc_mb={mem_alloc:.1f} cuda_mem_reserved_mb={mem_reserved:.1f}")
            except Exception as e:
                logging.info(f"[{tag}] cuda_mem_unavailable: {e}")
    except Exception as e:
        logging.info(f"[{tag}] state_log_failed: {e}")


def _install_signal_handlers():
    def _handler(signum, frame):
        signame = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
        logging.warning(f"[Signal] Caught {signame} ({signum}).")
        _log_runtime_state(f"Signal:{signum}")
        logging.warning("[Signal] Exiting after signal.")
        sys.exit(0)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass

# ==========================================
# 3. 优化器类 (训练/验证封装)
# ==========================================
class Optimizer:
    def __init__(self, args, device, writer):
        self.args = args
        self.device = device
        self.writer = writer

        # 1. 准备分类数据集
        logging.info("[*] 正在加载数据集...")
        img_paths, labels = load_dataset_from_folders()

        train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
            img_paths,
            labels,
            test_size=args.split,
            random_state=args.seed,
            stratify=labels if args.num_classes == 2 else None
        )
        logging.info(f"[*] 数据集划分完毕: 训练集 {len(train_img_paths)}，验证集 {len(val_img_paths)}")

        if 0 < args.cls_downsample < 1.0:
            train_total = len(train_img_paths)
            keep = max(1, int(train_total * args.cls_downsample))
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(train_total, size=keep, replace=False)
            train_img_paths = train_img_paths[idx]
            train_labels = train_labels[idx]
            logging.info(f"[*] 分类训练集下采样: {keep}/{train_total}")

            val_total = len(val_img_paths)
            keep_val = max(1, int(val_total * args.cls_downsample))
            rng_val = np.random.RandomState(args.seed + 1)
            idx_val = rng_val.choice(val_total, size=keep_val, replace=False)
            val_img_paths = val_img_paths[idx_val]
            val_labels = val_labels[idx_val]
            logging.info(f"[*] 分类验证集下采样: {keep_val}/{val_total}")

        train_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ResNetDataset(train_img_paths, train_labels, transform=train_transform)
        val_dataset = ResNetDataset(val_img_paths, val_labels, transform=val_transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size/2),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size/2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 2. 准备分割数据集
        seg_img_kv, seg_mask_kv = load_kvasir_img_mask()
        seg_img_cvc, seg_mask_cvc = load_CVC_img_mask()
        seg_img_all = list(seg_img_kv) + list(seg_img_cvc)
        seg_mask_all = list(seg_mask_kv) + list(seg_mask_cvc)

        if len(seg_img_all) == 0:
            raise RuntimeError("Segmentation dataset is empty. Check Kvasir/CVC paths.")

        seg_train_imgs, seg_val_imgs, seg_train_masks, seg_val_masks = train_test_split(
            seg_img_all,
            seg_mask_all,
            test_size=args.split,
            random_state=args.seed
        )

        seg_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        seg_train_dataset = SegDataset(seg_train_imgs, seg_train_masks, transform=seg_transform, mask_size=args.crop_size)
        seg_val_dataset = SegDataset(seg_val_imgs, seg_val_masks, transform=seg_transform, mask_size=args.crop_size)

        self.seg_train_loader = DataLoader(
            seg_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.seg_val_loader = DataLoader(
            seg_val_dataset,
            batch_size=int(args.batch_size/2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 3. 初始化模型
        logging.info("[*] 初始化 Encoder 和 Student 模型...")
        self.encoder = PretrainedResNet18_Encoder(freeze=False).to(device)
        self.student = Student_Head(input_dim=512, num_classes=args.num_classes).to(device)

        if args.encoder_ckpt:
            load_checkpoint(self.encoder, args.encoder_ckpt, device)
        if args.student_ckpt:
            load_checkpoint(self.student, args.student_ckpt, device)

        # 4. 优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.student.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})
        self.best_score = -1.0

        # 5. CAM 引擎
        self.cam_method = args.cam_method
        self.cam_model = FullResNet(self.encoder, self.student).to(device)
        self.cam_engine = self._build_cam_engine()
        self.cam_activations = None
        self._register_cam_hook()

    def _build_cam_engine(self):
        target_layers = [self._get_target_layer()]
        if self.cam_method == "layercam":
            return LayerCAM(model=self.cam_model, target_layers=target_layers)
        return GradCAMPlusPlus(model=self.cam_model, target_layers=target_layers)

    def _get_target_layer(self):
        return self.encoder.features[-2]

    def _register_cam_hook(self):
        target_layer = self._get_target_layer()

        def _hook(module, inp, out):
            self.cam_activations = out

        self._cam_hook = target_layer.register_forward_hook(_hook)

    def _cam_to_tensor(self, cam_np):
        cam_np = np.clip(cam_np, 0, 1)
        return torch.from_numpy(cam_np).unsqueeze(1).float().to(self.device)

    def _compute_cam_autograd(self, logits, activations, cam_method):
        score = logits[:, 1]
        grads = torch.autograd.grad(score.sum(), activations, create_graph=True, retain_graph=True)[0]
        if cam_method == "layercam":
            cam = torch.relu(grads) * activations
            cam = cam.sum(dim=1, keepdim=True)
        else:
            grad2 = grads ** 2
            grad3 = grads ** 3
            sum_act = (activations * grad3).sum(dim=(2, 3), keepdim=True)
            alpha = grad2 / (2 * grad2 + sum_act + 1e-6)
            weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)

        b = cam.size(0)
        cam_min = cam.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        cam = F.interpolate(cam, size=(self.args.crop_size, self.args.crop_size), mode="bilinear", align_corners=False)
        return cam

    def train(self, epoch):
        self.encoder.train()
        self.student.train()
        cls_loss_total = 0.0
        cam_loss_total = 0.0

        warmup = max(1, self.args.cam_warmup_epochs)
        if self.args.cam_warmup_epochs <= 0:
            cam_w = 0.5
        else:
            t = min(epoch + 1, warmup) / warmup
            cam_w = self.args.cam_weight + (0.5 - self.args.cam_weight) * t

        cls_iter = iter(self.train_loader)
        seg_iter = iter(self.seg_train_loader)
        steps = max(len(self.train_loader), len(self.seg_train_loader))
        pbar = tqdm(range(steps), desc=f"[Train] Epoch [{epoch+1}/{self.args.epochs}]")

        for _ in pbar:
            try:
                imgs_c, labels = next(cls_iter)
            except StopIteration:
                cls_iter = iter(self.train_loader)
                imgs_c, labels = next(cls_iter)

            try:
                imgs_s, masks = next(seg_iter)
            except StopIteration:
                seg_iter = iter(self.seg_train_loader)
                imgs_s, masks = next(seg_iter)

            imgs_c, labels = imgs_c.to(self.device).float(), labels.to(self.device).long()
            imgs_s, masks = imgs_s.to(self.device).float(), masks.to(self.device).float()

            self.optimizer.zero_grad()

            feats_c = self.encoder(imgs_c)
            logits_c = self.student(feats_c)
            cls_loss = self.criterion(logits_c, labels)

            feats_s = self.encoder(imgs_s)
            logits_s = self.student(feats_s)
            cam = self._compute_cam_autograd(logits_s, self.cam_activations, self.cam_method)
            bce = F.binary_cross_entropy(cam, masks)
            dice = dice_loss(cam, masks)
            cam_loss = 0.5 * bce + 0.5 * dice

            loss = cls_loss + cam_w * cam_loss
            loss.backward()
            self.optimizer.step()
            self.cam_activations = None

            cls_loss_total += cls_loss.item()
            cam_loss_total += cam_loss.item()
            pbar.set_postfix({'cls_loss': f"{cls_loss.item():.4f}", 'cam_loss': f"{cam_loss.item():.4f}", 'cam_w': f"{cam_w:.3f}"})

        cls_loss_avg = cls_loss_total / max(1, len(self.train_loader))
        cam_loss_avg = cam_loss_total / max(1, len(self.seg_train_loader))
        logging.info(f"[Epoch {epoch+1} Train] ClsLoss: {cls_loss_avg:.4f} | CamLoss: {cam_loss_avg:.4f} | CamW: {cam_w:.3f}")
        self.writer.add_scalar('Train/Cls_Loss', cls_loss_avg, epoch + 1)
        self.writer.add_scalar('Train/Cam_Loss', cam_loss_avg, epoch + 1)
        return cls_loss_avg, cam_loss_avg

    def validate(self, epoch):
        self.encoder.eval()
        self.student.eval()
        y_true, y_scores, y_preds = [], [], []
        cam_loss_total = 0.0
        dice_scores = []
        miou_scores = []

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="[Cls Validating]", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device).long()

                feats = self.encoder(imgs)
                logits = self.student(feats)

                probs = torch.softmax(logits, dim=1)[:, 1] if self.args.num_classes == 2 else torch.softmax(logits, dim=1).max(1)[0]
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
                y_preds.extend(preds.cpu().numpy())

        # CAM-分割一致性验证 + Dice
        for imgs, masks in tqdm(self.seg_val_loader, desc="[Seg Validating]", leave=False):
            imgs = imgs.to(self.device).float()
            masks = masks.to(self.device).float()
            feats = self.encoder(imgs)
            logits = self.student(feats)
            cam = self._compute_cam_autograd(logits, self.cam_activations, self.cam_method)
            bce = F.binary_cross_entropy(cam, masks)
            dice = dice_loss(cam, masks)
            cam_loss_total += (bce + dice).item()

            targets = [ClassifierOutputTarget(1) for _ in range(imgs.size(0))]
            cam_np = self.cam_engine(input_tensor=imgs, targets=targets)
            mask_np = masks.detach().cpu().numpy()
            for i in range(cam_np.shape[0]):
                pred_mask = cam2mask(cam_np[i])
                gt_mask = (mask_np[i, 0] > 0.5).astype(np.uint8)
                inter = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                dice_scores.append(dice_score(pred_mask, gt_mask))
                miou_scores.append(inter / (union + 1e-6))

        y_true, y_scores, y_preds = np.array(y_true), np.array(y_scores), np.array(y_preds)

        if len(np.unique(y_true)) < 2:
            auc = 0.0
        else:
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = 0.0

        acc = accuracy_score(y_true, y_preds)
        cam_loss_avg = cam_loss_total / max(1, len(self.seg_val_loader))
        dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        miou = float(np.mean(miou_scores)) if miou_scores else 0.0

        self.encoder.train()
        self.student.train()
        self.writer.add_scalar('Val/ACC', acc, epoch + 1)
        self.writer.add_scalar('Val/AUC', auc, epoch + 1)
        self.writer.add_scalar('Val/Cam_Loss', cam_loss_avg, epoch + 1)
        self.writer.add_scalar('Val/Dice', dice, epoch + 1)
        self.writer.add_scalar('Val/mIoU', miou, epoch + 1)
        logging.info(f"[Epoch {epoch+1} Validation] ACC: {acc:.4f} | AUC: {auc:.4f} | Dice: {dice:.4f} | mIoU: {miou:.4f} | CamLoss: {cam_loss_avg:.4f}")

        return acc, auc, dice, miou, cam_loss_avg

    def optimize(self):
        logging.info("[*] 开始 ResNet 训练 (Cls + CAM-Seg)...")
        for epoch in range(self.args.epochs):
            self.train(epoch)
            acc, auc, dice, miou, _ = self.validate(epoch)

            encoder_to_save = self.encoder.state_dict()
            student_to_save = self.student.state_dict()

            score = 0.35 * acc + 0.3 * auc + 0.25 * dice + 0.1 * miou
            ckpt_dir = os.path.join(self.args.work_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)

            if score > self.best_score:
                self.best_score = score
                best_enc_path = os.path.join(ckpt_dir, "best_resnet_encoder.pth")
                best_stu_path = os.path.join(ckpt_dir, "best_resnet_student.pth")

                torch.save(encoder_to_save, best_enc_path)
                torch.save(student_to_save, best_stu_path)
                logging.info(f"[*] 发现更优的 score(AUC+Dice): {self.best_score:.4f}，Encoder 和 Student 均已独立保存。")

            if (epoch + 1) % 5 == 0:
                last_enc_path = os.path.join(ckpt_dir, f"encoder_epoch_{epoch+1}.pth")
                last_stu_path = os.path.join(ckpt_dir, f"student_epoch_{epoch+1}.pth")

                torch.save(encoder_to_save, last_enc_path)
                torch.save(student_to_save, last_stu_path)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    args = get_args()
    set_seed(args.seed)
    
    # 设置 device（单卡，不依赖环境变量）
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    
    # 目录与日志初始化
    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, f'{timestamp}_{args.cam_method}')
    os.makedirs(args.work_dir, exist_ok=True)
    
    setup_logger(args.work_dir)
    _install_signal_handlers()
    tb_dir = os.path.join(args.work_dir, "tensorboard_log")
    writer = SummaryWriter(tb_dir)
    
    logging.info(f"[*] 训练参数: {args}")
    logging.info(f"[*] 工作目录: {args.work_dir}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()

    writer.close()

if __name__ == '__main__':
    print('-----------start training-----------')
    main()
