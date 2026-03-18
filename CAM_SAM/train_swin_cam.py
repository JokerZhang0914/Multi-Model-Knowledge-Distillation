import os
import argparse
import datetime
import logging
import random
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

from dataset.load_data import load_dataset_from_folders, load_kvasir_img_mask, load_CVC_img_mask
from dataset.dataset_cam import SwinDataset, SegDataset
from utils import cam2mask, load_checkpoint, dice_loss, dice_score
from model.swin_transformer import SwinTransformer
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser(description='Swin Transformer (Cls + CAM-Seg) 训练脚本')
    parser.add_argument("--seed", default=311, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
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
    parser.add_argument("--work_dir", default="runs/cam_swin", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--gpu", default="2", type=str)

    # Swin config
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--embed_dim", default=96, type=int)
    parser.add_argument("--depths", default="2,2,6,2", type=str)
    parser.add_argument("--num_heads", default="3,6,12,24", type=str)
    parser.add_argument("--window_size", default=7, type=int)
    parser.add_argument("--mlp_ratio", default=4.0, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--attn_drop_rate", default=0.0, type=float)
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    parser.add_argument("--ape", action="store_true")
    parser.add_argument("--patch_norm", action="store_true", default=True)
    parser.add_argument("--use_checkpoint", action="store_true")

    parser.add_argument("--ckpt", default='/home/zhaokaizhang/code/test_code/runs/cam_swin/2026-0311-1809_layercam/checkpoint/best_swin.pth', type=str, help="Swin 预训练权重路径")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(work_dir):
    log_file = os.path.join(work_dir, 'train_swin_cam.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )


class Optimizer:
    def __init__(self, args, device, writer):
        self.args = args
        self.device = device
        self.writer = writer

        # 1. 分类数据集
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

        train_dataset = SwinDataset(train_img_paths, train_labels, transform=train_transform)
        val_dataset = SwinDataset(val_img_paths, val_labels, transform=val_transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 2. 分割数据集
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
            batch_size=max(1, int(args.batch_size / 2)),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.seg_val_loader = DataLoader(
            seg_val_dataset,
            batch_size=max(1, int(args.batch_size / 2)),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 3. 模型
        logging.info("[*] 初始化 SwinTransformer 模型...")
        depths = tuple(int(x) for x in args.depths.split(','))
        num_heads = tuple(int(x) for x in args.num_heads.split(','))
        self.model = SwinTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            ape=args.ape,
            patch_norm=args.patch_norm,
            use_checkpoint=args.use_checkpoint
        ).to(device)

        if args.ckpt:
            load_checkpoint(self.model, args.ckpt, device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})
        self.best_score = -1.0

        # 4. CAM
        self.cam_method = args.cam_method
        self.cam_engine = self._build_cam_engine()

    def _get_cam_target_layer(self):
        return self.model.layers[-1]

    def _get_cam_hw(self):
        h, w = self.model.layers[-1].input_resolution
        return int(h), int(w)

    def _reshape_transform(self, tensor):
        h, w = self._get_cam_hw()
        if tensor.ndim == 3:
            b, l, c = tensor.shape
            if l == h * w:
                tensor = tensor.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return tensor

    def _build_cam_engine(self):
        target_layers = [self._get_cam_target_layer()]
        if self.cam_method == "layercam":
            return LayerCAM(model=self.model, target_layers=target_layers, reshape_transform=self._reshape_transform)
        return GradCAMPlusPlus(model=self.model, target_layers=target_layers, reshape_transform=self._reshape_transform)

    def _forward_with_features(self, x):
        mdl = self.model
        x = mdl.patch_embed(x)
        if mdl.ape:
            x = x + mdl.absolute_pos_embed
        x = mdl.pos_drop(x)
        for layer in mdl.layers:
            x = layer(x)
        features = x
        x = mdl.norm(x)
        x = mdl.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        logits = mdl.head(x)
        return logits, features

    def _compute_cam_autograd(self, logits, activations, cam_method):
        score = logits[:, 1]
        grads = torch.autograd.grad(
            score.sum(),
            activations,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if grads is None:
            raise RuntimeError("CAM grads is None. Check Swin feature path for CAM autograd.")

        if activations.ndim == 3:
            h, w = self._get_cam_hw()
            b, l, c = activations.shape
            if l != h * w:
                raise RuntimeError(f"CAM activations length {l} != h*w {h*w}.")
            activations = activations.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            grads = grads.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
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
        self.model.train()
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

            logits_c = self.model(imgs_c)
            cls_loss = self.criterion(logits_c, labels)

            logits_s, feats_s = self._forward_with_features(imgs_s)
            cam = self._compute_cam_autograd(logits_s, feats_s, self.cam_method)
            bce = F.binary_cross_entropy(cam, masks)
            dice = dice_loss(cam, masks)
            cam_loss = 0.5 * bce + 0.5 * dice

            loss = cls_loss + cam_w * cam_loss
            loss.backward()
            self.optimizer.step()

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
        self.model.eval()
        y_true, y_scores, y_preds = [], [], []
        cam_loss_total = 0.0
        dice_scores = []
        miou_scores = []

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="[Cls Validating]", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device).long()
                logits = self.model(imgs)

                probs = torch.softmax(logits, dim=1)[:, 1] if self.args.num_classes == 2 else torch.softmax(logits, dim=1).max(1)[0]
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
                y_preds.extend(preds.cpu().numpy())

        for imgs, masks in tqdm(self.seg_val_loader, desc="[Seg Validating]", leave=False):
            imgs = imgs.to(self.device).float()
            masks = masks.to(self.device).float()
            logits, feats = self._forward_with_features(imgs)
            cam = self._compute_cam_autograd(logits, feats, self.cam_method)
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

        self.model.train()
        self.writer.add_scalar('Val/ACC', acc, epoch + 1)
        self.writer.add_scalar('Val/AUC', auc, epoch + 1)
        self.writer.add_scalar('Val/Cam_Loss', cam_loss_avg, epoch + 1)
        self.writer.add_scalar('Val/Dice', dice, epoch + 1)
        self.writer.add_scalar('Val/mIoU', miou, epoch + 1)
        logging.info(f"[Epoch {epoch+1} Validation] ACC: {acc:.4f} | AUC: {auc:.4f} | Dice: {dice:.4f} | mIoU: {miou:.4f} | CamLoss: {cam_loss_avg:.4f}")

        return acc, auc, dice, miou, cam_loss_avg

    def optimize(self):
        logging.info("[*] 开始 Swin 训练 (Cls + CAM-Seg)...")
        for epoch in range(self.args.epochs):
            self.train(epoch)
            acc, auc, dice, miou, _ = self.validate(epoch)

            model_to_save = self.model.state_dict()

            score = 0.35 * acc + 0.3 * auc + 0.25 * dice + 0.1 * miou
            ckpt_dir = os.path.join(self.args.work_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)

            if score > self.best_score:
                self.best_score = score
                best_path = os.path.join(ckpt_dir, "best_swin.pth")
                torch.save(model_to_save, best_path)
                logging.info(f"[*] 发现更优的 score(AUC+Dice): {self.best_score:.4f}，模型已保存。")

            if (epoch + 1) % 5 == 0:
                last_path = os.path.join(ckpt_dir, f"swin_epoch_{epoch+1}.pth")
                torch.save(model_to_save, last_path)


def main():
    args = get_args()
    set_seed(args.seed)

    if args.use_checkpoint:
        logging.warning("[!] use_checkpoint=True may break CAM autograd. Forcing use_checkpoint=False.")
        args.use_checkpoint = False

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, f'{timestamp}_{args.cam_method}')
    os.makedirs(args.work_dir, exist_ok=True)

    setup_logger(args.work_dir)
    tb_dir = os.path.join(args.work_dir, "tensorboard_log")
    writer = SummaryWriter(tb_dir)

    if args.img_size is None:
        args.img_size = args.crop_size
    if args.img_size != args.crop_size:
        logging.warning(f"[!] img_size ({args.img_size}) != crop_size ({args.crop_size}), force to crop_size.")
        args.img_size = args.crop_size
    if args.img_size % args.patch_size != 0:
        raise ValueError(f"img_size ({args.img_size}) must be divisible by patch_size ({args.patch_size}).")
    patches_res = args.img_size // args.patch_size
    if patches_res % args.window_size != 0:
        new_w = None
        for w in range(args.window_size, 0, -1):
            if patches_res % w == 0:
                new_w = w
                break
        if new_w is None:
            new_w = 1
        logging.warning(f"[!] window_size ({args.window_size}) not divisible into patches_res ({patches_res}), use {new_w}.")
        args.window_size = new_w

    logging.info(f"[*] 训练参数: {args}")
    logging.info(f"[*] 工作目录: {args.work_dir}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()

    writer.close()


if __name__ == '__main__':
    main()
