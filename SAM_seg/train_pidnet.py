import argparse
import datetime
import importlib.util
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CsvPolypDataset, load_pairs_from_root, prepare_public_train_val_pairs, prepare_train_val_pairs


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIDNET_ROOT = os.path.join(THIS_DIR, "model", "PIDNet-main")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)

from models.pidnet import PIDNet
try:
    from configs import config as pid_cfg
    from utils.criterion import BondaryLoss, OhemCrossEntropy
    from utils.utils import FullModel

    HAS_PIDNET_FULLMODEL = True
except ModuleNotFoundError as _import_err:
    if _import_err.name != "yacs":
        raise
    pid_cfg = None
    HAS_PIDNET_FULLMODEL = False

    def _weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = logits.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = target_t == 1
        neg_index = target_t == 0
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        if sum_num.item() == 0:
            return F.binary_cross_entropy_with_logits(log_p, target_t, reduction="mean")

        weight = torch.zeros_like(log_p)
        weight[pos_index] = neg_num.float() / sum_num.float()
        weight[neg_index] = pos_num.float() / sum_num.float()
        return F.binary_cross_entropy_with_logits(log_p, target_t, weight=weight, reduction="mean")

    class BondaryLoss(nn.Module):
        def __init__(self, coeff_bce=20.0):
            super().__init__()
            self.coeff_bce = coeff_bce

        def forward(self, bd_pre, bd_gt):
            return self.coeff_bce * _weighted_bce(bd_pre, bd_gt)

    class OhemCrossEntropy(nn.Module):
        def __init__(
            self,
            ignore_label=-1,
            thres=0.9,
            min_kept=100000,
            weight=None,
            balance_weights=None,
            sb_weight=1.0,
        ):
            super().__init__()
            self.thresh = thres
            self.min_kept = max(1, int(min_kept))
            self.ignore_label = ignore_label
            self.balance_weights = list(balance_weights or [0.5, 1.0])
            self.sb_weight = float(sb_weight)
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction="none",
            )

        def _ce_forward(self, score, target):
            return self.criterion(score, target).mean()

        def _ohem_forward(self, score, target):
            pred = F.softmax(score, dim=1)
            pixel_losses = self.criterion(score, target).contiguous().view(-1)
            mask = target.contiguous().view(-1) != self.ignore_label
            if not torch.any(mask):
                return pixel_losses.mean()

            tmp_target = target.clone()
            tmp_target[tmp_target == self.ignore_label] = 0
            pred = pred.gather(1, tmp_target.unsqueeze(1))
            pred, ind = pred.contiguous().view(-1)[mask].contiguous().sort()
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
            threshold = max(min_value, self.thresh)
            pixel_losses = pixel_losses[mask][ind]
            pixel_losses = pixel_losses[pred < threshold]
            if pixel_losses.numel() == 0:
                return self.criterion(score, target).mean()
            return pixel_losses.mean()

        def forward(self, score, target):
            if not isinstance(score, (list, tuple)):
                score = [score]
            if len(self.balance_weights) == len(score):
                funcs = [self._ce_forward] * (len(score) - 1) + [self._ohem_forward]
                return sum(w * fn(x, target) for w, x, fn in zip(self.balance_weights, score, funcs))
            if len(score) == 1:
                return self.sb_weight * self._ohem_forward(score[0], target)
            raise ValueError("lengths of prediction and target are not identical!")

    class FullModel(nn.Module):
        def __init__(self, model, sem_loss, bd_loss, align_corners=False, ignore_label=-1):
            super().__init__()
            self.model = model
            self.sem_loss = sem_loss
            self.bd_loss = bd_loss
            self.align_corners = bool(align_corners)
            self.ignore_label = int(ignore_label)

        @staticmethod
        def _pixel_acc(pred, label):
            _, preds = torch.max(pred, dim=1)
            valid = (label >= 0).long()
            acc_sum = torch.sum(valid * (preds == label).long())
            pixel_sum = torch.sum(valid)
            return acc_sum.float() / (pixel_sum.float() + 1e-10)

        def forward(self, inputs, labels, bd_gt, *args, **kwargs):
            outputs = self.model(inputs, *args, **kwargs)
            h, w = labels.size(1), labels.size(2)
            ph, pw = outputs[0].size(2), outputs[0].size(3)
            if ph != h or pw != w:
                outputs = [
                    F.interpolate(x, size=(h, w), mode="bilinear", align_corners=self.align_corners)
                    for x in outputs
                ]

            acc = self._pixel_acc(outputs[-2], labels)
            loss_s = self.sem_loss(outputs[:-1], labels)
            loss_b = self.bd_loss(outputs[-1], bd_gt)

            filler = torch.ones_like(labels) * self.ignore_label
            bd_label = torch.where(torch.sigmoid(outputs[-1][:, 0, :, :]) > 0.8, labels, filler)
            loss_sb = self.sem_loss(outputs[-2], bd_label)
            loss = loss_s + loss_b + loss_sb

            return torch.unsqueeze(loss, 0), outputs[:-1], acc, [loss_s, loss_b]


LOCAL_UTILS_PATH = os.path.join(THIS_DIR, "utils.py")
_spec = importlib.util.spec_from_file_location("sam_utils", LOCAL_UTILS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load local utils module: {LOCAL_UTILS_PATH}")
sam_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sam_utils)


def get_args():
    parser = argparse.ArgumentParser(description="PIDNet-S training via official FullModel wrapper")
    parser.add_argument("--seed", default=324, type=int)
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true", help="enable DataLoader pin_memory")
    parser.add_argument("--split", default=0.01, type=float, help="validation split ratio")
    parser.add_argument("--downsample", default=1.0, type=float, help="random downsample ratio in (0,1]")
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--augmentation", action="store_true", help="enable random flip/rotation")

    parser.add_argument("--trainset", default="public", choices=["pretrain", "public"], type=str)
    parser.add_argument(
        "--csv_path",
        default="/mnt/nas1/disk03/zhaokaizhang/code/test_code/sam_pseudo_mask_pairs.csv",
        type=str,
        help="CSV path with columns image_path,pseudo_mask_path when --trainset pretrain",
    )
    parser.add_argument(
        "--kvasir_root",
        default="/mnt/nas1/disk03/zhaokaizhang/data/kvasir-seg",
        type=str,
        help="Kvasir-SEG root for --trainset public",
    )
    parser.add_argument(
        "--cvc_root",
        default="/mnt/nas1/disk03/zhaokaizhang/data/CVC-ClinicDB",
        type=str,
        help="CVC-ClinicDB root for --trainset public",
    )
    parser.add_argument(
        "--test_root",
        default="/mnt/nas1/disk03/zhaokaizhang/data/CVC-ColonDB",
        type=str,
        help="CVC-ColonDB root used as test set",
    )

    parser.add_argument("--optimizer", default="SGD", choices=["AdamW", "SGD"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--clip", default=0.5, type=float, help="gradient clipping value")
    parser.add_argument("--decay_rate", default=0.1, type=float)
    parser.add_argument("--decay_epoch", default=30, type=int)

    parser.add_argument("--aux_weight", default=0.4, type=float, help="semantic loss weight of aux branch")
    parser.add_argument("--main_weight", default=1.0, type=float, help="semantic loss weight of main branch")
    parser.add_argument("--boundary_bce_coef", default=20.0, type=float, help="boundary BCE coefficient")
    parser.add_argument("--ohem_thres", default=0.9, type=float)
    parser.add_argument("--ohem_keep", default=100000, type=int)

    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--ci_bootstrap", default=2000, type=int)
    parser.add_argument("--ci_seed", default=42, type=int)

    parser.add_argument("--save_best_by", default="test", choices=["val", "test"], type=str)
    parser.add_argument("--save_epochs", default=5, type=int)
    parser.add_argument("--test_interval", default=1, type=int)

    parser.add_argument(
        "--pidnet_pretrained",
        default="/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/seg_pidnet/2026-0324-2215_pidnet_s/checkpoint/best_pidnet_s.pth",
        type=str,
        help="optional PIDNet pretrained checkpoint",
    )
    parser.add_argument("--resume", default="", type=str, help="optional checkpoint to resume")
    parser.add_argument("--work_dir", default="runs/seg_pidnet", type=str)
    return parser.parse_args()


def configure_pidnet_config(args):
    if not HAS_PIDNET_FULLMODEL:
        return
    if hasattr(pid_cfg, "defrost"):
        pid_cfg.defrost()

    pid_cfg.MODEL.ALIGN_CORNERS = False
    pid_cfg.MODEL.NUM_OUTPUTS = 2
    pid_cfg.TRAIN.IGNORE_LABEL = -1
    pid_cfg.LOSS.BALANCE_WEIGHTS = [float(args.aux_weight), float(args.main_weight)]
    pid_cfg.LOSS.SB_WEIGHTS = float(args.main_weight)

    if hasattr(pid_cfg, "freeze"):
        pid_cfg.freeze()


def dice_iou_from_fg_probs(fg_probs: torch.Tensor, gt_masks: torch.Tensor, threshold=0.5, eps=1e-6):
    preds = (fg_probs > threshold).float()
    masks = (gt_masks > 0.5).float()

    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.mean().item()), float(iou.mean().item())


class Trainer:
    def __init__(self, args, device, writer):
        self.args = args
        self.device = device
        self.writer = writer
        self.pin_memory = bool(args.pin_memory and torch.cuda.is_available())

        train_pairs, val_pairs, stats = self._build_train_val_pairs()
        test_pairs = self._build_test_pairs()

        logging.info(
            f"[*] Trainset={args.trainset} | "
            f"Total={stats['original_total']} | AfterDownsample={stats['after_downsample']} | "
            f"Train={stats['train_size']} | Val={stats['val_size']} | "
            f"ValSource={stats.get('val_source', 'split')} | Test={len(test_pairs)}"
        )

        train_dataset = CsvPolypDataset(train_pairs, args.trainsize, args.augmentation)
        val_dataset = CsvPolypDataset(val_pairs, args.trainsize, False)
        test_dataset = CsvPolypDataset(test_pairs, args.trainsize, False)

        if int(args.batch_size) < 2:
            raise ValueError(
                "PIDNet training requires --batch_size >= 2 due to BatchNorm statistics in the backbone."
            )
        if len(train_dataset) < int(args.batch_size):
            raise ValueError(
                f"Train samples ({len(train_dataset)}) < batch_size ({args.batch_size}) with drop_last=True. "
                "Increase --downsample/--split data or reduce --batch_size."
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=self.pin_memory,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=self.pin_memory,
        )

        configure_pidnet_config(args)

        self.model = PIDNet(
            m=2,
            n=3,
            num_classes=2,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        ).to(device)

        if args.pidnet_pretrained.strip():
            info = sam_utils.load_model_weights(
                self.model,
                args.pidnet_pretrained.strip(),
                device,
                strict=False,
                match_shape=True,
            )
            logging.info(
                f"[*] Loaded pretrained: {args.pidnet_pretrained.strip()} | "
                f"matched={info['loaded']} / candidate={info['candidate']} / total={info['total']}"
            )

        if HAS_PIDNET_FULLMODEL:
            sem_criterion = OhemCrossEntropy(
                ignore_label=pid_cfg.TRAIN.IGNORE_LABEL,
                thres=args.ohem_thres,
                min_kept=args.ohem_keep,
                weight=None,
            )
            bd_criterion = BondaryLoss(coeff_bce=args.boundary_bce_coef)
            self.full_model = FullModel(self.model, sem_criterion, bd_criterion).to(device)
        else:
            sem_criterion = OhemCrossEntropy(
                ignore_label=-1,
                thres=args.ohem_thres,
                min_kept=args.ohem_keep,
                weight=None,
                balance_weights=[float(args.aux_weight), float(args.main_weight)],
                sb_weight=float(args.main_weight),
            )
            bd_criterion = BondaryLoss(coeff_bce=args.boundary_bce_coef)
            self.full_model = FullModel(
                self.model,
                sem_criterion,
                bd_criterion,
                align_corners=False,
                ignore_label=-1,
            ).to(device)

        if args.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

        self.start_epoch = 0
        self.best_val_dice = -1.0
        self.best_test_dice = 0.31
        if args.resume:
            self._resume(args.resume)

        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})

    def _build_train_val_pairs(self):
        if self.args.trainset == "pretrain":
            train_pairs, val_pairs, stats = prepare_train_val_pairs(
                csv_path=self.args.csv_path,
                split=self.args.split,
                seed=self.args.seed,
                downsample=self.args.downsample,
            )
            stats["val_source"] = "csv_split"
            return train_pairs, val_pairs, stats

        return prepare_public_train_val_pairs(
            kvasir_root=self.args.kvasir_root,
            cvc_root=self.args.cvc_root,
            split=self.args.split,
            seed=self.args.seed,
            downsample=self.args.downsample,
        )

    def _build_test_pairs(self):
        return load_pairs_from_root(
            self.args.test_root,
            candidates=[("images", "masks"), ("Original", "Ground Truth")],
            name="CVC-ColonDB",
            check_readable=True,
        )

    @staticmethod
    def _prepare_targets(masks: torch.Tensor):
        labels = (masks[:, 0] > 0.5).long()
        bd_gts = sam_utils.make_boundary_target(masks).squeeze(1).float()
        return labels, bd_gts

    @staticmethod
    def _extract_main_logits(pred_outputs):
        if isinstance(pred_outputs, (list, tuple)):
            main_logits = pred_outputs[-1]
        else:
            main_logits = pred_outputs
        if main_logits.ndim != 4 or main_logits.shape[1] < 2:
            raise RuntimeError(f"Unexpected main logits shape: {tuple(main_logits.shape)}")
        return main_logits

    def _resume(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported resume format: {ckpt_path}")

        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise RuntimeError(f"Checkpoint missing model_state_dict: {ckpt_path}")

        model_state = sam_utils.normalize_state_dict_keys(model_state)
        self.model.load_state_dict(model_state, strict=False)

        opt_state = checkpoint.get("optimizer_state_dict")
        if isinstance(opt_state, dict):
            self.optimizer.load_state_dict(opt_state)

        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        self.best_test_dice = float(checkpoint.get("best_test_dice", -1.0))

        logging.info(
            f"[*] Resumed from {ckpt_path}, start_epoch={self.start_epoch}, "
            f"best_val_dice={self.best_val_dice:.4f}, best_test_dice={self.best_test_dice:.4f}"
        )

    def _adjust_lr(self, epoch: int):
        lr_now = self.args.lr * (self.args.decay_rate ** (epoch // self.args.decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
        return lr_now

    def _save_ckpt(self, epoch: int, name: str):
        ckpt_dir = os.path.join(self.args.work_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_dice": self.best_val_dice,
                "best_test_dice": self.best_test_dice,
                "args": vars(self.args),
            },
            ckpt_path,
        )
        return ckpt_path

    def train_one_epoch(self, epoch: int):
        self.full_model.train()

        total_meter = 0.0
        sem_meter = 0.0
        bd_meter = 0.0
        sb_meter = 0.0
        acc_meter = 0.0

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}/{self.args.epochs}")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True).float()
            masks = masks.to(self.device, non_blocking=True).float()
            labels, bd_gts = self._prepare_targets(masks)

            self.optimizer.zero_grad()
            losses, _, acc, loss_list = self.full_model(images, labels, bd_gts)
            loss = losses.mean()
            loss.backward()
            sam_utils.clip_gradient(self.optimizer, self.args.clip)
            self.optimizer.step()

            sem_loss = float(loss_list[0].mean().item())
            bd_loss = float(loss_list[1].mean().item())
            total_loss = float(loss.item())
            sb_loss = total_loss - sem_loss - bd_loss

            total_meter += total_loss
            sem_meter += sem_loss
            bd_meter += bd_loss
            sb_meter += sb_loss
            acc_meter += float(acc.mean().item())

            pbar.set_postfix({"loss": f"{total_loss:.4f}", "acc": f"{float(acc.mean().item()):.4f}"})

        denom = max(1, len(self.train_loader))
        avg_total = total_meter / denom
        avg_sem = sem_meter / denom
        avg_bd = bd_meter / denom
        avg_sb = sb_meter / denom
        avg_acc = acc_meter / denom

        self.writer.add_scalar("Train/Loss", avg_total, epoch + 1)
        self.writer.add_scalar("Train/Loss_sem", avg_sem, epoch + 1)
        self.writer.add_scalar("Train/Loss_bd", avg_bd, epoch + 1)
        self.writer.add_scalar("Train/Loss_sb", avg_sb, epoch + 1)
        self.writer.add_scalar("Train/PixelAcc", avg_acc, epoch + 1)
        return avg_total, avg_sem, avg_bd, avg_sb, avg_acc

    def validate_one_epoch(self, epoch: int):
        self.full_model.eval()
        loss_meter = 0.0
        dice_all = []
        iou_all = []

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"[Val] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).float()
                labels, bd_gts = self._prepare_targets(masks)

                losses, preds, _, _ = self.full_model(images, labels, bd_gts)
                loss_meter += float(losses.mean().item())

                main_logits = self._extract_main_logits(preds)
                fg_probs = torch.softmax(main_logits, dim=1)[:, 1:2]
                dice, iou = dice_iou_from_fg_probs(fg_probs, masks, threshold=self.args.threshold)
                dice_all.append(dice)
                iou_all.append(iou)

        val_loss = loss_meter / max(1, len(self.val_loader))
        val_dice = float(np.mean(dice_all)) if dice_all else 0.0
        val_iou = float(np.mean(iou_all)) if iou_all else 0.0

        self.writer.add_scalar("Val/Loss", val_loss, epoch + 1)
        self.writer.add_scalar("Val/Dice", val_dice, epoch + 1)
        self.writer.add_scalar("Val/IoU", val_iou, epoch + 1)
        return val_loss, val_dice, val_iou

    def test_one_epoch(self, epoch: int):
        self.full_model.eval()
        metric_names = ["dice", "precision", "recall", "hd", "hd95"]
        values = {k: [] for k in metric_names}

        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc=f"[Test] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).float()
                labels, bd_gts = self._prepare_targets(masks)

                _, preds, _, _ = self.full_model(images, labels, bd_gts)
                main_logits = self._extract_main_logits(preds)
                fg_probs = torch.softmax(main_logits, dim=1)[:, 1:2]

                pred_bin = (fg_probs > self.args.threshold).float()
                gt_bin = (masks > 0.5).float()

                for b in range(pred_bin.shape[0]):
                    pred_mask = pred_bin[b, 0].detach().cpu().numpy().astype(np.uint8)
                    gt_mask = gt_bin[b, 0].detach().cpu().numpy().astype(np.uint8)
                    m = sam_utils.evaluate_metrics_with_type(pred_mask, gt_mask, type="test")
                    for k in metric_names:
                        values[k].append(float(m[k]))

        dice_summary = sam_utils.dice_ci95(
            values["dice"],
            type="test",
            confidence=0.95,
            n_bootstrap=self.args.ci_bootstrap,
            seed=self.args.ci_seed,
        )

        finite_hd = [x for x in values["hd"] if np.isfinite(x)]
        finite_hd95 = [x for x in values["hd95"] if np.isfinite(x)]

        report = {
            "dice": float(np.mean(values["dice"])) if values["dice"] else 0.0,
            "precision": float(np.mean(values["precision"])) if values["precision"] else 0.0,
            "recall": float(np.mean(values["recall"])) if values["recall"] else 0.0,
            "hd": float(np.mean(finite_hd)) if finite_hd else float("inf"),
            "hd95": float(np.mean(finite_hd95)) if finite_hd95 else float("inf"),
            "hd_inf_cases": int(np.sum(~np.isfinite(np.asarray(values["hd"], dtype=np.float64)))) if values["hd"] else 0,
            "hd95_inf_cases": int(np.sum(~np.isfinite(np.asarray(values["hd95"], dtype=np.float64)))) if values["hd95"] else 0,
            "dice_ci95": dice_summary["dice_ci95"],
            "samples": len(values["dice"]),
        }

        self.writer.add_scalar("Test/Dice", report["dice"], epoch + 1)
        self.writer.add_scalar("Test/Precision", report["precision"], epoch + 1)
        self.writer.add_scalar("Test/Recall", report["recall"], epoch + 1)
        if np.isfinite(report["hd"]):
            self.writer.add_scalar("Test/HD", report["hd"], epoch + 1)
        if np.isfinite(report["hd95"]):
            self.writer.add_scalar("Test/HD95", report["hd95"], epoch + 1)

        return report

    def optimize(self):
        logging.info("[*] Start training PIDNet-S with FullModel ...")
        for epoch in range(self.start_epoch, self.args.epochs):
            lr_now = self._adjust_lr(epoch)
            train_total, train_sem, train_bd, train_sb, train_acc = self.train_one_epoch(epoch)
            val_loss, val_dice, val_iou = self.validate_one_epoch(epoch)

            test_report = None
            improved_test = False
            if self.args.test_interval > 0 and ((epoch + 1) % self.args.test_interval == 0 or epoch + 1 == self.args.epochs):
                test_report = self.test_one_epoch(epoch)
                if float(test_report["dice"]) > self.best_test_dice:
                    self.best_test_dice = float(test_report["dice"])
                    improved_test = True

            improved_val = False
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                improved_val = True

            msg = (
                f"[Epoch {epoch + 1}/{self.args.epochs}] lr={lr_now:.6e} "
                f"train_loss={train_total:.4f} (sem={train_sem:.4f}, bd={train_bd:.4f}, sb={train_sb:.4f}, acc={train_acc:.4f}) "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            )
            if test_report is not None:
                msg += (
                    f" | test_dice={test_report['dice']:.4f} "
                    f"precision={test_report['precision']:.4f} recall={test_report['recall']:.4f} "
                    f"hd95={test_report['hd95']:.4f}(inf={test_report['hd95_inf_cases']})"
                )
            logging.info(msg)

            should_save_best = improved_val if self.args.save_best_by == "val" else improved_test
            if should_save_best:
                best_value = self.best_val_dice if self.args.save_best_by == "val" else self.best_test_dice
                best_path = self._save_ckpt(epoch + 1, "best_pidnet_s.pth")
                logging.info(f"[*] New best ({self.args.save_best_by})={best_value:.4f}, saved: {best_path}")

            if (epoch + 1) % self.args.save_epochs == 0:
                save_path = self._save_ckpt(epoch + 1, f"pidnet_s_epoch_{epoch + 1}.pth")
                logging.info(f"[*] Periodic checkpoint saved: {save_path}")

        final_path = self._save_ckpt(self.args.epochs, "last_pidnet_s.pth")
        logging.info(f"[*] Training completed. Final checkpoint: {final_path}")


def main():
    args = get_args()
    sam_utils.set_seed(args.seed)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, f"{timestamp}_pidnet_s")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "tensorboard_log"), exist_ok=True)

    sam_utils.setup_logger(args.work_dir, "train_pidnet.log")
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard_log"))

    logging.info(f"[*] args: {args}")
    logging.info(f"[*] work_dir: {args.work_dir}")
    logging.info(f"[*] device: {device}")
    if HAS_PIDNET_FULLMODEL:
        logging.info("[*] FullModel source: PIDNet official utils/utils.py")
    else:
        logging.warning("[!] yacs not found, using in-script FullModel-compatible fallback.")

    trainer = Trainer(args, device, writer)
    trainer.optimize()
    writer.close()


if __name__ == "__main__":
    main()
