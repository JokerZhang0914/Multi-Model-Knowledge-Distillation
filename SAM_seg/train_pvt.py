import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")
if PVT_ROOT not in sys.path:
    sys.path.insert(0, PVT_ROOT)

from lib.pvt import PolypPVT
from dataset import CsvPolypDataset, prepare_train_val_pairs


def get_args():
    parser = argparse.ArgumentParser(description="Polyp-PVT segmentation training with pseudo masks from CSV")
    parser.add_argument("--seed", default=318, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--split", default=0.2, type=float, help="validation split ratio")
    parser.add_argument("--downsample", default=0.9, type=float, help="random downsample ratio in (0,1]")
    parser.add_argument("--trainsize", default=352, type=int)
    
    parser.add_argument("--augmentation", default=False, help="enable random flip/rotation like official")
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--clip", default=0.5, type=float, help="gradient clipping value")
    parser.add_argument("--decay_rate", default=0.1, type=float)
    parser.add_argument("--decay_epoch", default=30, type=int)
    parser.add_argument("--save_epochs", default=8, type=int)
    parser.add_argument(
        "--csv_path",
        default="/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/data/sam_pseudo_mask_pairs.csv",
        type=str,
    )
    parser.add_argument("--pvt_pretrained", 
                        default="/home/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/SAM_seg/model/Polyp-PVT/PolypPVT_pre.pth",
                         type=str, help="optional path to pth")
    parser.add_argument("--resume", default="", type=str, help="optional checkpoint to resume")
    parser.add_argument("--work_dir", default="runs/seg_pvt", type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(work_dir):
    log_file = os.path.join(work_dir, "train_pvt.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def dice_iou_from_logits(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = (masks > 0.5).float()

    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.mean().item()), float(iou.mean().item())


class Optimizer:
    def __init__(self, args, device, writer):
        self.args = args
        self.device = device
        self.writer = writer

        train_pairs, val_pairs, stats = prepare_train_val_pairs(
            csv_path=args.csv_path,
            split=args.split,
            seed=args.seed,
            downsample=args.downsample,
        )

        logging.info(
            f"[*] Total pairs: {stats['original_total']} | "
            f"AfterDownsample: {stats['after_downsample']} | "
            f"Train: {stats['train_size']} | Val: {stats['val_size']}"
        )

        train_dataset = CsvPolypDataset(train_pairs, args.trainsize, args.augmentation)
        val_dataset = CsvPolypDataset(val_pairs, args.trainsize, False)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        pretrained_path = args.pvt_pretrained.strip() if args.pvt_pretrained else None
        if pretrained_path and not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"--pvt_pretrained not found: {pretrained_path}")
        self.model = PolypPVT(pretrained_path=pretrained_path).to(device)

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
        self.best_dice = -1.0
        if args.resume:
            self._resume(args.resume)

        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})

    def _resume(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_dice = float(checkpoint.get("best_dice", -1.0))
        logging.info(f"[*] Resumed from {ckpt_path}, start_epoch={self.start_epoch}, best_dice={self.best_dice:.4f}")

    def _adjust_lr(self, epoch):
        lr_now = self.args.lr * (self.args.decay_rate ** (epoch // self.args.decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
        return lr_now

    def _save_ckpt(self, epoch, name):
        ckpt_dir = os.path.join(self.args.work_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_dice": self.best_dice,
                "args": vars(self.args),
            },
            ckpt_path,
        )
        return ckpt_path

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = 0.0
        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}/{self.args.epochs}")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True).float()
            masks = masks.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad()
            p1, p2 = self.model(images)
            loss = structure_loss(p1, masks) + structure_loss(p2, masks)
            loss.backward()
            clip_gradient(self.optimizer, self.args.clip)
            self.optimizer.step()

            loss_meter += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = loss_meter / max(1, len(self.train_loader))
        self.writer.add_scalar("Train/Loss", avg_loss, epoch + 1)
        return avg_loss

    def validate_one_epoch(self, epoch):
        self.model.eval()
        loss_meter = 0.0
        dice_all = []
        iou_all = []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="[Val] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).float()
                p1, p2 = self.model(images)
                logits = p1 + p2
                loss = structure_loss(p1, masks) + structure_loss(p2, masks)
                loss_meter += loss.item()
                dice, iou = dice_iou_from_logits(logits, masks)
                dice_all.append(dice)
                iou_all.append(iou)

        val_loss = loss_meter / max(1, len(self.val_loader))
        val_dice = float(np.mean(dice_all)) if dice_all else 0.0
        val_iou = float(np.mean(iou_all)) if iou_all else 0.0

        self.writer.add_scalar("Val/Loss", val_loss, epoch + 1)
        self.writer.add_scalar("Val/Dice", val_dice, epoch + 1)
        self.writer.add_scalar("Val/IoU", val_iou, epoch + 1)
        return val_loss, val_dice, val_iou

    def optimize(self):
        logging.info("[*] Start training Polyp-PVT ...")
        for epoch in range(self.start_epoch, self.args.epochs):
            lr_now = self._adjust_lr(epoch)
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_dice, val_iou = self.validate_one_epoch(epoch)

            logging.info(
                f"[Epoch {epoch + 1}/{self.args.epochs}] "
                f"lr={lr_now:.6e} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                best_path = self._save_ckpt(epoch + 1, "best_pvt.pth")
                logging.info(f"[*] New best dice={self.best_dice:.4f}, saved: {best_path}")

            if (epoch + 1) % self.args.save_epochs == 0:
                save_path = self._save_ckpt(epoch + 1, f"pvt_epoch_{epoch + 1}.pth")
                logging.info(f"[*] Periodic checkpoint saved: {save_path}")

        final_path = self._save_ckpt(self.args.epochs, "last_pvt.pth")
        logging.info(f"[*] Training completed. Final checkpoint: {final_path}")


def main():
    args = get_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, f"{timestamp}_polyp_pvt")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "tensorboard_log"), exist_ok=True)

    setup_logger(args.work_dir)
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard_log"))

    logging.info(f"[*] args: {args}")
    logging.info(f"[*] work_dir: {args.work_dir}")
    logging.info(f"[*] device: {device}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()
    writer.close()


if __name__ == "__main__":
    main()
