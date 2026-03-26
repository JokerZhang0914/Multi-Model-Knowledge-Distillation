import argparse
import datetime
import logging
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CsvPolypDataset, load_pairs_from_root, read_csv_pairs
from utils import (
    clip_gradient,
    dice_iou_from_logits,
    evaluate_metrics_with_type,
    set_seed,
    setup_logger,
    structure_loss,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PVT_ROOT = os.path.join(THIS_DIR, "model", "Polyp-PVT")
if PVT_ROOT not in sys.path:
    sys.path.insert(0, PVT_ROOT)

from lib.pvt import PolypPVT


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def make_safe_tag(text: str) -> str:
    tag = str(text).strip()
    if not tag:
        return "unknown"
    out = []
    for ch in tag:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    cleaned = "".join(out).strip("-")
    return cleaned or "unknown"


def get_args():
    parser = argparse.ArgumentParser(description="Polyp-PVT segmentation training")
    parser.add_argument("--seed", default=326, type=int)
    parser.add_argument("--gpu", default="1", type=str)

    parser.add_argument("--train_source", default="csv", choices=["csv", "public"], type=str)
    parser.add_argument(
        "--csv_path",
        default="data/sam_pseudo_mask_pairs.csv",
        type=str,
        help="used when --train_source csv",
    )
    parser.add_argument(
        "--public_root",
        default="data/seg_data/TrainDataset",
        type=str,
        help="used when --train_source public",
    )
    parser.add_argument(
        "--testdataset_root",
        default="data/seg_data/TestDataset",
        type=str,
        help="root for seg_data/TestDataset",
    )
    parser.add_argument(
        "--val_dataset",
        default="all",
        type=str,
        choices=["CVC-ColonDB", "CVC-300", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir", "all"],
        help="subfolder under --testdataset_root, e.g. CVC-ColonDB/Kvasir/CVC-300; use 'all' for all subsets",
    )
    parser.add_argument("--downsample", default=1, type=float, help="random downsample ratio in (0,1]")
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--augmentation", default=False, type=str2bool)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--clip", default=1.0, type=float, help="gradient clipping value")
    parser.add_argument("--decay_rate", default=0.1, type=float)
    parser.add_argument("--decay_epoch", default=20, type=int)
    parser.add_argument("--save_epochs", default=15, type=int)
    parser.add_argument(
        "--pvt_pretrained",
        default="runs/seg_pvt/2026-0319-1248_polyp_pvt/checkpoint/best_pvt.pth",
        type=str,
        help="optional path to pth",
    )
    parser.add_argument("--resume", default="", type=str, help="optional checkpoint to resume")
    parser.add_argument("--work_dir", default="runs/seg_pvt", type=str)
    return parser.parse_args()


def _shuffle_pairs(pairs: Sequence[Tuple[str, str]], seed: int) -> List[Tuple[str, str]]:
    if len(pairs) <= 1:
        return list(pairs)
    perm = np.random.RandomState(seed).permutation(len(pairs))
    return [pairs[i] for i in perm]


def _downsample_pairs(pairs: Sequence[Tuple[str, str]], downsample: float, seed: int) -> List[Tuple[str, str]]:
    if downsample <= 0 or downsample > 1:
        raise ValueError("--downsample must be in (0, 1].")
    pairs = list(pairs)
    if downsample >= 1.0:
        return pairs
    keep = int(round(len(pairs) * downsample))
    keep = max(1, keep)
    keep = min(len(pairs), keep)
    idx = np.random.RandomState(seed).choice(len(pairs), size=keep, replace=False)
    return [pairs[i] for i in idx]


def _resolve_dataset_root(user_root: str, fallback_roots: Sequence[str], dataset_name: str) -> str:
    candidates = []
    if user_root:
        candidates.append(resolve_path(user_root))
    for path in fallback_roots:
        p = resolve_path(path)
        if p not in candidates:
            candidates.append(p)
    for path in candidates:
        if os.path.isdir(path):
            return path
    cand_text = "\n".join([f"  - {p}" for p in candidates]) if candidates else "  (none)"
    raise FileNotFoundError(f"{dataset_name} root not found. Checked:\n{cand_text}")


class Optimizer:
    def __init__(self, args, device, writer):
        self.args = args
        self.device = device
        self.writer = writer

        train_pairs, val_pairs, stats = self._build_train_val_pairs()
        logging.info(
            f"[*] TrainSource: {stats['train_source']} | "
            f"OriginalTrain: {stats['original_total']} | AfterDownsample: {stats['after_downsample']} | "
            f"Train: {stats['train_size']} | Val({stats['val_name']}): {stats['val_size']}"
        )

        train_dataset = CsvPolypDataset(train_pairs, args.trainsize, args.augmentation)
        val_dataset = CsvPolypDataset(val_pairs, args.trainsize, False)

        drop_last = len(train_dataset) >= args.batch_size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pretrained_path = args.pvt_pretrained.strip() if args.pvt_pretrained else ""
        if pretrained_path:
            pretrained_path = resolve_path(pretrained_path)
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"--pvt_pretrained not found: {pretrained_path}")
            self.model = PolypPVT(pretrained_path=pretrained_path).to(device)
        else:
            self.model = PolypPVT(pretrained_path=None).to(device)

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

    def _build_train_pairs_from_csv(self) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        csv_path = resolve_path(self.args.csv_path)
        train_pairs = read_csv_pairs(csv_path)
        original_total = len(train_pairs)
        train_pairs = _downsample_pairs(train_pairs, self.args.downsample, self.args.seed)
        train_pairs = _shuffle_pairs(train_pairs, self.args.seed + 1)
        stats = {
            "train_source": "csv",
            "original_total": original_total,
            "after_downsample": len(train_pairs),
            "train_size": len(train_pairs),
        }
        return train_pairs, stats

    def _build_train_pairs_from_public(self) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        public_root = _resolve_dataset_root(
            self.args.public_root,
            fallback_roots=["data/seg_data/TrainDataset"],
            dataset_name="TrainDataset",
        )
        train_pairs = load_pairs_from_root(
            public_root,
            candidates=[
                ("image", "mask"),
                ("images", "masks"),
                ("Image", "Mask"),
                ("img", "gt"),
            ],
            name="TrainDataset",
            verbose=True,
            check_readable=True,
        )
        original_total = len(train_pairs)
        train_pairs = _downsample_pairs(train_pairs, self.args.downsample, self.args.seed)
        train_pairs = _shuffle_pairs(train_pairs, self.args.seed + 1)
        stats = {
            "train_source": "public",
            "original_total": original_total,
            "after_downsample": len(train_pairs),
            "train_size": len(train_pairs),
        }
        return train_pairs, stats

    def _load_single_val_subset(self, root: str, subset: str) -> List[Tuple[str, str]]:
        subset_root = os.path.join(root, subset)
        if not os.path.isdir(subset_root):
            raise FileNotFoundError(f"Validation subset not found: {subset_root}")
        return load_pairs_from_root(
            subset_root,
            candidates=[("images", "masks"), ("image", "mask"), ("Original", "Ground Truth")],
            name=f"TestDataset/{subset}",
            verbose=True,
            check_readable=True,
        )

    def _build_testdataset_val_pairs(self) -> Tuple[List[Tuple[str, str]], str]:
        test_root = _resolve_dataset_root(
            self.args.testdataset_root,
            fallback_roots=["data/seg_data/TestDataset"],
            dataset_name="TestDataset",
        )

        val_name = str(self.args.val_dataset).strip()
        if not val_name:
            raise ValueError("--val_dataset cannot be empty.")

        if val_name.lower() == "all":
            subset_names = sorted(
                [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
            )
            if not subset_names:
                raise RuntimeError(f"No subsets found under {test_root}")
            val_pairs = []
            for subset in subset_names:
                val_pairs.extend(self._load_single_val_subset(test_root, subset))
            return val_pairs, "all"

        val_pairs = self._load_single_val_subset(test_root, val_name)
        return val_pairs, val_name

    def _build_train_val_pairs(self):
        if self.args.train_source == "csv":
            train_pairs, stats = self._build_train_pairs_from_csv()
        else:
            train_pairs, stats = self._build_train_pairs_from_public()

        if len(train_pairs) == 0:
            raise RuntimeError("No valid training pairs found.")

        val_pairs, val_name = self._build_testdataset_val_pairs()
        if len(val_pairs) == 0:
            raise RuntimeError(f"No valid validation pairs found from TestDataset/{val_name}.")

        stats["val_size"] = len(val_pairs)
        stats["val_name"] = val_name
        return train_pairs, val_pairs, stats

    def _resume(self, ckpt_path):
        ckpt_path = resolve_path(ckpt_path)
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
        dice_all = []
        iou_all = []

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}/{self.args.epochs}")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True).float()
            masks = masks.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad()
            p1, p2 = self.model(images)
            logits = p1 + p2
            loss = structure_loss(p1, masks) + structure_loss(p2, masks)
            loss.backward()
            clip_gradient(self.optimizer, self.args.clip)
            self.optimizer.step()

            loss_meter += float(loss.item())
            dice, iou = dice_iou_from_logits(logits, masks)
            dice_all.append(dice)
            iou_all.append(iou)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = loss_meter / max(1, len(self.train_loader))
        train_dice = float(np.mean(dice_all)) if dice_all else 0.0
        train_iou = float(np.mean(iou_all)) if iou_all else 0.0
        self.writer.add_scalar("Train/Loss", avg_loss, epoch + 1)
        self.writer.add_scalar("Train/Dice", train_dice, epoch + 1)
        self.writer.add_scalar("Train/IoU", train_iou, epoch + 1)
        return avg_loss, train_dice, train_iou

    def validate_one_epoch(self, epoch):
        self.model.eval()
        loss_meter = 0.0
        metrics = {"dice": [], "precision": [], "recall": [], "hd": [], "hd95": []}

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"[Val] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).float()

                p1, p2 = self.model(images)
                logits = p1 + p2
                loss = structure_loss(p1, masks) + structure_loss(p2, masks)
                loss_meter += float(loss.item())

                pred_mask = (torch.sigmoid(logits) > 0.5).float()
                gt_mask = (masks > 0.5).float()
                for b in range(pred_mask.shape[0]):
                    pred_np = pred_mask[b, 0].detach().cpu().numpy().astype(np.uint8)
                    gt_np = gt_mask[b, 0].detach().cpu().numpy().astype(np.uint8)
                    m = evaluate_metrics_with_type(pred_np, gt_np, type="test")
                    for key in metrics:
                        metrics[key].append(float(m[key]))

        val_loss = loss_meter / max(1, len(self.val_loader))

        metric_means = {}
        for key in ("dice", "precision", "recall"):
            vals = metrics[key]
            metric_means[key] = float(np.mean(vals)) if vals else 0.0

        hd_arr = np.asarray(metrics["hd"], dtype=np.float64) if metrics["hd"] else np.array([], dtype=np.float64)
        hd95_arr = np.asarray(metrics["hd95"], dtype=np.float64) if metrics["hd95"] else np.array([], dtype=np.float64)
        hd_finite = hd_arr[np.isfinite(hd_arr)]
        hd95_finite = hd95_arr[np.isfinite(hd95_arr)]

        metric_means["hd"] = float(hd_finite.mean()) if hd_finite.size > 0 else float("inf")
        metric_means["hd95"] = float(hd95_finite.mean()) if hd95_finite.size > 0 else float("inf")
        metric_means["hd_inf_cases"] = int((~np.isfinite(hd_arr)).sum()) if hd_arr.size > 0 else 0
        metric_means["hd95_inf_cases"] = int((~np.isfinite(hd95_arr)).sum()) if hd95_arr.size > 0 else 0

        self.writer.add_scalar("Val/Loss", val_loss, epoch + 1)
        self.writer.add_scalar("Val/Dice", metric_means["dice"], epoch + 1)
        self.writer.add_scalar("Val/Precision", metric_means["precision"], epoch + 1)
        self.writer.add_scalar("Val/Recall", metric_means["recall"], epoch + 1)
        if np.isfinite(metric_means["hd"]):
            self.writer.add_scalar("Val/HD", metric_means["hd"], epoch + 1)
        if np.isfinite(metric_means["hd95"]):
            self.writer.add_scalar("Val/HD95", metric_means["hd95"], epoch + 1)
        self.writer.add_scalar("Val/HD_inf_cases", metric_means["hd_inf_cases"], epoch + 1)
        self.writer.add_scalar("Val/HD95_inf_cases", metric_means["hd95_inf_cases"], epoch + 1)

        return val_loss, metric_means

    def optimize(self):
        logging.info("[*] Start training Polyp-PVT ...")
        for epoch in range(self.start_epoch, self.args.epochs):
            lr_now = self._adjust_lr(epoch)
            train_loss, train_dice, train_iou = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate_one_epoch(epoch)
            val_dice = val_metrics["dice"]

            logging.info(
                f"[Epoch {epoch + 1}/{self.args.epochs}] "
                f"lr={lr_now:.6e} "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} "
            )
            logging.info(
                f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
                f"val_hd95={val_metrics['hd95']:.4f}"
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
        device = torch.device(f"cuda:{int(str(args.gpu).split(',')[0].strip())}")
    else:
        device = torch.device("cpu")

    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    base_work_dir = resolve_path(args.work_dir)
    val_tag_raw = os.path.basename(os.path.normpath(str(args.val_dataset)))
    val_tag = make_safe_tag(val_tag_raw)
    args.work_dir = os.path.join(base_work_dir, f"{timestamp}_{args.train_source}_{val_tag}")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "tensorboard_log"), exist_ok=True)

    setup_logger(args.work_dir, "train_pvt.log")
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard_log"))

    logging.info(f"[*] args: {args}")
    logging.info(f"[*] work_dir: {args.work_dir}")
    logging.info(f"[*] device: {device}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()
    writer.close()


if __name__ == "__main__":
    main()
