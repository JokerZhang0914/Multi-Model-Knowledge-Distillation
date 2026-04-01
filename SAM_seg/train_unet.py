import argparse
import datetime
import importlib.util
import logging
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
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
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
UNET_ROOT = os.path.join(THIS_DIR, "model", "UNet")
if UNET_ROOT not in sys.path:
    sys.path.insert(0, UNET_ROOT)

from unet import UNet


def _import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_unet_dice_mod = _import_module_from_path(
    "unet_dice_score_module",
    os.path.join(UNET_ROOT, "utils", "dice_score.py"),
)
dice_loss = _unet_dice_mod.dice_loss


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
    parser = argparse.ArgumentParser(description="UNet segmentation training")
    parser.add_argument("--seed", default=326, type=int)
    parser.add_argument("--gpu", default="0", type=str)

    parser.add_argument("--train_source", default="public", choices=["csv", "public"], type=str)
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
        default="CVC-ColonDB",
        type=str,
        choices=["CVC-ColonDB", "CVC-300", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir", "all"],
        help="subfolder under --testdataset_root, e.g. CVC-ColonDB/Kvasir/CVC-300; use 'all' for all subsets",
    )
    parser.add_argument("--downsample", default=1.0, type=float, help="random downsample ratio in (0,1]")
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--augmentation", default=False, type=str2bool)

    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--optimizer", default="RMSprop", choices=["RMSprop", "AdamW", "SGD"])
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-8, type=float)
    parser.add_argument("--momentum", default=0.7, type=float)
    parser.add_argument("--clip", default=1.5, type=float, help="gradient clipping value")
    parser.add_argument("--save_epochs", default=25, type=int)

    parser.add_argument("--amp", default=False, type=str2bool, help="mixed precision training")
    parser.add_argument("--bilinear", default=False, type=str2bool, help="UNet upsample mode")
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--n_classes", default=1, type=int)

    parser.add_argument("--unet_pretrained", 
        default="/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/seg_unet/2026-0326-1942_csv_all_unet/checkpoint/best_unet.pth",
        type=str, 
        help="optional checkpoint to initialize weights"
    )
    parser.add_argument("--resume", default="", type=str, help="optional training checkpoint to resume")
    parser.add_argument("--work_dir", default="runs/seg_unet", type=str)
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
        self.use_amp = bool(args.amp) and device.type == "cuda"

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

        self.model = UNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            bilinear=args.bilinear,
        ).to(device=device, memory_format=torch.channels_last)

        pretrained_path = args.unet_pretrained.strip()
        if pretrained_path:
            loaded = self._load_model_weights(pretrained_path)
            logging.info(f"[*] Loaded UNet pretrained params: {loaded}/{len(self.model.state_dict())}")

        if args.optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        elif args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", patience=5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.criterion = nn.CrossEntropyLoss() if args.n_classes > 1 else nn.BCEWithLogitsLoss()

        self.start_epoch = 0
        self.best_dice = 0.623
        if args.resume:
            self._resume(args.resume)

        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise RuntimeError("Unsupported checkpoint format.")

    def _load_model_weights(self, ckpt_path: str) -> int:
        ckpt_path = resolve_path(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"weights file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        return self._load_model_state_dict(state_dict)

    def _load_model_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        if "mask_values" in state_dict and not torch.is_tensor(state_dict["mask_values"]):
            state_dict = {k: v for k, v in state_dict.items() if k != "mask_values"}

        model_dict = self.model.state_dict()
        matched = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            nk = key[7:] if key.startswith("module.") else key
            if nk in model_dict and model_dict[nk].shape == value.shape:
                matched[nk] = value
        if not matched:
            raise RuntimeError(f"No matched UNet parameters found in weights: {ckpt_path}")
        model_dict.update(matched)
        self.model.load_state_dict(model_dict, strict=False)
        return len(matched)

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
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        loaded = self._load_model_state_dict(state_dict)
        logging.info(f"[*] Resume weights loaded: {loaded}/{len(self.model.state_dict())}")

        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_dice = float(checkpoint.get("best_dice", -1.0))
        logging.info(f"[*] Resumed from {ckpt_path}, start_epoch={self.start_epoch}, best_dice={self.best_dice:.4f}")

    def _save_ckpt(self, epoch: int, name: str) -> str:
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

    def _compute_loss_and_fg_logit(self, logits: torch.Tensor, masks: torch.Tensor):
        if self.args.n_classes == 1:
            logits_1 = logits.squeeze(1)
            true_masks = masks.squeeze(1).float()
            loss = self.criterion(logits_1, true_masks)
            loss = loss + dice_loss(torch.sigmoid(logits_1), true_masks, multiclass=False)
            fg_logit = logits
            return loss, fg_logit

        labels = (masks[:, 0] > 0.5).long()
        ce_loss = self.criterion(logits, labels)
        dice_target = F.one_hot(labels, self.args.n_classes).permute(0, 3, 1, 2).float()
        d_loss = dice_loss(F.softmax(logits, dim=1).float(), dice_target, multiclass=True)
        loss = ce_loss + d_loss
        if self.args.n_classes >= 2:
            fg_logit = logits[:, 1:2] - logits[:, 0:1]
        else:
            fg_logit = logits[:, :1]
        return loss, fg_logit

    def train_one_epoch(self, epoch: int):
        self.model.train()
        loss_meter = 0.0
        dice_all = []
        iou_all = []

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}/{self.args.epochs}")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True).float(memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type if self.device.type != "mps" else "cpu", enabled=self.use_amp):
                logits = self.model(images)
                loss, fg_logit = self._compute_loss_and_fg_logit(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_gradient(self.optimizer, self.args.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter += float(loss.item())
            dice, iou = dice_iou_from_logits(fg_logit, masks)
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

    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        loss_meter = 0.0
        metrics = {"dice": [], "precision": [], "recall": [], "hd": [], "hd95": []}

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"[Val] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float(memory_format=torch.channels_last)
                masks = masks.to(self.device, non_blocking=True).float()

                with torch.autocast(self.device.type if self.device.type != "mps" else "cpu", enabled=self.use_amp):
                    logits = self.model(images)
                    loss, fg_logit = self._compute_loss_and_fg_logit(logits, masks)
                loss_meter += float(loss.item())

                pred_mask = (torch.sigmoid(fg_logit) > 0.5).float()
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
        logging.info("[*] Start training UNet ...")
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss, train_dice, train_iou = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate_one_epoch(epoch)
            val_dice = val_metrics["dice"]

            self.scheduler.step(val_dice)
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/LR", lr_now, epoch + 1)

            logging.info(
                f"[Epoch {epoch + 1}/{self.args.epochs}] "
                f"lr={lr_now:.6e} "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
                f"val_hd95={val_metrics['hd95']:.4f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                best_path = self._save_ckpt(epoch + 1, "best_unet.pth")
                logging.info(f"[*] New best dice={self.best_dice:.4f}, saved: {best_path}")

            if (epoch + 1) % self.args.save_epochs == 0:
                save_path = self._save_ckpt(epoch + 1, f"unet_epoch_{epoch + 1}.pth")
                logging.info(f"[*] Periodic checkpoint saved: {save_path}")

        final_path = self._save_ckpt(self.args.epochs, "last_unet.pth")
        logging.info(f"[*] Training completed. Final checkpoint: {final_path}")


def main():
    args = get_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        gpu_index = int(str(args.gpu).split(",")[0].strip())
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    timestamp = "{0:%Y-%m%d-%H%M}".format(datetime.datetime.now())
    base_work_dir = resolve_path(args.work_dir)
    val_tag_raw = os.path.basename(os.path.normpath(str(args.val_dataset)))
    val_tag = make_safe_tag(val_tag_raw)
    args.work_dir = os.path.join(base_work_dir, f"{timestamp}_{args.train_source}_{val_tag}_unet")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "tensorboard_log"), exist_ok=True)

    setup_logger(args.work_dir, "train_unet.log")
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard_log"))

    logging.info(f"[*] args: {args}")
    logging.info(f"[*] work_dir: {args.work_dir}")
    logging.info(f"[*] device: {device}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()
    writer.close()


if __name__ == "__main__":
    main()
