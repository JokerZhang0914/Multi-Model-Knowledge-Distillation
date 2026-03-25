import argparse
import datetime
import importlib.util
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
    load_model_weights,
    make_boundary_target,
    set_seed,
    setup_logger,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PIDNET_ROOT = os.path.join(THIS_DIR, "model", "PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)

from models.pidnet import PIDNet
from configs import config as pid_config


def _import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pid_criterion_mod = _import_module_from_path(
    "pidnet_criterion_module",
    os.path.join(PIDNET_ROOT, "utils", "criterion.py"),
)
_pid_fullmodel_mod = _import_module_from_path(
    "pidnet_fullmodel_module",
    os.path.join(PIDNET_ROOT, "utils", "utils.py"),
)

PIDCrossEntropy = _pid_criterion_mod.CrossEntropy
PIDOhemCrossEntropy = _pid_criterion_mod.OhemCrossEntropy
PIDBoundaryLoss = _pid_criterion_mod.BondaryLoss
PIDFullModel = _pid_fullmodel_mod.FullModel


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


def get_args():
    parser = argparse.ArgumentParser(description="PIDNet-S (lightweight) polyp segmentation training")
    parser.add_argument("--seed", default=318, type=int)
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
        "--colondb_root",
        default="data/CVC-ColonDB",
        type=str,
        help="validation dataset root, supports (images,masks) or (Original,Ground Truth)",
    )
    parser.add_argument("--downsample", default=1.0, type=float, help="random downsample ratio in (0,1]")
    parser.add_argument("--trainsize", default=352, type=int)
    parser.add_argument("--augmentation", default=False, type=str2bool)

    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW"])
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--poly_power", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--clip", default=0.5, type=float, help="gradient clipping value")

    parser.add_argument("--aux_weight", default=0.4, type=float, help="aux segmentation branch weight")
    parser.add_argument("--main_weight", default=1.0, type=float, help="main segmentation branch weight")
    parser.add_argument("--boundary_weight", default=20.0, type=float, help="boundary BCE weight")
    parser.add_argument("--sb_weight", default=1.0, type=float, help="semantic-on-boundary weight in FullModel")
    parser.add_argument("--use_ohem", default=True, type=str2bool)
    parser.add_argument("--ohem_thresh", default=0.9, type=float)
    parser.add_argument("--ohem_keep", default=100000, type=int)
    parser.add_argument("--ignore_label", default=255, type=int)

    parser.add_argument("--pid_pretrained", 
                        default="/mnt/nas1/disk03/zhaokaizhang/code/Multi-Model-Knowledge-Distillation/runs/seg_pidnet/2026-0324-2215_pidnet_s/checkpoint/best_pidnet_s.pth", 
                        type=str, help="optional PIDNet checkpoint for init")
    parser.add_argument("--resume", default="", type=str, help="optional checkpoint to resume")
    parser.add_argument("--save_epochs", default=8, type=int)
    parser.add_argument("--work_dir", default="runs/seg_pid", type=str)
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
            f"Train: {stats['train_size']} | Val(ColonDB): {stats['val_size']}"
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

        self.backbone = PIDNet(
            m=2,
            n=3,
            num_classes=2,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        ).to(self.device)

        pretrained_path = args.pid_pretrained.strip()
        if pretrained_path:
            pretrained_path = resolve_path(pretrained_path)
            info = load_model_weights(
                self.backbone,
                pretrained_path,
                self.device,
                strict=False,
                match_shape=True,
            )
            logging.info(
                f"[*] Loaded PID pretrained: {pretrained_path} "
                f"(loaded={info['loaded']}/{info['candidate']}, total={info['total']})"
            )

        self._configure_pidnet_runtime()
        if args.use_ohem:
            sem_criterion = PIDOhemCrossEntropy(
                ignore_label=args.ignore_label,
                thres=args.ohem_thresh,
                min_kept=args.ohem_keep,
                weight=None,
            )
        else:
            sem_criterion = PIDCrossEntropy(
                ignore_label=args.ignore_label,
                weight=None,
            )
        bd_criterion = PIDBoundaryLoss(coeff_bce=args.boundary_weight)
        self.model = PIDFullModel(self.backbone, sem_criterion, bd_criterion).to(self.device)

        if args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=False,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        self.start_epoch = 0
        self.best_dice = -1.0
        if args.resume:
            self._resume(args.resume)

        self.total_iters = max(1, args.epochs * max(1, len(self.train_loader)))
        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})

    def _configure_pidnet_runtime(self):
        pid_config.defrost()
        pid_config.MODEL.NUM_OUTPUTS = 2
        pid_config.MODEL.ALIGN_CORNERS = False
        pid_config.LOSS.BALANCE_WEIGHTS = [self.args.aux_weight, self.args.main_weight]
        pid_config.LOSS.SB_WEIGHTS = self.args.sb_weight
        pid_config.TRAIN.IGNORE_LABEL = self.args.ignore_label
        pid_config.freeze()

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

    def _build_colondb_val_pairs(self) -> List[Tuple[str, str]]:
        colondb_root = _resolve_dataset_root(
            self.args.colondb_root,
            fallback_roots=[
                "data/CVC-ColonDB",
                "data/CVC_ColonDB",
                "data/CVC-ColonDB-300",
                "data/TestDataset/CVC-ColonDB",
                "data/TestDataset/CVC-ColonDB/CVC-ColonDB",
            ],
            dataset_name="CVC-ColonDB",
        )
        val_pairs = load_pairs_from_root(
            colondb_root,
            candidates=[("images", "masks"), ("Original", "Ground Truth")],
            name="CVC-ColonDB",
            verbose=True,
            check_readable=True,
        )
        return val_pairs

    def _build_train_val_pairs(self):
        if self.args.train_source == "csv":
            train_pairs, stats = self._build_train_pairs_from_csv()
        else:
            train_pairs, stats = self._build_train_pairs_from_public()

        if len(train_pairs) == 0:
            raise RuntimeError("No valid training pairs found.")

        val_pairs = self._build_colondb_val_pairs()
        if len(val_pairs) == 0:
            raise RuntimeError("No valid validation pairs found from CVC-ColonDB.")

        stats["val_size"] = len(val_pairs)
        return train_pairs, val_pairs, stats

    def _resume(self, ckpt_path):
        ckpt_path = resolve_path(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        loaded = self._load_backbone_state_dict(state_dict)
        logging.info(f"[*] Resume weights loaded: {loaded}/{len(self.backbone.state_dict())}")

        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_dice = float(checkpoint.get("best_dice", checkpoint.get("best_mIoU", -1.0)))
        logging.info(f"[*] Resumed from {ckpt_path}, start_epoch={self.start_epoch}, best_dice={self.best_dice:.4f}")

    def _load_backbone_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        model_dict = self.backbone.state_dict()
        matched = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            nk = key[7:] if key.startswith("module.") else key
            nk = nk[6:] if nk.startswith("model.") else nk
            if nk in model_dict and model_dict[nk].shape == value.shape:
                matched[nk] = value
        if not matched:
            raise RuntimeError("No matched PIDNet backbone parameters found in checkpoint.")
        model_dict.update(matched)
        self.backbone.load_state_dict(model_dict, strict=False)
        return len(matched)

    def _adjust_lr(self, global_iter: int) -> float:
        ratio = 1.0 - float(global_iter) / float(self.total_iters)
        ratio = max(0.0, ratio)
        lr_now = self.args.lr * (ratio ** self.args.poly_power)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
        return lr_now

    def _save_ckpt(self, epoch: int, name: str) -> str:
        ckpt_dir = os.path.join(self.args.work_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.backbone.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_dice": self.best_dice,
                "args": vars(self.args),
            },
            ckpt_path,
        )
        return ckpt_path

    @staticmethod
    def _mask_to_labels(masks: torch.Tensor) -> torch.Tensor:
        return (masks[:, 0] > 0.5).long()

    @staticmethod
    def _main_binary_logit(main_logits: torch.Tensor) -> torch.Tensor:
        if main_logits.shape[1] == 1:
            return main_logits
        return main_logits[:, 1:2] - main_logits[:, 0:1]

    def train_one_epoch(self, epoch: int):
        self.model.train()
        meters = {"total": 0.0, "sem": 0.0, "boundary": 0.0, "sb": 0.0, "acc": 0.0}
        dice_all = []
        iou_all = []
        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}/{self.args.epochs}")

        lr_now = self.optimizer.param_groups[0]["lr"]
        for step, (images, masks) in enumerate(pbar):
            global_iter = epoch * len(self.train_loader) + step
            lr_now = self._adjust_lr(global_iter)

            images = images.to(self.device, non_blocking=True).float()
            masks = masks.to(self.device, non_blocking=True).float()
            labels = self._mask_to_labels(masks)
            bd_gts = make_boundary_target(masks).squeeze(1)

            self.optimizer.zero_grad()
            losses, preds, acc, loss_list = self.model(images, labels, bd_gts)
            loss = losses.mean()
            sem_loss = loss_list[0].mean()
            boundary_loss = loss_list[1].mean()
            sb_loss = loss - sem_loss - boundary_loss

            loss.backward()
            clip_gradient(self.optimizer, self.args.clip)
            self.optimizer.step()

            main_logits = preds[-1] if isinstance(preds, (list, tuple)) else preds
            fg_logit = self._main_binary_logit(main_logits)
            dice, iou = dice_iou_from_logits(fg_logit, masks)
            dice_all.append(dice)
            iou_all.append(iou)

            meters["total"] += float(loss.item())
            meters["sem"] += float(sem_loss.item())
            meters["boundary"] += float(boundary_loss.item())
            meters["sb"] += float(sb_loss.item())
            meters["acc"] += float(acc.mean().item())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "sem": f"{sem_loss.item():.4f}",
                    "bd": f"{boundary_loss.item():.4f}",
                    "lr": f"{lr_now:.2e}",
                }
            )

        num_batches = max(1, len(self.train_loader))
        avg = {k: v / num_batches for k, v in meters.items()}
        train_dice = float(np.mean(dice_all)) if dice_all else 0.0
        train_iou = float(np.mean(iou_all)) if iou_all else 0.0
        self.writer.add_scalar("Train/Loss", avg["total"], epoch + 1)
        self.writer.add_scalar("Train/LossSemantic", avg["sem"], epoch + 1)
        self.writer.add_scalar("Train/LossBoundary", avg["boundary"], epoch + 1)
        self.writer.add_scalar("Train/LossSB", avg["sb"], epoch + 1)
        self.writer.add_scalar("Train/PixelAcc", avg["acc"], epoch + 1)
        self.writer.add_scalar("Train/Dice", train_dice, epoch + 1)
        self.writer.add_scalar("Train/IoU", train_iou, epoch + 1)
        self.writer.add_scalar("Train/LR", lr_now, epoch + 1)
        return avg, lr_now, train_dice, train_iou

    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        meters = {"total": 0.0, "sem": 0.0, "boundary": 0.0, "sb": 0.0, "acc": 0.0}
        dice_all = []
        iou_all = []

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"[Val] Epoch {epoch + 1}/{self.args.epochs}"):
                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).float()
                labels = self._mask_to_labels(masks)
                bd_gts = make_boundary_target(masks).squeeze(1)

                losses, preds, acc, loss_list = self.model(images, labels, bd_gts)
                loss = losses.mean()
                sem_loss = loss_list[0].mean()
                boundary_loss = loss_list[1].mean()
                sb_loss = loss - sem_loss - boundary_loss

                main_logits = preds[-1] if isinstance(preds, (list, tuple)) else preds
                fg_logit = self._main_binary_logit(main_logits)
                dice, iou = dice_iou_from_logits(fg_logit, masks)
                dice_all.append(dice)
                iou_all.append(iou)

                meters["total"] += float(loss.item())
                meters["sem"] += float(sem_loss.item())
                meters["boundary"] += float(boundary_loss.item())
                meters["sb"] += float(sb_loss.item())
                meters["acc"] += float(acc.mean().item())

        num_batches = max(1, len(self.val_loader))
        avg = {k: v / num_batches for k, v in meters.items()}
        val_dice = float(np.mean(dice_all)) if dice_all else 0.0
        val_iou = float(np.mean(iou_all)) if iou_all else 0.0

        self.writer.add_scalar("Val/Loss", avg["total"], epoch + 1)
        self.writer.add_scalar("Val/LossSemantic", avg["sem"], epoch + 1)
        self.writer.add_scalar("Val/LossBoundary", avg["boundary"], epoch + 1)
        self.writer.add_scalar("Val/LossSB", avg["sb"], epoch + 1)
        self.writer.add_scalar("Val/PixelAcc", avg["acc"], epoch + 1)
        self.writer.add_scalar("Val/Dice", val_dice, epoch + 1)
        self.writer.add_scalar("Val/IoU", val_iou, epoch + 1)
        return avg, val_dice, val_iou

    def optimize(self):
        logging.info("[*] Start training PIDNet-S (lightweight) ...")
        for epoch in range(self.start_epoch, self.args.epochs):
            train_avg, lr_now, train_dice, train_iou = self.train_one_epoch(epoch)
            val_avg, val_dice, val_iou = self.validate_one_epoch(epoch)

            logging.info(
                f"[Epoch {epoch + 1}/{self.args.epochs}] "
                f"lr={lr_now:.6e} "
                f"train_loss={train_avg['total']:.4f} (sem={train_avg['sem']:.4f}, bd={train_avg['boundary']:.4f}, sb={train_avg['sb']:.4f}, acc={train_avg['acc']:.4f}) "
                f"val_loss={val_avg['total']:.4f} (sem={val_avg['sem']:.4f}, bd={val_avg['boundary']:.4f}, sb={val_avg['sb']:.4f}, acc={val_avg['acc']:.4f}) "
                f"train_dice={train_dice:.4f} train_iou={train_iou:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                best_path = self._save_ckpt(epoch + 1, "best_pidnet_s.pth")
                logging.info(f"[*] New best dice={self.best_dice:.4f}, saved: {best_path}")

            if (epoch + 1) % self.args.save_epochs == 0:
                save_path = self._save_ckpt(epoch + 1, f"pidnet_s_epoch_{epoch + 1}.pth")
                logging.info(f"[*] Periodic checkpoint saved: {save_path}")

        final_path = self._save_ckpt(self.args.epochs, "last_pidnet_s.pth")
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
    args.work_dir = os.path.join(base_work_dir, f"{timestamp}_pidnet_s")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "tensorboard_log"), exist_ok=True)

    setup_logger(args.work_dir, "train_pid.log")
    writer = SummaryWriter(os.path.join(args.work_dir, "tensorboard_log"))

    logging.info(f"[*] args: {args}")
    logging.info(f"[*] work_dir: {args.work_dir}")
    logging.info(f"[*] device: {device}")

    optimizer = Optimizer(args, device, writer)
    optimizer.optimize()
    writer.close()


if __name__ == "__main__":
    main()
