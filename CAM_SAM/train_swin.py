import os
import argparse
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.load_data import load_dataset_from_folders
from dataset.dataset_cam import SwinDataset
from utils import load_checkpoint
from model.swin_transformer import SwinTransformer


def get_args():
    parser = argparse.ArgumentParser(description='Swin Transformer 训练脚本')
    parser.add_argument("--seed", default=310, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--work_dir", default="runs/swin_train", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--gpu", default="0", type=str)

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

    parser.add_argument("--ckpt", default='/home/zhaokaizhang/code/test_code/runs/swin_train/2026-0310-1804-09/best_swin.pth', type=str, help="Swin 预训练权重路径")
    return parser.parse_args()


def setup_logger(work_dir):
    log_file = os.path.join(work_dir, 'train_swin.log')
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

        logging.info("[*] 正在加载数据集...")
        img_paths, labels = load_dataset_from_folders()

        train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
            img_paths,
            labels,
            test_size=0.25,
            random_state=args.seed,
            stratify=labels if args.num_classes == 2 else None
        )
        logging.info(f"[*] 数据集划分完毕: 训练集 {len(train_img_paths)}，验证集 {len(val_img_paths)}")

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

        gpu_list = args.gpu.split(',')
        if len(gpu_list) > 1:
            logging.info(f"[*] 开启 DataParallel 训练，使用 GPU: {args.gpu}")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.writer.add_hparams(vars(args), {"hparam/init": 0.0})
        self.best_auc = 0.0

    def train_cls(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.args.epochs}]")

        for imgs, labels in pbar:
            imgs, labels = imgs.cuda().float(), labels.cuda().long()

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        logging.info(f"[Epoch {epoch+1} Summary] Train Loss: {avg_loss:.4f}")
        self.writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch + 1)
        return avg_loss

    def validate_cls(self, epoch):
        self.model.eval()
        y_true, y_scores, y_preds = [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="[Validating]", leave=False):
                imgs, labels = imgs.cuda(), labels.cuda().long()
                logits = self.model(imgs)

                probs = torch.softmax(logits, dim=1)[:, 1] if self.args.num_classes == 2 else torch.softmax(logits, dim=1).max(1)[0]
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
                y_preds.extend(preds.cpu().numpy())

        y_true, y_scores, y_preds = np.array(y_true), np.array(y_scores), np.array(y_preds)

        if len(np.unique(y_true)) < 2:
            auc, ap = 0.0, 0.0
        else:
            try:
                auc = roc_auc_score(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
            except Exception:
                auc, ap = 0.0, 0.0

        acc = accuracy_score(y_true, y_preds)

        self.model.train()
        self.writer.add_scalar('Val/ACC', acc, epoch + 1)
        self.writer.add_scalar('Val/AUC', auc, epoch + 1)
        self.writer.add_scalar('Val/AP', ap, epoch + 1)
        logging.info(f"[Epoch {epoch+1} Validation] ACC: {acc:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")

        return acc, auc, ap

    def optimize(self):
        logging.info("[*] 开始 Swin 分类训练...")
        print("[*] 开始 Swin 分类训练...")
        for epoch in range(self.args.epochs):
            self.train_cls(epoch)

            acc, auc, ap = self.validate_cls(epoch)

            model_to_save = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

            if auc > self.best_auc:
                self.best_auc = auc
                best_path = os.path.join(self.args.work_dir, "best_swin.pth")
                torch.save(model_to_save, best_path)
                logging.info(f"[*] 发现更优的 AUC: {self.best_auc:.4f}，模型已保存。")

            if (epoch + 1) % 5 == 0:
                last_path = os.path.join(self.args.work_dir, f"swin_epoch_{epoch+1}.pth")
                torch.save(model_to_save, last_path)


def main():
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = "{0:%Y-%m%d-%H%M-%S}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
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
