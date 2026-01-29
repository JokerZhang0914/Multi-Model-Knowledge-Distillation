import argparse
import datetime
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from dataset import dataset_loader_public_data
from dataset.dataloader_public import load_img_label_info_from_csv, load_test_img_mask,load_img_label_info_from_public_data
from model.losses import get_masked_ptc_loss, get_seg_loss, get_seg_loss_update, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
from model.model_cam import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer,imutils2
from utils.camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

