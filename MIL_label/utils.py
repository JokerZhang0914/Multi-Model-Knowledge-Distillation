import torch
import numpy as np
import random
import os
from sklearn import metrics

def cal_auc(label, pred, pos_label=1, return_fpr_tpr=False, save_fpr_tpr=False):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=pos_label, drop_intermediate=False)
    auc_score = metrics.auc(fpr, tpr)
    if save_fpr_tpr:
        if auc_score > 0.5:
            np.save("./ROC_reinter/{:.0f}".format(auc_score * 10000),
                    np.concatenate([np.expand_dims(fpr, axis=1), np.expand_dims(tpr, axis=1)], axis=1))
    if return_fpr_tpr:
        return fpr, tpr, auc_score
    return auc_score

def cal_acc(label, pred, threshold=0.5):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    pred_logit = pred>threshold
    pred_logit = pred_logit.astype(np.long)
    acc = np.sum(pred_logit == label)/label.shape[0]
    return acc

def setup_runtime(seed=0, cuda_dev_id=[0]):
    """Initialize CUDA, CuDNN and the random seeds."""
    # Setup CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(cuda_dev_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
        for i in cuda_dev_id[1:]:
            os.environ["CUDA_VISIBLE_DEVICES"] += "," + str(i)

    # global cuda_dev_id
    _cuda_device_id = cuda_dev_id
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False 
    # Fix random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
