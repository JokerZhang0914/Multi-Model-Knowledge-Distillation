import torch
import numpy as np

def cal_acc(label, pred, threshold=0.5):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    pred_logit = pred>threshold
    pred_logit = pred_logit.astype(np.long)
    acc = np.sum(pred_logit == label)/label.shape[0]
    return acc


