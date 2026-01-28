import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score
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


def calculate_metrics(y_true, y_pred_prob):
    """
    计算基础指标: AUC, ACC (阈值0.5)
    """
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = 0.0
    
    y_pred_bin = (np.array(y_pred_prob) > 0.5).astype(int)
    acc = (y_true == y_pred_bin).mean()
    return roc_auc, acc

def get_roc_figure(y_true, y_pred_prob):
    """
    绘制 ROC 曲线并返回 matplotlib figure 对象
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return fig

def get_pr_figure(y_true, y_pred_prob):
    """
    绘制 PR 曲线并返回 matplotlib figure 对象
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    
    fig = plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    return fig

def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"[*] Model saved to {path}")


def get_features_chunked(encoder, imgs, batch_size=64):
    """
    分块特征提取：避免将整个 Bag (N=1000+) 一次性送入 ResNet 导致显存爆炸。
    imgs: [N, 3, H, W]
    """
    feats_list = []
    num_imgs = imgs.size(0)
    
    # 只需要推理，不需要梯度
    with torch.no_grad():
        for i in range(0, num_imgs, batch_size):
            batch_imgs = imgs[i : i + batch_size]
            # encoder 输出 [B, 512]
            batch_feats = encoder(batch_imgs)
            feats_list.append(batch_feats)
            
    # 拼接所有分块特征
    return torch.cat(feats_list, dim=0)