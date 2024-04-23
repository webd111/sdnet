# From original author
import numpy as np
import torch
from tlib import tpfp, pr, roc


def fpr95(labels, scores, use_yurun_tian=False):
    if use_yurun_tian:
        distances = - scores
        dist_pos = distances[labels == 1]
        dist_neg = distances[labels != 1]
        dist_pos, indice = torch.sort(dist_pos)
        loc_thr = int(np.ceil(dist_pos.numel() * 0.95))  # 真例中被预测为真的概率定为95％时的阈值
        thr = dist_pos[loc_thr]
        fpr95 = float(dist_neg.le(thr).sum()) / dist_neg.numel()  # 假例中被预测为真的概率
        # print("fpr95:{}".format(fpr95))
        return fpr95

    distances = - scores
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)


def fprx(labels, scores, recall_point):
    distances = 1.0 / (scores + 1e-8)
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)
