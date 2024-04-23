import numpy as np
import torch


def fpr95(labels, distances, ):
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


def tpfp(scores, labels, numpos=None):
    assert type(scores) == type(labels), "a, b should be same type."
    is_numpy = (type(scores) == np.ndarray)
    if is_numpy:
        # from hpatches
        # count labels
        p = int(np.sum(labels))
        n = len(labels) - p

        if numpos is not None:
            assert (numpos >= p), 'numpos smaller that number of positives in labels'
            extra_pos = numpos - p
            p = numpos
            scores = np.hstack((scores, np.repeat(-np.inf, extra_pos)))
            labels = np.hstack((labels, np.repeat(1, extra_pos)))

        perm = np.argsort(-scores, kind='mergesort', axis=0)

        scores = scores[perm]
        # assume that data with -INF score is never retrieved
        stop = np.max(np.where(scores > -np.inf))

        perm = perm[0:stop + 1]

        labels = labels[perm]
        # accumulate true positives and false positives by scores
        tp = np.hstack((0, np.cumsum(labels == 1)))
        fp = np.hstack((0, np.cumsum(labels == 0)))
    else:
        device = scores.device
        p = torch.sum(labels).to(torch.int64)
        n = labels.shape[0] - p

        if numpos is not None:
            assert (numpos >= p), 'numpos smaller that number of positives in labels'
            extra_pos = torch.tensor(numpos - p, dtype=torch.int64)
            p = torch.tensor(numpos, dtype=torch.int64)
            scores = torch.hstack([scores, torch.repeat_interleave(torch.tensor(-np.inf).to(device), extra_pos)])
            labels = torch.hstack([labels, torch.repeat_interleave(torch.tensor(1).to(device), extra_pos)])

        perm = torch.argsort(-scores, dim=0, descending=False)

        scores = scores[perm]
        # assume that data with -INF score is never retrieved
        stop = torch.sum(~torch.isinf(scores), dtype=torch.int64)

        perm = perm[0:stop]

        labels = labels[perm]
        # accumulate true positives and false positives by scores
        tp = torch.hstack((torch.tensor(0).to(device), torch.cumsum(labels == 1, dim=0))).to(torch.float)
        fp = torch.hstack((torch.tensor(0).to(device), torch.cumsum(labels == 0, dim=0))).to(torch.float)

    return tp, fp, p, n, perm


# from hpatches
def pr(scores, labels, numpos=None):
    assert type(scores) == type(labels), "a, b should be same type."
    is_numpy = (type(scores) == np.ndarray)

    [tp, fp, p, n, perm] = tpfp(scores, labels, numpos)

    if is_numpy:
        # compute precision and recall
        small = 1e-10
        recall = tp / float(np.maximum(p, small))
        precision = np.maximum(tp, small) / np.maximum(tp + fp, small)
    else:
        # compute precision and recall
        small = 1e-10
        recall = tp / p.clamp(small)
        precision = tp.clamp(small) / (tp + fp).clamp(small)

    return precision, recall, torch.trapz(precision, recall)


# # from hpatches
# def tpfp(scores, labels, numpos=None):
#     # count labels
#     p = int(np.sum(labels))
#     n = len(labels) - p
#
#     if numpos is not None:
#         assert (numpos >= p), 'numpos smaller that number of positives in labels'
#         extra_pos = numpos - p
#         p = numpos
#         scores = np.hstack((scores, np.repeat(-np.inf, extra_pos)))
#         labels = np.hstack((labels, np.repeat(1, extra_pos)))
#
#     perm = np.argsort(-scores, kind='mergesort', axis=0)
#
#     scores = scores[perm]
#     # assume that data with -INF score is never retrieved
#     stop = np.max(np.where(scores > -np.inf))
#
#     perm = perm[0:stop + 1]
#
#     labels = labels[perm]
#     # accumulate true positives and false positives by scores
#     tp = np.hstack((0, np.cumsum(labels == 1)))
#     fp = np.hstack((0, np.cumsum(labels == 0)))
#
#     return tp, fp, p, n, perm
#
#
# # from hpatches
# def pr(scores, labels, numpos=None):
#     [tp, fp, p, n, perm] = tpfp(scores, labels, numpos)
#
#     # compute precision and recall
#     small = 1e-10
#     recall = tp / float(np.maximum(p, small))
#     precision = np.maximum(tp, small) / np.maximum(tp + fp, small)
#
#     return precision, recall, np.trapz(precision, recall)


# from hpatches
def roc(scores, labels, numpos=None):
    [tp, fp, p, n, perm] = tpfp(scores, labels, numpos)

    # compute tpr and fpr
    small = 1e-10
    tpr = tp / float(np.maximum(p, small))
    fpr = fp / float(np.maximum(n, small))

    return fpr, tpr, np.trapz(tpr, fpr)
