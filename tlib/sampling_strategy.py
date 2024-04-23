import torch
import torch.nn as nn
from typing import List
import numpy as np


def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...

    INPUT:
    dmat: distance matrix, NOT NEGETIVE
    output_index: bool, whether output index
    OUTPUT1:
    pos: N * N, matrix
    OUTPUT2:
    a_idx:
    p_idx:
    n_idx: hardest negetive sample idx in x and its
    """

    is_numpy = (type(dmat) == np.ndarray)
    if is_numpy:
        dmat = torch.from_numpy(dmat)

    cnt = dmat.size(0)

    if not output_index:
        pos = dmat.diag()

    dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
    dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset
    min_a, min_a_idx = torch.min(dmat, dim=0)
    min_p, min_p_idx = torch.min(dmat, dim=1)

    neg = torch.min(min_a, min_p)
    if not output_index:
        if is_numpy:
            return pos.numpy(), neg.numpy()
        else:
            return pos, neg

    mask = min_a < min_p
    a_idx = torch.cat([torch.arange(cnt, device=dmat.device)[mask] + cnt,
                       torch.arange(cnt, device=dmat.device)[~mask]], dim=0)
    p_idx = (a_idx + cnt) % (2 * cnt)
    n_idx = torch.cat((min_a_idx[mask], min_p_idx[~mask] + cnt))
    #
    idx_pair = a_idx % cnt
    idx_inv = torch.argsort(idx_pair)
    if is_numpy:
        return a_idx[idx_inv].numpy(), p_idx[idx_inv].numpy(), n_idx[idx_inv].numpy()
    else:
        return a_idx[idx_inv], p_idx[idx_inv], n_idx[idx_inv]


def find_hard_negatives_full(dmat_ap, dmat_aa, dmat_pp, output_index=True, empirical_thresh=0.0):
    """
    dmat_ap = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...

    dmat_aa = A * A'
    dmat_pp = P * P'

    INPUT:
    dmat_ap: distance matrix, NOT NEGETIVE
    dmat_aa: distance matrix, NOT NEGETIVE
    dmat_pp: distance matrix, NOT NEGETIVE
    output_index: bool, whether output index
    OUTPUT1:
    pos: N * N, matrix
    OUTPUT2:
    idx_a:
    idx_p:
    idx_n: hardest negetive sample idx in x and its
    """
    assert (type(dmat_ap) == type(dmat_aa)) & (type(dmat_ap) == type(dmat_pp))
    is_numpy = (type(dmat_ap) == np.ndarray)
    if is_numpy:
        dmat_ap = torch.from_numpy(dmat_ap)
        dmat_aa = torch.from_numpy(dmat_aa)
        dmat_pp = torch.from_numpy(dmat_pp)

    cnt = dmat_ap.size(0)

    if not output_index:
        pos = dmat_ap.diag()

    dmat_ap = dmat_ap + torch.eye(cnt).to(dmat_ap.device) * 9999  # filter diagonal
    dmat_aa = dmat_aa + torch.eye(cnt).to(dmat_aa.device) * 9999  # filter diagonal
    dmat_pp = dmat_pp + torch.eye(cnt).to(dmat_pp.device) * 9999  # filter diagonal
    dmat_ap[dmat_ap < empirical_thresh] = 9999  # filter outliers in brown dataset
    dmat_aa[dmat_aa < empirical_thresh] = 9999  # filter outliers in brown dataset
    dmat_pp[dmat_pp < empirical_thresh] = 9999  # filter outliers in brown dataset
    min_a, min_a_idx = torch.min(torch.cat([dmat_ap, dmat_pp], dim=0), dim=0)       # [a, p] to p
    min_p, min_p_idx = torch.min(torch.cat([dmat_aa, dmat_ap], dim=1), dim=1)       # a to [a, p]

    neg = torch.min(min_a, min_p)
    if not output_index:
        if is_numpy:
            return pos.numpy(), neg.numpy()
        else:
            return pos, neg

    mask = min_a < min_p        # p is anchor when mask=True
    idx_a = torch.cat([torch.arange(cnt, device=dmat_ap.device)[mask] + cnt,
                       torch.arange(cnt, device=dmat_ap.device)[~mask]], dim=0)
    idx_p = (idx_a + cnt) % (2 * cnt)
    idx_n = torch.cat((min_a_idx[mask], min_p_idx[~mask]))
    # sort it back to keep aligned with output_index=False
    idx_pair = idx_a % cnt
    idx_inv = torch.argsort(idx_pair)
    if is_numpy:
        return idx_a[idx_inv].numpy(), idx_p[idx_inv].numpy(), idx_n[idx_inv].numpy()
    else:
        return idx_a[idx_inv], idx_p[idx_inv], idx_n[idx_inv]


def find_semi_hard_negatives(dmat, knn_start, knn_end, output_index=True, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...

    INPUT:
    dmat: distance matrix
    knn_start:
    knn_end:
    output_index: bool, whether output index
    OUTPUT1:
    pos: N, matrix
    neg: N * (knn_end - knn_start), matrix
    OUTPUT2:
    a_idx: N * (knn_end - knn_start), matrix
    p_idx: N * (knn_end - knn_start), matrix
    n_idx: N * (knn_end - knn_start), matrix, semi-hard negetive sample idx in x and its
    """

    raise NotImplementedError

    assert knn_start < knn_end, "knn_start must be smaller than knn_end"
    is_numpy = (type(dmat) == np.ndarray)
    if is_numpy:
        dmat = torch.from_numpy(dmat)

    cnt = dmat.size(0)

    if not output_index:
        pos = dmat.diag()

    dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
    dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset

    dmat_ext = torch.cat([dmat, torch.transpose(dmat, 0, 1)], dim=0)
    dmat_ext_sort, idx_ext_sort = torch.sort(dmat_ext, dim=0, descending=False)
    neg = torch.transpose(dmat_ext_sort[knn_start:knn_end, :], 0, 1)

    if not output_index:
        if is_numpy:
            return pos.numpy(), neg.numpy()
        else:
            return pos, neg

    n_idx = torch.transpose(idx_ext_sort[knn_start:knn_end, :], 0, 1)

    mask = n_idx < cnt
    a_idx = torch.tensor(range(cnt)).unsqueeze(-1).expand(cnt, knn_end - knn_start).to(dmat.device)
    p_idx = torch.tensor(range(cnt)).unsqueeze(-1).expand(cnt, knn_end - knn_start).to(dmat.device)
    a_idx[mask] += cnt
    p_idx[~mask] += cnt

    if is_numpy:
        return a_idx.numpy(), p_idx.numpy(), n_idx.numpy()
    else:
        return a_idx, p_idx, n_idx


def find_k_global_hard_negatives(dmat, k, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...

    INPUT:
    dmat: distance matrix
    k: return k samples
    output_index: bool, whether output index
    OUTPUT1:
    pos: N * N, matrix
    OUTPUT2:
    a_idx:
    p_idx:
    n_idx: hardest negetive sample idx in x and its
    """

    raise NotImplementedError

    is_numpy = (type(dmat) == np.ndarray)
    if is_numpy:
        dmat = torch.from_numpy(dmat)

    cnt = dmat.size(0)

    with torch.no_grad():
        pos = dmat.diag()
        dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
        dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset
        dmat2 = dmat - pos.expand_as(dmat)  # when using p as anchor
        dmat1 = dmat - pos.expand_as(dmat).transpose(1, 0)  # when using a as anchor
        dmats = torch.stack([dmat1, dmat2], dim=2)
        thres, _ = torch.kthvalue(dmats.reshape([-1]), k=k)

        dmats_numpy = dmats.cpu().numpy()
        idx = np.argwhere(dmats_numpy <= thres.item())  # find k-th hardest samples
        idx = torch.tensor(idx)
        mask_is_p = idx[:, 2] > 0.5
        idx_a = torch.cat([idx[:, 1][mask_is_p] + cnt, idx[:, 0][~mask_is_p]], dim=0)
        idx_p = torch.cat([idx[:, 1][mask_is_p], idx[:, 0][~mask_is_p] + cnt], dim=0)
        idx_n = torch.cat([idx[:, 0][mask_is_p], idx[:, 1][~mask_is_p] + cnt], dim=0)

    if is_numpy:
        return idx_a.numpy(), idx_p.numpy(), idx_n.numpy()
    return idx_a, idx_p, idx_n


if __name__ == "__main__":
    num = 6
    a = torch.rand([num, num])
    idx_a, idx_p, idx_n = find_k_global_hard_negatives(a, k=num, empirical_thresh=0.008)
