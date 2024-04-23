import torch
import torch.nn as nn
import torch.nn.functional as F

import tlib
from tlib import *


def dot_to_dist(dot, alpha, eps=1e-12):
    dist = alpha * (1 - dot) + (1 - alpha) * torch.sqrt(2 * (1 - dot) + eps)
    return dist


def estimate_alpha(dots, labels, weights):
    # using all samples to estimate alpha and pair mask selected samples to estimate current
    # INPORTANT: dots should be [pos, neg]
    # changed alpha_range to [-1,1]
    with torch.no_grad():
        cnt = dots.size(0) // 2
        alpha_range = torch.arange(-1, 1, 0.01)
        BCE_dists = torch.zeros(len(alpha_range))
        # pair mask
        for k in range(len(alpha_range)):
            dists = dot_to_dist(dots, alpha_range[k])
            # normalize
            mean = torch.mean(dists, dim=0)
            std = torch.std(dists, dim=0)
            probs = 1 / (1 + torch.exp((dists - mean) / std))
            BCE_dists[k] = torch.mean(- labels * torch.log(probs) * weights -
                                      (1 - labels) * torch.log(1 - probs) * weights)
        alpha_opt = alpha_range[torch.argmin(BCE_dists)]
    return alpha_opt


class SDNetLoss(nn.Module):
    """
    using alpha estimate with \alpha in [-1,1], use train.py to train
    """

    def __init__(self, margin=0.36, alpha_init=0, alpha_moment=0.995, lambda_clean=1, lambda_adv=1):
        super().__init__()
        self.margin = margin
        self.alpha = alpha_init
        self.alpha_moment = alpha_moment
        self.lambda_clean = lambda_clean
        self.lambda_adv = lambda_adv

    def forward(self, input):
        x = input["descs"]
        cnt = x.size(0) // 6

        xn = F.normalize(x, p=2, dim=1)

        a = xn[:cnt]
        p = xn[cnt:2 * cnt]
        a_rec = xn[2 * cnt:3 * cnt]
        p_rec = xn[3 * cnt:4 * cnt]
        a_adv = xn[4 * cnt:5 * cnt]
        p_adv = xn[5 * cnt:6 * cnt]

        l_a_adv = input["l_a_adv"]
        l_p_adv = input["l_p_adv"]

        if l_a_adv is None:
            l_a_adv = torch.tensor(1).float()
        if l_p_adv is None:
            l_p_adv = torch.tensor(1).float()

        eps = 1e-2
        with torch.no_grad():
            smat_ap = compute_similarity_matrix_dot(a, p)
            smat_arpa = compute_similarity_matrix_dot(a_rec, p_adv)
            smat_aapr = compute_similarity_matrix_dot(a_adv, p_rec)
            idx_a, idx_p, idx_n = find_hard_negatives(1 / smat_ap + eps, output_index=True)
            idx_a_arpa, idx_p_arpa, idx_n_arpa = find_hard_negatives(1 / smat_arpa + eps, output_index=True)
            idx_a_aapr, idx_p_aapr, idx_n_aapr = find_hard_negatives(1 / smat_aapr + eps, output_index=True)

        xn_ap = torch.cat([a, p], dim=0)
        xn_arpa = torch.cat([a_rec, p_adv], dim=0)
        xn_aapr = torch.cat([a_adv, p_rec], dim=0)

        pos_ap = tlib.compute_similarity_dot(xn_ap[idx_a], xn_ap[idx_p])
        neg_ap = tlib.compute_similarity_dot(xn_ap[idx_a], xn_ap[idx_n])
        pos_arpa = tlib.compute_similarity_dot(xn_arpa[idx_a_arpa], xn_arpa[idx_p_arpa])
        neg_arpa = tlib.compute_similarity_dot(xn_arpa[idx_a_arpa], xn_arpa[idx_n_arpa])
        pos_aapr = tlib.compute_similarity_dot(xn_aapr[idx_a_aapr], xn_aapr[idx_p_aapr])
        neg_aapr = tlib.compute_similarity_dot(xn_aapr[idx_a_aapr], xn_aapr[idx_n_aapr])
        mask = (self.margin + torch.arccos(pos_ap) - torch.arccos(neg_ap)) > 0

        with torch.no_grad():
            # estimate alpha
            dots = torch.cat([pos_ap, pos_arpa, pos_aapr, neg_ap, neg_arpa, neg_aapr], dim=0)
            labels = torch.cat([torch.ones(cnt * 3, device=x.device),
                                torch.zeros(cnt * 3, device=x.device)], dim=0)
            weights = torch.cat([torch.ones(cnt, device=x.device) * self.lambda_clean,
                                 torch.ones(cnt * 2, device=x.device) * self.lambda_adv,
                                 torch.ones(cnt, device=x.device) * self.lambda_clean,
                                 torch.ones(cnt * 2, device=x.device) * self.lambda_adv], dim=0)
            alpha = estimate_alpha(dots=dots, labels=labels, weights=weights)
            self.alpha = self.alpha * self.alpha_moment + alpha * (1 - self.alpha_moment)

        dist_ap_pos = dot_to_dist(pos_ap, self.alpha)
        dist_ap_neg = dot_to_dist(neg_ap, self.alpha)
        dist_arpa_pos = dot_to_dist(pos_arpa, self.alpha)
        dist_arpa_neg = dot_to_dist(neg_arpa, self.alpha)
        dist_aapr_pos = dot_to_dist(pos_aapr, self.alpha)
        dist_aapr_neg = dot_to_dist(neg_aapr, self.alpha)

        loss_triplet_clean = ((dist_ap_pos - dist_ap_neg) * mask).mean() * self.lambda_clean
        loss_triplet_arpa = ((l_p_adv * dist_arpa_pos - dist_arpa_neg) * mask).mean() * self.lambda_adv
        loss_triplet_aapr = ((l_a_adv * dist_aapr_pos - dist_aapr_neg) * mask).mean() * self.lambda_adv
        loss = torch.mean(torch.stack([loss_triplet_clean, loss_triplet_arpa, loss_triplet_aapr]))

        out_dict = {"loss": loss, "loss_triplet_clean": loss_triplet_clean,
                    "loss_triplet_arpa": loss_triplet_arpa, "loss_triplet_aapr": loss_triplet_aapr,
                    "l_mean": (l_a_adv.mean() + l_p_adv.mean()) / 2,
                    "alpha": self.alpha, "alpha_opt": alpha, "mask_ratio": mask.sum() / cnt}
        return out_dict
