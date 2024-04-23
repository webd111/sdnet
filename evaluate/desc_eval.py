import os
import os.path
import sys
import logging
import time
from collections import defaultdict

import tlib
from evaluate import metrics
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision

logger = logging.getLogger(__name__)


class GenericLearnedDescriptorExtractor:
    def __init__(self, patch_size, model, batch_size, transform=None, device="cuda"):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.transform = transform
        self.model = model.to(self.device)

    def __call__(self, patches):
        if self.transform is not None:
            patches = self.transform(patches)
        n_batches = (patches.size(0) + self.batch_size - 1) // self.batch_size
        descs = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(n_batches):
                s = batch_idx * self.batch_size
                e = min((batch_idx + 1) * self.batch_size, patches.size(0))
                batch_data = patches[s:e, ...].to(self.device, )

                batch_data = self.model(batch_data)
                descs.append(batch_data)

        descs = torch.cat(descs, dim=0)

        return descs


class hpatch_descr:
    """Class for loading an HPatches descriptor result .csv file"""
    itr = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5',
           't1', 't2', 't3', 't4', 't5']

    def __init__(self):
        self.name = None
        self.N = None
        self.dim = None


class DescriptorEvaluator(object):
    def __init__(self, extractor, datasets, dataset_type="brown", batch_size=1024, binarize=False, metric="l2",
                 out_dim=128):
        self.datasets = datasets
        self.dataset_type = dataset_type
        self.loader = DataLoader(datasets, batch_size=batch_size,
                                 shuffle=False, num_workers=0 if sys.platform == "win32" else 8,
                                 drop_last=False, pin_memory=True, )
        self.extractor = extractor
        self.device = self.extractor.device
        self.binarize = binarize
        self.metric = metric
        self.out_dim = out_dim
        if self.binarize:
            logger.info("descriptor evaluator is set to binary mode")
        # HP only
        self.descs = {}
        self.hp_sequence = []
        self.itr = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5',
                    't1', 't2', 't3', 't4', 't5']
        self.tp = ['e', 'h', 't']
        # self.itr_dict = dict(zip(range(len(self.itr)), self.itr))
        # TODO: only split a is supported
        self.pos = pd.read_csv(os.path.join('./data/hpatches/tasks/verif_pos_split-a.csv')).to_numpy()
        self.neg_intra = pd.read_csv(os.path.join('./data/hpatches/tasks/verif_neg_intra_split-a.csv')).to_numpy()
        self.neg_inter = pd.read_csv(os.path.join('./data/hpatches/tasks/verif_neg_inter_split-a.csv')).to_numpy()

    def run(self):
        if self.dataset_type == "brown":
            labels = []
            descs_a = []
            descs_b = []

            for patches_a, patches_b, label in tqdm(self.loader):
                descs_a.append(self.extractor(patches_a))
                descs_b.append(self.extractor(patches_b))
                labels.append(label)

            self.labels = torch.cat(labels, dim=0)
            self.descs_a = torch.cat(descs_a, dim=0)
            self.descs_b = torch.cat(descs_b, dim=0)

            if self.binarize:
                self.descs_a = (self.descs_a > 0).to(torch.float32)
                self.descs_b = (self.descs_b > 0).to(torch.float32)

            if self.metric == "l2":
                self.dist = tlib.compute_distance_l2(self.descs_a, self.descs_b, eps=1e-12)
                self.score = 1.0 / (self.dist + 1e-8)
            elif self.metric == "l1":
                self.dist = (self.descs_a - self.descs_b).abs().sum(dim=1)
                self.score = 1.0 / (self.dist + 1e-8)
            elif self.metric == "cos":
                self.score = torch.cosine_similarity(self.descs_a, self.descs_b, dim=1)
                self.dist = torch.arccos(self.score)      # theta distance
            else:
                raise NotImplementedError

        elif self.dataset_type == "HP":
            sequence_len = self.datasets.sequence_len

            self.descs["distance"] = self.metric
            self.descs["dim"] = self.out_dim

            for seq_name in sequence_len.keys():
                if not seq_name in self.descs.keys():
                    self.hp_sequence.append(seq_name)
                    self.descs[seq_name] = hpatch_descr()
                    self.descs[seq_name].N = sequence_len[seq_name]
                    self.descs[seq_name].dim = self.out_dim
                    self.descs[seq_name].name = seq_name
                    for t in self.itr:
                        setattr(self.descs[seq_name], t, torch.zeros([self.descs[seq_name].N, self.descs[seq_name].dim],
                                                                     dtype=torch.float32, device=self.device))

            for patches, seq, pt_label in tqdm(self.loader):
                desc = self.extractor(patches)
                for i in range(len(seq)):  # batchsize
                    for j in range(16):  # patch_types
                        getattr(self.descs[seq[i]], self.itr[j])[pt_label[i], :] = desc[i * 16 + j, :]

    def computeROC(self):
        fpr, tpr, auc = metrics.roc(self.score, self.labels)
        logger.info(f"Area under ROC: {auc}")
        return fpr, tpr, auc

    def computeGrad(self, criterion, bs):
        if type(self.descs_a) is np.ndarray:
            desc_a = torch.from_numpy(self.descs_a)
        else:
            desc_a = self.descs_a
        if type(self.descs_b) is np.ndarray:
            desc_b = torch.from_numpy(self.descs_b)
        else:
            desc_b = self.descs_b
        desc_a.requires_grad_(True)
        desc_b.requires_grad_(True)
        num_idx = len(desc_a) // bs + 1
        loss_sum = None
        for idx in range(len(desc_a) // bs + 1):
            if idx == num_idx - 1:  # last batch
                data = torch.cat([desc_a[idx * bs:, :], desc_b[idx * bs:, :]], dim=0)
            else:
                data = torch.cat([desc_a[idx * bs:(idx + 1) * bs, :],
                                  desc_b[idx * bs:(idx + 1) * bs, :]], dim=0)
            loss = criterion(data)["loss"]
            if loss_sum is not None:
                loss_sum = loss_sum + loss
            else:
                loss_sum = loss
        loss_sum.backward()
        dist = (((self.descs_a - self.descs_b) ** 2).sum(axis=1) + 1e-8) ** 0.5
        idx_ascent = torch.argsort(dist)
        dist = dist.clone().detach()
        grad_desc_a = desc_a.grad[idx_ascent, :].clone().detach()
        grad_desc_b = desc_b.grad[idx_ascent, :].clone().detach()
        return dist[idx_ascent], self.labels[idx_ascent], grad_desc_a, grad_desc_b

    def computeFPR95(self):
        if self.dataset_type == "brown":
            fpr95 = metrics.fpr95(self.labels, self.score, use_yurun_tian=True)
            logger.info(f"FPR95: {fpr95 * 100}%")
            return fpr95
        elif self.dataset_type == "HP":
            return None
        else:
            raise NotImplementedError

    # from hpatches
    def get_verif_dists(self, descr, pairs, op):
        d = {}
        id2t = {0: {'e': 'ref', 'h': 'ref', 't': 'ref'}, 1: {'e': 'e1', 'h': 'h1', 't': 't1'},
                2: {'e': 'e2', 'h': 'h2', 't': 't2'}, 3: {'e': 'e3', 'h': 'h3', 't': 't3'},
                4: {'e': 'e4', 'h': 'h4', 't': 't4'}, 5: {'e': 'e5', 'h': 'h5', 't': 't5'}}
        for t in ['e', 'h', 't']:
            d[t] = torch.empty((pairs.shape[0], 1)).to(self.device)
        idx = 0
        pbar = tqdm(pairs)
        pbar.set_description("Processing verification task %i/3 " % op)
        for p in pbar:
            [t1, t2] = [id2t[p[1]], id2t[p[4]]]
            for t in self.tp:
                d1 = getattr(descr[p[0]], t1[t])[p[2]]
                d2 = getattr(descr[p[3]], t2[t])[p[5]]
                distance = descr['distance']
                # TODO: only l2 distance is supported now
                if distance == "l2":
                    dist = torch.norm(d1 - d2, p=2)
                elif distance == "cos":
                    dist = torch.arccos(torch.cosine_similarity(d1, d2))
                else:
                    raise NotImplementedError
                d[t][idx] = dist
            idx += 1
        return d

    def computeVerificationScore(self):
        if self.dataset_type == "brown":
            return None
        elif self.dataset_type == "HP":
            # Using imbalanced split a
            logger.info('>> Evaluating verification task')
            start = time.time()
            d_pos = self.get_verif_dists(self.descs, self.pos, 1)
            d_neg_intra = self.get_verif_dists(self.descs, self.neg_intra, 2)
            d_neg_inter = self.get_verif_dists(self.descs, self.neg_inter, 3)

            results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            for t in self.tp:
                l = torch.vstack((torch.zeros_like(d_pos[t]), torch.ones_like(d_pos[t]))).to(self.device)
                d_intra = torch.vstack((d_neg_intra[t], d_pos[t])).to(self.device)
                d_inter = torch.vstack((d_neg_inter[t], d_pos[t])).to(self.device)

                # get results for the imbalanced protocol: 0.2M Positives - 1M Negatives
                N_imb = d_pos[t].shape[0] + int(d_pos[t].shape[0] * 0.2)  # 1M + 0.2*1M
                _, _, ap = metrics.pr(-d_intra[0:N_imb], l[0:N_imb])
                results[t]['intra']['imbalanced']['ap'] = ap

                _, _, ap = metrics.pr(-d_inter[0:N_imb], l[0:N_imb])
                results[t]['inter']['imbalanced']['ap'] = ap
            end = time.time()
            logger.info(f">> Verification task finished in {(end - start):.0f} secs")

            AP = {'e': (results['e']['inter']['imbalanced']['ap'] + results['e']['intra']['imbalanced']['ap']) / 2,
                  'h': (results['h']['inter']['imbalanced']['ap'] + results['h']['intra']['imbalanced']['ap']) / 2,
                  't': (results['t']['inter']['imbalanced']['ap'] + results['t']['intra']['imbalanced']['ap']) / 2}
            AP['mean'] = (AP['e'] + AP['h'] + AP['t']) / 3

            logger.info(f"AP: Easy: {AP['e']}, Hard: {AP['h']}, Tough: {AP['t']}, Mean: {AP['mean']}")
            return AP['mean']
        else:
            raise NotImplementedError

    def computeFPRX(self, recall_point, output_split=False):
        if output_split:
            fprxs = []
            for score in self.scores:
                fprx = metrics.fprx(self.labels, score, recall_point)
                fprxs.append(fprx)
                logger.info(f"FPR{recall_point * 100}: {fprx * 100}%")
            return fprxs
        else:
            fprx = metrics.fprx(self.labels, self.score_all, recall_point)
            logger.info(f"FPR{recall_point * 100}: {fprx * 100}%")
            return fprx

    def computeMatchingScore(self):
        if self.dataset_type == "brown":
            # combining every 1000 patches to one matching block
            labels = self.labels[self.labels == True]
            descs1 = self.descs_a[self.labels == True]
            descs2 = self.descs_b[self.labels == True]
            num_block = (self.labels[self.labels == True]).shape[0] // 1000
            APs = torch.zeros(num_block).to(self.device)
            match_rates = torch.zeros(num_block).to(self.device)
            for i in range(num_block):
                # label = labels[i * 1000:(i + 1) * 1000]
                desc1 = descs1[i * 1000:(i + 1) * 1000, :]
                desc2 = descs2[i * 1000:(i + 1) * 1000, :]
                if self.metric == "l2":
                    dmat = tlib.compute_distance_matrix_l2(desc1, desc2)
                elif self.metric == "cos":
                    dmat = torch.arccos(tlib.compute_similarity_matrix_cosine(desc1, desc2))
                else:
                    raise NotImplementedError
                match_pred = torch.argmin(dmat, dim=1)
                match_label = torch.cumsum(torch.ones(1000).to(self.device), dim=0, dtype=torch.int64) - 1
                res = (match_pred == match_label)
                match_rate = torch.sum(res) / float(res.shape[0])
                match_dist = dmat[match_label, match_pred]
                precision, recall, ap = tlib.pr(-match_dist, res, numpos=match_label.shape[0])
                APs[i] = ap
                match_rates[i] = match_rate
            mAP = torch.mean(APs)
            # mMR = np.mean(match_rates)
            return mAP.cpu().numpy()
        elif self.dataset_type == "HP":
            logger.info('>> Evaluating matching task')
            start = time.time()
            results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            pbar = tqdm(self.hp_sequence)
            for seq in pbar:
                d_ref = getattr(self.descs[seq], 'ref')
                gt_l = torch.arange(d_ref.shape[0]).to(self.device)
                for t in self.tp:
                    for i in range(1, 6):
                        d = getattr(self.descs[seq], t + str(i))
                        # TODO: Only L2 distance is supported
                        # D = spatial.distance.cdist(d_ref, d, 'euclidean')
                        if self.metric == "l2":
                            D = tlib.compute_distance_matrix_l2(d_ref, d, eps=0)
                        elif self.metric == "cos":
                            D = torch.arccos(tlib.compute_similarity_matrix_cosine(d_ref, d))
                        else:
                            raise NotImplementedError
                        idx = torch.argmin(D, dim=1)
                        m_l = (idx == gt_l)
                        results[seq][t][i]['sr'] = torch.sum(m_l) / float(m_l.shape[0])
                        m_d = D[gt_l, idx]
                        pr, rc, ap = metrics.pr(-m_d, m_l, numpos=m_l.shape[0])
                        results[seq][t][i]['ap'] = ap
                        results[seq][t][i]['pr'] = pr
                        results[seq][t][i]['rc'] = rc
                        # print(t,i,ap,results[seq][t][i]['sr'])
            end = time.time()
            logger.info(f">> Matching task finished in {end - start} secs")

            mAP = {'e': 0, 'h': 0, 't': 0}
            k_mAP = 0
            for seq in results:
                for t in ['e', 'h', 't']:
                    for idx in range(1, 6):
                        mAP[t] += results[seq][t][idx]['ap']
                        k_mAP += 1
            k_mAP = k_mAP / 3.0
            results = [mAP['e'] / k_mAP, mAP['h'] / k_mAP, mAP['t'] / k_mAP]
            results.append(sum(results) / float(len(results)))
            logger.info(f'mAP: Easy: {results[0]}, Hard: {results[1]}, Tough: {results[2]}, Mean: {results[3]}')
            return np.ndarray.item(results[3].cpu().numpy())
        else:
            raise NotImplementedError

    def computeRetrievalScore(self):
        raise NotImplementedError

    def computePR(self):
        precisions = []
        recalls = []
        aucs = []
        for score in self.scores:
            precision, recall, auc = metrics.pr(score, self.labels)
            precisions.append(precision)
            recalls.append(recall)
            aucs.append(auc)
            logger.info(f"Area under PR: {auc}")
        return precisions, recalls, aucs

    def viz_tsne(self, with_pca=False, out_dif=False):
        if with_pca:
            pass
        else:
            if out_dif:
                desc_embedded = TSNE(n_components=2, learning_rate=100).fit_transform(
                    self.descs_a - self.descs_b)
            else:
                desc_embedded = TSNE(n_components=2, learning_rate=100).fit_transform(
                    np.concatenate([self.descs_a, self.descs_b], axis=0))
            plt.scatter(desc_embedded[:, 0], desc_embedded[:, 1])
            plt.show()

    def pca_fpr95_test(self, k_start, k_end, eigenvector=None, dim_reverse=False):
        fpr95s = np.zeros(k_end-k_start,)
        with torch.no_grad():
            if eigenvector is None:
                idx_rand = np.random.choice(range(self.descs_a.shape[0]), 4096, replace=False)
                descs_sample = torch.cat([self.descs_a[idx_rand, :], self.descs_b[idx_rand, :]], dim=0)
                _, eigenvalue, eigenvector = tlib.pca(descs_sample, k1=0, k2=self.out_dim, return_eigen=True)
            x_descs_a_proj = torch.transpose(torch.mm(eigenvector, torch.transpose(self.descs_a, 1, 0)), 1, 0)
            x_descs_b_proj = torch.transpose(torch.mm(eigenvector, torch.transpose(self.descs_b, 1, 0)), 1, 0)
            for i in tqdm(range(k_start, k_end)):
                if self.metric == "l2":
                    if dim_reverse:
                        dist = tlib.compute_distance_l2(x_descs_a_proj[:, i:], x_descs_b_proj[:, i:], eps=1e-12)
                    else:
                        dist = tlib.compute_distance_l2(x_descs_a_proj[:, :i], x_descs_b_proj[:, :i], eps=1e-12)
                    score = 1.0 / (dist + 1e-8)
                else:
                    raise NotImplementedError

                if self.dataset_type == "brown":
                    fpr95 = metrics.fpr95(self.labels, score, use_yurun_tian=True)
                    # logger.info(f"dim {i} FPR95: {fpr95 * 100}%")
                    fpr95s[i-k_start] = fpr95
                elif self.dataset_type == "HP":
                    return None
                else:
                    raise NotImplementedError

            return fpr95s

    def viz_samples(self, viz_type="hp", n=1, show_knn=False, k=1):
        """
        viz_type: "hp" for "hardest pos", "ep" for "easiest pos" and so on
        n: viz number
        show_knn: show Knn
        k: Knn number
        """
        idx = np.argsort(self.score_all)
        labels_sorted = self.labels[idx]
        idx_p = np.argsort(self.score_all)[labels_sorted > 0.5]
        idx_n = np.argsort(self.score_all)[labels_sorted < 0.5]
        img = None
        for i in range(n):
            if viz_type == "hp":
                patch_a, patch_b, label = self.datasets[idx_p[i]]
            elif viz_type == "en":
                patch_a, patch_b, label = self.datasets[idx_n[i]]
            elif viz_type == "ep":
                patch_a, patch_b, label = self.datasets[idx_p[-i - 1]]
            elif viz_type == "hn":
                patch_a, patch_b, label = self.datasets[idx_n[-i - 1]]
            else:
                pass

            if img is None:
                img = torch.cat([patch_a, patch_b], dim=2).unsqueeze(0)
            else:
                img_temp = torch.cat([patch_a, patch_b], dim=2).unsqueeze(0)
                img = torch.cat([img, img_temp], dim=0)
        torchvision.utils.save_image(tensor=img, fp=viz_type + str(n) + ".tif", normalize=True)

    def viz_umap(self):
        desc_embedded = TSNE(n_components=2, learning_rate=100).fit_transform(
            np.concatenate([self.descs_a, self.descs_b], axis=0))
        plt.scatter(desc_embedded[:, 0], desc_embedded[:, 1])
        plt.show()

    # def viz_histogram(self, nbins=100):
    #     d_max = np.max(self.dist_all)
    #     d_min = np.min(self.dist_all)
    #     idx_bin = (self.dist_all - d_min) * nbins // (d_max - d_min)

    def viz_learner_pred(self, viz_type="hp", n=1):
        """
        viz_type: "hp" for "hardest pos", "ep" for "easiest pos" and so on
        n: viz number
        """
        dist_p = self.dist_all[self.labels > 0.5]
        dist_n = self.dist_all[self.labels < 0.5]

        idx_p = np.argsort(dist_p)
        idx_n = np.argsort(dist_n)

        plt.title("Learners' distance prediction " + viz_type)
        if viz_type == "hp":
            idx_c = idx_p[-n:]
            dist = dist_p
        elif viz_type == "hn":
            idx_c = idx_n[:n]
            dist = dist_n
        elif viz_type == "ep":
            idx_c = idx_p[:n]
            dist = dist_p
        elif viz_type == "en":
            idx_c = idx_n[-n:]
            dist = dist_n
        else:
            pass

        # for i in range(len(self.split_ratios)):
        #     plt.plot(range(n), dist[i][idx_c], label="learner" + str(i))

        plt.plot(range(n), dist[idx_c], color="black", label="all")
        plt.legend()

        plt.xlabel("index")
        plt.ylabel("distance")
        plt.show()

    def calc_hist(self):
        for i in range(np.size(self.descs_a, axis=1)):
            # hist, bins = np.histogram(np.concatenate([self.descs_a, self.descs_b], axis=0)[:, i], bins=50, density=True)
            plt.hist(np.concatenate([self.descs_a, self.descs_b], axis=0)[:, i], bins=50)
            plt.savefig("hist" + str(i) + ".png")
            plt.close()

    def calc_l2_hist(self):
        pos_dist = self.dist[0][self.labels > 0.5]
        plt.hist(pos_dist, bins=50, range=(0, 2), weights=np.ones_like(pos_dist) / len(pos_dist))
        plt.savefig("l2_dist_pos.png")
        plt.close()
        neg_dist = self.dist[0][self.labels < 0.5]
        plt.hist(neg_dist, bins=50, range=(0, 2), weights=np.ones_like(neg_dist) / len(neg_dist))
        plt.savefig("l2_dist_neg.png")
        plt.close()

    def calc_hard_l2_hist(self):
        # combining every 1000 patches to one matching block
        labels = self.labels[self.labels == True]
        descs1 = self.descs_a[self.labels == True]
        descs2 = self.descs_b[self.labels == True]
        num_block = len(self.labels[self.labels == True]) // 1000
        pos_dist = np.zeros(num_block * 1000)
        neg_dist = np.zeros(num_block * 1000)
        for i in range(num_block):
            desc1 = descs1[i * 1000:(i + 1) * 1000, :]
            desc2 = descs2[i * 1000:(i + 1) * 1000, :]
            dmat = tlib.compute_distance_matrix_l2(desc1, desc2)
            pos, neg = tlib.find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)
            pos_dist[i * 1000:(i + 1) * 1000] = pos
            neg_dist[i * 1000:(i + 1) * 1000] = neg
        plt.hist(pos_dist, bins=50, range=(0, 2), weights=np.ones_like(pos_dist) / len(pos_dist))
        plt.savefig("l2_dist_pos.png")
        plt.close()
        plt.hist(neg_dist, bins=50, range=(0, 2), weights=np.ones_like(neg_dist) / len(neg_dist))
        plt.savefig("l2_dist_neg_hard.png")
        plt.close()

    # def calc_dist_p2n_hist(self):
    #     # combining every 1000 patches to one matching block
    #     labels = self.labels[self.labels == True]
    #     descs1 = self.descs_a[self.labels == True]
    #     descs2 = self.descs_b[self.labels == True]
    #     num_block = len(self.labels[self.labels == True]) // 1000
    #     dist_p2n = np.zeros(num_block*1000)
    #     for i in range(num_block):
    #         desc1 = descs1[i * 1000:(i + 1) * 1000, :]
    #         desc2 = descs2[i * 1000:(i + 1) * 1000, :]
    #         dmat = tlib.compute_distance_matrix_l2(desc1, desc2)
    #         pos_mat1 = dmat.diag().repeat(1000, axis=1)
    #         pos_mat2 = pos_mat1.transpose([0, 1])
    #         pos_dist[i * 1000:(i + 1) * 1000] = pos
    #         neg_dist[i * 1000:(i + 1) * 1000] = neg
    #         plt.hist(pos_dist, bins=50, range=(0, 2), weights=np.ones_like(pos_dist) / len(pos_dist))
    #         plt.savefig("l2_dist_pos.png")
    #         plt.close()
    #         plt.hist(neg_dist, bins=50, range=(0, 2), weights=np.ones_like(neg_dist) / len(neg_dist))
    #         plt.savefig("l2_dist_neg_hard.png")
    #         plt.close()
