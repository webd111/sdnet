import numpy
import torch
import numpy as np
from typing import List


def compute_distance_matrix_unit_l2(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes pairwise Euclidean distance for NORMALIZED descriptors and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dmat: N * N, matrix
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dis_mat = torch.matmul(a, torch.transpose(b, 0, 1))
    dis_mat = (((1.0 - dis_mat).clamp(0) + eps) * 2.0).pow(0.5)
    dis_mat[torch.isnan(dis_mat)] = 0

    if is_numpy:
        dis_mat = dis_mat.numpy()
    return dis_mat


def compute_distance_matrix_l2(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    normalizes descriptor and computes pairwise Euclidean distance and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dmat: N * N, matrix
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dis_mat = torch.matmul(a, torch.transpose(b, 0, 1))
    dis_aa = torch.mul(a, a).sum(dim=1)
    dis_bb = torch.mul(b, b).sum(dim=1)
    # Using boardcast
    dis_mat = torch.sqrt((dis_aa + dis_bb - 2 * dis_mat).clamp(0) + eps)
    dis_mat[torch.isnan(dis_mat)] = 0

    if is_numpy:
        dis_mat = dis_mat.numpy()

    return dis_mat


def compute_distance_matrix_unit_hybrid(a, p, alpha, eps=1e-12):
    assert type(a) == type(p), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        p = torch.from_numpy(p)

    # inner product matrix
    si = torch.mm(a, torch.transpose(p, 0, 1))

    # L2 matrix
    sl = torch.sqrt(2 * (1 - si) + eps)

    # hybrid similarity matrix
    sh = (alpha * (1 - si) + sl)

    if is_numpy:
        sh = sh.numpy()
    return sh


# TODO: NEED VALIDATION
def compute_distance_matrix_hybrid(a, b, alpha, eps=1e-12):
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    # inner product matrix
    si = torch.mm(a, torch.transpose(b, 1, 0))

    dis_aa = torch.mul(a, a).sum(dim=1).expand_as(si)
    dis_bb = torch.mul(b, b).sum(dim=1).expand_as(si).transpose(1, 0)

    # L2 matrix
    sl = torch.sqrt(dis_aa + dis_bb - 2 * si + eps)

    # hybrid similarity matrix
    sh = (alpha * (1 - si) + sl)

    if is_numpy:
        sh = sh.numpy()
    return sh


def compute_distance_matrix_scalar_hybrid(a, b, alpha, eps=1e-12):
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    # inner product matrix
    si = torch.mm(a.reshape(a.shape[0], 1), b.reshape(1, b.shape[0]))

    dis_aa = torch.mul(a, a).expand_as(si).transpose(1, 0)
    dis_bb = torch.mul(b, b).expand_as(si)

    # L2 matrix
    sl = torch.sqrt(dis_aa + dis_bb - 2 * si + eps)

    # hybrid similarity matrix
    sh = (alpha * (1 - si) + sl)

    if is_numpy:
        sh = sh.numpy()
    return sh


def compute_distance_unit_hybrid(a, p, alpha, eps=1e-12):
    # Be aware of the fact that when using backward, the gradient calculated
    # may be different with compute_distance_hybrid before normalization layer.
    # But if gradients backward goes through normalization, the gradient will be the same.
    assert type(a) == type(p), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        p = torch.from_numpy(p)

    # inner product matrix
    si = torch.sum(torch.mul(a, p), dim=1)

    # L2 vector
    sl = torch.sqrt(2 * (1 - si) + eps)

    # hybrid similarity vector
    sh = (alpha * (1 - si) + sl)

    if is_numpy:
        sh = sh.numpy()
    return sh


def compute_distance_hybrid(a, p, alpha, eps=1e-12):
    assert type(a) == type(p), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        p = torch.from_numpy(p)

    # inner product matrix
    si = torch.sum(torch.mul(a, p), dim=1)
    dif_ap = a - p

    # L2 vector
    sl = torch.sqrt(torch.sum(torch.mul(dif_ap, dif_ap), dim=1) + eps)

    # hybrid similarity vector
    sh = (alpha * (1 - si) + sl)

    if is_numpy:
        sh = sh.numpy()
    return sh


def compute_distance_unit_l2(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes corresponding Euclidean distance between two UNIT descriptors and return a N x 1 scalar
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dis_l2: N * 1, scalar
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dis_dot = torch.sum(torch.mul(a, b), dim=1)
    dis_l2 = (((1.0 - dis_dot).clamp(0)) * 2.0 + eps).pow(0.5)
    dis_l2[torch.isnan(dis_l2)] = 0

    if is_numpy:
        dis_l2 = dis_l2.numpy()
    return dis_l2


def compute_distance_l2(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes corresponding Euclidean distance between descriptors and return a N x 1 vector
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dis_l2: N * 1, vector
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dis_dot = torch.sum(torch.mul(a, b), dim=1)
    dis_aa = torch.mul(a, a).sum(dim=1)
    dis_bb = torch.mul(b, b).sum(dim=1)
    dis_l2 = ((dis_aa + dis_bb - 2 * dis_dot).clamp(0) + eps).pow(0.5)
    dis_l2[torch.isnan(dis_l2)] = 0

    if is_numpy:
        dis_l2 = dis_l2.numpy()
    return dis_l2


def compute_similarity_dot(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes corresponding dot similarity between descriptors and return a N x 1 scalar
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    sim_cos: N * 1, vector
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    sim_dot = torch.sum(torch.mul(a, b), dim=1)

    if is_numpy:
        sim_dot = sim_dot.numpy()

    return sim_dot


def compute_similarity_matrix_dot(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes Dot similarity matrix and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dot_mat: N * N, matrix
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dot_mat = torch.matmul(a, torch.transpose(b, 0, 1))

    if is_numpy:
        dot_mat = dot_mat.numpy()
    return dot_mat


def compute_similarity_cosine(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes corresponding Cosine distance between descriptors and return a N x 1 scalar
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    sim_cos: N * 1, vector
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    sim_dot = torch.sum(torch.mul(a, b), dim=1)
    norm_aa = torch.norm(a, p=2, dim=1)
    norm_bb = torch.norm(b, p=2, dim=1)
    sim_cos = sim_dot / (norm_aa * norm_bb)

    if is_numpy:
        sim_cos = sim_cos.numpy()

    return sim_cos


def compute_similarity_matrix_cosine(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    """
    computes corresponding Cosine distance matrix between descriptors and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    sim_cos: N * N, vector
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    a = a.unsqueeze(0).expand([a.shape[0], a.shape[0], a.shape[1]])
    b = b.unsqueeze(1).expand([b.shape[0], b.shape[0], b.shape[1]])
    smat = torch.cosine_similarity(a, b, dim=2)
    if is_numpy:
        smat = smat.numpy()

    return smat


def compute_similarity_kernel_matrix(a: torch.Tensor, b: torch.Tensor, kernel_type, params, eps=1e-12):
    """
    computes corresponding kernel distance between descriptors and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    kernel_type: str
    params: for rbf kernel, params[0] is sigma
    OUTPUT:
    sim: N * 1, scalar
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    sim_dot_mat = torch.matmul(a, torch.transpose(b, 1, 0))
    if kernel_type == "rbf":
        sigma = params[0]
        sim_kernel = torch.exp((sim_dot_mat - 1) / (sigma * sigma))
    elif kernel_type == "linear":
        sim_kernel = sim_dot_mat
    else:
        raise NotImplementedError

    if is_numpy:
        sim_kernel = sim_kernel.numpy()

    return sim_kernel


def compute_similarity_kernel(a: torch.Tensor, b: torch.Tensor, kernel_type, params, eps=1e-12):
    """
    computes corresponding kernel distance between descriptors and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    kernel_type: str
    params: for rbf kernel, params[0] is sigma
    OUTPUT:
    sim: N * 1, scalar
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    sim_dot = torch.mul(a, b).sum(dim=1)
    if kernel_type == "rbf":
        sigma = params[0]
        sim_kernel = torch.exp((sim_dot - 1) / (sigma * sigma))
    elif kernel_type == "linear":
        sim_kernel = sim_dot
    else:
        raise NotImplementedError

    if is_numpy:
        sim_kernel = sim_kernel.numpy()

    return sim_kernel


def compute_distance_matrix_hamming(a, b):
    """
    computes pairwise Hamming distance and return a N x N matrix
    INPUT:
    a: N * dim, matrix
    b: N * dim, matrix
    OUTPUT:
    dis_l2: N * N, matrix
    """
    assert type(a) == type(b), "a, b should be same type."
    is_numpy = (type(a) == numpy.ndarray)
    if is_numpy:
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    dims = a.size(1)
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = (dims - dmat) * 0.5

    if is_numpy:
        dmat = dmat.numpy()
    return dmat
