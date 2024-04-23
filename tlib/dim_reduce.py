import numpy
import torch
import numpy as np
from typing import List

import tlib


def pca(data: torch.Tensor, k2, k1=0, center=True, return_eigen=False, eps=1e-12):
    """
    perform PCA Dimensionality reduction
    INPUT:
    data: N * dim, matrix
    k2: k end index
    k1: k start index, default 0
    center: bool, whether original data needs centralization
    return_eigen: bool, whether func return eigenvalues and eigenvectors
    OUTPUT:
    desc_dim_reduce: N * reduced_dim, matrix
    eigenvalue: dim, vector
    eigenvector: dim(vector nums) * dim(original dim), matrix
    """
    is_numpy = (type(data) == numpy.ndarray)
    if is_numpy:
        data = torch.from_numpy(data)

    with torch.no_grad():
        N = data.size()[0]
        dim = data.size()[1]
        k1 = torch.tensor(k1, dtype=torch.int64)
        k2 = torch.tensor(k2, dtype=torch.int64)

        if center:
            data_mean = torch.mean(data, dim=0)
            data = data - data_mean.expand_as(data)

        # data = U*Diag(S)*V^T
        # data * data^T = U * Diag(S*S) * U^T
        U, S, V = torch.svd(torch.t(data))
        desc_dim_reduce = torch.mm(data, U[:, k1:k2])

        if return_eigen:
            eigenvalue = torch.sqrt(S)
            eigenvector = torch.transpose(U, 1, 0)

            # # for validation
            # data_proj = torch.transpose(torch.mm(eigenvector, torch.transpose(data, 1, 0)), 1, 0)
            # data_mat = tlib.compute_distance_matrix_l2(data[:N//2, :], data[N//2:N, :])
            # data_mat_proj = tlib.compute_distance_matrix_l2(data_proj[:N // 2, :], data_proj[N // 2:N, :])
            # print((data_mat - data_mat_proj).max().max())

            if is_numpy:
                desc_dim_reduce = desc_dim_reduce.numpy()
                eigenvalue = eigenvalue.numpy()
                eigenvector = eigenvector.numpy()

            return desc_dim_reduce, eigenvalue, eigenvector

        if is_numpy:
            desc_dim_reduce = desc_dim_reduce.numpy()

    return desc_dim_reduce

