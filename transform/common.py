# From original author
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor, Resize, ToPILImage
import torch.nn.functional as F

import tlib


def get_basic_input_transform(patch_size, mean, std):
    cv2_scale = (lambda x: cv2.resize(x, dsize=(patch_size, patch_size), interpolation=cv2.INTER_LINEAR) if x.shape[0] != patch_size else x.copy())

    def np_reshape(x): return np.reshape(x, (patch_size, patch_size, 1))

    tforms = [
        Lambda(cv2_scale),
        Lambda(np_reshape),
        ToTensor(),
        Normalize((mean,), (std,)),
    ]
    tforms = Compose(tforms)
    return tforms


def get_input_transform_no_norm(patch_size):
    cv2_scale = (lambda x: cv2.resize(x, dsize=(patch_size, patch_size), interpolation=cv2.INTER_LINEAR) if x.shape[0] != patch_size else x.copy())

    def np_reshape(x): return np.reshape(x, (patch_size, patch_size, 1))

    tforms = [Lambda(cv2_scale),
              Lambda(np_reshape),
              ToTensor()]
    tforms = Compose(tforms)
    return tforms


def get_input_transform_no_norm_torch(patch_size):
    def np_reshape(x): return np.reshape(x, (patch_size, patch_size, 1))

    tforms = [Lambda(np_reshape),
              Resize(patch_size),
              ToTensor()]
    tforms = Compose(tforms)
    return tforms


def get_input_transform_sdgm(patch_size):
    cv2_scale = (lambda x: cv2.resize(x, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) if x.shape[0] != patch_size else x.copy())

    def np_reshape64(x): return np.reshape(x, (64, 64, 1))

    tforms = [Lambda(cv2_scale),
              Lambda(np_reshape64),
              ToPILImage(),
              Resize(patch_size),
              ToTensor()]
    tforms = Compose(tforms)
    return tforms


def get_input_transform_norm_1(patch_size):
    cv2_scale = (lambda x: cv2.resize(x, dsize=(patch_size, patch_size), interpolation=cv2.INTER_LINEAR) if x.shape[0] != patch_size else x.copy())

    def np_reshape(x): return np.reshape(x, (patch_size, patch_size, 1))

    def torch_unsqueeze(x): return torch.unsqueeze(x, dim=0)

    def torch_squeeze(x): return torch.squeeze(x, dim=0)

    tforms = [Lambda(cv2_scale),
              Lambda(np_reshape),
              ToTensor(),
              Lambda(torch_unsqueeze),
              Lambda(tlib.torch_input_norm_1),
              Lambda(torch_squeeze), ]
    tforms = Compose(tforms)
    return tforms


def get_transform_tensor_c2b():
    def tensor_channel_to_batchsize(x): return torch.reshape(x, [-1, 1, x.shape[2], x.shape[3]])

    tforms = [Lambda(tensor_channel_to_batchsize), ]
    tforms = Compose(tforms)
    return tforms


def compute_multi_dataset_mean_std(data_lens, data_mean, data_std):
    N = float(sum(data_lens))

    avg_mean = 0
    for i, v in enumerate(data_mean):
        avg_mean += float(data_lens[i]) * v
    avg_mean /= N

    TGSS = 0
    ESS = 0
    for i, v in enumerate(data_std):
        TGSS += (data_mean[i] - avg_mean) ** 2.0 * float(data_lens[i])
        ESS += data_std[i] ** 2.0 * float(data_lens[i] - 1)

    avg_std = ((ESS + TGSS) / (N - 1)) ** 0.5

    return avg_mean, avg_std


# TODO: This module has some bugs.
def log_polar_transform(patch_size, mean, std):
    cv2_scale = (
        lambda x: cv2.resize(
            x, dsize=(patch_size, patch_size), interpolation=cv2.INTER_LINEAR
        )
        if x.shape[0] != patch_size
        else x.copy()
    )

    # Polar transformer network forward function
    def log_polar_sampling():
        maxR = np.floor(patch_size / 2)

        # get [self.batchSize x self.resolution x self.resolution x 2] grids with values in [-1,1],
        # define grids or call torch function and apply unit transform
        ident = torch.from_numpy(
            np.array(1 * [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        grid = F.affine_grid(ident, torch.Size((1, 1, patch_size, patch_size)))
        grid_y = grid[:, :, :, 0].view(1, -1)
        grid_x = grid[:, :, :, 1].view(1, -1)

        maxR = torch.unsqueeze(torch.tensor(
            maxR), -1).expand(1, grid_y.shape[-1])

        # get radius of polar grid with values in [1, maxR]
        normGrid = (grid_y + 1) / 2
        r_s_ = torch.exp(normGrid * torch.log(maxR))

        # convert radius values to [0, 2maxR/W] range
        r_s = (r_s_ - 1) / (maxR - 1)

        # y is from -1 to 1; theta is from 0 to 2pi
        # tmin_threshold_distance_s equals \frac{2 pi y^t}{H} in eq (9-10)
        t_s = (grid_x + 1) * torch.Tensor([np.pi])

        # see eq (9) : theta[:,0] shifts each batch entry by the kp's x-coords
        x_s = r_s * torch.cos(t_s)
        # see eq (10): theta[:,1] shifts each batch entry by the kp's y-coords
        y_s = r_s * torch.sin(t_s)

        # tensorflow grid is of shape [self.batchSize x 3 x self.resolution**2],
        # pytorch grid is of shape [self.batchSize x self.resolution x self.resolution x 2]
        # x_s and y_s are of shapes [1 x self.resolution**2]
        # bilinear interpolation in tensorflow takes _interpolate(input_dim,
        # x_s_flat, y_s_flat, out_size)

        # reshape polar coordinates to square tensors and append to obtain
        # [self.batchSize x self.resolution x self.resolution x 2] grid
        polargrid = torch.cat(
            (x_s.view(1, patch_size, patch_size, 1),
             y_s.view(1, patch_size, patch_size, 1)),
            -1).float()

        return polargrid

    def apply_sampling(x):
        x = x.view(1, 1, patch_size, patch_size)
        x = F.grid_sample(
            x, sampling_grid
        )  # do bilinear interpolation to sample values on the grid
        x = x.view(1, patch_size, patch_size)

        return x

    sampling_grid = log_polar_sampling()

    tforms = [
        Lambda(cv2_scale),
        ToTensor(),
        Lambda(apply_sampling),
        Normalize((mean,), (std,)),
    ]
    tforms = Compose(tforms)

    return tforms
