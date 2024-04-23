import torch

def torch_input_norm_1(x):
    flat = x.view(x.size(0), -1)
    # min
    min_v = flat.min(dim=1)[0]
    min_v = min_v.reshape(x.shape[0], 1)
    min_v = min_v.repeat(1, x.shape[2] * x.shape[3])
    min_v = min_v.reshape(x.shape)
    # max
    max_v = flat.max(dim=1)[0]
    max_v = max_v.reshape(x.shape[0], 1)
    max_v = max_v.repeat(1, x.shape[2] * x.shape[3])
    max_v = max_v.reshape(x.shape)
    # mapminmax
    res = 2 * (x - min_v) / (max_v - min_v) - 1
    res[torch.isnan(res)] = 0.5
    return res