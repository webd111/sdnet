import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calc_iou(theta, patch_size):
    # slightly different from grid_sample result for interpolation mode
    with torch.no_grad():
        grid = F.affine_grid(theta, [theta.shape[0], 1, patch_size, patch_size], align_corners=True)
        grid_mask = (grid >= -1-1/patch_size) & (grid <= 1+1/patch_size)    # consider nearest interpolation mode
        grid_valid = grid_mask[:, :, :, 0] * grid_mask[:, :, :, 1]
        iou = torch.sum(torch.sum(grid_valid, dim=2), dim=1) / (patch_size * patch_size)
    return iou


# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


if __name__ == "__main__":
    # test calc_iou()
    img = torch.ones([65, 65], dtype=torch.float)
    # img = torch.zeros([65, 65])
    # img_x = torch.arange(0, 65, dtype=torch.float).expand([65, 65])
    # img_y = torch.arange(0, 65, dtype=torch.float).unsqueeze(-1).expand([65, 65])
    # img = img_x + img_y + 1
    img = img.expand([6, 1, 65, 65])

    with torch.no_grad():
        theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
        theta = theta.expand(6, 2, 3).clone()
        theta[1:6, :, :] += torch.rand([5, 2, 3]) / 10
        grid = F.affine_grid(theta, [6, 1, 65, 65], align_corners=True)
        img_trans = F.grid_sample(img, grid.float(), mode="nearest", padding_mode="zeros", align_corners=True)
        iou = calc_iou(theta, patch_size=65)
        iou_test = torch.sum(torch.sum(img_trans == 1, dim=3), dim=2).squeeze() / (65*65)
        # imgs = img[0, 0, :, :].numpy()
        # for i in range(6):
        #     imgs = np.concatenate([imgs, img_trans[i, 0, :, :].numpy()], axis=0)
        # plt.figure()
        # plt.imshow(imgs)
        # plt.show()
        print(theta)
        print(iou)
        print(iou_test)

