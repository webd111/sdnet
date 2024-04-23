import numpy as np
import cv2
import skimage.feature


def calc_principle_direction(patches, ):
    pds = []
    for p in patches:
        p = p.numpy().squeeze()
        hist = skimage.feature.hog(p, orientations=36, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)
        pd = np.argmax(hist) * 10
        pds += pd.tolist()
    pds = np.stack(pds)

    return pds

