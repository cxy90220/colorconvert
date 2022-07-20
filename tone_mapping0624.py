"""
参考论文：Tone mapping high dynamic 3D scenes with global lightness coherency
"""

import numpy as np
import cv2
import math


def tp(img):
    row, col, dim = img.shape
    cin_r = img[:, :, 0]
    cin_g = img[:, :, 1]
    cin_b = img[:, :, 2]
    lin = 0.27 * cin_r + 0.67 * cin_g + 0.06 * cin_b
    lin_log = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            lin_log[i, j] = math.log(1e-10 + lin[i, j])
    l_mean = math.exp(lin_log.sum() / (row * col))
    lout = np.zeros((row, col))
    ldr = np.zeros((row, col, dim))
    for i in range(row):
        for j in range(col):
            lout[i, j] = 1 - 1 / (1 + lin[i, j] * 0.18 / l_mean)
            cin = img[i, j, :]
            s = 1 - cin.min() / cin.max()
            cout = ((cin / lin[i, j] - 1) * s + 1) * lout[i, j]
            for k in range(dim):
                ldr[i, j, k] = cout[k] ** (1.0 / 2.2)
    return ldr


if __name__ == '__main__':
    imghdr = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    result = tp(imghdr)
    result = (np.clip(result * 255.0, 0, 255)).astype('uint8')
    cv2.imwrite('hmd_tmo0624.png', result)
    # cv2.imshow('ldr', result)
    cv2.waitKey(0)