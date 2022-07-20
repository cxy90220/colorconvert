"""
参考论文：Color correction for tone mapping
"""

import numpy as np
import cv2
import math


def tp(img, n=1000, lamb=400, gamma=50):
    row, col, dim = img.shape
    # cin_r = img[:, :, 0]
    # cin_g = img[:, :, 1]
    # cin_b = img[:, :, 2]
    # lin = 0.27 * cin_r + 0.67 * cin_g + 0.06 * cin_b
    img = img / img.max()
    lab = rgb2lab(img)
    lin = lab[:, :, 0] / 100.0
    l_median = np.median(lin)
    l_mean = lin.mean()
    l_max = lin.max()
    l_min = lin.min()
    l_avg = (l_median ** 0.5) * (l_mean ** 0.5)
    k = (2 * math.log(l_avg, 2) - math.log(l_max, 2) - math.log(l_min, 2)) / (math.log(l_max, 2) - math.log(l_min, 2))
    alpha = 0.18 * 4 ** k
    b = - math.log(1 - alpha, 2)
    # ftm = np.zeros((row, col))
    # for i in range(row):
    #     for j in range(col):
    #         ftm[i, j] = 1 - (l_avg / (lin[i, j] + l_avg)) ** b
    # lout = ftm  # asc
    # ldr = np.zeros((row, col, dim))
    # for i in range(row):
    #     for j in range(col):
    #         cin = img[i, j, :]
    #         s = 1 - cin.min() / cin.max()
    #         cout = ((cin / lin[i, j] - 1) * s + 1) * lout[i, j]
    #         for k in range(dim):
    #             if cout[k] > 0.0031308:
    #                 ldr[i, j, k] = 1.055 * (cout[k] ** (1.0 / 2.2)) - 0.055
    #             else:
    #                 ldr[i, j, k] = 12.92 * cout[k]
    # return ldr
    log_l = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            log_l[i, j] = math.log(lin[i, j])
    log_l_max = log_l.max()
    log_l_min = log_l.min()
    ftm_discret = np.zeros(n)
    log_l_discret = np.zeros(n)
    for i in range(n):
        log_l_discret[i] = log_l_min + i * (log_l_max - log_l_min) / (n-1)
        ftm_discret[i] = 1 - (l_avg / (math.exp(log_l_discret[i]) + l_avg)) ** b
    hx = np.zeros(n)
    x = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            judge = True
            for k in range(n - 1):
                if log_l_discret[k] <= log_l[i, j] < log_l_discret[k + 1]:
                    hx[k] = hx[k] + 1
                    x[i, j] = k
                    judge = False
                    break
            if judge:
                hx[n - 1] = hx[n - 1] + 1
                x[i, j] = n - 1
    d = np.eye(n) - np.eye(n, k=-1)
    k_lambda = np.linalg.inv(np.eye(n) + lamb * np.dot(d.T, d))
    k_gamma = np.linalg.inv(np.eye(n) + gamma * np.dot(d.T, d))
    hx_normal = hx / hx.sum()
    ftm = np.dot(k_lambda, ftm_discret + lamb * np.dot(np.dot(d.T, k_gamma), hx_normal))
    lout = np.zeros((row, col))
    ldr = np.zeros((row, col, dim))
    k1, k2 = 38.7889, 1.5856
    for i in range(row):
        for j in range(col):
            lout[i, j] = ftm[int(x[i, j])]
            c = (l_avg ** b) * b / ((lin[i, j] + l_avg) ** (b + 1)) * lin[i, j] / lout[i, j]
            s = (1.0 + k1) * (c ** k2) / (1.0 + k1 * (c ** k2))
            lab[i, j, 0] = lout[i, j] * 100.0
            lab[i, j, 1] = lab[i, j, 1] * s * lout[i, j] / lin[i, j]
            lab[i, j, 2] = lab[i, j, 2] * s * lout[i, j] / lin[i, j]
    rgb = lab2rgb(lab)
    for i in range(row):
        for j in range(col):
            for k in range(dim):
                ldr[i, j, k] = rgb[i, j, k] ** (1.0 / 2.2)
    return ldr


def rgb2lab(rgb):
    row, col, dim = rgb.shape
    lab = np.zeros((row, col, dim))
    mat = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    xyzn = [0.950489, 1.0, 1.08840]
    for i in range(row):
        for j in range(col):
            fxyz = np.zeros(dim)
            xyz = np.dot(mat, rgb[i, j, :])
            for k in range(dim):
                xyz[k] = xyz[k] / xyzn[k]
                if xyz[k] > 0.008856:
                    fxyz[k] = xyz[k] ** 0.33333
                else:
                    fxyz[k] = xyz[k] * 7.787 + 0.13793
            lab[i, j, 0] = 116.0 * fxyz[1] - 16.0
            lab[i, j, 1] = 500.0 * (fxyz[0] - fxyz[1])
            lab[i, j, 2] = 200.0 * (fxyz[1] - fxyz[2])
    return lab


def lab2rgb(lab):
    row, col, dim = lab.shape
    rgb = np.zeros((row, col, dim))
    mat = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
    xyzn = [0.950489, 1.0, 1.08840]
    for i in range(row):
        for j in range(col):
            fxyz = np.zeros(dim)
            xyz = np.zeros(dim)
            fxyz[1] = (lab[i, j, 0] + 16.0) / 116.0
            fxyz[0] = lab[i, j, 1] / 500.0 + fxyz[1]
            fxyz[2] = fxyz[1] - lab[i, j, 2] / 200.0
            for k in range(dim):
                if fxyz[k] > 0.206893:
                    xyz[k] = fxyz[k] ** 3.0
                else:
                    xyz[k] = (fxyz[k] - 0.13793) / 7.787
                xyz[k] = xyz[k] * xyzn[k]
            rgb[i, j, :] = np.dot(mat, xyz)
    return rgb


if __name__ == '__main__':
    imgraw = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    result = tp(imgraw)
    result = (np.clip(result * 255.0, 0, 255)).astype('uint8')
    # cv2.imwrite('en_contrast.png', result)
    cv2.imshow('ldr', result)
    cv2.waitKey(0)
