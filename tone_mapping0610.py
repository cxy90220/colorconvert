"""
参考论文：High dynamic range image tone mapping based on asymmetric model of retinal adaptation
"""

import numpy as np
import cv2
import math


def tp(img, n=1000, lamb=400, gamma=50):
    row, col, dim = img.shape
    cin_r = img[:, :, 0]
    cin_g = img[:, :, 1]
    cin_b = img[:, :, 2]
    lin = 0.27 * cin_r + 0.67 * cin_g + 0.06 * cin_b
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
    for i in range(row):
        for j in range(col):
            lout[i, j] = ftm[int(x[i, j])]
            cin = img[i, j, :]
            s = 1 - cin.min() / cin.max()
            cout = ((cin / lin[i, j] - 1) * s + 1) * lout[i, j]
            for k in range(dim):
                if cout[k] > 0.0031308:
                    ldr[i, j, k] = (cout[k] ** (1.0 / 2.4) - 0.05214) / 0.9479
                else:
                    ldr[i, j, k] = cout[k]
    return ldr


if __name__ == '__main__':
    imghdr = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    result = tp(imghdr)
    result = (np.clip(result * 255.0, 0, 255)).astype('uint8')
    cv2.imshow('ldr', result)
    cv2.waitKey(0)
