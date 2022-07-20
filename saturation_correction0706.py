"""
参考论文：Automatic saturation correction for dynamic range management algorithms
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
    for i in range(row):
        for j in range(col):
            lout[i, j] = ftm[int(x[i, j])]
            cin = img[i, j, :]
            ldr[i, j, :] = cin / lin[i, j] * lout[i, j]
    return ldr


def rgb2ipt(img):
    row, col, dim = img.shape
    ipt = np.zeros((row, col, dim))
    mat = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    for i in range(row):
        for j in range(col):
            rgb = img[i, j, :]
            xyz = np.dot(mat, rgb)
            l = 0.4002 * xyz[0] + 0.7075 * xyz[1] - 0.0807 * xyz[2]
            if l > 0:
                l = l ** 0.43
            else:
                l = - (- l) ** 0.43
            m = -0.2280 * xyz[0] + 1.1500 * xyz[1] + 0.0612 * xyz[2]
            if m > 0:
                m = m ** 0.43
            else:
                m = - (- m) ** 0.43
            s = 0.9184 * xyz[2]
            if s > 0:
                s = s ** 0.43
            else:
                s = - (- s) ** 0.43
            ipt[i, j, 0] = (0.4000 * l + 0.4000 * m + 0.2000 * s) * 100
            ipt[i, j, 1] = (4.4550 * l - 4.8510 * m + 0.3960 * s) * 150
            ipt[i, j, 2] = (0.8056 * l + 0.3572 * m - 1.1628 * s) * 150
    return ipt


def ipt2rgb(img):
    row, col, dim = img.shape
    rgb = np.zeros((row, col, dim))
    img[:, :, 0] = img[:, :, 0] / 100
    img[:, :, 1] = img[:, :, 1] / 150
    img[:, :, 2] = img[:, :, 2] / 150
    for i in range(row):
        for j in range(col):
            l = 1 * img[i, j, 0] + 0.097569 * img[i, j, 1] + 0.205226 * img[i, j, 2] 
            if l < 0:
                l = - (-l) ** (1.0 / 0.43)
            else:
                l = l ** (1.0 / 0.43)
            m = 1 * img[i, j, 0] - 0.113876 * img[i, j, 1] + 0.133217 * img[i, j, 2]  
            if m < 0:
                m = - (-m) ** (1.0 / 0.43)
            else:
                m = m ** (1.0 / 0.43)
            s = 1 * img[i, j, 0] + 0.032615 * img[i, j, 1] - 0.676887 * img[i, j, 2] 
            if s < 0:
                s = - (-s) ** (1.0 / 0.43)
            else:
                s = s ** (1.0 / 0.43)
            x = (1.850243 * l - 1.138302 * m + 0.238435 * s)
            y = (0.366831 * l + 0.643885 * m - 0.010673 * s)
            z = 1.088850 * s
            mat = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
            rgb[i, j, :] = np.dot(mat, [x, y, z])
    return rgb


def sc(mo, mt):
    row, col, dim = mo.shape
    ipto = rgb2ipt(mo)
    iptt = rgb2ipt(mt)
    ipt = np.zeros((row, col, dim))
    ipt[:, :, 0] = iptt[:, :, 0]
    for i in range(row):
        for j in range(col):
            io = ipto[i, j, 0]
            it = iptt[i, j, 0]
            co = (ipto[i, j, 1] ** 2.0 + ipto[i, j, 2] ** 2.0) ** 0.5
            ct = (iptt[i, j, 1] ** 2.0 + iptt[i, j, 2] ** 2.0) ** 0.5
            ct_t = ct * io / it
            so = co / (co ** 2.0 + io ** 2.0) ** 0.5
            st = ct_t / (ct_t ** 2.0 + it ** 2.0) ** 0.5
            r = so / st * io / it
            if r > 1:
                d = min(min(mt[i, j, 0], 1-mt[i, j, 0]), min(mt[i, j, 1], 1-mt[i, j, 1]), min(mt[i, j, 2], 1-mt[i, j, 2]))
                if d < 0:
                    d = 0
                d = (2 * d) / (2 * d + 0.01)
                cc = ct * (d * r + 1 - d)
            else:
                cc = ct * r
            ipt[i, j, 1] = cc * ipto[i, j, 1] / co
            ipt[i, j, 2] = cc * ipto[i, j, 2] / co
    ldr = ipt2rgb(ipt)
    for i in range(row):
        for j in range(col):
            for k in range(dim):
                if ldr[i, j, k] > 0.0031308:
                    ldr[i, j, k] = 1.055 * (ldr[i, j, k] ** (1.0 / 2.2)) - 0.055
                else:
                    ldr[i, j, k] = 12.92 * ldr[i, j, k]
    return ldr

if __name__ == '__main__':
    imgo = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    imgo = imgo / imgo.max()
    imgt = tp(imgo)
    result = sc(imgo, imgt)
    result = (np.clip(result * 255.0, 0, 255)).astype('uint8')
    # cv2.imwrite('en_contrast.png', result)
    cv2.imshow('ldr', result)
    cv2.waitKey(0)