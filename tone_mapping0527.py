"""
参考论文：HDR图像色调映射的自适应色彩调节算法
"""

import numpy as np
import cv2
import math


def color_enhance(r, g, b):
    r_g = r / g
    b_g = b / g
    y = 2.73 * r_g * r_g - 4.76 * r_g + 2.91
    if y - 0.2 < b_g < y + 0.2:
        return True
    else:
        return False


def tp(img):
    var_r = img[:, :, 0]
    var_g = img[:, :, 1]
    var_b = img[:, :, 2]
    row, col, dim = img.shape
    lin = 0.299 * var_r + 0.587 * var_g + 0.114 * var_b
    lout = np.zeros((row, col))
    d = np.zeros((row, col))
    s = np.zeros((row, col))
    for m in range(row):
        for n in range(col):
            lout[m, n] = 1 * math.log(1 + lin[m, n], 2)
            d[m, n] = lout[m, n] / lin[m, n]
    dmax = np.max(d)
    d = d / dmax
    for m in range(row):
        for n in range(col):
            s[m, n] = math.log(d[m, n], 10) * 0.2 + 1
            var_r[m, n] = math.pow(var_r[m, n] / lin[m, n], s[m, n]) * lout[m, n]
            var_g[m, n] = math.pow(var_g[m, n] / lin[m, n], s[m, n]) * lout[m, n]
            var_b[m, n] = math.pow(var_b[m, n] / lin[m, n], s[m, n]) * lout[m, n]
    rij, gij, bij = 0, 0, 0
    for m in range(row):
        for n in range(col):
            if color_enhance(var_r[m, n], var_g[m, n], var_b[m, n]):
                rij = rij + var_r[m, n]
                gij = gij + var_g[m, n]
                bij = bij + var_b[m, n]
    kr = (rij + gij + bij) / (3 * rij)
    kg = (rij + gij + bij) / (3 * gij)
    kb = (rij + gij + bij) / (3 * bij)
    var_r = var_r * kr
    var_g = var_g * kg
    var_b = var_b * kb
    for m in range(row):
        for n in range(col):
            if var_r[m, n] > 0.0031308:
                var_r[m, n] = (var_r[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_g[m, n] > 0.0031308:
                var_g[m, n] = (var_g[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_b[m, n] > 0.0031308:
                var_b[m, n] = (var_b[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
    ldr = np.dstack([var_r, var_g, var_b])
    return ldr


if __name__ == '__main__':
    imghdr = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    result = tp(imghdr)
    result = (np.clip(result * 255.0, 0, 255)).astype('uint8')
    cv2.imshow('ldr', result)
    cv2.waitKey(0)
