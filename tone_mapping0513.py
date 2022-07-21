"""
参考论文：Tone-mapping high dynamic range images by novel histogram adjustment
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


def ft(t, ave, xmax, xmin, k):
    return(math.log(ave + t) - math.log(xmin + t)) - k * (math.log(xmax + t) - math.log(xmin + t))


def fft(t, ave, xmax, xmin, k):
    return(1.0 / (ave + t) - 1.0 / (xmin + t)) - k * (1.0 / (xmax + t) - 1.0 / (xmin + t))


def tp(img):
    var_r = img[:, :, 0]
    var_g = img[:, :, 1]
    var_b = img[:, :, 2]
    row, col, dim = img.shape
    lum = 0.299 * var_r + 0.587 * var_g + 0.114 * var_b
    xmax = lum[0, 0]
    xmin = lum[0, 0]
    ave = 0
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if xmax < lum[m, n]:
                xmax = lum[m, n]
            if xmin > lum[m, n]:
                xmin = lum[m, n]
            ave = ave + math.log(1e-10 + lum[m, n])
    ave = math.exp(ave / (img.shape[0] * img.shape[1]))
    k = 0.8 ** ((2 * math.log(1e-10 + ave) - math.log(1e-10 + xmax) - math.log(1e-10 + xmin)) / (math.log(1e-10 + xmax) - math.log(1e-10 + xmin)))
    t = 1e-5
    while abs(ft(t, ave, xmax, xmin, k)) < 1e-5:
        t = t - ft(t, ave, xmax, xmin, k) / fft(t, ave, xmax, xmin, k)
    d = np.zeros((img.shape[0], img.shape[1]))
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            d[m, n] = (math.log(lum[m, n] + t) - math.log(xmin + t)) / (math.log(xmax + t) - math.log(xmin + t))
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            var_r[m, n] = var_r[m, n] * d[m, n]
            var_g[m, n] = var_g[m, n] * d[m, n]
            var_b[m, n] = var_b[m, n] * d[m, n]
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
    # cv2.imwrite('histogram.png', result)
    cv2.imwrite('histogram_encolor.png', result)
    cv2.waitKey(0)
