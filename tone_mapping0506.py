import numpy as np
import cv2


def aces(img):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    ldr = ((img * (a * img + b)) / (img * (c * img + d) + e))
    var_r = ldr[:, :, 0]
    var_g = ldr[:, :, 1]
    var_b = ldr[:, :, 2]
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.0031308:
                var_r[m, n] = (var_r[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_g[m, n] > 0.0031308:
                var_g[m, n] = (var_g[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_b[m, n] > 0.0031308:
                var_b[m, n] = (var_b[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
    imgldr = np.dstack([var_r, var_g, var_b])
    return imgldr


def aces2(img):
    ACESInputMat = [[0.59719, 0.35458, 0.04823],[0.07600, 0.90834, 0.01566],[0.02840, 0.13383, 0.83777]]
    ACESOutputMat = [[1.60475, -0.53108, -0.07367],[-0.10208, 1.10813, -0.00605],[-0.00327, -0.07276, 1.07602]]
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            img[m, n, :] = np.dot(ACESInputMat, img[m, n, :])
    a = img * (img + 0.0245786) - 0.000090537
    b = img * (0.983729 * img + 0.4329510) + 0.238081
    ldr = a / b
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            ldr[m, n, :] = np.dot(ACESOutputMat, ldr[m, n, :]) * 1.8
    var_r = ldr[:, :, 0]
    var_g = ldr[:, :, 1]
    var_b = ldr[:, :, 2]
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.0031308:
                var_r[m, n] = (var_r[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_g[m, n] > 0.0031308:
                var_g[m, n] = (var_g[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_b[m, n] > 0.0031308:
                var_b[m, n] = (var_b[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
    imgldr = np.dstack([var_r, var_g, var_b])
    return imgldr


if __name__ == '__main__':
    imghdr = cv2.imread('test/memorial.hdr', flags=cv2.IMREAD_ANYDEPTH)
    result1 = aces(imghdr)
    reslut1 = (np.clip(result1 * 255.0, 0, 255)).astype('uint8')
    result2 = aces2(imghdr)
    reslut2 = (np.clip(result2 * 255.0, 0, 255)).astype('uint8')
    # tonemapDurand = cv2.createTonemapReinhard(2.2, 0, 0, 0)
    # result2 = tonemapDurand.process(imghdr)
    # reslut2 = np.clip(result2 * 255.0, 0, 255).astype('uint8')
    cv2.imshow('aces', result1)
    cv2.imshow('aces2', reslut2)
    cv2.waitKey(0)
