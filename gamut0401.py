import numpy as np
from PIL import Image
import sys


def srgb2xyz(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # 归一化
    var_r = r / 255.0
    var_g = g / 255.0
    var_b = b / 255.0
    # gamma 校正公式
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.04045:
                var_r[m, n] = ((var_r[m, n] + 0.055) / 1.055) ** 2.4
            else:
                var_r[m, n] = var_r[m, n] / 12.92
            if var_g[m, n] > 0.04045:
                var_g[m, n] = ((var_g[m, n] + 0.055) / 1.055) ** 2.4
            else:
                var_g[m, n] = var_g[m, n] / 12.92
            if var_b[m, n] > 0.04045:
                var_b[m, n] = ((var_b[m, n] + 0.055) / 1.055) ** 2.4
            else:
                var_b[m, n] = var_b[m, n] / 12.92

    var_r = var_r * 100.0
    var_g = var_g * 100.0
    var_b = var_b * 100.0
    # 转化矩阵
    x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505
    imgxyz = np.dstack([x, y, z])
    return imgxyz


def p32xyz(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    var_r = r / 255.0
    var_g = g / 255.0
    var_b = b / 255.0

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.04045:
                var_r[m, n] = (0.9479 * var_r[m, n] + 0.05214) ** 2.4
            if var_g[m, n] > 0.04045:
                var_g[m, n] = (0.9479 * var_g[m, n] + 0.05214) ** 2.4
            if var_b[m, n] > 0.04045:
                var_b[m, n] = (0.9479 * var_b[m, n] + 0.05214) ** 2.4

    var_r = var_r * 100.0
    var_g = var_g * 100.0
    var_b = var_b * 100.0

    x = var_r * 0.4866 + var_g * 0.2657 + var_b * 0.1982
    y = var_r * 0.2290 + var_g * 0.6917 + var_b * 0.0793
    z = var_r * 0.0000 + var_g * 0.0451 + var_b * 1.0437
    imgxyz = np.dstack([x, y, z])
    return imgxyz


def adobe2xyz(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    var_r = r / 255.0
    var_g = g / 255.0
    var_b = b / 255.0

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            var_r[m, n] = var_r[m, n] ** 2.1992
            var_g[m, n] = var_g[m, n] ** 2.1992
            var_b[m, n] = var_b[m, n] ** 2.1992

    var_r = var_r * 100.0
    var_g = var_g * 100.0
    var_b = var_b * 100.0

    x = var_r * 0.5767 + var_g * 0.1856 + var_b * 0.1882
    y = var_r * 0.2974 + var_g * 0.6274 + var_b * 0.0753
    z = var_r * 0.0270 + var_g * 0.0707 + var_b * 0.9911
    imgxyz = np.dstack([x, y, z])
    return imgxyz


def xyz2srgb(img):
    x = img[:, :, 0]
    y = img[:, :, 1]
    z = img[:, :, 2]
    
    var_x = x / 100.0
    var_y = y / 100.0
    var_z = z / 100.0

    var_r = var_x * 3.2406 + var_y * -1.5372 + var_z * -0.4986
    var_g = var_x * -0.9689 + var_y * 1.8758 + var_z * 0.0415
    var_b = var_x * 0.0557 + var_y * -0.2040 + var_z * 1.0570

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.0031308:
                var_r[m, n] = 1.055 * (var_r[m, n] ** (1.0 / 2.4)) - 0.055
            else:
                var_r[m, n] = 12.92 * var_r[m, n]
            if var_g[m, n] > 0.0031308:
                var_g[m, n] = 1.055 * (var_g[m, n] ** (1.0 / 2.4)) - 0.055
            else:
                var_g[m, n] = 12.92 * var_g[m, n]
            if var_b[m, n] > 0.0031308:
                var_b[m, n] = 1.055 * (var_b[m, n] ** (1.0 / 2.4)) - 0.055
            else:
                var_b[m, n] = 12.92 * var_b[m, n]

    r = var_r * 255.0
    g = var_g * 255.0
    b = var_b * 255.0
    imgsrgb = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)
    return imgsrgb


def xyz2p3(img):
    x = img[:, :, 0]
    y = img[:, :, 1]
    z = img[:, :, 2]

    var_x = x / 100.0
    var_y = y / 100.0
    var_z = z / 100.0

    var_r = var_x * 2.4932 + var_y * -0.9313 + var_z * -0.4027
    var_g = var_x * -0.8295 + var_y * 1.7627 + var_z * 0.0236
    var_b = var_x * 0.0359 + var_y * -0.0762 + var_z * 0.9571

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if var_r[m, n] > 0.0031308:
                var_r[m, n] = (var_r[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_g[m, n] > 0.0031308:
                var_g[m, n] = (var_g[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479
            if var_b[m, n] > 0.0031308:
                var_b[m, n] = (var_b[m, n] ** (1.0 / 2.4) - 0.05214) / 0.9479

    r = var_r * 255.0
    g = var_g * 255.0
    b = var_b * 255.0
    imgp3 = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)
    return imgp3


def xyz2adobe(img):
    x = img[:, :, 0]
    y = img[:, :, 1]
    z = img[:, :, 2]

    var_x = x / 100.0
    var_y = y / 100.0
    var_z = z / 100.0

    var_r = var_x * 2.0414 + var_y * -0.5650 + var_z * -0.3447
    var_g = var_x * -0.9693 + var_y * 1.8760 + var_z * 0.0416
    var_b = var_x * 0.0135 + var_y * -0.1184 + var_z * 1.0154

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            var_r[m, n] = var_r[m, n] ** (1.0 / 2.1992)
            var_g[m, n] = var_g[m, n] ** (1.0 / 2.1992)
            var_b[m, n] = var_b[m, n] ** (1.0 / 2.1992)

    r = var_r * 255.0
    g = var_g * 255.0
    b = var_b * 255.0
    imgp3 = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)
    return imgp3


def gamut(filename, input, output):
    """
    :param filename: 待处理文件的名字
    :param input: 输入格式:sRGB,Display P3,Adobe RGB
    :param output: 输出格式sRGB,Display P3,Adobe RGB
    """
    img = Image.open(filename)
    img = np.array(img)
    # RGB2CIE
    if input == 'sRGB':
        imgxyz = srgb2xyz(img)
    elif input == 'Display P3':
        imgxyz = p32xyz(img)
    elif input == 'Adobe RGB':
        imgxyz = adobe2xyz(img)
    else:
        print('input有误')
        sys.exit(1)
    # CIE2RGB
    if output == 'sRGB':
        result = xyz2srgb(imgxyz)
    elif output == 'Display P3':
        result = xyz2p3(imgxyz)
    elif output == 'Adobe RGB':
        result = xyz2adobe(imgxyz)
    else:
        print('output有误')
        sys.exit(1)
    result = Image.fromarray(result)
    result.save('document/result' + output + '.jpg')
    return None


if __name__ == '__main__':
    gamut(filename='document/Display P3.jpg', input='Display P3', output='sRGB')
    gamut(filename='document/sRGB.jpg', input='sRGB', output='Adobe RGB')
    gamut(filename='document/Adobe RGB.jpg', input='Adobe RGB', output='Display P3')
