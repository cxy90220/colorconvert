import numpy as np


# 色适应矩阵
def adapt(target, source):
    """
    :param target: CIE 15.2中目标空间白点坐标
    :param source: CIE 15.2中源空间白点坐标
    """
    # ICC V4规范中的Bradford转换矩阵
    brad = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
    roetarget = np.dot(brad, target)
    roesource = np.dot(brad, source)
    midmat = np.zeros((3, 3))
    midmat[0, 0] = roesource[0]/roetarget[0]
    midmat[1, 1] = roesource[1]/roetarget[1]
    midmat[2, 2] = roesource[2]/roetarget[2]
    result = np.dot(np.dot(np.linalg.inv(brad), midmat), brad)
    return result


def rgb2cie(white, adaptmat, rgbmat):
    """
    :param white: CIE 15.2中目标空间白点坐标
    :param adaptmat: 色适应矩阵
    :param rgbmat: 源空间RGB三原色坐标
    """
    # 色适应到D65
    rgbxyz = np.dot(np.linalg.inv(adaptmat), np.transpose(rgbmat))
    xyz = [[rgbxyz[0, 0] / rgbxyz[1, 0], 1, rgbxyz[2, 0] / rgbxyz[1, 0]],
           [rgbxyz[0, 1] / rgbxyz[1, 1], 1, rgbxyz[2, 1] / rgbxyz[1, 1]],
           [rgbxyz[0, 2] / rgbxyz[1, 2], 1, rgbxyz[2, 2] / rgbxyz[1, 2]]]
    # 归一化
    w = np.array(white) / 100.0
    s = np.dot(np.linalg.inv(np.transpose(xyz)), w)
    result = np.transpose(np.transpose(np.array([s, s, s])) * xyz)
    return result


if __name__ == '__main__':
    D65 = [95.04, 100.0, 108.89]
    D50 = [96.42, 100.0, 82.49]
    adapt = adapt(D65, D50)
    print('adapt:')
    print(adapt)
    # icc文件参数
    p3 = [[0.515, 0.241, -0.001], [0.292, 0.692, 0.042], [0.157, 0.067, 0.784]]
    srgb = [[0.43607, 0.22249, 0.01392], [0.38515, 0.71687, 0.09708], [0.14307, 0.06061, 0.71410]]
    adobe = [[0.60974, 0.31111, 0.01947], [0.20528, 0.62567, 0.06087], [0.14919, 0.06322, 0.74457]]
    mp3 = rgb2cie(D65, adapt, p3)
    print('Display P3:')
    print(mp3)
    msrgb = rgb2cie(D65, adapt, srgb)
    print('sRGB:')
    print(msrgb)
    madobe = rgb2cie(D65, adapt, adobe)
    print('Adobe RGB:')
    print(madobe)



