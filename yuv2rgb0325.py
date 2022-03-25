import numpy as np
import sys


def yuv2bgr(filename, height, width, format1, format2='image', startfrm=0, endfrm=False):
    """
    :param filename: 待处理 YUV 文件的名字
    :param height: 图像的高
    :param width: 图像的宽
    :param format1: 存储格式:420p,422p,444p,yuyv
    :param format2: 文件类型:video，image，默认为image
    :param startfrm: 起始帧,默认为0
    :param endfrm: 结束帧，默认为最后一帧
    """
    fp = open(filename, 'rb')
    # 一帧图像所含的uv像素高和宽
    if format1 == '420p':
        h_h = height // 2
        h_w = width // 2
    elif format1 == '422p' or format1 == 'yuyv':
        h_h = height
        h_w = width // 2
    elif format1 == '444p':
        h_h = height
        h_w = width
    else:
        print('format1有误')
        sys.exit(1)
    if format2 == 'image':
        endfrm = 1
    elif format2 == 'video':
        framesize = height * width + h_h * h_w * 2  # 帧大小
        fp.seek(0, 2)  # 设置文件指针到文件流的尾部
        ps = fp.tell()  # 当前文件指针位置
        numfrm = ps // framesize  # 计算总帧数
        if not endfrm:
            endfrm = numfrm
        if endfrm > numfrm:
            print('endfrm有误')
            sys.exit(1)
        if startfrm < 0 or startfrm > endfrm:
            print('startfrm有误')
            sys.exit(1)
        fp.seek(framesize * startfrm, 0)
    else:
        print('format2有误')
        sys.exit(1)
    out = open('result'+format1+'.rgb', 'wb+')  # 输出文件名
    for i in range(endfrm - startfrm):
        yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        if format1 == 'yuyv':
            for m in range(h_h):
                for n in range(h_w):
                    yt[m, 2 * n] = ord(fp.read(1))
                    ut[m, n] = ord(fp.read(1))
                    yt[m, 2 * n + 1] = ord(fp.read(1))
                    vt[m, n] = ord(fp.read(1))
        else:
            for m in range(height):
                for n in range(width):
                    yt[m, n] = ord(fp.read(1))
            for m in range(h_h):
                for n in range(h_w):
                    ut[m, n] = ord(fp.read(1))
            for m in range(h_h):
                for n in range(h_w):
                    vt[m, n] = ord(fp.read(1))
        if format1 == '422p' or format1 == 'yuyv':
            ut = np.repeat(ut, 2, 1)
            vt = np.repeat(vt, 2, 1)
        if format1 == '420p':
            ut = np.repeat(ut, 2, 0)
            vt = np.repeat(vt, 2, 0)
            ut = np.repeat(ut, 2, 1)
            vt = np.repeat(vt, 2, 1)
        r = (yt + 1.4075 * (vt - 128.0))
        g = (yt - 0.3455 * (ut - 128.0) - 0.7169 * (vt - 128.0))
        b = (yt + 1.779 * (ut - 128.0))
        img = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)  # 防止溢出
        out.write(img)
        print("Extract frame %d " % (i + 1))
    fp.close()
    out.close()
    print("job done!")
    return None


if __name__ == '__main__':
    yuv2bgr(filename='422p.yuv', height=720, width=1280, format1='422p')
    