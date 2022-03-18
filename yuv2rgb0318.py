import cv2
import numpy as np

def yuv2bgr(filename, height, width, format, startfrm = 0, endfrm = False):
    """
    :param filename: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param startfrm: 起始帧
    :param endfrm: 结束帧
    :param format: YUV视频存储格式:420p,422p,444p,yuyv
    """
    fp = open(filename, 'rb')
    # 一帧图像所含的各类像素个数
    if format == '420p':
        framesize = height * width * 3 // 2
        h_h = height // 2
        h_w = width // 2
    elif format == '422p' or format == 'yuyv':
        framesize = height * width * 2
        h_h = height
        h_w = width // 2
    else:
        framesize = height * width * 3
        h_h = height
        h_w = width
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部
    ps = fp.tell()  # 当前文件指针位置
    numfrm = ps // framesize  # 计算总帧数
    if not endfrm:
        endfrm = numfrm
    fp.seek(framesize * startfrm, 0)
    out = open('result'+format+'.rgb', 'wb+')  # 输出文件名
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('result.avi', fourcc, 25.0, (1280, 720))
    for i in range(endfrm - startfrm):
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        if format == 'yuyv':
            for m in range(h_h):
                for n in range(h_w):
                    Yt[m, 2 * n] = ord(fp.read(1))
                    Ut[m, n] = ord(fp.read(1))
                    Yt[m, 2 * n + 1] = ord(fp.read(1))
                    Vt[m, n] = ord(fp.read(1))
        else:
            for m in range(height):
                for n in range(width):
                    Yt[m, n] = ord(fp.read(1))
            for m in range(h_h):
                for n in range(h_w):
                    Ut[m, n] = ord(fp.read(1))
            for m in range(h_h):
                for n in range(h_w):
                    Vt[m, n] = ord(fp.read(1))
        if format == '422p' or format == 'yuyv':
            Ut = np.repeat(Ut, 2, 1)
            Vt = np.repeat(Vt, 2, 1)
        if format == '420p':
            img = (np.concatenate([Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)])).astype(np.uint8)
            img = img.reshape(height * 3 // 2, width)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)
        else:
            img = (np.dstack([Yt, Ut, Vt])).astype(np.uint8)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

        out.write(bgr_img)
        print("Extract frame %d " % (i + 1))

    fp.close()
    print("job done!")
    return None

if __name__ == '__main__':
    yuv2bgr(filename='420.yuv', height=720, width=1280, format='444p', startfrm=0, endfrm=1)  # endfrm-startfrm=1时为图片




