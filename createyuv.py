
import numpy as np

def createyuyv(filename, height, width, startfrm = 0, endfrm = False):

    fp = open(filename, 'rb')
    framesize = height * width * 2
    h_h = height
    h_w = width // 2
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部
    ps = fp.tell()  # 当前文件指针位置
    numfrm = ps // framesize  # 计算总帧数
    if not endfrm:
        endfrm = numfrm
    fp.seek(framesize * startfrm, 0)
    file = open('yuyv' + '.yuv', 'wb+')
    for i in range(endfrm - startfrm):
        mat = np.zeros(shape=(height, width * 2), dtype='uint8', order='C')
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        for m in range(height):
            for n in range(width):
                Yt[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Ut[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Vt[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                mat[m, 4 * n] = Yt[m, 2 * n]
                mat[m, 4 * n + 1] = Ut[m, n]
                mat[m, 4 * n + 2] = Yt[m, 2 * n + 1]
                mat[m, 4 * n + 3] = Vt[m, n]
        img = mat.astype(np.uint8)
        file.write(img)
        print("Extract frame %d " % (i + 1))

    fp.close()
    print("job done!")
    return None

if __name__ == '__main__':
    createyuyv(filename='422.yuv', height=720, width=1280, endfrm=50)