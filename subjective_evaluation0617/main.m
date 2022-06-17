hdr = hdrread('memorial.hdr');
ldr = imread("reinhard.png");
Q_f = (FSITM (hdr,ldr,1) + FSITM(hdr,ldr,2) + FSITM(hdr,ldr,3)) / 3;
[Q_t, S, N] = TMQI(hdr, ldr);
Q_ft = FSITM_TMQI(hdr,ldr);
result = [Q_f, Q_t, S, N, Q_ft];
