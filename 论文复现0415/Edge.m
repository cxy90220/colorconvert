%% HDR算法复现
% Farbman, Fattal, Lischinski, Szeliski "Edge-Preserving Decompositions for Multi-Scale Tone and DetailManipulation"
% ACM Transactions on Graphics, 27(3), August 2008.
close all;
clc;
%% Load
filename = 'SIGGRAPH17_HDR_Testset/memorial.hdr';
hdr = double(hdrread(filename));
I = 0.299*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.114*hdr(:,:,3) + 1e-6;
R = hdr(:,:,1) ./ I;  %归一化R/G/B
G = hdr(:,:,2) ./ I;
B = hdr(:,:,3) ./ I;
 
LL = log(I);
 
figure,imshow(hdr);
% figure,imshow(I./max(I(:)));
%% Filter
u0 = LL;
u1 = wlsFilter(LL, 0.125, 1.2, LL );
u2 = wlsFilter(LL, 1.0,   1.2, LL );
u3 = wlsFilter(LL, 8.0,   1.2, LL );
 
% figure,imshow(exp(u1));
% figure,imshow(exp(u2));
% figure,imshow(exp(u3));
%% Compute detail layers
d0 = u0-u1;
d1 = u1-u2;
d2 = u2-u3;
 
%% Recombine the layers together while moderately boosting multiple scale detail
% 具有柔化图像效果的HDR技术演示
% look = 'Balanced'; 
w0 =0.2;  %抑制小尺度和大尺度的对比度，中尺度对比度不变，获得柔和效果
w1 = 1.0;
w2 = 0.8;
w3 = 0.4; %原图像幂律变换，压缩动态范围作为基底图像
sat = 1.0; %保持1.0,表示不是各通道的原始对比度进行压缩
exposure = 1.0; %保持1.0，通过调整百分比确保变换后在设定百分比内的像素不会过曝
gamma = 1.0/1.6;
 
cLL = w0*d0 + w1*d1 + w2*d2 + w3*u3;
 
% figure,imshow(exp(d0)-0.5); %显示各尺度的对比度图像
% figure,imshow(exp(d1)-0.5);
% figure,imshow(exp(d2)-0.5);
% figure,imshow(exp(w3*u3)); %动态范围压缩后的图像作为基底图像，在此基础上进行各尺度对比度的调整
%% Convert back to RGB
Inew = exposure*exp(cLL);
% figure,imshow(Inew);
sI = sort(Inew(:));
mx = sI(round(length(sI) * (98/100))); %保持98%的像素处理后不会过曝
Inew = Inew/mx;
 
% figure,imshow(Inew);
rgb = composeIRGB(Inew, R, G, B, sat, gamma);
figure,imshow(rgb);
 
%% Recombine the layers together with stronger emphasis on fine detail
% 小尺度对比度增强的HDR技术演示，保持核心源代码不变
% look = 'StrongFine'; 
w0 = 2.0;
w1 = 0.8;
w2 = 0.7;
w3 = 0.2;
sat = 0.6;
exposure = 0.9;
gamma = 1.0;
 
cLL = w0*d0 + w1*d1 + w2*d2 + w3*u3; %进行多尺度合成

% figure,imshow(exp(d0)-0.5); %显示各尺度的对比度图像
% figure,imshow(exp(d1)-0.5);
% figure,imshow(exp(d2)-0.5);
% figure,imshow(exp(w3*u3)); %动态范围压缩后的图像作为基底图像，在此基础上进行各尺度对比度的调整 
%% Convert back to RGB
Inew = exp(cLL);
sI = sort(Inew(:));
mx = sI(round(length(sI) * (99.95/100))); 
Inew = Inew/mx;
 
rgb = composeIRGB(exposure*Inew, R, G, B, sat, gamma);
figure,imshow(rgb);
%% 函数
function rgb = composeIRGB(Inew, r, g, b, sat, gamma)

rgb(:,:,1) = Inew .* (r .^ sat);
rgb(:,:,2) = Inew .* (g .^ sat);
rgb(:,:,3) = Inew .* (b .^ sat);
if gamma ~= 1.0    
    rgb = rgb .^ gamma;
end
 
end
 

function OUT = wlsFilter(IN, lambda, alpha, L)
%   Given an input image IN, we seek a new image OUT, which, on the one hand,
%   is as close as possible to IN, and, at the same time, is as smooth as
%   possible everywhere, except across significant gradients in L.
%
%
%   Input arguments:
%   ----------------
%     IN              Input image (2-D, double, N-by-M matrix). 
%       
%     lambda          Balances between the data term and the smoothness
%                     term. Increasing lambda will produce smoother images.
%                     Default value is 1.0
%       
%     alpha           Gives a degree of control over the affinities by non-
%                     lineary scaling the gradients. Increasing alpha will
%                     result in sharper preserved edges. Default value: 1.2
%       
%     L               Source image for the affinity matrix. Same dimensions
%                     as the input image IN. Default: log(IN)
% 
%
%   Example 
%   -------
%     RGB = imread('peppers.png'); 
%     I = double(rgb2gray(RGB));
%     I = I./max(I(:));
%     res = wlsFilter(I, 0.5);
%     figure, imshow(I), figure, imshow(res)
%     res = wlsFilter(I, 2, 2);
%     figure, imshow(res)
 
if(~exist('L', 'var')),
    L = log(IN+eps);
end
 
if(~exist('alpha', 'var')),
    alpha = 1.2;
end
 
if(~exist('lambda', 'var')),
    lambda = 1;
end
 
smallNum = 0.0001;
 
[r,c] = size(IN);
k = r*c;
 
% Compute affinities between adjacent pixels based on gradients of L
dy = diff(L, 1, 1);
dy = -lambda./(abs(dy).^alpha + smallNum);
dy = padarray(dy, [1 0], 'post');
dy = dy(:);
 
dx = diff(L, 1, 2); 
dx = -lambda./(abs(dx).^alpha + smallNum);
dx = padarray(dx, [0 1], 'post');
dx = dx(:);
 
 
% Construct a five-point spatially inhomogeneous Laplacian matrix
B(:,1) = dx;
B(:,2) = dy;
d = [-r,-1];
A = spdiags(B,d,k,k);
 
e = dx;
w = padarray(dx, r, 'pre'); w = w(1:end-r);
s = dy;
n = padarray(dy, 1, 'pre'); n = n(1:end-1);
 
D = 1-(e+w+s+n);
A = A + A' + spdiags(D, 0, k, k);
 
% Solve
OUT = A\IN(:);
OUT = reshape(OUT, r, c);
end