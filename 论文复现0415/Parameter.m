%% HDR算法复现
%Reinhard “Parameter Estimation for Photographic Tone Reproduction”
% Journal of Graphics Tools, 7:1, 45-51, DOI: 10.1080/10867651.2002.10487554
close all;
clc;
filename = 'SIGGRAPH17_HDR_Testset/memorial.hdr';
hdr = double(hdrread(filename));
gamma=1/1.6;
figure,imshow(hdr);
%figure,imshow(hdr.^gamma);
%imwrite(hdr.^gamma,'原始HDR图像.png');
%%  预处理，log域均值取指数结果映射到中性灰
Lw = 0.27*hdr(:,:,1) + 0.67*hdr(:,:,2) + 0.06*hdr(:,:,3) + 1e-6; %论文推荐的world luminance计算公式,小偏移量为避免被0除的情况
R = hdr(:,:,1) ./ Lw;
G = hdr(:,:,2) ./ Lw;
B = hdr(:,:,3) ./ Lw;
 
% maxL=max(L(:));
% settedMaxL=20;
% thL=10;
% mask=L>thL;
% settedMaxL=min(maxL,settedMaxL);
% L(mask)=(settedMaxL-thL)*(L(mask)-thL)/(maxL-thL)+thL;%高曝光部分压缩动态范围
 
LL=log(Lw+10^-9);
meanLw=exp(mean(LL(:))); %实践中发现将对数均值再反对数变换（取指数）结果映射到key值，通常可以得到更好的效果（主要是大区域之间的对比度更加合理而不会过低）
maxLw=quantile(Lw(:),0.99);
minLw=quantile(Lw(:),0.01);
zone=log2(maxLw)-log2(minLw);
a=0.18*4^((2*log2(meanLw)-log2(minLw)-log2(maxLw))/zone);
L=Lw*a/meanLw;
if zone<11
    Lwhite=1.5*2^(log2(maxLw)-log2(minLw)-5);
    Ld=(L+L.^2/Lwhite^2)./(L+1);
else
    picNum=8;
    ratio=1.3;
    phi=8.0;
    eps=0.05;
    alpha=1/sqrt(8);
    refV=1.0;
    V1=zeros([size(L),picNum+1]); % V2的当前尺度与V1的下一尺度相同
    %-----------------------空域实现-------------------------------%
    tic
    for i=1:picNum+1 % picNum+1个尺度，以ratio为倍数递增
        s=ratio^(i-1);
        sigma=alpha*s;
        kernelRadius = ceil(2*sigma);
        kernelSize = 2*kernelRadius+1;
        gaussKernelHorizontal = fspecial('gaussian', [kernelSize 1], sigma);
        temp= conv2(L, gaussKernelHorizontal, 'same');
        gaussKernelVertical = fspecial('gaussian', [1 kernelSize], sigma);
        V1(:,:,i)= conv2(temp, gaussKernelVertical, 'same');
    end
    V=zeros([size(L),picNum]);
    for i=1:picNum
        s=ratio^(i-1);
        V(:,:,i)=abs(V1(:,:,i+1)-V1(:,:,i))./(a*2^phi/s^2+V1(:,:,i));
    end
    
    Vsm=V1(:,:,picNum); %给Vsm初始化，作为8个尺度都不满足条件下的预设值
    for i=1:size(L,1)
        for j=1:size(L,2)
            for k=1:picNum
                if V(i,j,k)>eps
                    Vsm(i,j)=V1(i,j,k);
                    break;
                end
            end
        end
    end
    
    toc
    %figure,imshow(Vsm);
    %imwrite(Vsm,'各点参考值图像.png');
    Ld=L./(refV+Vsm);
end
Ld(Ld>1)=1;
%figure,imshow(Ld);
%% 色彩还原
rgb=zeros(size(L,1),size(L,2),3);
rgb(:,:,1)=Ld.*R;
rgb(:,:,2)=Ld.*G;
rgb(:,:,3)=Ld.*B;
rgb=rgb.^gamma;
figure,imshow(rgb);