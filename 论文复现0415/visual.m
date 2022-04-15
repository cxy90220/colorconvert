%% HDR算法复现
% Zhengguo Li, Senior Member, IEEE, and Jinghong Zheng "Visual-Salience-Based Tone Mapping for High Dynamic Range Images"
% IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 61, NO. 12, DECEMBER 2014
close all;clc
filename = 'SIGGRAPH17_HDR_Testset/memorial.hdr';
hdr = double(hdrread(filename));
gamma=1/2.0;
figure,imshow(hdr);
% figure,imshow(hdr.^gamma);
%imwrite(hdr.^gamma,'原始HDR图像.png');
%%  预处理，log域均值取指数结果映射到中性灰
Lw = 0.299*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.144*hdr(:,:,3) + 1e-9; %论文推荐的world luminance计算公式,小偏移量为避免被0除的情况
R = hdr(:,:,1) ./ Lw;
G = hdr(:,:,2) ./ Lw;
B = hdr(:,:,3) ./ Lw;
 
%% 具有边缘意识的色调映射
Lh=log(Lw);
meanLw=exp(mean(Lh(:))); 
maxLw=quantile(Lw(:),0.99);
minLw=quantile(Lw(:),0.01);
zone=log2(maxLw)-log2(minLw);
a=0.18*4^((2*log2(meanLw)-log2(minLw)-log2(maxLw))/zone);
r=15;
eps=1;
Ge=varBasedWeight(Lh); %\Gamma_e
Lb=WGIF(Lh,Lh,r,eps,Ge); %加权引导滤波获得基底层
Ld=Lh-Lb; %细节层
% figure,imshow(exp(Lb));
% figure,imshow(exp(Ld));
meanLh=mean(Lh(:));
compLb=Lb+log(a)-meanLh-log(1+a*exp(Lb-meanLh));
% figure,imshow(exp(compLb));
theta=1.5; %细节缩放因子
Lt=exp(compLb+theta*Ld);
% figure,imshow(Lt);
% 色彩还原
rgb=zeros(size(Lw,1),size(Lw,2),3);
rgb(:,:,1)=Lt.*R;
rgb(:,:,2)=Lt.*G;
rgb(:,:,3)=Lt.*B;
rgb=rgb.^gamma;
figure,imshow(rgb);
 
%% 基于显著度的色调映射与色彩还原
Gb=SaliencyBasedICH(Lh,20,4); %\Gamma_b
temp=1./(Gb+10^-9);
Gb=(Gb+10^-9)*mean(temp(:));
lambda=0.75;
Gb=Gb.^lambda;
Gb(Gb<1)=1;
% figure,imshow(Gb/max(Gb(:)));
W2=Ge.*Gb;
Lb2=WGIF(Lh,Lh,r,eps,W2); %加权引导滤波获得基底层
Ld2=Lh-Lb2; %细节层
% figure,imshow(exp(Lb2));
% figure,imshow(exp(Ld2));
 
wLh=Gb.*Lh;
meanLh2=sum(wLh(:))/sum(Gb(:));
compLb2=Lb2+log(a)-meanLh2-log(1+a*exp(Lb2-meanLh2));
theta=1.5; %细节缩放因子
Lt2=exp(compLb2+theta*Ld2);
rgb2=zeros(size(Lw,1),size(Lw,2),3);
rgb2(:,:,1)=Lt2.*R;
rgb2(:,:,2)=Lt2.*G;
rgb2(:,:,3)=Lt2.*B;
rgb2=rgb2.^gamma;
figure,imshow(rgb2);
figure,imshow((rgb+rgb2)/2);
%% 函数
function wM = varBasedWeight(img,phi)
if ~exist('phi','var')
    phi=0.75;
end
img=img-min(img(:));
L=max(img(:))-min(img(:));
r=15;
paddedImg=padarray(img,[r r],'replicate','both');
sq_paddedImg=paddedImg.^2;
intMap=intergalMap(paddedImg);
sq_intMap=intergalMap(sq_paddedImg);
sumImg=zeros(size(img));
meanImg=zeros(size(img));
sq_sumImg=zeros(size(img));
varImg=zeros(size(img));
wM=zeros(size(img)); %edge-aware weight image
num=(2*r+1)^2;
for i=1:size(img,1)
    for j=1:size(img,2)
        sumImg(i,j)=intMap(i+2*r+1,j+2*r+1)+intMap(i,j)-intMap(i,j+2*r+1)-intMap(i+2*r+1,j);
        meanImg(i,j)=sumImg(i,j)/num;
        sq_sumImg(i,j)=sq_intMap(i+2*r+1,j+2*r+1)+sq_intMap(i,j)-sq_intMap(i,j+2*r+1)-sq_intMap(i+2*r+1,j);
        varImg(i,j)= sq_sumImg(i,j)-(sumImg(i,j)^2)/num;
    end
end
v1=(0.001*L)^2;
v2=10^-9;
wM=((varImg+v1)./(meanImg.^2+v2)).^phi;
temp=1./wM;
wM=wM*mean(temp(:));
end 
 
function q = WGIF(I, p, r, eps, W)
%   WEIGHTEDGUIDEDFILTER   O(1) time implementation of weighted guided filter.
%
%   - guidance image: I (should be a gray-scale/single channel image)
%   - filtering input image: p (should be a gray-scale/single channel image)
%   - local window radius: r
%   - regularization parameter: eps
%   - weight image: W 
 
[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
 
mean_I = boxfilter(I, r) ./ N;
mean_p = boxfilter(p, r) ./ N;
mean_Ip = boxfilter(I.*p, r) ./ N;
cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
 
mean_II = boxfilter(I.*I, r) ./ N;
var_I = mean_II - mean_I .* mean_I;
 
a = W.*cov_Ip ./ (W.*var_I + eps); % Eqn. (5) in the paper;
b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
 
mean_a = boxfilter(a, r) ./ N;
mean_b = boxfilter(b, r) ./ N;
 
q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
end


function S=SaliencyBasedICH(img,K,r)
%基于灰度共生矩阵的显著度矩阵计算程序
%K为量化等级
%r为衡量共生性质的邻域半径，灰度m,n在(2r+1,2r+1)的窗口内同时出现过即视为共生
M=ICH(img,K,r);
nM=M/sum(M(:));
nzMask=nM~=0;
nzMat=double(nzMask);
nzNum=sum(nzMat(:));
invPMF=nM;
mask=nzMask & nM<1/nzNum;
invPMF(mask)=1/nzNum-nM(mask);
invPMF(~mask)=0;
 
%求窗口w*w内的平均显著度
interval=(max(img(:))-min(img(:)))/(K-0.5);
Q=floor((img-min(img(:)))/interval)+1;
pQ=padarray(Q,[r r],'replicate','both');
num=(2*r+1)^2;
S=zeros(size(img,1),size(img,2));
for i=r+1:size(pQ,1)-r
    for j=r+1:size(pQ,2)-r
        sumS=0;
        for p=-r:r
            for q=-r:r
                sumS=sumS+invPMF(pQ(i,j),pQ(i+p,j+q));
            end
        end
        S(i-r,j-r)=sumS/num;
    end
end
S=S/max(S(:));
end
 
function M=ICH(img,K,r)
%求灰度共生矩阵程序
%K为量化等级
%r为衡量共生性质的邻域半径，灰度m,n在(2r+1,2r+1)的窗口内同时出现过即视为共生
if ~exist('K','var')
    K=20;
end
if ~exist('r','var')
    r=4;
end
interval=(max(img(:))-min(img(:)))/(K-0.5);
Q=floor((img-min(img(:)))/interval)+1;
rowNum=size(Q,1);
M=zeros(K,K);
idxShift=zeros(r+1,r+1); %参考像素相对于当前像素的索引偏移量
for i=0:r
    for j=0:r
        idxShift(i+1,j+1)=i+j*rowNum;
    end
end
idxShift=idxShift(:);
for p=2:length(idxShift) %idxShift(1)为0，表示当前进行统计的两像素实际为同一像素，跳过
    for i=1:size(Q,1)-r
        for j=1:size(Q,2)-r
            idx=(j-1)*rowNum+i;
            s=idxShift(p);
            M(Q(idx),Q(idx+s))=M(Q(idx),Q(idx+s))+1;
            M(Q(idx+s),Q(idx))=M(Q(idx+s),Q(idx))+1;
        end
    end
end
end