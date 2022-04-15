function Yim=intergalMap(img)
% 生成积分图
% Yim为原始值积分图，YYim为平方值积分图
% 可以只保留原始值积分图，通过输入图像与自己的点乘获得平方值积分图
paddedImg=padarray(img,[1 1],0,'pre');
Yim=zeros(size(paddedImg));
for i=2:size(paddedImg,1)
    for j=2:size(paddedImg,2)
        Yim(i,j)=Yim(i,j-1)+Yim(i-1,j)-Yim(i-1,j-1)+paddedImg(i,j);
    end
end