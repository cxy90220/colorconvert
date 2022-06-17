function Q = FSITM (HDR, LDR, CH)
% Feature similarity index for tone mapped images (FSITM)
% By: Hossein Ziaei Nafchi, November 2014
% hossein.zi@synchromedia.ca
% Synchromedia Lab, ETS, Canada
% The code can be modified, rewritten and used without obtaining permission
% of the authors.
% Please refer to the following paper:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Reza Farrahi Moghaddam, Mohamed Cheriet, IEEE Signal Processing Letters, vol. 22, no. 8, pp. 1026-1029, 2015.
%%
% HDR: High dynamic range image
% LDR: Low dynamic range image
% CH = 1 --> Red channel, CH = 2 --> Green channel, CH = 3 --> Blue channel
% Q: Quality index
% Needs phasecong100 and Lowpassfilter functions
%%
[row, col, ~] = size(LDR);
NumPixels = row * col;
r = floor(NumPixels / (2 ^ 18));
if r > 1
    alpha = 1 - (1 / r);
else
    alpha = 0;
end
HDR_CH = HDR(:, :, CH);
LDR_CH = LDR(:, :, CH);
LogH = HDR_CH;
minNonzero = min(HDR_CH(HDR_CH ~= 0));
LogH(HDR_CH == 0) = minNonzero;
LogH = log(LogH);
LogH = im2uint8(mat2gray(LogH)); 
if alpha~=0
    HDR_CH = HDR(:, :, CH); 
    PhaseHDR_CH = phasecong100(HDR_CH, 2, 2, 8, 8);
    PhaseLDR_CH8 = phasecong100(LDR_CH, 2, 2, 8, 8);
else
    PhaseHDR_CH = 0;
    PhaseLDR_CH8 = 0;
end
PhaseLogH = phasecong100(LogH, 2, 2, 2, 2);
PhaseH = alpha * PhaseHDR_CH + (1 - alpha) * PhaseLogH; 
PhaseLDR_CH2 = phasecong100(LDR_CH, 2, 2, 2, 2);
PhaseL = alpha * PhaseLDR_CH8 + (1 - alpha) * PhaseLDR_CH2;
index = (PhaseL <= 0 & PhaseH <= 0) | (PhaseL > 0 & PhaseH > 0);
Q = sum(index(:)) / NumPixels;