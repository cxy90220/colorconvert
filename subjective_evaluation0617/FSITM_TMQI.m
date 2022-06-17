function Q = FSITM_TMQI (HDR, LDR)
% Feature similarity index for tone mapped images (FSITM) combined with the
% tone mapped quality index (TMQI)
% By: Hossein Ziaei Nafchi, November 2014
% hossein.zi@synchromedia.ca
% Synchromedia Lab, ETS, Canada
% The code can be modified, rewritten and used without obtaining permission
% of the authors. 
% Please refer to the following papers:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Reza Farrahi Moghaddam, Mohamed Cheriet, IEEE Signal Processing Letters, vol. 22, no. 8, pp. 1026-1029, 2015.
% H. Yeganeh and Z. Wang, "Objective Quality Assessment of Tone Mapped
% Images," IEEE Transactios on Image Processing, vol. 22, no. 2, pp. 657- 
% 667, Feb. 2013.
%%
% HDR: High dynamic range image
% LDR: Low dynamic range image
% CH = 1 --> Red channel, CH = 2 --> Green channel, CH = 3 --> Blue channel
% Q: Quality index
% Needs FSITM and TMQI functions
%%
% Q1 = FSITM(HDR, LDR, CH);
Q1 = (FSITM(HDR, LDR, 1) + FSITM(HDR, LDR, 2) + FSITM(HDR, LDR, 3)) / 3;
Q2 = TMQI(HDR, LDR);
Q = (Q1 + Q2) / 2;