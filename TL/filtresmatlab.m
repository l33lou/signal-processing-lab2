%%% tested under "matlab drive"
%%%
%%% Stéphane Rossignol -- 2025

clear all;
close all;

fs=8000;                      %%% sampling rate (Hz)
xx=cos(2*pi*3000*[0:1/fs:1]); %%% signal to be analyzed

[yy,dd]=lowpass(xx,2000,fs,ImpulseResponse="iir",Steepness=0.75);  %%% filter design and filtering ; 
                              %%% the signal after filtering is in "yy" and "dd" contains the filter

%%% signal obtained after filtering
figure(1);
clf;
plot(yy);
hold off;

%%% frequency response of the filter
figure(2);
clf;
filterAnalyzer(dd,Analysis="magnitude");
hold off;

zz=hilbert(yy);

