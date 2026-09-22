%%% Mon implantation de Welch
%%%
%%% Stéphane Rossignol -- retravaillé en août 2026

clear all;
close all;

NN=500;       %%% taille du signal
MM=400;         %%% taille de la fenêtre d'analyse (pour Welch)
KK=NN-MM;       %%% on décale d'un échantillon entre 2 fenêtres d'analyse (pour Welch)
taillefft=2^16; %%% taille des FFT (pour le périodogramme et Welch)

fe=8000; %%% fréquence d'échantillonnage

fep=hanning(NN)'; %%% fenêtre pour le périodogramme
fen=hanning(MM)'; %%% fenêtre pour Welch
%fep=ones(NN,1)'; %%% fenêtre pour le périodogramme
%fen=ones(MM,1)'; %%% fenêtre pour Welch

freqp=[0:taillefft-1]/taillefft*fe;
freqw=freqp;

stdb=2; %%% écart-type du bruit

ncoups=1;
for ii=1:ncoups %%% nombre de coups pour les statistiques
  sig = stdb*randn(1,NN);

  %%% périodogramme
  perio = abs(fft(sig.*fep,taillefft)).^2/NN;
  [valp,posp]=max(perio(1:taillefft/2));

  %%% Welch
  www=zeros(1,taillefft);
  for kk=1:KK
    sigi=sig(1+(kk-1):MM+(kk-1));
    www=www+abs(fft(sigi.*fen,taillefft)).^2/MM;
  end;
  www=www/KK*(MM/sum(fen));

  vm=max([www perio]);

  dofig=1;
  if dofig==1
    figure(1);
    clf;
    plot(freqp(1:taillefft/2),perio(1:taillefft/2));
    hold on;
    plot([1 fe/2],[stdb stdb],'r');
    xlabel('frequency (Hz)','FontSize',15);
    ylabel('Periodogram','FontSize',15);
    legend('Periodogram','DSP white noise','FontSize',15);
    ylim([0 vm]);
    hold off;
%    print -depsc2 periowhite.eps
  end;

  dofig=1;
  if dofig==1
    figure(2);
    clf;
    plot(freqw(1:taillefft/2),www(1:taillefft/2),'g');
    hold on;
    plot([1 fe/2],[stdb stdb],'r');
    xlabel('frequency (Hz)','FontSize',15);
    ylabel('Welch','FontSize',15);
    legend('Welch','DSP white noise','FontSize',15);
    ylim([0 vm]);
    hold off;
%    print -depsc2 periowelch.eps
  end;
end;

