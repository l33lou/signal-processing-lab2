%%% Algorithme de Morf pour la d\'etermination des param\`etres AR
%%%
%%% entr\'ees :
%%%   - xx : signal
%%%   - pp : ordre du mod\`ele AR (choisi de mani\`ere ind\'ependante)
%%%   - fe : fr\'equence d'\'echantillonnage
%%%
%%% sorties :
%%%   - aa     : les param\`etres AR
%%%   - sigma2 : variance du bruit
%%%   - ref    : coefficients de r\'eflexion
%%%   - ff     : fr\'equences auxquelles la dsp a \'et\'e calcul\'ee
%%%   - mydsp  : la dsp elle-m\^eme
%%%
%%% exemple : fe=32000;f0=440;xx=cos(2*pi*f0/fe*[1:1280]+2*pi*rand(1,1));mymorf(xx,4,fe);
%%%
%%% S. Rossignol -- 2012


function [aa, sigma2, ref, ff, mydsp] = mymorf (xx, pp, fe)
%clear all;fe=32000;f0=440;xx=cos(2*pi*f0/fe*[1:1280]+2*pi*rand(1,1));pp=4;


N1=length(xx);


%%% initialisation pour p1=0
fenerg=0;
for ii=1:N1
  fenerg = fenerg + xx(ii)^2;
end;
benerg = fenerg ;
aa(1)=1;
cc(1) = xx(1)/fenerg;
dd(1) = xx(N1)/benerg;
gg = cc(1)*xx(1);
ww = dd(1)*xx(N1);
hh = dd(1)*xx(1);
epri = fenerg - xx(1)^2;
rpri = benerg - xx(N1)^2;
cpri(1) = cc(1) + hh*dd(1) / (1-ww);
dpri(1) = dd(1) + hh*cc(1) / (1-gg);


%%% initialisation pour p1=1
help = 0;
for ii = 2:N1
  help = help + xx(ii) * xx(ii-1);
end;
Rmatr(1) = help;
ROmatr(1) = help;
rcf(2) = Rmatr(1)/rpri;
rcb(2) = Rmatr(1)/epri;
aa(2) = -rcf(2);
bb(1) = -rcb(2);
bb(2) = 1;

fenerg = epri - rcf(2)*Rmatr(1);
benerg = rpri - rcb(2)*Rmatr(1);

ref=[]; %%% todo


for p1=1:pp-1

  %%% mise \`a jour du filtre pr\'edicteur
  epp = xx(p1+1);
  for ii=1:p1
    epp = epp + aa(ii+1)*xx(p1-ii+1);
  end;
  rpnm1 = xx(N1-p1);
  for ii=0:p1-1
    rpnm1 = rpnm1 + bb(p1-ii)*xx(N1-p1+ii+1);
  end;

  %%% mise \`a jour du vecteur auxiliaire
  alpha2 = epp/fenerg ;
  alpha3 = rpnm1/benerg;
  cc(1) = alpha2;
  dd(p1+1) = alpha3;
  for ii=1:p1
    cc(ii+1) = cpri(ii) + alpha2*aa(ii+1);
  end;
  for ii=0:p1-1
    dd(ii+1) = dpri(ii+1) + alpha3*bb(ii+1);
  end;

  %%% mise \`a jour des scalaires
  gg = gg + epp^2/fenerg + hh^2/(1-ww);
  ww = ww + rpnm1^2/benerg + hh^2/(1-gg);
  hh = 0;
  for ii=0:p1
    hh = hh + xx(N1-ii)*cc(ii+1);
  end;

  %%% mise \`a jour des d\'ecalages temporels
  alphapr = 1/(1 + epp^2/(fenerg*(1-gg)));
  alphadpr = 1/(1 + rpnm1^2/(benerg*(1-ww)));
  for ii = 0:p1
    apri(ii+1) = alphapr*( aa(ii+1) + epp*cc(ii+1)/(1-gg));
    bpri(ii+1) = alphadpr*( bb(ii+1) + rpnm1*dd(ii+1)/(1-ww));
    cpri(ii+1) = cc(ii+1) + hh*dd(ii+1)/(1-ww);
    dpri(ii+1) = dd(ii+1) + hh*cc(ii+1)/(1-gg);
  end;
  epri = alphapr*fenerg;
  rpri = alphadpr*benerg;

  %%% mise \`a jour de l'ordre
  for ii=p1:-1:1
    Rmatr(ii+1) = Rmatr(ii) - xx(N1-p1)*xx(N1-ii-1);
    ROmatr(ii) = ROmatr(ii) - xx(p1+1)*xx(ii);
  end;
  help = 0 ;
  for ii=p1+1:N1-1
    help = help + xx(ii-p1-1+1)*xx(ii+1);
  end;
  Rmatr(1) = help;
  ROmatr(p1+1) = help;
  epsilonp = 0 ;
  epsilonm = 0 ;
  for ii=0:p1
    epsilonp = epsilonp + apri(ii+1)*Rmatr(ii+1);
    epsilonm = epsilonm + bpri(p1-ii+1)*ROmatr(p1-ii+1);
  end;

  gamma = epsilonp/rpri;
  beta = epsilonm/epri;
  rcf(p1+1) = gamma;
  rcb(p1+1) = beta;
  fenerg = epri - gamma*epsilonm;
  benerg = rpri - beta*epsilonp;
  for ii=1:p1
    aa(ii+1) = apri(ii+1) - gamma*bpri(ii);
    bb(ii+1) = bpri(ii)   - beta*apri(ii+1);
  end;

  aa(p1+1+1) = -gamma;
  bb(1) = -beta;
  bb(p1+1+1) = 1 ;
end;

sigma2 = abs(fenerg); %%% we choose the side 'aa'
rcb(1) = benerg;


%%% densit\'e spectrale de puissance
interm2=-j*2*pi/fe*[1:pp];
df=0.9765625;      %%% la dsp est calcul\'ee tous les df Hz
ff=-fe/2:df:fe/2;

interm3=interm2'*ff;
interm=1.+aa(2:pp+1)*exp(interm3);
mydsp = sigma2./(interm.*conj(interm));

figure(1);
clf;
grid on;
hold on;
plot(ff,mydsp,'linewidth',2);
xlabel('frequency (in Hz)','fontsize',20);
ylabel('magnitude','fontsize',20);
hold off;
drawnow;

