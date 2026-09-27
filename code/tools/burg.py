"""
Code MATLAB

function [aa, sigma2, kk, ff, mydsp] = myburg(xx, pp, fe)
% Initializations
N = length(xx);
xx = xx(:);
ef = xx;
eb = xx;
aa = 1;
EE = xx’*xx./N;
kk = zeros(1, pp);% Pre-allocation of ’kk’ (to improve execution speed)

for tt=1:pp
    % Compute the following reflection coefficient
    efp = ef(2:end);
    ebp = eb(1:end-1);
    num = -2.*ebp’*efp; %%% it’s -Nk (F5)
    den = efp’*efp+ebp’*ebp; %%% it’s Dk (F6)
    kk(1,tt) = num./den; %%% it’s −γ in the slides (F7)
    % Updating ’forward’ and ’backward’ prediction errors
    ef = efp + kk(tt)*ebp; %%% (F3)
    eb = ebp + kk(tt)’*efp; %%% (F4)
    aa=[aa;0] + kk(1,tt)*[0;conj(flipud(aa))];% Updating the AR coefficients (F1 and F2)
    EE(tt+1) = (1 - kk(1,tt)’*kk(1,tt))*EE(tt);% Global prediction error update (E5)
end
aa = aa(:).’;
sigma2=EE(end); %%% not necessarily well estimated; note: it’s EE

%%% power spectral density
interm2=-j*2*pi/fe*[1:pp];
ff=-fe/2:10:fe/2;

for ii=1:length(ff)
    interm=1.+aa(2:end)*exp(interm2*ff(ii))’;
    mydsp(ii) = sigma2./(interm.*conj(interm));
end;
"""
import numpy as np
import matplotlib.pyplot as plt

def burg(y, p, sr, energy_threshold, visualize=False):
    #Filtration of silence noise 
    rms = np.sqrt(np.mean(y**2))
    if rms < energy_threshold:
        return np.nan

    #Hanning windowing to reduce leakage
    y = y*np.hanning(len(y))
    
    N = len(y)
    ef = y.copy()
    eb = y.copy()
    a = 1
    E = 1/N*(y @ y.T)
    k = np.zeros((1, p))

    for t in range(p):
        efp = ef[1:]
        ebp = eb[:-1]
        num = -2*np.vdot(ebp, efp)
        den = np.vdot(efp, efp).real + np.vdot(ebp, ebp).real
        k_val = num / den if den !=0 else 0.0

        #Updating forward and backward prediction errors
        ef = efp + k_val*ebp
        eb = ebp + np.conjugate(k_val) * efp
        a = np.append(a, 0.0) + k_val*np.append(0.0, np.conjugate(np.flip(a)))
        E *= 1.0 - np.abs(k_val)**2

    sigma2 = E.real
    a_real = a.real

    freq = np.arange(80, 1500 + 1, 1)  # 1 Hz resolution
    interm2 = -1j * 2 * np.pi / sr * np.arange(1, p + 1)
    A_matrix = np.exp(np.outer(freq, interm2))
    interm = 1.0 + np.dot(A_matrix, a_real[1:])

    mydsp = sigma2 / (np.abs(interm) ** 2)

    arg_f0 = np.argmax(mydsp)
    f0 = freq[arg_f0]

    if visualize : 
        plt.plot(freq, mydsp)
        plt.plot(f0, mydsp[arg_f0], 'x')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.title('Levinson-Durbin')
        plt.show()

    return f0