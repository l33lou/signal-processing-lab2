"""
Code MATLAB :

function [aa, sigma2, ref, ff, mydsp] = mylevinsondurbin (xx, pp, fe)
acf = xcorr(xx, pp+1, ’biased’);
acf(1:pp+1) = []; %% we remove the negative part
acf(1) = real(acf(1)); %% Levinson-Durbin requires c(1)==conj(c(1))
ref = zeros(pp,1);
gg = -acf(2)/acf(1);
aa = [ gg ];
sigma2 = real( ( 1 - gg*conj(gg)) * acf(1) ); %% real: removes any potential residual imaginary part
ref(1) = gg;
for tt = 2 : pp
gg = -(acf(tt+1) + aa * acf(tt:-1:2)’) / sigma2;
aa = [ aa + gg*conj(aa(tt-1:-1:1)), gg ];
sigma2 = sigma2 * ( 1 - real(gg*conj(gg)) ) ;
ref(tt) = gg;
end;

aa = [1, aa];
interm2=-j*2*pi/fe*[1:pp];
ff=-fe/2:10:fe/2;
for ii=1:length(ff)
interm=1.+aa(2:end)*exp(interm2*ff(ii))’;
mydsp(ii) = sigma2./(interm.*conj(interm)); %%% power spectral density
end;
figure(1);
clf;
grid on;
hold on;
plot(ff,mydsp);
hold off;

"""
import numpy as np
import matplotlib.pyplot as plt


def levinson_durbin(x, p, fe, energy_threshold, visualize=False):
    #Filtration of silence noise 
    rms = np.sqrt(np.mean(x**2))
    if rms < energy_threshold:
        return np.nan

    #Windowing
    xw =  x*np.hanning(len(x))
        
    #biased autocorrelation of x 
    autocorr_x = np.correlate(xw, xw, 'full')/len(xw)
    center = len(xw)-1
    acf = autocorr_x[center:center+p+2] #keeping only the positive part

    acf[0] = np.real(acf[0]) #levinson-durbin requires c(0)==c*(0)
    ref = np.zeros(p, dtype=complex)
    g = -acf[1]/acf[0]
    a = np.array([g])
    sigma2 = np.real( ( 1 - g*np.conjugate(g)) * acf[0] )
    ref[0] = g

    #Levinson-Durbin algorithm
    for t in range(2, p+1):
        g = -(acf[t] + np.dot(a, acf[1:t][::-1])) / sigma2
        a = np.append(a + g * np.conjugate(a[::-1]), g)
        sigma2 = sigma2*(1 - np.real(g*np.conjugate(g)))
        ref[t-1] = g

    a = np.insert(a, 0, 1.0)

    #Positive frequencies restricted to musical pitch range: 80 Hz to 1500 Hz
    fmin = 80.0
    fmax = 1500.0
    freq = np.arange(fmin, fmax + 1, 1)  # Or np.arange(0, fe / 2 + 1, 1)

    #Power spectral density 
    interm2 = -1j * 2 * np.pi / fe * np.arange(1, p + 1)
    A_matrix = np.exp(np.outer(freq, interm2))
    interm = 1.0 + np.dot(A_matrix, a[1:])

    mydsp = sigma2 / (np.abs(interm) ** 2)

    # 4. Direct peak finding without slicing
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