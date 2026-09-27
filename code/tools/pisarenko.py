import numpy as np
import matplotlib.pyplot as plt

def pisarenko(xx, pp, fe, visualize=False):
    MM1 = 2 * pp
    MM = MM1 + 1
    NN = len(xx)
    xx = xx - np.mean(xx)

    # Autocorrélation biaisée (équivalent à xcorr(xx, MM1, 'biased'))
    autocorr = np.correlate(xx, xx, mode="full") / NN
    center = NN - 1
    acf = autocorr[center - MM1 : center + MM1 + 1]

    # Construction de la matrice d'autocorrélation (Toeplitz)
    lMM = len(acf)
    rrr1 = acf[MM1:lMM].reshape(-1, 1)
    for ii in range(1, MM1 + 1):
        col = acf[MM1 - ii : lMM - ii].reshape(-1, 1)
        rrr1 = np.hstack((rrr1, col))

    rrr = rrr1.T
    irrr = np.linalg.inv(rrr)

    # Méthode de la puissance inverse pour trouver le vecteur propre
    # associé à la plus petite valeur propre
    mu = 1.0
    muold = mu
    aaa = np.random.rand(1, 2 * pp + 1)
    kk = 1

    while kk == 1:
        mu = float((aaa @ rrr @ aaa.T) / (aaa @ aaa.T))
        aaa = (irrr @ (mu * aaa.T)).T
        if abs(muold - mu) < 1e-10:
            kk = 0
        muold = mu

    aaa = aaa.flatten()

    # Extraction des racines (polynôme des fréquences)
    racines = np.roots(aaa)
    frequ = np.log(racines) / (1j * 2 * np.pi) * fe
    frequ = np.abs(frequ[::2])

    # Grille fréquentielle
    df = 0.9765625
    ff = np.arange(-fe / 2, fe / 2 + df, df)
    mydsp = np.zeros(len(ff)) + 1e-10

    # Résolution du système pour les amplitudes des sinusoïdes
    coscos = np.zeros((pp, pp))
    for ii in range(1, pp + 1):
        for jj in range(1, pp + 1):
            coscos[ii - 1, jj - 1] = np.cos(2.0 * np.pi * ii * frequ[jj - 1] / fe)

    # Correspondance des indices MATLAB acf(2*pp+1:3*pp)
    acf_slice = acf[2 * pp : 3 * pp]
    alpha = np.linalg.inv(coscos) @ acf_slice
    amp = np.sqrt(2.0 * np.abs(alpha))

    # Construction du spectre de raies (fréquences positives et négatives)
    for ii in range(pp):
        posi_pos = np.argmin(np.abs(ff - frequ[ii]))
        mydsp[posi_pos] = amp[ii]

        posi_neg = np.argmin(np.abs(ff + frequ[ii]))
        mydsp[posi_neg] = amp[ii]

    if visualize : 
        plt.plot(ff, mydsp)
        #plt.plot(f0, mydsp[arg_f0], 'x')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.title('Levinson-Durbin')
        plt.show()

    return ff, mydsp