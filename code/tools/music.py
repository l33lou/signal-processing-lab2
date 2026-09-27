import numpy as np


def mymusic(xx, pp, MM, fe):
    # Verification des parametres
    if MM <= pp:
        raise ValueError("It is absolutely essential that MM > pp !!!")

    MM1 = MM - 1
    NN = len(xx)
    xx = xx - np.mean(xx)

    # 1. Autocorrelation non biaisee (xcorr 'unbiased')
    autocorr = np.correlate(xx, xx, mode="full")
    center = NN - 1
    lags = np.arange(-MM1, MM1 + 1)
    scale = NN - np.abs(lags)
    acf = autocorr[center - MM1 : center + MM1 + 1] / scale

    # 2. Construction de la matrice d'autocorrelation Toeplitz (MM x MM)
    col_0 = acf[MM1 : 2 * MM1 + 1].reshape(-1, 1)
    cols = [col_0]
    for ii in range(1, MM1 + 1):
        cols.append(acf[MM1 - ii : 2 * MM1 + 1 - ii].reshape(-1, 1))

    rrr = np.hstack(cols).T

    # 3. Decomposition en valeurs/vecteurs propres et tri decroissant
    lambda_val, v = np.linalg.eig(rrr)
    pl = np.argsort(lambda_val)[::-1]

    # 4. Construction de la matrice de projection du sous-espace bruit
    deni = np.zeros((MM, MM), dtype=v.dtype)
    for ii in range(pp, MM):
        vec = v[:, pl[ii]].reshape(-1, 1)
        deni += vec @ np.conj(vec).T

    # 5. Grille frequentielle
    df = 0.9765625
    ff = np.arange(-fe / 2, fe / 2 + df, df)

    # 6. Calcul du pseudo-spectre MUSIC (vectorise)
    n_vec = np.arange(MM)
    # Matrice des vecteurs d'orientation / steering vectors (len(ff) x MM)
    E = np.cos(2 * np.pi * np.outer(ff, n_vec) / fe)

    # den = diag(E * deni * E^T)
    den = np.real(np.sum((E @ deni) * E, axis=1))
    mydsp = np.abs(1.0 / den)

    # Soustraction du composant non nul minimal
    mydsp = mydsp - np.min(mydsp)

    return ff, mydsp