import numpy as np


def myprony(xx, pp, nco, fe, rec):
    """Méthode de Prony pour l'estimation spectrale et la recherche de poles.

    Parameters:
        xx  : array-like, signal d'entrée
        pp  : int, ordre du modèle (nombre de pôles complexes)
        nco : int, nombre maximal de tentatives en cas de matrice mal
        conditionnée
        fe  : float, fréquence d'échantillonnage (Hz)
        rec : int, 0 pour la DSP continue de Prony, 1 pour le spectre de raies

    Returns:
        aa    : ndarray, coefficients du filtre AR [1, a1, a2, ..., app]
        ff    : ndarray, vecteur des fréquences (Hz)
        mydsp : ndarray, densité spectrale de puissance ou spectre de raies
    """
    tsig = len(xx)
    df = 0.9765625
    ff = np.arange(-fe / 2, fe / 2 + df, df)

    strt = 0
    count = 0
    done = 0
    aa = np.array([])

    while done < 1:
        ok1 = False
        ok2 = False

        # 1. Construction de la matrice yy (pp x pp) et du vecteur y1
        cols_y = []
        for ii in range(pp):
            # Extraction rétrograde des échantillons pour chaque colonne
            col_ii = xx[strt + ii : strt + pp + ii][::-1]
            cols_y.append(col_ii)
        yy = np.column_stack(cols_y)

        y1 = xx[strt + pp : strt + 2 * pp]

        # Conditionnement de la matrice yy (1 / cond(yy))
        cond_yy = 1.0 / np.linalg.cond(yy) if np.linalg.cond(yy) != 0 else 0

        if cond_yy > 1.52721e-17:
            ok1 = True
            invaa = np.linalg.inv(yy)
            aa_rest = -invaa @ y1
            aa = np.concatenate(([1.0], aa_rest))

            # Extraction des racines (pôles)
            racines = np.roots(aa)
            frequ = (1.0 / (2.0 * np.pi)) * np.arctan2(
                np.imag(racines), np.real(racines)
            )

            # 2. Construction de la matrice zz (Vandermonde basée sur les racines)
            cols_z = [racines**ii for ii in range(pp)]
            zz = np.column_stack(cols_z)

            y2 = xx[0:pp]  # Conforme à xx(1:pp) du code MATLAB original

            cond_zz = 1.0 / np.linalg.cond(zz) if np.linalg.cond(zz) != 0 else 0
            if cond_zz > 1.52721e-17:
                ok2 = True
                invzz = np.linalg.inv(zz)

        if ok1 and ok2:
            hh = invzz @ y2
            done = 1
        else:
            print("pc ", end="", flush=True)
            count += 1
            if count == nco:
                done = 2
            else:
                # Tirage d'un nouveau point de départ aléatoire
                strt = int(np.round(np.random.rand() * (tsig - 2 * pp)))

    # 3. Calcul de la DSP ou du spectre de raies
    ff_norm = ff / fe  # Fréquences normalisées (-0.5 à 0.5)

    if done == 1:
        if rec == 0:
            # DSP Prony continue
            sff = np.zeros(len(ff_norm), dtype=complex)
            za = np.exp(-1j * 2 * np.pi * ff_norm)
            zb = np.conj(za)
            for ii in range(pp):
                term1 = 1.0 / (1.0 - racines[ii] * za)
                term2 = 1.0 / (1.0 - (np.conj(racines[ii]) * zb) ** (-1))
                sff += hh[ii] * (term1 - term2)
            mydsp = np.abs(sff) ** 2 / fe
        else:
            # Spectre de raies
            mydsp = np.zeros(len(ff_norm)) + 1e-10
            for ii in range(pp):
                posi_pos = np.argmin(np.abs(ff_norm - frequ[ii]))
                mydsp[posi_pos] = np.abs(hh[ii])

                posi_neg = np.argmin(np.abs(ff_norm + frequ[ii]))
                mydsp[posi_neg] = np.abs(hh[ii])
    else:
        print("unable to comply ", end="", flush=True)
        mydsp = np.zeros(len(ff_norm)) + 1e-15
        mydsp[int(np.round(len(ff_norm) / 2))] = 1e-5
        aa = np.array([])

    return aa, ff, mydsp