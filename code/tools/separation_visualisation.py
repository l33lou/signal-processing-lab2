import matplotlib.pyplot as plt
import numpy as np


def build_scalogram_matrix(coeffs, N_samples):
    """
    Construit la matrice temps-sous-bandes [d1, d2, ..., aN] en dB avec
    centrage par rembourrage de zéros (reproduit le profil en entonnoir de separation1.m).
    """
    # Réordonnancement : [cD1, cD2, ..., cDN, aN] du haut vers le bas
    subbands = coeffs[1:][::-1] + [coeffs[0]]
    
    matrix = []
    for sb in subbands:
        M = len(sb)
        if M < N_samples:
            # Centrage des M coefficients dans un vecteur de taille N_samples
            pad_left = (N_samples - M) // 2
            pad_right = N_samples - M - pad_left
            centered_sb = np.pad(sb, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            centered_sb = sb[:N_samples]
            
        matrix.append(centered_sb)
        
    mag = np.abs(np.array(matrix))
    max_val = np.max(mag)
    if max_val == 0:
        max_val = 1.0
        
    with np.errstate(divide='ignore'):
        db = 20 * np.log10(mag / max_val)
        
    # Masquage strict du fond à -80 dB (noir)
    db[mag == 0] = -80.0
    return np.clip(db, -80, 0)


def plot_single_scalogram(coeffs, N_samples, duration, level, title="scalogram -- signal 1 + signal 2"):
    """Affiche le scalogramme du signal combiné (diapositive 403)."""
    db_data = build_scalogram_matrix(coeffs, N_samples)
    labels = [f'd{i}' for i in range(1, level + 1)] + [f'a{level}']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(db_data, aspect='auto', cmap='inferno', vmin=-80, vmax=0,
                   interpolation='nearest', extent=[0, N_samples, level + 1, 0])
    
    ax.set_yticks(np.arange(0.5, level + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel('Subbands')
    ax.set_xlabel('Time (samples)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('dB')
    plt.tight_layout()
    plt.show()


def plot_dual_scalograms(coeffs1, coeffs2, N_samples, duration, level, titles=("scalogram -- signal 1", "scalogram -- signal 2")):
    """Affiche une paire de scalogrammes côte à côte (diapositive 404)."""
    db1 = build_scalogram_matrix(coeffs1, N_samples)
    db2 = build_scalogram_matrix(coeffs2, N_samples)
    labels = [f'd{i}' for i in range(1, level + 1)] + [f'a{level}']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    for ax, db_data, title in zip([ax1, ax2], [db1, db2], titles):
        im = ax.imshow(db_data, aspect='auto', cmap='inferno', vmin=-80, vmax=0,
                       interpolation='nearest', extent=[0, N_samples, level + 1, 0])
        ax.set_yticks(np.arange(0.5, level + 1))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylabel('Subbands')
        ax.set_xlabel('Time (samples)')
        ax.set_title(title, fontweight='bold', fontsize=10)
        fig.colorbar(im, ax=ax, label='dB')

    plt.tight_layout()
    plt.show()