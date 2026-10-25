import matplotlib.pyplot as plt
import numpy as np
from tools.denoising import compute_snr, optimal_threshold_calculation


def plot_scalograms_comparison(coeffs_dict, N_samples, duration, wavelet_level):
    """
    1. Affiche les scalogrammes DWT en dB pour plusieurs signaux (type diapo 399).
    coeffs_dict : dictionnaire {'Titre du signal': coeffs_dwt}
    """
    num_plots = len(coeffs_dict)
    
    # Disposition en grille 2x2 si 3 signaux sont transmis (conforme diapo 399)
    if num_plots == 3:
        fig = plt.figure(figsize=(11, 7))
        axes = [fig.add_subplot(2, 2, i + 1) for i in range(3)]
    else:
        fig, axes = plt.subplots(1, num_plots, figsize=(4.5 * num_plots, 4), sharey=True)
        if num_plots == 1:
            axes = [axes]
        
    global_max = max([max([np.max(np.abs(c)) for c in coeffs]) for coeffs in coeffs_dict.values()])
    if global_max == 0:
        global_max = 1.0
        
    labels = [f'd{i}' for i in range(1, wavelet_level + 1)] + [f'a{wavelet_level}']

    for ax, (title, coeffs) in zip(axes, coeffs_dict.items()):
        # Réordonnancement : [cD1, cD2, ..., cDN, aN] du haut vers le bas
        subbands = coeffs[1:][::-1] + [coeffs[0]]
        
        matrix = []
        for sb in subbands:
            interp_sb = np.interp(np.linspace(0, 1, N_samples), np.linspace(0, 1, len(sb)), sb)
            matrix.append(interp_sb)
            
        mag = np.abs(np.array(matrix))
        
        # Calcul logarithmique en dB
        with np.errstate(divide='ignore'):
            db = 20 * np.log10(mag / global_max)
        
        # Masque strict pour les zones à énergie nulle (-90 dB -> Noir)
        db[mag == 0] = -90.0
        db = np.clip(db, -90, 0)

        # Affichage avec interpolation 'nearest' pour des bandes bien délimitées
        im = ax.imshow(db, aspect='auto', cmap='inferno', vmin=-90, vmax=0,
                       interpolation='nearest', extent=[0, duration, wavelet_level + 1, 0])
        ax.set_yticks(np.arange(0.5, wavelet_level + 1))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Subbands')
        ax.set_title(title, fontweight='bold', fontsize=10)
        fig.colorbar(im, ax=ax, label='dB')

    plt.tight_layout()
    plt.show()

def plot_temporal_signals(time, clean_signal, noisy_signal, denoised_signal, zoom_range=(0.5, 0.514)):
    """2. Affiche la comparaison temporelle (superposition) avec zoom."""
    mask = (time >= zoom_range[0]) & (time <= zoom_range[1])
    snr_in = compute_snr(clean_signal, noisy_signal)
    snr_out = compute_snr(clean_signal, denoised_signal)

    plt.figure(figsize=(9, 4.5))
    plt.plot(time[mask], noisy_signal[mask], 'r', label=f'signal with noise (SNR = {snr_in:.2f} dB)', alpha=0.8)
    plt.plot(time[mask], denoised_signal[mask], 'b', label=f'signal noise removed (SNR = {snr_out:.2f} dB)', linewidth=1.5)
    plt.plot(time[mask], clean_signal[mask], 'g', label='clean signal', linewidth=1.5)
    
    plt.xlim(zoom_range)
    plt.ylim([-1.2, 1.2])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparaison temporelle des signaux', fontweight='bold')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_coefficients_histogram(coeffs):
    """3. Affiche l'histogramme de la magnitude des coefficients d'ondelettes."""
    all_coeffs_mag = np.abs(np.concatenate(coeffs))
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(all_coeffs_mag, bins=60, range=(0, 2), color='#4a0052', edgecolor='black')
    plt.title('histogram of wavelet coefficients (signal with noise)', fontweight='bold')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_denoising_ratio_optimization(threshold_vals, snr_vals, best_threshold):

    plt.plot(threshold_vals, snr_vals, '#0072BD', linewidth=2)
    plt.plot(best_threshold, max(snr_vals), 'ro', label=f'Optimal threshold = {best_threshold:.3f}')
    plt.title('Threshold optimization', fontweight='bold')
    plt.xlabel('threshold')
    plt.ylabel('Denoising ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()