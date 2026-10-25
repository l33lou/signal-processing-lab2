import numpy as np
import pywt


def generate_two_sines(f1=100, f2=500, fs=8000, duration=1.5):
    time = np.arange(0, duration, 1/fs)
    s1 = np.sin(2 * np.pi * f1 * time)  # Basse fréquence (100 Hz)
    s2 = np.sin(2 * np.pi * f2 * time)  # Haute fréquence (500 Hz)
    s_sum = s1 + s2
    return time, s1, s2, s_sum


def separate_sources_dwt(s_sum, split_band_idx=4, wavelet='db4', level=8):
    coeffs_sum = pywt.wavedec(s_sum, wavelet, level=level)
    
    # Initialization
    coeffs_sep1 = [np.zeros_like(c) for c in coeffs_sum]  
    coeffs_sep2 = [np.zeros_like(c) for c in coeffs_sum]  

    for i in range(1, split_band_idx + 1):
        coeffs_sep2[-i] = coeffs_sum[-i].copy()
        
    for i in range(split_band_idx + 1, level + 1):
        coeffs_sep1[-i] = coeffs_sum[-i].copy()
    coeffs_sep1[0] = coeffs_sum[0].copy()  # Conservation de l'approximation aN

    # Reconstruction
    s1_sep = pywt.waverec(coeffs_sep1, wavelet)[:len(s_sum)]
    s2_sep = pywt.waverec(coeffs_sep2, wavelet)[:len(s_sum)]

    return s1_sep, s2_sep, coeffs_sep1, coeffs_sep2


def compute_separation_error(original, separated):
    """Calcul de l'erreur relative de reconstruction (norme L2 relative)."""
    return np.linalg.norm(original - separated) / np.linalg.norm(original)