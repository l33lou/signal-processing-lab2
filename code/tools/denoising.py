import numpy as np
import pywt

#### Functions for wavelet decomposition, denoising, and reconstruction

def generate_signal(f0, fs=8000, duration=1.5, noise_level=0.4):
    time = np.arange(0, duration, 1/fs)
    clean_sig = np.sin(2 * np.pi * f0 * time)
    noise = np.random.normal(0, noise_level, len(time))
    noisy_sig = clean_sig + noise
    return time, clean_sig, noisy_sig

def wavelet_decompose(signal, wavelet='db4', level=4):
    return pywt.wavedec(signal, wavelet, level=level)

def wavelet_denoise(coeffs, threshold, mode='soft'):
    coeffs_thresh = [coeffs[0]]  # Conservation de l'approximation a_N intacte
    for cd in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(cd, threshold, mode=mode))
    return coeffs_thresh

def wavelet_reconstruct(coeffs, wavelet='db4', target_len=None):
    rec = pywt.waverec(coeffs, wavelet)
    if target_len is not None:
        rec = rec[:target_len]
    return rec

def compute_snr(clean_signal, test_signal):
    noise = clean_signal - test_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

# Optimal threshold for our noisy signal
def optimal_threshold_calculation(clean_signal, coeffs_noisy, wavelet='db4', level=4):
    threshold_vals = np.linspace(0.1, 1.0, 30)
    snr_vals = []
    for threshold in threshold_vals:
        coeffs_denoised = wavelet_denoise(coeffs_noisy, threshold=threshold, mode='soft')
        denoised_signal = wavelet_reconstruct(coeffs_denoised, wavelet, target_len=len(clean_signal))
        snr_val = compute_snr(clean_signal, denoised_signal)
        snr_vals.append(snr_val)
    best_threshold = threshold_vals[np.argmax(snr_vals)]

    return threshold_vals, snr_vals, best_threshold

# Optimal threshold for any noisy signal, using Monte-Carlo simulation
def monte_carlo_optimal_threshold(clean_signal, wavelet='db4', level=4, num_trials=100):
    thresholds = np.linspace(0.1, 1.0, 30)
    denoising_ratios = []

    for threshold in thresholds:
        err_noisy_accum = 0.0
        err_denoised_accum = 0.0
        
        for _ in range(num_trials):
            noise = np.random.normal(0, 0.4, len(clean_signal))
            noisy_signal = clean_signal + noise
            coeffs_noisy = wavelet_decompose(noisy_signal, wavelet=wavelet, level=level)
            coeffs_denoised = wavelet_denoise(coeffs_noisy, threshold=threshold, mode='soft')
            denoised_signal = wavelet_reconstruct(coeffs_denoised, wavelet=wavelet, target_len=len(clean_signal))
            
            err_noisy_accum += np.sum((noisy_signal - clean_signal) ** 2)
            err_denoised_accum += np.sum((denoised_signal - clean_signal) ** 2)
            
        denoising_ratios.append(err_noisy_accum / err_denoised_accum)

    best_threshold = thresholds[np.argmax(denoising_ratios)]
    return thresholds, denoising_ratios, best_threshold
