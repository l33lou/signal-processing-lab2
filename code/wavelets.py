from tools.denoising import (generate_signal, 
                             wavelet_decompose, 
                             wavelet_denoise, 
                             wavelet_reconstruct, 
                             monte_carlo_optimal_threshold )
from tools.denoising_visualisation import ( plot_scalograms_comparison, 
                                           plot_temporal_signals, 
                                           plot_coefficients_histogram, 
                                           optimal_threshold_calculation, 
                                           plot_denoising_ratio_optimization )

### Execution of wavelet decomposition, denoising, and reconstruction               

# Generating noisy signal
F0 = 500    
FS = 8000 
DURATION = 1.5
NOISE_LEVEL = 0.4
THRESHOLD = 0.45

time, clean, noisy = generate_signal(f0=F0, fs=FS, duration=DURATION, noise_level=NOISE_LEVEL)

# Applying wavelet decomposition to the noisy signal
WAVELET = 'db4'  # wavelet type
LEVEL = 4        # decomposition level

coeffs_clean = wavelet_decompose(clean, wavelet=WAVELET, level=LEVEL)
coeffs_noisy = wavelet_decompose(noisy, wavelet=WAVELET, level=LEVEL)

# Denoising the signal using wavelet thresholding and reconstructing it
coeffs_denoised = wavelet_denoise(coeffs_noisy, threshold=THRESHOLD, mode='soft')
denoised = wavelet_reconstruct(coeffs_denoised, wavelet=WAVELET, target_len=len(clean))

scalograms_data = {
    'scalogram of clean signal': coeffs_clean,
    'scalogram of signal with noise': coeffs_noisy,
    'scalogram of denoised signal': coeffs_denoised
}

# Finding the optimal threshold for denoising 
threshold_vals, snr_vals, best_threshold = optimal_threshold_calculation(clean, coeffs_noisy, WAVELET, LEVEL)

# Comparison with optimal threshold using Monte-Carlo simulation
threshold_vals_mc, snr_vals_mc, best_threshold_mc = monte_carlo_optimal_threshold(clean, wavelet=WAVELET, level=LEVEL)


if __name__ == '__main__':
    plot_scalograms_comparison(scalograms_data, N_samples=len(time), duration=DURATION, wavelet_level=LEVEL)
    plot_temporal_signals(time, clean, noisy, denoised, zoom_range=(0.5, 0.514))
    plot_coefficients_histogram(coeffs_noisy)
    plot_denoising_ratio_optimization(threshold_vals, snr_vals, best_threshold)
    plot_denoising_ratio_optimization(threshold_vals_mc, snr_vals_mc, best_threshold_mc)