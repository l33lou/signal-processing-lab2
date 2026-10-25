import numpy as np
from tools.separation import (
    compute_separation_error,
    generate_two_sines,
    separate_sources_dwt,
)
from tools.separation_visualisation import (
    plot_dual_scalograms,
    plot_single_scalogram,
)
from tools.denoising import wavelet_decompose

### Execution of source separation using wavelet decomposition and coefficient masking     

F1 = 100               
F2 = 500              
FS = 8000           
DURATION = 0.3125     
WAVELET = 'db4'        
LEVEL = 20            
SPLIT_BAND = 7        

# Generating the two sine waves and their sum
time, s1, s2, s_sum = generate_two_sines(f1=F1, f2=F2, fs=FS, duration=DURATION)
N_samples = len(time)

# Decomposing the signals using wavelet transform
coeffs_s1 = wavelet_decompose(s1, wavelet=WAVELET, level=LEVEL)
coeffs_s2 = wavelet_decompose(s2, wavelet=WAVELET, level=LEVEL)
coeffs_sum = wavelet_decompose(s_sum, wavelet=WAVELET, level=LEVEL)

# Source separation using wavelet coefficient masking
s1_sep, s2_sep, coeffs_sep1, coeffs_sep2 = separate_sources_dwt(s_sum, split_band_idx=SPLIT_BAND, wavelet=WAVELET, level=LEVEL)

# Computing the relative reconstruction errors for both separated signals
err1 = compute_separation_error(s1, s1_sep)
err2 = compute_separation_error(s2, s2_sep)

if __name__ == '__main__':
    plot_single_scalogram(coeffs_sum, N_samples=N_samples, duration=DURATION, level=LEVEL, title="scalogram of signal 1 + signal 2"    )
    plot_dual_scalograms(coeffs_s1, coeffs_s2, N_samples=N_samples, duration=DURATION, level=LEVEL, titles=("scalogram of signal 1", "scalogram of signal 2"))
    plot_dual_scalograms(coeffs_sep1, coeffs_sep2, N_samples=N_samples, duration=DURATION, level=LEVEL,titles=("scalogram of denoised signal 1", "scalogram of denoised signal 2"))
    print(f"Erreur relative de reconstruction (Signal 1 - 100 Hz) : {err1:.4f}")
    print(f"Erreur relative de reconstruction (Signal 2 - 500 Hz) : {err2:.4f}")