import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

def spectrogram(y, sr, f_min=0, f_max=15000, nperseg=2048, nfft=4096, overlap_ratio=0.75):
    noverlap = int(nperseg * overlap_ratio)

    freqs, times, Sxx = scipy.signal.spectrogram(y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs_cut = freqs[mask]
    Sxx_db = 10*np.log10(Sxx[mask, :] + 1e-12)

    plt.figure(figsize=(9, 4.5))
    plt.pcolormesh(times, freqs_cut, Sxx_db, cmap='plasma', shading='nearest')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Temps (s)')
    plt.ylabel('Fréquence (Hz)')
    plt.title('Spectrogramme (SciPy)')
    plt.tight_layout()
    plt.show()
