import numpy as np
import scipy
import matplotlib.pyplot as plt

#fft method on y signal sampled at rate sr 
def fft(y, sr, n, energy_threshold, visualize=False, vals=False) :

    T = np.linspace(0, len(y)/sr, len(y))

    #Filtration of silence noise 
    rms = np.sqrt(np.mean(y**2))
    if rms < energy_threshold:
        return np.nan
    
    #Hanning windowing to reduce leakage
    windowed = y * np.hanning(len(y))

    #FFT
    fourier = scipy.fft.rfft(windowed, 2**n)
    mag = np.abs(fourier)
    freq = scipy.fft.rfftfreq(2**n, 1/sr)

    #Fundamental frequency
    max_index = np.argmax(mag)
    f0 = freq[max_index]

    if visualize :

        #Visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(T, y)
        ax1.set_title('Time domain signal')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        ax2.plot(freq, mag)
        ax2.plot(f0, mag[max_index], 'x')
        ax2.set_title("Magnitude spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    if vals :
        return T, freq, mag

    return f0