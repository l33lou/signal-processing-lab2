import librosa
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt



#fft methods
def ffrequency(y, sr, n) :

    T = np.linspace(0, len(y)/sr, len(y))

    #FFT
    fourier = scipy.fft.rfft(y, 2**n)
    mag = np.abs(fourier)
    freq = scipy.fft.rfftfreq(2**n, 1/sr)

    #Fundamental frequency
    max_index = np.argmax(mag)
    f0 = freq[max_index]

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

    return f0

if __name__=="__main__":
    y1, sr1 = librosa.load('../audios/fluteircam.wav')   
    y2, sr2 = librosa.load('../audios/voiceP.wav')   
    print(len(y1), len(y2))

    print("f01 =", ffrequency(y1, sr1, 20))
    print("f02 =", ffrequency(y2, sr2, 19))


