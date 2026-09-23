import librosa
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

#fft method on y signal sampled at rate sr 
def ffrequency(y, sr, n, energy_threshold, visualize=False) :

    T = np.linspace(0, len(y)/sr, len(y))

    # 1. Calcul de l'énergie RMS pour filtrer le silence/bruit
    rms = np.sqrt(np.mean(y**2))
    if rms < energy_threshold:
        return np.nan  # Silence ou bruit de fond
    
    # 2. Fenêtrage (Hanning) pour réduire le rayonnement spectral (leakage)
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

    return f0

#splitting the signal into 10ms long frames
def framing(y, sr, Tf):
    Nf = int(sr*Tf)
    frames = [ (y[i*Nf:(i+1)*Nf], sr) for i in range(len(y)//Nf) ]
    return frames

def traitement(s, fe, n, energy_threshold, Tf):
    frames = framing(s, fe, Tf)

    y0, sr0 = frames[0]
    f00 = ffrequency(y0, sr0, n, energy_threshold)
    T = [[0, f00]]

    for k in range(len(frames)) :
        y, sr = frames[k]
        f0 = ffrequency(y, sr, n, energy_threshold)
        temps = k*Tf
        if not np.isnan(f0):
            if len(T) == 0 or not np.isclose(f0, T[-1][1], atol=5.0):
                T.append([temps, f0])
        df = pd.DataFrame(T, columns=["Temps (s)", "Fréquence (Hz)"])
    return df

def df_to_tabularx(df, column_width='\\textwidth'):
    """
    Convertit un DataFrame Pandas en code LaTeX utilisant tabularx.
    """
    # Générer le code LaTeX standard avec Pandas
    latex_str = df.to_latex(index=False, escape=True)
    
    # Remplacer l'environnement tabular par tabularx
    latex_tabularx = latex_str.replace(
        "\\begin{tabular}", 
        f"\\begin{{tabularx}}{{{column_width}}}{{{'|'.join(['X']*len(df.columns))}}}"
    ).replace("\\end{tabular}", "\\end{tabularx}")
    
    return latex_tabularx


if __name__=="__main__":
    y1, sr1 = librosa.load('../audios/fluteircam.wav')   
    y2, sr2 = librosa.load('../audios/voiceP.wav')
    print(traitement(y1, sr1, 19, 0.01, 10E-3))
    print(traitement(y2, sr2, 19, 0.01, 10E-3))

