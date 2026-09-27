import librosa
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from scipy.signal import find_peaks
import time

#splitting the signal into 10ms long frames
def framing(y, sr, Tf):
    Nf = int(sr*Tf)
    frames = [ (y[i*Nf:(i+1)*Nf], sr) for i in range(len(y)//Nf) ]
    return frames

#fft method on y signal sampled at rate sr 
def fft(y, sr, n, energy_threshold, visualize=False) :

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

    return f0


def autocorrelation(y, sr, energy_threshold, visualization=False):
    #Filtration of silence noise 
    rms = np.sqrt(np.mean(y**2))
    if rms < energy_threshold:
        return np.nan

    #Compute autocorrelation of y
    Gamma = np.correlate(y, y, 'full')
    Time = np.linspace(0, len(y)*(1/sr), len(y))
    y_corr_pos = Gamma[-len(y):]
    y = np.insert(y_corr_pos, 0, 0)

    #Deduce fundamental frequency y0
    peaks, _ = find_peaks(y)
    peaks_sorted = peaks[np.argsort(y[peaks])[::-1]]
    first_peak = peaks_sorted[0]-1
    second_peak = peaks_sorted[1]-1
    f0 = sr/(second_peak-first_peak)

    #Visualization
    if visualization == True :
        plt.plot(Time, y_corr_pos, label="Autocorrelation")
        plt.plot(first_peak/sr, y_corr_pos[first_peak], 'rx', label="Highest peak")     #-1 bc inserted 0
        plt.plot(second_peak/sr, y_corr_pos[second_peak], 'mx', label="Second highest peak")
        plt.xlabel('Time offset (s)')
        plt.ylabel('?')
        plt.legend()
        plt.grid(True)
        plt.show()
            
    return f0    

def traitement(s, fe, n, energy_threshold, Tf, frequency_tolerance, method='autocorrelation'):
    t0 = time.time()

    frames = framing(s, fe, Tf)
    T = []

    for k, (y, sr) in enumerate(frames) :
        t_start = round(k*Tf, 2)
        t_end = round((k+1)*Tf, 2)
        if method == 'fft':
            f0 = fft(y, sr, n, energy_threshold)
        elif method == 'autocorrelation':
            f0 = autocorrelation(y, sr, energy_threshold)
        else:
            raise ValueError("Invalid method : please choose either 'fft' or 'autocorrelation'.")

        if not np.isnan(f0):
            if len(T) == 0 or not np.isclose(f0, T[-1][2], atol=frequency_tolerance):
                T.append([t_start, t_end, round(f0, 2)])
            else : 
                T[-1][1] = t_end

    #Filtering notes not long enough to exist and fixing time continuity
    T = [ note for note in T if (note[1]-note[0]) >= 50E-3 ]
    for i in range(1, len(T)):
        T[i][0] = T[i - 1][1]

    df = pd.DataFrame(T, columns=["Start time (s)", "End time (s)", "Frequency (Hz)"])

    tf = time.time()
    computing_time = round(tf - t0, 2)

    print("Computing time : ", computing_time, "s")
    print("Method :", method)
    print(df)

    return df

def df_to_tabularx(df, column_width='\\textwidth'):
    latex_str = df.to_latex(index=False, escape=True)
    latex_tabularx = latex_str.replace(
        "\\begin{tabular}", 
        f"\\begin{{tabularx}}{{{column_width}}}{{{'|'.join(['X']*len(df.columns))}}}"
        ).replace("\\end{tabular}", "\\end{tabularx}")
    
    print(latex_tabularx)

if __name__=="__main__":
    y1, sr1 = librosa.load('../audios/fluteircam.wav')   
    #y2, sr2 = librosa.load('../audios/voiceP.wav')
    #print(traitement(y1, sr1, 19, 0.01, 10E-3))
    #print(traitement(y2, sr2, 19, 0.01, 10E-3))
    frames = framing(y1, sr1, 10E-3)
    y1i, sri = frames[int(len(frames)//2)]
    #autocorrelation(y1i, sri, True)
    df1 = traitement(y1, sr1, 19, 0.01, 10E-3, 20, 'fft')
    df2 = traitement(y1, sr1, 19, 0.01, 10E-3, 20, 'autocorrelation')

    tab1 = df_to_tabularx(df1)
    tab2 = df_to_tabularx(df2)



