import librosa
import numpy as np
import pandas as pd
import time

#Custom functions
from tools.fft import fft
from tools.autocorrelation import autocorrelation
from tools.levinson_durbin import levinson_durbin


#Splitting the signal into 10ms long frames
def framing(y, sr, Tf):
    Nf = int(sr*Tf)
    frames = [ (y[i*Nf:(i+1)*Nf], sr) for i in range(len(y)//Nf) ]
    return frames

#Main script
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
    y2, sr2 = librosa.load('../audios/voiceP.wav')

    #frames = framing(y1, sr1, 10E-3)
    #y1i, sri = frames[int(len(frames)//2)]
    #autocorrelation(y1i, sri, True)

    df1 = traitement(y1, sr1, 19, 0.01, 10E-3, 20, 'fft')
    df2 = traitement(y1, sr1, 19, 0.01, 10E-3, 20, 'autocorrelation')

    tab1 = df_to_tabularx(df1)
    tab2 = df_to_tabularx(df2)



