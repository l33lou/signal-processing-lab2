import numpy as np
import soundfile as sf
import scipy.fft
import scipy.signal
import librosa

from tools.filtrepython import apply_filter
from tools.fft import fft
from tools.spectrogram import spectrogram

def generate_sound(f0, sr, Tf, namefile, beta=0, fv=0):
    Time = np.arange(0, Tf, 1/sr)
    S = np.zeros_like(Time)
    for j in range(1, 5):
        S += (1/(j**2))*np.sin(2*np.pi*(j*f0)*Time + beta*np.sin(2*np.pi*fv*Time))
        
    sf.write('../audios/'+namefile+'.wav', S, sr)

    return Time, S


def single_tuned(s, n, threshold):

    # Windowing and FFT
    fourier = scipy.fft.rfft(s*np.hanning(len(s)), 2**n)
    mag = np.abs(fourier)
    max_mag = np.max(mag)

    if max_mag == 0:
        return True
    
    peaks, _ = scipy.signal.find_peaks(mag, height=threshold * max_mag, distance=10)
    
    return len(peaks) <= 1

def cascading_filtration( Time, S, n, threshold):
    cascade = [[(Time, S)]]

    while True : 
        current_level = cascade[-1]
        next_level = []
        all_single = True

        for time_sig, sig in current_level :
            if not single_tuned(sig, n, threshold):
                all_single = False
                Time_l, S_l = apply_filter(time_sig, sig, filter_type='low')
                Time_h, S_h = apply_filter(time_sig, sig, filter_type='high')
                next_level.extend([(Time_l, S_l), (Time_h, S_h)])
            else :
                next_level.append((time_sig, sig))

        if all_single :
            break

        cascade.append(next_level)

        print("Necessary number of layers to decompose the signal into multiple single-tuned ones : ", len(cascade)-1)

    return cascade[-1]

if __name__=="__main__":
    f0 = 880 #Hz
    sr = 8000 #Hz
    Tf = 1 #s
    
    time, s = generate_sound(f0, sr, Tf, 'test3')
    sr = time[1]-time[0]
    #fft(s, sr, 12, 0.01, visualize=True)
    #cascading_filtration(time, s, n=12, threshold=0.03)
    flute, srf = librosa.load('../audios/fluteircam.wav')
    voice, srp = librosa.load('../audios/voiceP.wav')
    #spectrogram(flute, srf)
    #spectrogram(voice, srp)
    #spectrogram(s, sr)
