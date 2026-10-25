import numpy as np
import soundfile as sf

def generate_sound(f0, sr, Tf, namefile, beta=0, fv=0):
    Time = np.arange(0, Tf, 1/sr)
    S = np.zeros_like(Time)
    for j in range(1, 5):
        S += (1/(j**2))*np.sin(2*np.pi*(j*f0)*Time + j*beta*np.sin(2*np.pi*fv*Time))
        
    sf.write('../audios/'+namefile+'.wav', S, sr)

    return Time, S