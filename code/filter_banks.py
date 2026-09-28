import numpy as np
import soundfile as sf

def generate_sound(f0, sr, Tf, namefile):
    Time = np.arange(0, Tf, 1/sr)
    S = []
    for i in range(len(Time)):
        s = 0
        for j in range(1, 5) : 
            s += (1/(j**2))*np.sin(2*np.pi*(j*f0)*Time[i])
        S.append(s)

    sf.write('../audios/'+namefile+'.wav', S, sr)

    return Time, S

if __name__=="__main__":
    f0 = 880 #Hz
    sr = 8000 #Hz
    Tf = 1 #s
    
    generate_sound(f0, sr, Tf, 'test3')