import numpy as np 
import matplotlib.pyplot as plt
import scipy as sc 

def getS(f0, phi, fe, Tend):

    T = np.arange(0, Tend, 1/fe)
    F = np.sin(2*np.pi*f0*T + phi)

    #plt.plot(T, F, 'x')
    #plt.show()

    return T, F

def magnitude_spectrum(f0, phi, fe, Tend):

    T, F = getS(f0, phi, fe, Tend)
    N = len(F)

    fourier_transform = sc.fft.rfft(F)
    mag = np.abs(fourier_transform)*2.0/N
    freq = sc.fft.rfftfreq(N, 1/fe)

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time Domain Plot
    ax1.plot(T, F, 'x-', color='tab:blue', label="s(t)")
    ax1.set_title("Time Domain Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Frequency Domain Plot (Discrete spectrum representation)
    ax2.plot(freq, mag)
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return None

def magnitude_spectrum_padded(f0, phi, fe, Tend):

    N = Tend/fe
    Npad = 2**(int(np.log2(N))+2)
    Tend_pad = Npad*fe

    T = np.linspace(0, (Npad-1)/fe, Npad)
    _, Fpadded = getS(f0, phi, fe, Tend_pad)
    
    F_padded = np.concatenate((F, np.zeros(Npad-N)))
    print("len(F_padded)=Npad", len(F_padded)==Npad)

    fourier_transform = sc.fft.rfft(F_padded)
    mag = np.abs(fourier_transform)*2.0/Npad
    freq = sc.fft.rfftfreq(Npad, 1/fe)

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time Domain Plot
    ax1.plot(T, F_padded, 'x-', color='tab:blue', label="s(t)")
    ax1.set_title("Time Domain Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Frequency Domain Plot (Discrete spectrum representation)
    ax2.plot(freq, mag)
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return None

if __name__=="__main__":
    f0 = 50         #Hz
    phi = 2*np.pi*(50/512)
    fe = 512        #Hz
    Tend = 0.06     #s

    #getS(f0, phi, fe, Tend)
    magnitude_spectrum(f0, phi, fe, Tend)
    magnitude_spectrum_padded(f0, phi, fe, Tend)

