import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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