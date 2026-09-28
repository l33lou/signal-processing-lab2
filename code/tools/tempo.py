#Tempo detection

import pandas as pd
import numpy as np

def tempo_detection(df):
    df['Duration (s)'] = df['End time (s)'] - df['Start time (s)']

    mean_note_duration = df['Duration (s)'].mean()
    bpm = int(60/mean_note_duration)
    rythmic_values = ["\U0001D161", "\U0001D160", "\U0001D15F", "\U0001D15E", "\U0001D15D"]
    
    L = []
    for k, note_duration in enumerate(df['Duration (s)']) :
        q = 4*note_duration/mean_note_duration
        if q > 4 :
            qq = q//4
            r = q%4
        idx = int(np.clip(np.round(np.log2(q)), 0, 4))
        L.append(rythmic_values[idx])

    df['Rythmic value'] = L

    print("Tempo : \U0001D15F = ", bpm)
    print(df)

    return df