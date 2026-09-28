#Note detection

import numpy as np

def freq_to_midi(f):
    if f<= 0:
        return 0
    return 69 + 12*np.log2(f/440.0)

def midi_to_scale(midi):
    scale = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note_index = int(midi % 12)
    octave = int(np.floor(midi / 12) - 1)
    return scale[note_index]+f"{octave}"

def freq_to_scale(f):
    return midi_to_scale(freq_to_midi(f))

def add_notes(df):
    df['Note'] = [ freq_to_scale(f) for f in df['Frequency (Hz)']]
    df = df.drop('Duration (s)', axis=1)
    print(df)
    return df