import sys
import os
sys.path.append(os.getcwd() + '/..')

import buffer
import numpy as np

def mix_tracks(tracks): 
    y = np.zeros(tracks[0]['y'].shape)
    for track in tracks:
        y += track['amp'] * track['y']
    y = y / len(tracks)
    return y

def normalize_signal(y):
    max_val = np.max(np.abs(y))
    if max_val == 0: return y  # avoids division by zero
    return y / max_val

def fade_in(t_0, dt, t_max, sample_rate=44100):
    low  = np.zeros(int(t_0 * sample_rate))
    fade = np.linspace(0,1, int(dt * sample_rate))
    high = np.ones(int((t_max-t_0-dt) * sample_rate))
    joined = np.concatenate((low, fade, high))
    fill = np.zeros(t_max*sample_rate - len(joined))
    return np.concatenate((joined, fill))

def fade_out(t_0, dt, t_max, sample_rate=44100):
    high = np.ones(int(t_0 * sample_rate))
    fade = np.linspace(1,0, int(dt * sample_rate))
    low  = np.zeros(int((t_max-t_0-dt) * sample_rate))
    joined = np.concatenate((high, fade, low))
    fill = np.zeros(t_max*sample_rate - len(joined))
    return np.concatenate((joined, fill))

length = 20
t = np.linspace(0, length, length*44100)
tracks = list()

division = 6

for k in range(2):
    frequency = np.random.randint(1000) + 100
    decay = (2 + np.random.randint(10)) * 0.01
    beats = [np.random.randint(2) for i in range(np.random.randint(8)+2)]
    print(beats, frequency, decay)
    
    beats = (beats * division * length)[0:length*division]
    for i, e in enumerate(beats):
        y  = np.sin(t * frequency * (np.pi * 2))
        y  = np.random.rand(len(t)) * 2 - 1
        set_in = (i/division) + 0.0
        amp = fade_in(set_in,.01,length) * fade_out(set_in,decay,length)  / 2 
        amp = amp * e
        tracks.append({'amp':amp, 'y':y})

y = normalize_signal(mix_tracks(tracks))

bf = buffer.MonoAudioBuffer(encoder=buffer.AmplitudeBinaryEncoder_unsignedchar())
bf.add_audio_data(t,y)
bf.write_to_wav('03.wav')