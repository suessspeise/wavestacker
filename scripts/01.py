import sys
import os
sys.path.append(os.getcwd() + '/..')

import buffer3 as buffer
import numpy as np
import matplotlib.pyplot as plt

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

freq = 220
set_in = 0
duration = 4
y  = np.sin(t * freq * (np.pi * 2))
amp = fade_in(set_in,1,length) * fade_out(set_in+duration,1,length) 
tracks.append({'amp':amp, 'y':y})

freq = 222
y  = np.sin(t * freq * (np.pi * 2))
amp = fade_in(2,1,length) * fade_out(7,1,length)
tracks.append({'amp':amp, 'y':y})

freq = 210
y  = np.sin(t * freq * (np.pi * 2))
amp = fade_in(3,1,length) * fade_out(10,1,length)
tracks.append({'amp':amp, 'y':y})

freq = 250
y  = np.sin(t * freq * (np.pi * 2))
amp = fade_in(9,1,length) * fade_out(15,1,length)
tracks.append({'amp':amp, 'y':y})

freq = 250
y  = np.sin(t * freq * (np.pi * 2))
amp = fade_in(9,1,length) * fade_out(15,1,length)
tracks.append({'amp':amp, 'y':y})

y = normalize_signal(mix_tracks(tracks))
bf = buffer.MonoAudioBuffer(encoder=buffer.AmplitudeBinaryEncoder_unsignedchar())

fig, ax = plt.subplots()
for tr in tracks:
    ax.plot(tr['y'])
ax.set_xlim(2e5, 2e5+1000)

bf.add_audio_data(t,y)
bf.write_to_wav('01.wav')