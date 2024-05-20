import sys
import os
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd())

import numpy as np
import buffer

mixer = buffer.MonoMixer()

for i in range(50):
    freq = 190 + np.random.randint(20)
    offset = i*2
    duration = 4
    y  = np.sin(np.linspace(0,3,(duration+2)*44100) * freq * (np.pi * 2))
    amp = np.concatenate(( np.linspace(0,1,44100), np.ones(44100*duration), np.linspace(1,0,44100)))
    mixer.add(y,amp,offset)
    
t,y = mixer.get_mix()
print(mixer)
print(mixer.tracks[0])

bf = buffer.MonoAudioBuffer()
bf.add_audio_data(y)
bf.write_to_wav('04.wav')
