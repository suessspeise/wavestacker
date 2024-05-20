# from abc import ABC, abstractmethod

import numpy as np
import struct

# for plotting in a notebook
import matplotlib.pyplot as plt

# for replay inside a notebook
from IPython.display import Audio
import tempfile

class WavPlayer:
    """
    A class to replay audio inside a jupyter notebook.

    Example:
        >>> Wavplayer('output.wav).play()
    """
    def __init__(self, file_path):
        self.file_path = file_path
    
    def play(self):
        """
        Loads and plays the WAV file.
        """
        return Audio(filename=self.file_path)
    
    
class MonoTrack:
    """
    A class representing an audio track.
    """

    def __init__(self, data=None, amplitude=1.0, position=0.0, name=''):
        """
        Initializes a Track object.

        Args:
            data (numpy.ndarray): An array of float representing the audio data.
            amplitude (float or numpy.ndarray): The amplitude of the audio track.
            position (float): The time offset indicating when the track starts.
            name (str): The name of the track.
        """
        self.data = data
        self.amplitude = amplitude
        self.position = position
        self.name = name
        
    def __repr__(self):
        return f'MonoTrack, {len(self.tracks)} tracks:\n' + str([t.name for t in self.tracks])

    
class MonoMixer:
    """
    A class representing a mixer for audio tracks.
    """

    def __init__(self, sample_rate=44100):
        self.tracks = []
        self.sample_rate = sample_rate

    def __repr__(self):
        duration = max(len(t.data) + t.position * self.sample_rate for t in self.tracks)/self.sample_rate
        return f'MonoMixer, {len(self.tracks)} tracks, {duration:.1f}s:\n' #+ str([t.name for t in self.tracks])
        
    def add(self, data, amplitude=1.0, position=0.0, name=None):
        if name == None: name = f'track{str(len(self.tracks)).rjust(3,"0")}'
        self.tracks.append(MonoTrack(data, amplitude, position, name))
        
    def _normalise_signal(self, y):
        max_val = np.max(np.abs(y))
        if max_val == 0: return y  # avoids division by zero
        return y / max_val
    
    def append(self, track):
        """
        Adds a Track object to the mixer.

        Args:
            track (Track): A Track object to be added to the mixer.
        """
        self.tracks.append(track)    

    def get_mix(self):
        """
        Mixes all tracks together and returns the mixed audio data.

        Returns:
            numpy.ndarray: The mixed audio data.
        """
        # Find the maximum length of tracks
        max_length = max(track.data.shape[0] + int(track.position * self.sample_rate) for track in self.tracks)
        
        # Initialize an array to hold the mixed audio data
        mixed_data = np.zeros(max_length)
        
        # Mix each track
        for track in self.tracks:
            start_idx = int(track.position * self.sample_rate)
            end_idx = start_idx + track.data.shape[0]
            mixed_data[start_idx:end_idx] += track.data * track.amplitude
        
        return np.linspace(0,len(mixed_data)/self.sample_rate, len(mixed_data)), self._normalise_signal(mixed_data)

    def plot(self):
        fig, axs = plt.subplots(len(self.tracks), 1, figsize=(20,2*len(self.tracks)), sharex=True)
        for ax, track in zip(axs, self.tracks):
            ax.set_ylim(-1.1,1.1)
            ax.set_title(track.name, loc='left')
            ax.axhline(0, lw=0.5, color='black')
            ax.grid(axis='x')
            y = track.data * track.amplitude
            t = np.linspace(0.0+track.position, 
                            track.position + len(track.data)/self.sample_rate,  
                            len(track.data))
            ax.plot(t,y)
        if len(axs) > 1:
            for ax in axs[0:-1]: ax.axes.xaxis.set_ticklabels([])
        return fig, axs
    
    
class AmplitudeBinaryEncoder_unsignedchar:
    """
    A class to encode amplitude arrays into binary data for audio generation.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize the Encoder with the given sample rate.

        Args:
            sample_rate (int): The sample rate of the audio data.
        """
        self.sample_rate = sample_rate
        self.encoding_format = 'B'
        self.bits_per_sample = 8
        
    def __repr__(self):
        return f'AmplitudeBinaryEncoder (sample rate={self.sample_rate}Hz)'

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        
    def get_format(self):
        return self.encoding_format
        
    def plot(self, data):
        fig, ax = plt.subplots(figsize=(20,2))
        ax.axhline(0, lw=0.5, color='black')
        data_bytes = bytearray(data)  # Convert to bytearray if it's not already
        # Convert data from 0-255 to -1.0 to 1.0
        y = np.asarray(data_bytes, dtype=np.uint8)
        y = y / 127.5 - 1.0
        x = np.linspace(0, len(y) / self.sample_rate, len(y))
        ax.plot(x, y)
        ax.set_xlim(0, max(x))
        return fig, ax
    
    def encode(self, amplitude_array):
        """
        Encode amplitude arrays into binary data.

        Args:
            amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

        Returns:
            bytearray: The encoded binary data.
        """
        
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_byte = int((amplitude + 1.0) * 127.5)
            packed_data = struct.pack(self.encoding_format, amplitude_byte)
            binary_data.extend(packed_data)
        return binary_data
    



# class BaseAmplitudeEncoder(ABC):
#     def __init__(self, sample_rate=44100):
#         self.sample_rate = sample_rate

#     @abstractmethod
#     def encode(self, amplitude_array):
#         pass

#     @abstractmethod
#     def plot(self, data):
#         pass
    
    
class AmplitudeBinaryEncoder_short(AmplitudeBinaryEncoder_unsignedchar):
    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
        self.bits_per_sample = 16  # 16 bits per sample
        self.encoding_format = 'h'  # Short type (16-bit signed integer)

    def plot(self, data):
        fig, ax = plt.subplots(figsize=(20,2))
        ax.axhline(0, lw=0.5, color='black')
        data_bytes = bytearray(data)  # Convert to bytearray if it's not already
        # Convert binary data to 16-bit integers and then to -1.0 to 1.0
        y = np.frombuffer(data_bytes, dtype=np.int16)
        y = y / 32767.0
        x = np.linspace(0, len(y) / self.sample_rate, len(y))
        ax.plot(x, y)
        ax.set_xlim(0, max(x))
        return fig, ax

    def encode(self, amplitude_array):
        """
        Encode amplitude arrays into 16-bit signed integer binary data.

        Args:
            amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

        Returns:
            bytearray: The encoded binary data.
        """
    
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_int = int(amplitude * 32767.0)
            packed_data = struct.pack(self.encoding_format, amplitude_int)
            binary_data.extend(packed_data)
        return binary_data
    
    
class AmplitudeBinaryEncoder(AmplitudeBinaryEncoder_short):
    pass # defaulting to short


class MonoAudioBuffer:
    """
    A class to generate audio data and store it for playback or further processing.

    Attributes:
        sample_rate (int): The sample rate of the audio data.
        audio_buffer (bytearray): The buffer to store the generated audio data.
        encoder (Encoder): The encoder used to encode audio data.
    """
    def __init__(self, encoder=AmplitudeBinaryEncoder_short(), sample_rate=44100):
        """
        Initialize the AudioBuffer with the given sample rate and baud rate.

        Args:
            sample_rate (int): The sample rate of the audio data.
            encoding_format (str): The encoding format of the audio data ('B' for unsigned integer, 'f' for float, etc.).
        """
        self.class_description = 'MonoAudioBuffer'
        self.encoder = encoder
        self.sample_rate = sample_rate
        self.encoder.set_sample_rate(self.sample_rate)
        self.data = []
        self.checksum = 0
    
    def __repr__(self):
        return f'{self.class_description}, encoder = {self.encoder}, contains {len(self.data)/self.sample_rate}s'
        
    def add_audio_data(self, amplitude_array):
        """
        Encoded and add audio data to the buffer.

        Args:
            audio_data (bytearray): The audio data to be added.
        """
        encoded_data = self.encoder.encode(amplitude_array)
        self.data.extend(encoded_data)
            
    def write_to_wav(self, filename):
        encoding_format = self.encoder.get_format()
        num_channels = 1  # Mono
        bits_per_sample = self.encoder.bits_per_sample
        byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_chunk_size = len(self.data)
        file_size = 36 + data_chunk_size  # 36 bytes for the header + size of data
        
        with open(filename, 'wb+') as file:
            # RIFF header
            file.write(b'RIFF')
            file.write(struct.pack('<I', file_size))
            # file.write((file_size).to_bytes(4, byteorder='little'))
            file.write(b'WAVE')
            # fmt subchunk
            file.write(b'fmt ')
            file.write(struct.pack('<I', 16))  # Subchunk size
            file.write(struct.pack('<H', 1))  # Audio format (1 is PCM)
            file.write(struct.pack('<H', num_channels))  # Number of channels
            file.write(struct.pack('<I', self.sample_rate))  # Sample rate
            file.write(struct.pack('<I', byte_rate))  # Byte rate
            file.write(struct.pack('<H', block_align))  # Block align
            file.write(struct.pack('<H', bits_per_sample))  # Bits per sample
            # data subchunk
            file.write(b'data')
            file.write(struct.pack('<I', data_chunk_size))
            # file.write(data_chunk_size.to_bytes(4, byteorder='little'))
            file.write(bytearray(self.data))
            
    def play(self):
        # Create a temporary file using the tempfile library
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_file:
            self.write_to_wav(tmp_file.name)
            return WavPlayer(tmp_file.name).play()

    def plot(self):
        return self.encoder.plot(self.data)
    
    def estimate_disk_space(self):
        """
        Estimate the required disk space for the audio data.

        Returns:
            int: Estimated disk space required in bytes.
        """
        data_subchunk_size = len(self.data) * struct.calcsize(self.encoder.encoding_format)
        return 44 + data_subchunk_size #WAV header is fixed 44 bytes

