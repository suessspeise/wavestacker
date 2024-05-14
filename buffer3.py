import numpy as np
import struct
import matplotlib.pyplot as plt
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
        
    def append(self, track):
        """
        Adds a Track object to the mixer.

        Args:
            track (Track): A Track object to be added to the mixer.
        """
        self.tracks.append(track)
        
    def _normalise_signal(self, y):
        max_val = np.max(np.abs(y))
        if max_val == 0: return y  # avoids division by zero
        return y / max_val

    def get_mix(self):
        """
        Mixes all tracks together and returns the mixed audio data.

        Returns:
            numpy.ndarray: The mixed audio data.
        """
        if len(self.tracks) < 1: return None, None
        max_length = max(track.data.shape[0] + int(track.position * self.sample_rate) for track in self.tracks)
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
    A class to encode time and amplitude arrays into binary data for audio generation.
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
    
    def add_initial_sequence(self):
        pass # not needed for Audio data
    def add_final_sequence(self):
        pass # not needed for Audio data
        
#     def encode(self, time_array, amplitude_array):
#         """
#         Encode time and amplitude arrays into binary data.

#         Args:
#             time_array (numpy.ndarray): Array representing time in seconds.
#             amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

#         Returns:
#             bytearray: The encoded binary data.
#         """
#         assert len(time_array) == len(amplitude_array), "Time and amplitude arrays must have the same length."
        
#         binary_data = bytearray()
#         for time, amplitude in zip(time_array, amplitude_array):
#             # Convert amplitude to 8-bit unsigned integer (assuming 8-bit encoding)
#             amplitude_byte = int((amplitude + 1.0) * 127.5)
#             binary_data.append(amplitude_byte)
#         return binary_data
    
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
    
    # def xxx(self, data):
    #     return struct.pack(encoding_format * len(data), * data)
    
    def encode(self, time_array, amplitude_array):
        """
        Encode time and amplitude arrays into binary data.

        Args:
            time_array (numpy.ndarray): Array representing time in seconds.
            amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

        Returns:
            bytearray: The encoded binary data.
        """
        assert len(time_array) == len(amplitude_array), "Time and amplitude arrays must have the same length."
        
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_byte = int((amplitude + 1.0) * 127.5)
            packed_data = struct.pack(self.encoding_format, amplitude_byte)
            binary_data.extend(packed_data)
        return binary_data
    

class AmplitudeBinaryEncoder_short(AmplitudeBinaryEncoder_unsignedchar):
    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
        self.bits_per_sample = 16  # 16 bits per sample
        self.encoding_format = 'h'  # Short type (16-bit signed integer)

#     def encode(self, time_array, amplitude_array):
#         """
#         Encode time and amplitude arrays into 16-bit signed integer binary data.

#         Args:
#             time_array (numpy.ndarray): Array representing time in seconds.
#             amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

#         Returns:
#             bytearray: The encoded binary data.
#         """
#         assert len(time_array) == len(amplitude_array), "Time and amplitude arrays must have the same length."
    
#         binary_data = bytearray()
#         linear_data = list()
#         for amplitude in amplitude_array:
#             amplitude_int = int(amplitude * 32767.0)
#             linear_data.append(amplitude_int)
#             # binary_data.append(struct.pack(self.encoding_format, amplitude_int))
#             binary_data.extend(struct.pack(self.encoding_format, amplitude_int))
#         return binary_data
    

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
    
    # def xxx(self, data):
    #     data = bytearray(data)
    #     samples = np.frombuffer(data, dtype=np.int16)
    #     return samples.tobytes()

    def encode(self, time_array, amplitude_array):
        """
        Encode time and amplitude arrays into 16-bit signed integer binary data.

        Args:
            time_array (numpy.ndarray): Array representing time in seconds.
            amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

        Returns:
            bytearray: The encoded binary data.
        """
        assert len(time_array) == len(amplitude_array), "Time and amplitude arrays must have the same length."
    
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_int = int(amplitude * 32767.0)
            packed_data = struct.pack(self.encoding_format, amplitude_int)
            binary_data.extend(packed_data)
        return binary_data
    
    
class AmplitudeBinaryEncoder(AmplitudeBinaryEncoder_short):
    pass


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
        # self.encoded_data = []
        self.checksum = 0
    
    def __repr__(self):
        return f'{self.class_description}, encoder = {self.encoder}, contains {len(self.data)/self.sample_rate}s'

#     def initialise(self):
#         """
#         Finalize the audio buffer by adding the final sequence.
#         """
#         self.encoder.add_initial_sequence()
    
#     def finalise(self):
#         """
#         Finalize the audio buffer by adding the final sequence.
#         """
#         self.encoder.add_final_sequence()
        
    def add_audio_data(self, time_array, amplitude_array):
        """
        Encoded and add audio data to the buffer.

        Args:
            audio_data (bytearray): The audio data to be added.
        """
        encoded_data = self.encoder.encode(time_array, amplitude_array)
        self.data.extend(encoded_data)
            
    def write_to_wav(self, filename):
        encoding_format = self.encoder.get_format()
        num_channels = 1  # Mono
        bits_per_sample = self.encoder.bits_per_sample
        byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_chunk_size = len(self.data)
        file_size = 36 + data_chunk_size  # 36 bytes for the header + size of data
        # self.encoded_data = self.encoder.xxx(self.data)
        
        with open(filename, 'wb+') as file:
            # Write WAV header
            file.write(b'RIFF')
            file.write((file_size).to_bytes(4, byteorder='little'))
            file.write(b'WAVEfmt ')
            file.write((16).to_bytes(4, byteorder='little'))  # Length of format data
            file.write((1).to_bytes(2, byteorder='little'))  # PCM format
            file.write((num_channels).to_bytes(2, byteorder='little'))
            file.write((self.sample_rate).to_bytes(4, byteorder='little'))
            file.write(byte_rate.to_bytes(4, byteorder='little'))
            file.write(block_align.to_bytes(2, byteorder='little'))
            file.write(bits_per_sample.to_bytes(2, byteorder='little'))
            file.write(b'data')
            file.write(data_chunk_size.to_bytes(4, byteorder='little'))
            file.write(bytearray(self.data))
            
    #         # tempfile.mkdtemp(suffix='.wav')
    # def play(self):
    #     filename = 'temp_file.wav'
    #     self.write_to_wav(filename)
    #     return WavPlayer(filename).play()

    def play(self):
        # Create a temporary file using the tempfile library
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_file:
            self.write_to_wav(tmp_file.name)
            # print(tmp_file.name)
            return WavPlayer(tmp_file.name).play()
            # Since the file is automatically deleted upon closing, we disable deletion here
            # to allow the WavPlayer to access it. It will be manually deleted later.

            # Play the WAV file
            # result = 
        # # Manually delete the temporary file after playing
        # os.remove(tmp_file.name)

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

