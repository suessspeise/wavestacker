import numpy as np
import struct
import matplotlib.pyplot as plt
from IPython.display import Audio

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
    pass # defaulting to short

# class MonoAudioBuffer:
#     """
#     A class to generate audio data and store it for playback or further processing.

#     Attributes:
#         sample_rate (int): The sample rate of the audio data.
#         audio_buffer (bytearray): The buffer to store the generated audio data.
#         encoder (Encoder): The encoder used to encode audio data.
#     """
#     def __init__(self, encoder=AmplitudeBinaryEncoder(), sample_rate=44100):
#         """
#         Initialize the AudioBuffer with the given sample rate and baud rate.

#         Args:
#             sample_rate (int): The sample rate of the audio data.
#             encoding_format (str): The encoding format of the audio data ('B' for unsigned integer, 'f' for float, etc.).
#         """
#         self.class_description = 'MonoAudioBuffer'
#         self.encoder = encoder
#         self.sample_rate = sample_rate
#         self.encoder.set_sample_rate(self.sample_rate)
#         self.data = []
#         self.checksum = 0
    
#     def __repr__(self):
#         return f'{self.class_description}, encoder = {self.encoder}, contains {len(self.data)/self.sample_rate}s'

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
        
#     def add_audio_data(self, time_array, amplitude_array):
#         """
#         Encoded and add audio data to the buffer.

#         Args:
#             audio_data (bytearray): The audio data to be added.
#         """
#         encoded_data = self.encoder.encode(time_array, amplitude_array)
#         self.data.extend(encoded_data)
        
#     def write_to_wav(self, filename):
#         """
#         Write the audio data to a WAV file.

#         Args:
#             filename (str): The filename of the WAV file to write.
#         """
#         encoding_format = self.encoder.get_format()
#         with open(filename, 'wb+') as output_file:
#             # Calculate the size of the data subchunk
#             data_subchunk_size = len(self.data) * struct.calcsize(encoding_format)
#             print(data_subchunk_size)
#             # Write WAV header
#             output_file.write(b'RIFF')
#             output_file.write((36 + data_subchunk_size).to_bytes(4, byteorder='little'))  # Length in bytes
#             output_file.write(b'WAVEfmt ')
#             output_file.write((16).to_bytes(4, byteorder='little'))  # Length of format data
#             if encoding_format == 'f':
#                 output_file.write((3).to_bytes(2, byteorder='little'))  # Floating-point PCM
#             else:
#                 output_file.write((1).to_bytes(2, byteorder='little'))  # PCM
#             output_file.write((1).to_bytes(2, byteorder='little'))  # Number of channels
#             output_file.write((self.sample_rate).to_bytes(4, byteorder='little'))  # Sample Rate
#             output_file.write((self.sample_rate).to_bytes(4, byteorder='little'))  # Sample Rate * bits * channels / 8
#             output_file.write((1).to_bytes(2, byteorder='little'))  # 8-bit mono
#             output_file.write((8).to_bytes(2, byteorder='little'))  # Bits per sample
#             output_file.write(b'data')
#             output_file.write(data_subchunk_size.to_bytes(4, byteorder='little'))  # Length in bytes
#             output_file.write(struct.pack(encoding_format * len(self.data), *self.data))  # Write audio data
#             # output_file.write(struct.pack(encoding_format * len(self.data), *self.data))  # Write audio data
            
#     def write_to_wav(self, filename):
#         """
#         Write the audio data to a WAV file.

#         Args:
#             filename (str): The filename of the WAV file to write.
#         """ 
#         encoding_format = self.encoder.get_format()
#         sample_size = 8 * struct.calcsize(encoding_format)
#         data_subchunk_size = len(self.data) * struct.calcsize(encoding_format)
#         # byte_rate = (self.sample_rate * sample_size * 1) // 8  # Calculate byte rate
#         byte_rate = self.sample_rate * struct.calcsize(encoding_format) # Calculate byte rate

#         with open(filename, 'wb+') as output_file:
#             output_file.write(b'RIFF')
#             output_file.write((36 + data_subchunk_size).to_bytes(4, byteorder='little'))
#             output_file.write(b'WAVEfmt ')
#             output_file.write((16).to_bytes(4, byteorder='little'))  # Length of format data
#             if encoding_format == 'f':
#                 output_file.write((3).to_bytes(2, byteorder='little'))  # Floating-point PCM
#             else:
#                 output_file.write((1).to_bytes(2, byteorder='little'))  # PCM
#             output_file.write((1).to_bytes(2, byteorder='little'))  # Number of channels
#             output_file.write((self.sample_rate).to_bytes(4, byteorder='little'))  # Sample Rate
#             output_file.write((byte_rate).to_bytes(4, byteorder='little'))  # Byte Rate
#             output_file.write((1).to_bytes(2, byteorder='little'))  # 1 channel * bits per sample / 8
#             output_file.write((sample_size).to_bytes(2, byteorder='little'))  # Bits per sample
#             output_file.write(b'data')
#             output_file.write(data_subchunk_size.to_bytes(4, byteorder='little'))
#             output_file.write(struct.pack(encoding_format * len(self.data), *self.data))

#     def write_to_wav(self, filename):
#         with open(filename, 'wb') as file:
#             # Write the RIFF header
#             file.write(b'RIFF')
#             file.write(struct.pack('<I', 36 + len(self.data)))  # File size
#             file.write(b'WAVE')

#             # Write the fmt subchunk
#             file.write(b'fmt ')
#             file.write(struct.pack('<I', 16))  # Subchunk size
#             file.write(struct.pack('<H', 1))  # Audio format (1 is PCM)
#             file.write(struct.pack('<H', 1))  # Number of channels
#             file.write(struct.pack('<I', self.sample_rate))  # Sample rate
#             file.write(struct.pack('<I', self.sample_rate * 1 * 8 // 8))  # Byte rate
#             file.write(struct.pack('<H', 1 * 8 // 8))  # Block align
#             file.write(struct.pack('<H', 8))  # Bits per sample

#             # Write the data subchunk
#             file.write(b'data')
#             file.write(struct.pack('<I', len(self.data)))
#             file.write(bytes(self.data))
            
#     def write_to_wav(self, filename):
#         num_channels = 1  # Mono
#         bits_per_sample = self.encoder.bits_per_sample
#         byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
#         block_align = num_channels * bits_per_sample // 8

#         with open(filename, 'wb+') as file:
#             # RIFF header
#             file.write(b'RIFF')
#             file.write((36 + len(self.data)).to_bytes(4, byteorder='little'))
#             file.write(b'WAVE')
#             # fmt subchunk
#             file.write(b'fmt ')
#             file.write((16).to_bytes(4, byteorder='little'))  # Length of format data
#             file.write((1).to_bytes(2, byteorder='little'))  # PCM format
#             file.write((num_channels).to_bytes(2, byteorder='little'))
#             file.write((self.sample_rate).to_bytes(4, byteorder='little'))
#             file.write(byte_rate.to_bytes(4, byteorder='little'))
#             file.write(block_align.to_bytes(2, byteorder='little'))
#             file.write(bits_per_sample.to_bytes(2, byteorder='little'))
#             # data subchunk
#             file.write(b'data')
#             file.write(len(self.data).to_bytes(4, byteorder='little'))
#             file.write(self.data)
            
#     def play(self):
#         filename = 'temp_file.wav'
#         self.write_to_wav(filename)
#         return WavPlayer(filename).play()
    
#     def plot(self):
#         fig, ax = plt.subplots(figsize=(20,2))
#         ax.axhline(0, lw=0.5, color='black')
#         y = np.asarray(self.data)
#         y = y/(2**7) - 1
#         x = np.linspace(0, len(y)/self.sample_rate, len(y))
#         ax.plot(x,y)
#         ax.set_xlim(0,max(x))
#         return fig, ax
    
#     def estimate_disk_space(self):
#         """
#         Estimate the required disk space for the audio data.

#         Returns:
#             int: Estimated disk space required in bytes.
#         """
#         data_subchunk_size = len(self.data) * struct.calcsize('B')
#         return 44 + data_subchunk_size #WAV header is fixed 44 bytes


# class StereoAudioBuffer(MonoAudioBuffer):
#     """
#     A class to generate stereo audio data and store it for playback or further processing.

#     Attributes:
#         sample_rate (int): The sample rate of the audio data.
#         encoder (Encoder): The encoder used to encode audio data.
#     """
#     def add_audio_data(self, time_array, left, right=None):
#         """
#         Encoded and add stereo audio data to the buffer.

#         Args:
#             time_array (numpy.ndarray): Array representing time in seconds.
#             left_amplitude_array (numpy.ndarray): Array representing left channel amplitude (between -1.0 and 1.0).
#             right_amplitude_array (numpy.ndarray): Array representing right channel amplitude (between -1.0 and 1.0).
#         """
#         if right == None: right = left
#         encoded_data_left = self.encoder.encode(time_array, left)
#         encoded_data_right = self.encoder.encode(time_array, right)
#         stereo_data = [byte for pair in zip(encoded_data_left, encoded_data_right) for byte in pair] # Interleave left and right channel data
#         self.data.extend(stereo_data)

#     def write_to_wav(self, filename):
#         """
#         Write the stereo audio data to a WAV file.

#         Args:
#             filename (str): The filename of the WAV file to write.
#         """
#         # Use the superclass method to write audio data to the WAV file
#         super().write_to_wav(filename)
#         # Update the header to reflect stereo audio
#         with open(filename, 'r+b') as output_file:
#             output_file.seek(22) # Move the file pointer to the 'Number of channels' field
#             output_file.write((2).to_bytes(2, byteorder='little'))  # Write the number of channels (stereo)
#             output_file.seek(28) # Move the file pointer to the 'Byte Rate' field
#             output_file.write((self.sample_rate * 2).to_bytes(4, byteorder='little')) # Write the byte rate (Sample Rate * Bits * Channels / 8)
#             output_file.seek(32) # Move the file pointer to the 'Block Align' field
#             output_file.write((2).to_bytes(2, byteorder='little')) # Write the block align (Bits * Channels / 8)


# def mono_example(filename='output.wav'):
#     sample_rate = 44100  # Sample rate in Hz
#     audio_buffer = MonoAudioBuffer(AmplitudeBinaryEncoder(), sample_rate)

#     for frequency in [440, 880, 220, 880, 440, 220, 1760, 110, 1760, 110]:
#         duration = 0.2 # seconds
#         time_array = np.linspace(0, duration, int(sample_rate * duration))
#         amplitude_array = np.sin(2 * np.pi * frequency * time_array)
#         audio_buffer.add_audio_data(time_array, amplitude_array)

#     audio_buffer.write_to_wav(filename)

# def stereo_example(filename='output.wav'):
#     def generate_sine_wave(frequency, duration, sample_rate):
#         time_array = np.linspace(0, duration, int(sample_rate * duration))
#         sine_wave = np.sin(2 * np.pi * frequency * time_array)
#         return sine_wave

#     sample_rate = 44100  # Sample rate in Hz
#     audio_buffer = StereoAudioBuffer(AmplitudeBinaryEncoder(), sample_rate)

#     # Generate large arrays of sine waves with different frequencies
#     frequencies = [440, 880, 220, 880, 440, 220, 1760, 110, 1760, 110]
#     frequencies += frequencies # repeat melody
#     duration = 0.2  # seconds
#     time_array = np.linspace(0, duration*len(frequencies), int(sample_rate * duration * len(frequencies)))
#     large_arrays = [generate_sine_wave(freq, duration, sample_rate) for freq in frequencies]
#     joined_array = np.concatenate(large_arrays)

#     # Apply modulation 
#     channel0 = apply_modulation(joined_array, ( np.cos(2 * np.pi * 2 * time_array) / 2) + .5)
#     channel1 = apply_modulation(joined_array, (-np.cos(2 * np.pi * 2 * time_array) / 2) + .5)

#     audio_buffer.add_audio_data(time_array, channel0, channel1)
#     audio_buffer.write_to_wav(filename)


    
    
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
            # RIFF header
            file.write(b'RIFF')
            file.write(struct.pack('<I', file_size))
            # file.write((file_size).to_bytes(4, byteorder='little'))
            file.write(b'WAVE')
            # fmt subchunk
            file.write(b'fmt ')
#             file.write((16).to_bytes(4, byteorder='little'))  # Length of format data
#             file.write((1).to_bytes(2, byteorder='little'))  # PCM format
#             file.write((num_channels).to_bytes(2, byteorder='little'))
#             file.write((self.sample_rate).to_bytes(4, byteorder='little'))
#             file.write(byte_rate.to_bytes(4, byteorder='little'))
#             file.write(block_align.to_bytes(2, byteorder='little'))
#             file.write(bits_per_sample.to_bytes(2, byteorder='little'))
            
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
            
    #         # tempfile.mkdtemp(suffix='.wav')
    # def play(self):
    #     filename = 'temp_file.wav'
    #     self.write_to_wav(filename)
    #     return WavPlayer(filename).play()
    
    
#     def write_to_wav(self, filename):
#         with open(filename, 'wb') as file:
#             # Write the RIFF header
#             file.write(b'RIFF')
#             file.write(struct.pack('<I', 36 + len(self.data)))  # File size
#             file.write(b'WAVE')

#             # Write the fmt subchunk
#             file.write(b'fmt ')
#             file.write(struct.pack('<I', 16))  # Subchunk size
#             file.write(struct.pack('<H', 1))  # Audio format (1 is PCM)
#             file.write(struct.pack('<H', 1))  # Number of channels
#             file.write(struct.pack('<I', self.sample_rate))  # Sample rate
#             file.write(struct.pack('<I', self.sample_rate * 1 * 8 // 8))  # Byte rate
#             file.write(struct.pack('<H', 1 * 8 // 8))  # Block align
#             file.write(struct.pack('<H', 8))  # Bits per sample

#             # Write the data subchunk
#             file.write(b'data')
#             file.write(struct.pack('<I', len(self.data)))
#             file.write(bytes(self.data))
            

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

