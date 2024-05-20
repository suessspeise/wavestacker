from abc import ABC, abstractmethod
import warnings
import struct
import numpy as np

# for plotting in a notebook
plotting_is_available = False
try:
    import matplotlib.pyplot as plt
    plotting_is_available = True
except ImportError:
    warnings.warn("Plotting is not available because matplotlib could not be imported.", ImportWarning)
    
# for replay inside a notebook
wavplayer_is_available = False
try: 
    from IPython.display import Audio
    import tempfile

    class WavPlayer:
        """
        A class to replay audio inside a Jupyter notebook.

        Example:
            >>> WavPlayer('output.wav').play()
        """
        def __init__(self, file_path):
            self.file_path = file_path

        def play(self):
            """
            Loads and plays the WAV file.
            """
            return Audio(filename=self.file_path)

    wavplayer_is_available = True
except ImportError:
    warnings.warn("WavPlayer is not available because IPython.display.Audio could not be imported. Playback functionality is disabled.", ImportWarning)

    
class MonoTrack:
    """
    A class representing an audio track.
    """

    def __init__(self, data=None, amplitude=1.0, position=0.0, name='', sample_rate=44100):
        """
        Initializes a Track object.

        Args:
            data (numpy.ndarray): An array of float representing the audio data.
            amplitude (float or numpy.ndarray): The amplitude of the audio track.
            position (float): The time offset indicating when the track starts.
            name (str): The name of the track.
        """
        self.sample_rate = sample_rate
        self.data = data
        self.amplitude = amplitude
        self.position = position
        self.name = name
        
    def __repr__(self):
        return f'MonoTrack, {len(self.data) / self.sample_rate:.1f}s'

    
class MonoMixer:
    """
    A class representing a mixer for audio tracks.
    """

    def __init__(self, sample_rate=44100):
        self.tracks = []
        self.sample_rate = sample_rate

    def __repr__(self):
        duration = max(len(t.data) + t.position * self.sample_rate for t in self.tracks)/self.sample_rate
        return f'MonoMixer, {len(self.tracks)} tracks, {duration:.1f}s'
        
    def add(self, data, amplitude=1.0, position=0.0, name=None):
        if name == None: name = f'track{str(len(self.tracks)).rjust(3,"0")}'
        self.tracks.append(MonoTrack(data, amplitude, position, name, self.sample_rate))
        
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

    def get_tracks_by_name(self, name):
        """
        Fetches tracks by name.

        Args:
            name (str): The name of the tracks to fetch.

        Returns:
            list: A list of tracks with the given name.
        """
<<<<<<< HEAD
        return [track for track in self.tracks if track.name == name]

    def get_mix(self, track_names=None):
        """
        Mixes all tracks together or a subset of tracks if track_names is provided.

        Args:
            track_names (list, optional): A list of track names to mix. Mixes all tracks if None.

        Returns:
            tuple: Time array and the mixed audio data as a numpy.ndarray.
        """
        if track_names is not None:
            # Filter tracks to include only those with names in track_names
            tracks_to_mix = [track for track in self.tracks if track.name in track_names]
        else:
            tracks_to_mix = self.tracks

        max_length = max((track.data.shape[0] + int(track.position * self.sample_rate) for track in tracks_to_mix), default=0) # Find the maximum length of tracks
        mixed_data = np.zeros(max_length)  # Initialize an array to hold the mixed audio data

        for track in tracks_to_mix:  # Mix each track into it
=======
        max_length = max(track.data.shape[0] + int(track.position * self.sample_rate) for track in self.tracks) # Find the maximum length of tracks
        mixed_data = np.zeros(max_length) # Initialize an array to hold the mixed audio data
        
        for track in self.tracks: # Mix each track
>>>>>>> 0d7906257ced62738f8618c1431d8dea502f49ac
            start_idx = int(track.position * self.sample_rate)
            end_idx = start_idx + track.data.shape[0]
            mixed_data[start_idx:end_idx] += track.data * track.amplitude

        return np.linspace(0, len(mixed_data) / self.sample_rate, len(mixed_data)), self._normalise_signal(mixed_data)

    def plot(self):
        if plotting_is_available:
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
        else:
            print("Plotting is not available. Please install matplotlib to enable this feature.")


class BaseAmplitudeEncoder(ABC):
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
        
    def __repr__(self):
        return f'AmplitudeEncoder (sample rate={self.sample_rate}Hz)'
    
    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        
    def get_format(self):
        return self.encoding_format
    
    def _validate_amplitude_range(self, amplitude_array):
        # Validate amplitude range
        if not np.all((-1.0 <= amplitude_array) & (amplitude_array <= 1.0)):
            raise ValueError("Amplitude values must be between -1.0 and 1.0.")
    @abstractmethod
    def encode(self, amplitude_array):
        """
        Encode amplitude arrays into 16-bit signed integer binary data.

        Args:
            amplitude_array (numpy.ndarray): Array representing amplitude (between -1.0 and 1.0).

        Returns:
            bytearray: The encoded binary data.
        """
        pass

    @abstractmethod
    def plot(self, data):
        pass
    
    
class AmplitudeEncoder_unsignedchar(BaseAmplitudeEncoder):
    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
        self.bits_per_sample = 8  # 8 bits per sample
        self.encoding_format = 'B'  # Short type (16-bit signed integer)
    
    def encode(self, amplitude_array):
        self._validate_amplitude_range(amplitude_array)
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_byte = int((amplitude + 1.0) * 127.5)
            packed_data = struct.pack(self.encoding_format, amplitude_byte)
            binary_data.extend(packed_data)
        return binary_data
        
    def plot(self, data):
        fig, ax = plt.subplots(figsize=(20,2))
        ax.axhline(0, lw=0.5, color='black')
        data_bytes = bytearray(data)  # Convert to bytearray if it's not already
        y = np.asarray(data_bytes, dtype=np.uint8)
        y = y / 127.5 - 1.0  # Convert data from 0-255 to -1.0 to 1.0
        x = np.linspace(0, len(y) / self.sample_rate, len(y))
        ax.plot(x, y)
        ax.set_xlim(0, max(x))
        return fig, ax
    
    
class AmplitudeEncoder_short(BaseAmplitudeEncoder):
    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
        self.bits_per_sample = 16  # 16 bits per sample
        self.encoding_format = 'h'  # Short type (16-bit signed integer)

    def encode(self, amplitude_array):
        self._validate_amplitude_range(amplitude_array)
        binary_data = bytearray()
        for amplitude in amplitude_array:
            amplitude_int = int(amplitude * 32767.0)
            packed_data = struct.pack(self.encoding_format, amplitude_int)
            binary_data.extend(packed_data)
        return binary_data

    def plot(self, data):
        fig, ax = plt.subplots(figsize=(20,2))
        ax.axhline(0, lw=0.5, color='black')
        data_bytes = bytearray(data)  # Convert to bytearray if it's not already
        y = np.frombuffer(data_bytes, dtype=np.int16)
        y = y / 32767.0 # Convert binary data to 16-bit integers and then to -1.0 to 1.0
        x = np.linspace(0, len(y) / self.sample_rate, len(y))
        ax.plot(x, y)
        ax.set_xlim(0, max(x))
        return fig, ax
    
# Setting AmplitudeEncoder_short as the default encoder
AmplitudeEncoder = AmplitudeEncoder_short


class MonoAudioBuffer:
    """
    A class to generate audio data and store it for playback or further processing.

    Attributes:
        sample_rate (int): The sample rate of the audio data.
        audio_buffer (bytearray): The buffer to store the generated audio data.
        encoder (Encoder): The encoder used to encode audio data.
    """
    def __init__(self, encoder=AmplitudeEncoder(), sample_rate=44100):
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

    def plot(self):
        if plotting_is_available:
            return self.encoder.plot(self.data)
        else:
            print("Plotting is not available. Please install matplotlib to enable this feature.")
    
    def estimate_disk_space(self):
        """
        Estimate the required disk space for the audio data.

        Returns:
            int: Estimated disk space required in bytes.
        """
        data_subchunk_size = len(self.data) * struct.calcsize(self.encoder.encoding_format)
        return 44 + data_subchunk_size #WAV header is fixed 44 bytes
    
    def play(self):
        if wavplayer_is_available:
            # Create a temporary file using the tempfile library
            with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_file:
                self.write_to_wav(tmp_file.name)
                return WavPlayer(tmp_file.name).play()
        else:
<<<<<<< HEAD
            print("WavPlayer is not available. Playback functionality is disabled.")


class StereoAudioBuffer(MonoAudioBuffer):
    def __init__(self, encoder=AmplitudeEncoder(), sample_rate=44100):
        super().__init__(encoder, sample_rate)
        # Initialize a separate list for right channel data
        self.right_channel_data = []

    def add_audio_data(self, left_channel_data, right_channel_data):
        """
        Encodes and adds stereo audio data to the buffer.

        Args:
            left_channel_data (numpy.ndarray): The left channel audio data.
            right_channel_data (numpy.ndarray): The right channel audio data.
        """
        assert len(left_channel_data) == len(right_channel_data), "Left and right channels must have the same length."
        stereo_data = self._interleave_stereo_data(left_channel_data, right_channel_data)
        encoded_data = self.encoder.encode(stereo_data)
        self.data.extend(encoded_data)

    def _interleave_stereo_data(self, left_channel, right_channel):
        """ 
        Interleaves left and right channel data for stereo audio.

        Args:
            left_channel (numpy.ndarray): The left channel audio data.
            right_channel (numpy.ndarray): The right channel audio data.

        Returns:
            numpy.ndarray: The interleaved stereo audio data.
        """
        stereo_data = np.empty((len(left_channel) + len(right_channel),), dtype=left_channel.dtype)
        stereo_data[0::2] = left_channel
        stereo_data[1::2] = right_channel
        return stereo_data

    def write_to_wav(self, filename):
        """ 
        Writes the encoded stereo audio data to a WAV file.

        Args:
            filename (str): The name of the file to write the audio data to.
        """
        num_channels = 2  # Stereo
        bits_per_sample = self.encoder.bits_per_sample
        byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_chunk_size = len(self.data)
        file_size = 36 + data_chunk_size  # 36 bytes for the header + size of data

        with open(filename, 'wb+') as file:
            # RIFF header
            file.write(b'RIFF')
            file.write(struct.pack('<I', file_size))
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
            file.write(bytearray(self.data))

    def plot(self):
        if plotting_is_available:
            left_channel, right_channel = self._uninterleave_stereo_data(bytearray(self.data))
            time_axis = np.linspace(0, len(left_channel) / self.sample_rate, len(left_channel))
            
            # Plotting
            fig, axs = plt.subplots(2, 1, figsize=(20, 4), sharex=True)
            axs[0].plot(time_axis, left_channel, label="Left Channel")
            axs[0].set_title("Left Channel")
            axs[0].set_ylabel("Amplitude")

            axs[1].plot(time_axis, right_channel, label="Right Channel")
            axs[1].set_title("Right Channel")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Amplitude")

            plt.tight_layout()
            return fig, axs
        else:
            print("Plotting is not available. Please install matplotlib to enable this feature.")

    def _uninterleave_stereo_data(self, interleaved_data):
        """
        Uninterleaves stereo audio data into separate left and right channels.

        Args:
            interleaved_data (bytearray): The interleaved stereo audio data.

        Returns:
            tuple: Two numpy.ndarrays representing the left and right channels, respectively.
        """
        interleaved_data_np = np.frombuffer(interleaved_data, dtype=np.int16)  # Ensure dtype matches encoding format
        left_channel = interleaved_data_np[0::2]
        right_channel = interleaved_data_np[1::2]
        return left_channel, right_channel
