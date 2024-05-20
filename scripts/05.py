import sys
import os
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd())

import numpy as np
import buffer

import struct

class StereoAudioBuffer(buffer.MonoAudioBuffer):
    def __init__(self, encoder=buffer.AmplitudeEncoder(), sample_rate=44100):
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
        # Ensure both channels have the same length
        assert len(left_channel_data) == len(right_channel_data), "Left and right channels must have the same length."

        # Interleave left and right channel data
        stereo_data = self._interleave_stereo_data(left_channel_data, right_channel_data)
        
        # Encode and add the interleaved data
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

    # The plot and play methods from MonoAudioBuffer can be reused without modification.
    # If specific stereo handling is desired for these methods, they can be overridden and modified accordingly.




def generate_sine_wave(frequency, duration, sample_rate):
    time_array = np.linspace(0, duration, int(sample_rate * duration))
    sine_wave = np.sin(2 * np.pi * frequency * time_array)
    return sine_wave

sample_rate = 44100  # Sample rate in Hz
audio_buffer = StereoAudioBuffer()

# Generate large arrays of sine waves with different frequencies
frequencies = [440, 880, 220, 880, 440, 220, 1760, 110, 1760, 110]
frequencies += frequencies # repeat melody
duration = 0.2  # seconds
time_array = np.linspace(0, duration*len(frequencies), int(sample_rate * duration * len(frequencies)))
large_arrays = [generate_sine_wave(freq, duration, sample_rate) for freq in frequencies]
joined_array = np.concatenate(large_arrays)

# Apply modulation 
channel0 = joined_array * (( np.cos(2 * np.pi * 2 * time_array) / 2) + .5)
channel1 = joined_array * ((-np.cos(2 * np.pi * 2 * time_array) / 2) + .5)

audio_buffer.add_audio_data(channel0, channel1)
audio_buffer.write_to_wav('05.wav')
