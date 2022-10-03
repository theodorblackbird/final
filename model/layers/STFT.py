import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

class STFT_tf(tf.keras.layers.Layer):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT_tf, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = tf.signal.fft(tf.eye(filter_length, dtype=tf.complex64))

        fourier_basis = tf.concat([tf.math.real(fourier_basis[:cutoff, :]),
                                    tf.math.imag(fourier_basis[:cutoff, :])], axis=0)
        forward_basis = fourier_basis[:, None, :]


        if window is not None:
            assert(filter_length >= win_length)
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = tf.convert_to_tensor(fft_window, dtype=tf.float32)

            forward_basis *= fft_window
    def transform(self, input_data):
        num_batches = tf.shape(input_data)[0]
        num_samples = tf.shape(input_data)[1]

        # similar to librosa, reflect-pad the input
        input_data = tf.expand_dims(input_data, 1)
        input_data = tf.pad(
            tf.expand_dims(input_data, 1),
            [[0, 0],[0, 0],[0, 0], [int(self.filter_length / 2), int(self.filter_length / 2)]],
            mode='REFLECT')
        input_data = tf.squeeze(input_data, 1)

        forward_transform = tf.nn.conv1d(
            tf.transpose(input_data, (0,2,1)),
            tf.transpose(forward_basis2, (2,1,0)),
            stride=hop_length,
            padding='VALID',)
        forward_transform = tf.transpose(forward_transform, (0,2,1))
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]


        magnitude = tf.math.sqrt(real_part**2 + imag_part**2)
        phase = tf.math.atan2(imag_part, real_part)

        return magnitude, phase
class TacotronSTFT(tf.keras.layers.Layer):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT_tf(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        self.mel_basis = tf.convert_to_tensor(mel_basis)

    def spectral_normalize(self, magnitudes):
        return tf.math.log(tf.clip_by_value(magnitudes, 1e-5, np.inf))

    def spectral_de_normalize(self, magnitudes):
        return tf.math.exp(magnitudes)

    def mel_spectrogram(self, y):

        magnitudes, phases = self.stft_fn.transform(y)
        mel_output = tf.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
