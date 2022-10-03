from tensorflow.data import Dataset
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from config.config import Tacotron2Config
from model.layers.MelSpec import MelSpec, TacotronSTFT
import librosa

class ljspeechDataset(object):
    def __init__(self, conf, train_conf) -> None:
        super().__init__()
        self.conf = conf
        self.train_conf = train_conf
        msc = conf["mel_spec"]
        self.mel_spec_gen = MelSpec(
                msc["frame_length"],
                msc["frame_step"],
                None,
                msc["sampling_rate"],
                msc["n_mel_channels"],
                msc["freq_min"],
                msc["freq_max"])
        self.tac_stft = TacotronSTFT(
                msc["frame_length"],
                msc["frame_step"],
                msc["frame_length"],
                msc["n_mel_channels"],
                msc["sampling_rate"],
                msc["freq_min"],
                msc["freq_max"])

        self.sigma = 0.4
        self.gradual_training = tf.constant(train_conf["train"]["gradual_training"])

    

    def __call__(self, x):

        split = tf.strings.split(x, sep='|')
        name = split[0]
        phon = split[1]
        path = self.train_conf["data"]["audio_dir"] + "/"+ name + ".wav"
        raw_audio = tf.io.read_file(path)
        audio, sr = tf.audio.decode_wav(raw_audio)
        audio = audio / tf.math.abs(tf.math.reduce_max(audio)) * 0.999
        audio = tf.squeeze(audio)

        #mel_spec = self.mel_spec_gen(audio)
        mel_spec = self.tac_stft.mel_spectrogram(tf.expand_dims(audio, 0))[0]
        mel_spec = tf.transpose(mel_spec)

        crop = tf.shape(mel_spec)[0] - tf.shape(mel_spec)[0]%self.conf["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step

        mel_spec = mel_spec[:crop, :]
        
        mel_len = len(mel_spec)
        
        string_len = tf.strings.length(phon, 'UTF8_CHAR')
        gate = tf.concat([tf.zeros((mel_len//self.conf["n_frames_per_step"] - 1)), [1.]], axis=0)

        X = tf.linspace(0.,1.,tf.strings.length(phon, 'UTF8_CHAR'))
        Y = tf.linspace(0.,1., mel_len//self.conf["n_frames_per_step"])
        X, Y = tf.meshgrid(X, Y)
        ga_mask = 1. - tf.math.exp( -((X-Y)**2) /(2. * (self.sigma**2)))
        mel_mask = tf.ones((mel_len, self.conf["n_mel_channels"]))

        return (phon, mel_spec), (mel_spec, gate, ga_mask, mel_mask)

if __name__ == "__main__":

    from model.Tacotron2 import Tacotron2
    import matplotlib.pyplot as plt
    import numpy as np
    conf = Tacotron2Config("config/configs/tacotron2_laptop.yaml")

    tac = Tacotron2(conf)
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    dataset_mapper = ljspeechDataset(conf)
    ljspeech = ljspeech_text.map(dataset_mapper).shuffle(100)
    x, y = next(iter(ljspeech))
    phon, mel, gate = x
    print(mel.shape)
    mel = tf.transpose(mel)
    fig, ax = plt.subplots(1)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    ax.imshow(mel)
    plt.show()
    print(np.max(mel))


