import argparse
import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2, split_func

import os
import platform
import tensorflow_addons as tfa
#if platform.node() != "jean-zay3":
#   import manage_gpus as gpl
from datetime import datetime
from pprint import pprint
import matplotlib.cm
import yaml
import os

@tf.keras.utils.register_keras_serializable()
class NoamLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.math.maximum(1., tf.cast(step, tf.float32))
        return self.initial_learning_rate * tf.math.sqrt(self.warmup_steps) * \
                tf.math.minimum(step * tf.math.pow(self.warmup_steps, -1.5), tf.math.pow(step, -0.5))
    def get_config(self):
        config = {'initial_learning_rate': self.initial_learning_rate,
                'warmup_steps': self.warmup_steps}
        return config


@tf.keras.utils.register_keras_serializable()
class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds, fw):
        self.g_step = 0
        self.ds = ds
        self.cm = tf.constant(matplotlib.cm.get_cmap('viridis').colors, dtype=tf.float32)
        self.fw = fw

    def on_train_epoch_end(self, batch, logs=None):
        x, y = next(iter(self.ds))
        true_mels, true_gates, ga_mask, mels_mask = y
        mels, mels_postnet, gates, alignments = self.model(x)

        true_mels = tf.expand_dims(true_mels, -1)
        alignments = tf.expand_dims(alignments, -1)
        mels= tf.expand_dims(mels, -1)
        
        
        normalize = lambda x: (x - tf.math.reduce_min(x))/(tf.math.reduce_max(x) - tf.math.reduce_min(x))
        mels_image = tf.squeeze(normalize(tf.transpose(mels, [0,2,1,3]))*255)
        true_mels_image = tf.squeeze(normalize(tf.transpose(true_mels, [0,2,1,3]))*255)
        alignments_image = tf.squeeze(normalize(tf.transpose(alignments, [0,2,1,3]))*255)

        mels_image = tf.cast(tf.round(mels_image), dtype=tf.int32)
        true_mels_image = tf.cast(tf.round(true_mels_image), dtype=tf.int32)
        alignments_image = tf.cast(tf.round(alignments_image), dtype=tf.int32)

        mels_image = tf.gather(self.cm, mels_image)
        true_mels_image = tf.gather(self.cm, true_mels_image)
        alignments_image = tf.gather(self.cm, alignments_image)
        with self.fw.as_default() :
            tf.summary.image("mels", mels_image, step=self.g_step)
            tf.summary.image("true_mels", true_mels_image, step=self.g_step)
            tf.summary.image("alignments", alignments_image, step=self.g_step)
    def get_config(self):
        config = {'g_step': self.g_step,
                'ds': self.ds,
                'cm': self.cm}
        return config



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_checkpoint', type=str, help="continue training from given checkpoint")
    args, leftovers = parser.parse_known_args()


    # silence verbose TF feedback
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    #if platform.node() != "jean-zay3":
    #    gpl.get_gpu_lock(gpu_device_id=2, soft=False)

    """
    initialize model
    """
    date_now =  datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/" + date_now

    conf = Tacotron2Config("config/configs/tacotron2_in_use.yaml")
    train_conf = Tacotron2Config("config/configs/train_in_use.yaml")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    config_dir = os.path.join("./config", "LJSpeech")
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    callbacks = []

    print("START : ", date_now)
    print("______________________")

    print("_______TRAIN HP_______")
    pprint(train_config)
    print("_______MODEL HP_______")
    pprint(model_config)

    """
    initalize dataset
    """

    batch_size = 16
    print(train_conf["data"]["transcript_path"])
    ljspeech_text = tf.data.TextLineDataset(train_conf["data"]["transcript_path"])
    tac = Tacotron2(preprocess_config, model_config, train_config)
    tac.adapt(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1]))
    dataset_mapper = ljspeechDataset(conf, train_conf)
    ljspeech = ljspeech_text.map(dataset_mapper)

    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """
    eval_ljspeech =ljspeech.padded_batch(3, 
            padding_values=((None, None), (0., 1., 0., 0.)),
            drop_remainder=True)

    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None), (0., 1., 0., 0.)),
            drop_remainder=True)

    epochs = 1000
    learning_rate = train_config["optimizer"]["init_lr"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, 
        update_freq=30))


    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=train_conf["data"]["checkpoint_path"] + date_now,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            ))
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=tac)
    batch = next(iter(ljspeech))
    x, y = batch
    tac(x)
    tac.compile(optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint("checkpoint"))
	


    tac.fit(ljspeech,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks)
