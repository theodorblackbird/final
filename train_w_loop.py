import argparse
import tensorflow as tf
import sys

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2, split, alignLoss, gateLoss

import os
import platform
import tensorflow_addons as tfa
#if platform.node() != "jean-zay3":
#   import manage_gpus as gpl
from datetime import datetime
from pprint import pprint
import time
import matplotlib.cm
import numpy as np

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
class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds):
        self.g_step = 0
        self.ds = ds
        self.cm = tf.constant(matplotlib.cm.get_cmap('viridis').colors, dtype=tf.float32)

    def on_train_batch_end(self, batch, logs=None):
        for key in logs.keys():
            tf.summary.scalar(key, logs[key], step=self.g_step)

        if self.g_step%self.model.train_conf["train"]["plot_every_n_batches"] == 0:
            x, y = next(iter(self.ds))
            
            y_hat = self.model(x, training=True)

            """
            compute loss
            """
            
            _, _, _, gate_mask = x
            true_mels, _, gate_gt, ga_mask = y

            mels = y_hat["mels_post"]
            alignments = y_hat["alignments"]

            true_mels = tf.expand_dims(true_mels, -1)

            alignments = tf.expand_dims(alignments, -1)
            ga_mask = tf.expand_dims(ga_mask, -1)
            
            normalize = lambda x: (x - tf.math.reduce_min(x))/(tf.math.reduce_max(x) - tf.math.reduce_min(x))

            mels_image = tf.squeeze(normalize(tf.transpose(mels, [0,2,1,3]))*255)
            true_mels_image = tf.squeeze(normalize(tf.transpose(true_mels, [0,2,1,3]))*255)
            alignments_image = tf.squeeze(normalize(tf.transpose(alignments, [0,2,1,3]))*255)
            ga_mask_image = tf.squeeze(normalize(tf.transpose(ga_mask, [0,2,1,3]))*255)

            mels_image = tf.cast(tf.round(mels_image), dtype=tf.int32)
            true_mels_image = tf.cast(tf.round(true_mels_image), dtype=tf.int32)
            alignments_image = tf.cast(tf.round(alignments_image), dtype=tf.int32)
            ga_mask_image = tf.cast(tf.round(ga_mask_image), dtype=tf.int32)

            mels_image = tf.gather(self.cm, mels_image)
            true_mels_image = tf.gather(self.cm, true_mels_image)
            alignments_image = tf.gather(self.cm, alignments_image)
            ga_mask_image = tf.gather(self.cm, ga_mask_image)

            tf.summary.image("mels", mels_image, step=self.g_step)
            tf.summary.image("true_mels", true_mels_image, step=self.g_step)
            tf.summary.image("alignments", alignments_image, step=self.g_step)
            tf.summary.image("ga_mask", ga_mask_image, step=self.g_step)
        self.g_step += 1
    def get_config(self):
        config = {'g_step': self.g_step,
                'ds': self.ds,
                'cm': self.cm}
        return config

@tf.function(experimental_relax_shapes=True)
def train_step(x, y):

    with tf.GradientTape() as tape:

        y_hat = tac(x, training=True)

        """
        compute loss
        """
        _, _, _, gates_mask = x
        true_mels, _, gate_gt, ga_mask = y

        pre_loss =  mse_loss(true_mels, y_hat['mels_pre']) 
        post_loss = mse_loss(true_mels, y_hat['mels_post'])
        ga_loss = align_loss(ga_mask, y_hat['alignments'])
        gate_norm = tf.math.count_nonzero(gates_mask, dtype=tf.float32)
        gate_l = gate_loss(gate_gt, y_hat['gates'])/gate_norm

        loss = pre_loss + post_loss + gate_l/gate_norm + ga_loss 

    grads = tape.gradient(loss, tac.trainable_weights)

    return y_hat['mels_post'], true_mels, y_hat['alignments'], ga_mask, (loss, pre_loss, post_loss, ga_loss, gate_l), grads

def log_step(mels, true_mels, alignments, ga_mask, loss):
    if g_step%train_conf["train"]["plot_every_n_batches"] == 0:
        cm = tf.constant(matplotlib.cm.get_cmap('viridis').colors, dtype=tf.float32)
        true_mels = tf.expand_dims(true_mels, -1)

        alignments = tf.expand_dims(alignments, -1)
        ga_mask = tf.expand_dims(ga_mask, -1)
        
        normalize = lambda x: (x - tf.math.reduce_min(x))/(tf.math.reduce_max(x) - tf.math.reduce_min(x))

        mels_image = tf.squeeze(normalize(tf.transpose(mels, [0,2,1,3]))*255)
        true_mels_image = tf.squeeze(normalize(tf.transpose(true_mels, [0,2,1,3]))*255)
        alignments_image = tf.squeeze(normalize(tf.transpose(alignments, [0,2,1,3]))*255)
        ga_mask_image = tf.squeeze(normalize(tf.transpose(ga_mask, [0,2,1,3]))*255)

        mels_image = tf.cast(tf.round(mels_image), dtype=tf.int32)
        true_mels_image = tf.cast(tf.round(true_mels_image), dtype=tf.int32)
        alignments_image = tf.cast(tf.round(alignments_image), dtype=tf.int32)
        ga_mask_image = tf.cast(tf.round(ga_mask_image), dtype=tf.int32)

        mels_image = tf.gather(cm, mels_image)
        true_mels_image = tf.gather(cm, true_mels_image)
        alignments_image = tf.gather(cm, alignments_image)
        ga_mask_image = tf.gather(cm, ga_mask_image)

        tf.summary.image("mels", mels_image, step=g_step)
        tf.summary.image("true_mels", true_mels_image, step=g_step)
        tf.summary.image("alignments", alignments_image, step=g_step)
        tf.summary.image("ga_mask", ga_mask_image, step=g_step)
        
    loss, pre_loss, post_loss, ga_loss, gate_l = loss
    tf.print(f"pre_loss : ", pre_loss, " | post_loss : ", post_loss, " | ga_loss : ", ga_loss, " | gate_loss", gate_l, output_stream=sys.stdout)
    tf.print( "total loss : ", loss)
    
    train_pre_loss(pre_loss)
    train_post_loss(post_loss)
    train_loss(loss)
    train_ga_loss(ga_loss)
    train_gate_loss(gate_l)

    with file_writer.as_default():
        tf.summary.scalar('pre_loss', train_pre_loss.result(), step=g_step)
        tf.summary.scalar('post_loss', train_post_loss.result(), step=g_step)
        tf.summary.scalar('ga_loss', train_ga_loss.result(), step=g_step)
        tf.summary.scalar('gate_loss', train_gate_loss.result(), step=g_step)
        tf.summary.scalar('loss', train_loss.result(), step=g_step)


    train_pre_loss.reset_states()
    train_post_loss.reset_states()
    train_loss.reset_states()
    train_ga_loss.reset_states()
    train_gate_loss.reset_states()



def save_checkpoint(model, optimizer, checkpoint_path, epoch, date):
    model.save_weights(checkpoint_path+"model_"+date+"-"+str(epoch))
    np.save(checkpoint_path+"optimizer_"+date+"-"+str(epoch), optimizer.get_weights())

def load_checkpoint(model, optimizer, checkpoint_path, epoch, date, dummy_input):

    model(dummy_input)

    grad_vars = model.trainable_weights
    
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    
    opt_weights = np.load(checkpoint_path+"optimizer_"+date+"-"+str(epoch)+".npy", allow_pickle=True)
    
    optimizer.apply_gradients(zip(zero_grads, grad_vars))
    optimizer.set_weights(opt_weights)

    model.load_weights(checkpoint_path+"model_"+date+"-"+str(epoch))



    




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_checkpoint', nargs='+', type=str, help="continue training from given checkpoint")
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

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    conf = Tacotron2Config("config/configs/tacotron2_in_use.yaml")
    train_conf = Tacotron2Config("config/configs/train_in_use.yaml")
    tac = Tacotron2(conf, train_conf)
    callbacks = []

    print("START : ", date_now)
    print("______________________")
    print("without gate length normalization")
    print("_______TRAIN HP_______")
    pprint(train_conf["train"])
    print("_______MODEL HP_______")
    pprint(conf.conf)

    """
    initalize dataset
    """

    batch_size = train_conf["train"]["batch_size"]
    ljspeech_text = tf.data.TextLineDataset(train_conf["data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1])) #initialize tokenizer and char. embedding
    dataset_mapper = ljspeechDataset(conf, train_conf)
    ljspeech = ljspeech_text.map(dataset_mapper)

    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """
    eval_ljspeech =ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None, 0.), (0., 0., 1., 0.)),
            drop_remainder=train_conf["train"]["drop_remainder"])
    loss_callback = LossCallback(eval_ljspeech)

    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None, 0.), (0., 0., 1., 0.)),
            drop_remainder=train_conf["train"]["drop_remainder"])
    ljspeech = ljspeech.prefetch(tf.data.AUTOTUNE)

    epochs = train_conf["train"]["epochs"]
    learning_rate = train_conf["train"]["lr"]
    warmup_steps = train_conf["train"]["warmup_steps"]

    optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=NoamLR(learning_rate, warmup_steps),
            weight_decay=train_conf["train"]["weight_decay"],
            clipnorm=train_conf["train"]["clip_norm"],)

    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    if train_conf["train"]["tensorboard"]:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, 
            update_freq=30))

    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    align_loss = alignLoss()
    gate_loss = gateLoss(train_conf["train"]["bce_weight"])
    
    if args.continue_checkpoint is not None:
        print("continue from checkpoint : ", args.continue_checkpoint)
        load_checkpoint(tac, optimizer, train_conf["data"]["checkpoint_path"], args.continue_checkpoint[0], args.continue_checkpoint[1], next(iter(eval_ljspeech))[0] )
        start_epoch = int(args.continue_checkpoint[0])
    else: 
        start_epoch = 0
    g_step = 0
    
    start_time = time.perf_counter()

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_ga_loss = tf.keras.metrics.Mean('train_ga_loss', dtype=tf.float32)
    train_gate_loss = tf.keras.metrics.Mean('train_gate_loss', dtype=tf.float32)
    train_pre_loss = tf.keras.metrics.Mean('train_pre_loss', dtype=tf.float32)
    train_post_loss = tf.keras.metrics.Mean('train_post_loss', dtype=tf.float32)


    for epoch in range(start_epoch, epochs) :
        print("_______", "START EPOCH ", epoch, " _______")
        print("Execution time : ", time.perf_counter() - start_time)
        load_time_step = time.perf_counter()
        for i, batch in enumerate(ljspeech):
            load_time_step = time.perf_counter() - load_time_step
            print("BATCH : ", i)
            print("STEP : ", g_step)

            g_step += 1
            x, y = batch

            start_train_step = time.perf_counter()
            mels, true_mels, alignments, ga_mask, loss, grads = train_step(x,y)
            
            optimizer.apply_gradients(zip(grads, tac.trainable_weights))

            print("BATCH LOAD TIME : ", load_time_step)
            load_time_step = time.perf_counter()
            time_train_step = load_time_step - start_train_step
            print("TRAIN STEP TIME : ", time_train_step)

            log_step(mels, true_mels, alignments, ga_mask, loss)

        if epoch%10 == 0:
            print(" END EPOCH ... SAVE CHECKPOINT ")
            save_checkpoint(tac, optimizer, train_conf["data"]["checkpoint_path"], epoch, date_now)


                
           





