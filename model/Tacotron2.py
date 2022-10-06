from ctypes import alignment
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.python.ops.math_ops import reduce_mean



import numpy as np

import os
import json
from math import sqrt

from .blocks import LinearNorm
from .modules import Encoder, Decoder, Postnet

@tf.keras.utils.register_keras_serializable()
def split_func(x):
    return tf.strings.unicode_split(x, 'UTF-8')

class Tacotron2(tf.keras.Model):
    """ Tacotron2 """

    
    def adapt(self, dataset):
        self.tokenizer.adapt(dataset.batch(64))

    def __init__(self, preprocess_config, model_config, train_config, vocabulary=None):
        super(Tacotron2, self).__init__()
        self.model_config = model_config
        self.tokenizer = tf.keras.layers.TextVectorization(split=split_func, standardize=None)
        if vocabulary:
            self.tokenizer.set_vocabulary(vocabulary)
        self.train_config = train_config
        self.preprocess_config = preprocess_config
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.decoder = Decoder(preprocess_config, model_config)
        self.postnet = Postnet(preprocess_config, model_config)
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.embedding = tf.keras.layers.Embedding(self.tokenizer.vocabulary_size(),
                self.model_config["encoder"]["encoder_embedding_dim"],
                mask_zero=True)

        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]

    def get_config(self):
        return {'vocabulary': self.tokenizer.get_vocabulary(), 'preprocess_config': self.preprocess_config, 'model_config': self.model_config, 'train_config': self.train_config}

    @classmethod
    def from_config(cls, config):
        return cls(config['preprocess_config'], config['model_config'], config['train_config'], vocabulary=config['vocabulary'])


    def call(self, batch):
        phon, mels = batch
        mels = tf.transpose(mels, perm=[0,2,1])
        phon = self.tokenizer(phon)
        mask = self.embedding.compute_mask(phon)
        embedded_inputs = self.embedding(phon)
        encoder_outputs = self.encoder(embedded_inputs, mask)

        mels, gates, alignments = self.decoder(encoder_outputs, mels, mask)
        mels = tf.transpose(mels, (1, 0, 2))
        alignments = tf.transpose(alignments, (1, 0, 2))
        gates = tf.squeeze(tf.transpose(gates, (1, 0, 2)),-1)
    
        mels = tf.reshape(mels, (tf.shape(mels)[0], -1, self.n_mel_channels))
        mels_postnet = self.postnet(mels)

        return mels, mels_postnet, gates, alignments
    @tf.function
    def inference(self, phon):
        
        phon = self.tokenizer(phon)
        mask = self.embedding.compute_mask(phon)
        embedded_inputs = self.embedding(phon)
        encoder_outputs = self.encoder(embedded_inputs, mask)

        mels, gates, alignments = self.decoder.inference(encoder_outputs, mask)
        mels = tf.transpose(mels, (1, 0, 2))
        alignments = tf.transpose(alignments, (1, 0, 2))
        gates = tf.squeeze(tf.transpose(gates, (1, 0, 2)), -1)
        mels = tf.reshape(mels, (tf.shape(mels)[0], -1, self.n_mel_channels))
        mels_postnet = self.postnet(mels)

        return mels, mels_postnet, gates, alignments


    def train_step(self, data):
        x, y = data
        true_mels, true_gates, ga_mask, mels_mask = y
        with tf.GradientTape() as tape:
            mels, mels_postnet, gates, alignments = self(x)
            mels *= mels_mask
            mels_postnet *= mels_mask
            align_penalty = tf.math.reduce_mean(alignments * ga_mask)
            mels_loss = self.mse_loss(mels, true_mels)
            mels_postnet_loss = self.mse_loss(mels + mels_postnet, true_mels)
            gates_loss = self.bce_loss(true_gates, gates)
            loss = mels_loss + mels_postnet_loss + gates_loss + align_penalty

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return  {'loss' : loss, 'mels_postnet_loss' : mels_postnet_loss, 'mels_loss' : mels_loss, 'gates_loss' : gates_loss, 'align_penalty' : align_penalty}

