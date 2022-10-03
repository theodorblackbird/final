from tensorflow.keras.layers import Layer, Dense, Conv1D, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout

import tensorflow as tf


class Prenet(Layer):
    def __init__(self, units, n, dropout_rate):
        super().__init__()
        self.layers = Sequential()
        for i in range(n):
            self.layers.add(Dense(units, activation='relu'))
            self.layers.add(Dropout(dropout_rate))

    def call(self, x):
        return self.layers(x)

class LSAttention(Layer):

    def __init__(self, 
            rnn_dim,
            embed_dim,
            att_dim,
            att_n_filters,
            att_ker_size):
        super(LSAttention, self).__init__()

        self.query_dense = Dense(att_dim, 
                use_bias=False)
        self.memory_dense = Dense(att_dim, 
                use_bias=False)

        self.location_dense = Dense(att_dim,
                use_bias=False)
        self.location_conv = Conv1D(att_n_filters, 
                att_ker_size,
                padding='same',
                use_bias=False)

        self.energy_dense = Dense(1, 
                use_bias=False)


    def prepare_attention(self, batch):
        batch_size = tf.shape(batch)[0]
        max_len = tf.shape(batch)[1]
        encoder_dim = tf.shape(batch)[2]

        self.att_weights = tf.zeros([batch_size, max_len])
        self.cum_att_weights = tf.zeros_like(self.att_weights)
        self.att_context = tf.zeros([batch_size, encoder_dim])
    
    def process_memory(self, memory):
        return self.memory_dense(memory)

    def alignment_score(self, query, W_memory, att_weights, cum_att_weights):
        
        cat_att_weights = tf.concat([tf.expand_dims(att_weights, 1), tf.expand_dims(cum_att_weights, 1)],
                1)
        cat_att_weights = tf.transpose(cat_att_weights, perm=[0,2,1])
        
        W_query = self.query_dense(tf.expand_dims(query, 1))
        W_att_weights = self.location_conv(cat_att_weights)
        W_att_weights = self.location_dense(W_att_weights)
        alignment = self.energy_dense(tf.math.tanh(W_query + W_att_weights + W_memory))
        return tf.squeeze(alignment, axis=-1)

    def call(self, att_hs, memory, W_memory, memory_mask, att_weights, cum_att_weights):


        alignment = self.alignment_score(att_hs, W_memory, att_weights, cum_att_weights)
        alignment = tf.where(memory_mask, alignment, -float("inf"))
        att_weights = tf.nn.softmax(alignment, axis=1)
        tf.print(tf.shape(memory))
        tf.print(tf.shape(tf.expand_dims(att_weights, 1)))
        att_context = tf.matmul(tf.expand_dims(att_weights, 1), memory)
        att_context = tf.squeeze(att_context)


        return att_context, att_weights 

class DecConvLayer(Layer):
    def __init__(self,
            filters,
            kernel_size,
            dropout_rate,
            activation=None) -> None:
        super().__init__()
        self.conv = Conv1D(
                filters,
                kernel_size,
                padding="same")
        self.bn = BatchNormalization()
        self.dropout = Dropout(
                rate=dropout_rate)
        self.support_masking = True
        self.activation = tf.keras.layers.Activation(activation)
    def call(self, x, training=True):
        y = self.conv(x)
        y = self.bn(y, training=training)
        y = self.activation(y)
        y = self.dropout(y, training=training)
        return y


class Postnet(Layer):
    def __init__(self, 
            filters, 
            n,
            n_mel_channels,
            kernel_size,
            dropout_rate,
            n_frames_per_step):
        super().__init__()
        self.layers = Sequential()
        for i in range(0, n-1):
            self.layers.add(DecConvLayer(filters, kernel_size, dropout_rate, activation='tanh'))
        self.layers.add(DecConvLayer(n_mel_channels, kernel_size, dropout_rate))
    
    def call(self, x):
        return self.layers(x)




