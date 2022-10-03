import math
import tensorflow as tf
import numpy as np
from einops import rearrange, reduce, repeat

from .blocks import (
    LinearNorm,
    ConvNorm,
)



class Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.convolutions = []
        for _ in range(config["encoder"]["encoder_n_convolutions"]):
            conv_layer = tf.keras.Sequential(
                [ConvNorm(config["encoder"]["encoder_embedding_dim"],
                         kernel_size=config["encoder"]["encoder_kernel_size"], strides=1,
                         dilation_rate=1, w_init_gain='relu'),
                tf.keras.layers.BatchNormalization()])
            self.convolutions.append(conv_layer)

        self.lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(config["encoder"]["encoder_embedding_dim"],
                            return_sequences=True)
                )

    def call(self, x, mask):
        for conv in self.convolutions:
            x = tf.nn.dropout(tf.nn.relu(conv(x)), 0.5)

        outputs = self.lstm(x, mask=mask)

        return outputs

class Decoder(tf.keras.layers.Layer):
    def __init__(self, preprocess_config, model_config):
        super(Decoder, self).__init__()
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
        self.encoder_embedding_dim = model_config["encoder"]["encoder_embedding_dim"]
        self.attention_rnn_dim = model_config["attention"]["attention_rnn_dim"]
        self.decoder_rnn_dim = model_config["decoder"]["decoder_rnn_dim"]
        self.prenet_dim = model_config["decoder"]["prenet_dim"]
        self.max_decoder_steps = model_config["decoder"]["max_decoder_steps"]
        self.gate_threshold = model_config["decoder"]["gate_threshold"]
        self.p_attention_dropout = model_config["decoder"]["p_attention_dropout"]
        self.p_decoder_dropout = model_config["decoder"]["p_decoder_dropout"]
        attention_dim = model_config["attention"]["attention_dim"]
        attention_location_n_filters = model_config["location_layer"]["attention_location_n_filters"]
        attention_location_kernel_size = model_config["location_layer"]["attention_location_kernel_size"]

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = tf.keras.layers.LSTMCell(
            self.attention_rnn_dim)

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        self.decoder_rnn = tf.keras.layers.LSTMCell(
            self.decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = LinearNorm(
            1,
            use_bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        B = tf.shape(memory)[0]
        decoder_input = tf.zeros((B, self.n_mel_channels * self.n_frames_per_step))
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        B = tf.shape(memory)[0]
        MAX_TIME = tf.shape(memory)[1]

        attention_rnn_state = self.attention_rnn.get_initial_state(
                None, B, dtype=tf.float32
                )
        decoder_rnn_state = self.attention_rnn.get_initial_state(
                None, B, dtype=tf.float32
                )

        attention_weights = tf.zeros((B, MAX_TIME))
        attention_weights_cum = tf.zeros((B, MAX_TIME))
        attention_context = tf.zeros((B, self.encoder_embedding_dim*2))

        processed_memory = self.attention_layer.memory_layer(memory)

        return attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context, processed_memory

    def parse_decoder_inputs(self, decoder_inputs):
        B = tf.shape(decoder_inputs)[0]
        T_out = tf.shape(decoder_inputs)[2]
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = tf.transpose(decoder_inputs, (0, 2, 1))
        # Grouping multiple frames if necessary: (B, n_mel_channels, T_out) -> (B, T_out/r, n_mel_channels*r)
        decoder_inputs = tf.reshape(decoder_inputs, (B, T_out//self.n_frames_per_step, -1))
        # (B, T_out/r, n_mel_channels*r) -> (T_out/r, B, n_mel_channels*r)
        decoder_inputs = tf.transpose(decoder_inputs, (1, 0, 2))

        return decoder_inputs


    def decode(self,
            decoder_input,
            attention_rnn_state,
            decoder_rnn_state,
            attention_weights,
            attention_weights_cum,
            attention_context,
            memory,
            processed_memory,
            mask):

        cell_input = tf.concat((decoder_input, attention_context), -1)
        attention_hidden, attention_rnn_state = self.attention_rnn(
            cell_input, attention_rnn_state)
        attention_hidden = tf.nn.dropout(
            attention_hidden, self.p_attention_dropout)
        attention_rnn_state = [attention_hidden, attention_rnn_state[1]]

        attention_weights_cat = tf.concat(
            (tf.expand_dims(attention_weights, 1),
             tf.expand_dims(attention_weights_cum,1)), axis=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = tf.concat(
            (attention_hidden, attention_context), -1)
        decoder_hidden, decoder_rnn_state = self.decoder_rnn(
            decoder_input, decoder_rnn_state)
        decoder_hidden = tf.nn.dropout(
            decoder_hidden, self.p_decoder_dropout)
        decoder_rnn_state = [decoder_hidden, decoder_rnn_state[1]]

        decoder_hidden_attention_context = tf.concat(
            (decoder_hidden, attention_context), axis=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, attention_weights, attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context

    def call(self, memory, decoder_inputs, mask):
        batch_size = tf.shape(memory)[0]
        max_enc = tf.shape(memory)[1]

        decoder_input = tf.expand_dims(self.get_go_frame(memory), 0) # (1, B, n_mel_channels)
        #decoder_inputs = self.parse_decoder_inputs(decoder_inputs) # (T_out/r, B, n_mel_channels*r)
        decoder_inputs = rearrange(decoder_inputs, 'b c (h r) -> h b (c r)', r=self.n_frames_per_step)
        decoder_inputs = tf.concat((decoder_input, decoder_inputs), axis=0) 
        decoder_inputs = self.prenet(decoder_inputs) # (1+(T_out/r), B, prenet_dim)
        decoder_inputs = decoder_inputs[:-1, :, :]

        attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context, processed_memory = self.initialize_decoder_states(memory, mask=mask)

        input_len = tf.shape(decoder_inputs)[0]

        first_gate = tf.zeros((batch_size, 1))
        first_mel = tf.zeros((batch_size, self.n_frames_per_step * self.n_mel_channels))

        initializer = (first_mel, first_gate, attention_weights, attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context)

        fn = lambda acc, x: self.decode(x,
                acc[3],
                acc[4],
                acc[5],
                acc[6],
                acc[7],
                memory,
                processed_memory,
                mask)

        results = tf.scan(fn,
                decoder_inputs,
                initializer=initializer)
        mel_outputs, gate_outputs, alignments, _, _, _, _, _ = results

        return mel_outputs, gate_outputs, alignments

"""
        i = tf.constant(0, dtype=tf.int32)

        mel_outputs, gate_outputs, alignments = tf.TensorArray(tf.float32, input_len-1),tf.TensorArray(tf.float32, input_len-1), tf.TensorArray(tf.float32, input_len-1)
        max_len = tf.constant(input_len - 1, dtype=tf.int32)
        while tf.less(i, max_len) :
            decoder_input = decoder_inputs[i]
            mel_output, gate_output, attention_weights = self.decode(decoder_input, attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context, memory, processed_memory, mask)
            mels_outputs = mel_outputs.write(i, mel_output)
            gate_outputs = gate_outputs.write(i, gate_output)
            alignments = alignments.write(i, attention_weights)
            i += 1

        return mel_outputs.stack(), gate_outputs.stack(), alignments.stack()
"""
        

    def inference(self, memory):
        batch_size = tf.shape(memory)[0]
        max_enc = tf.shape(memory)[1]

        decoder_input = tf.expand_dims(self.get_go_frame(memory), 0) 
        decoder_inputs = rearrange(decoder_inputs, 'b c (h r) -> h b (c r)', r=self.n_frames_per_step)
        decoder_inputs = tf.concat((decoder_input, decoder_inputs), axis=0) 
        decoder_inputs = self.prenet(decoder_inputs) 
        decoder_inputs = decoder_inputs[:-1, :, :]

        attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context, processed_memory = self.initialize_decoder_states(memory, mask=mask)

        input_len = tf.shape(decoder_inputs)[0]

        first_gate = tf.zeros((batch_size, 1))
        first_mel = tf.zeros((batch_size, self.n_frames_per_step * self.n_mel_channels))

        initializer = (first_mel, first_gate, attention_weights, attention_rnn_state, decoder_rnn_state, attention_weights, attention_weights_cum, attention_context)
        
        cond = lambda i, gate, : i
        body = lambda i, gate,\
                decoder_input,\
                attention_weights,\
                attention_weights_cum,\
                attention_rnn_state,\
                decoder_rnn_state,\
                attention_weights,\
                attention_context,\
                memory,\
                processed_memory,\
                mask:\
                self.decode(decoder_input,
                        attention_rnn_state,
                        decoder_rnn_state,
                        attention_weights,
                        attention_weights_cum,
                        attention_context,
                        memory,
                        processed_memory,
                        mask)

        results = tf.while_loop(cond,
                decoder_inputs,
                initializer=initializer)
        mel_outputs, gate_outputs, alignments, _, _, _, _, _ = results

        return mel_outputs, gate_outputs, alignments


class LocationLayer(tf.keras.layers.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = ConvNorm(attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      use_bias=False, strides=1,
                                      dilation_rate=1)
        self.location_dense = LinearNorm(attention_dim,
                                         use_bias=False, w_init_gain='tanh')

    def call(self, attention_weights_cat):
        attention_weights_cat = tf.transpose(attention_weights_cat, (0, 2, 1))
        processed_attention = self.location_conv(attention_weights_cat)
        attention_weights_cat = tf.transpose(attention_weights_cat, (0, 2, 1))
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_dim,
                                      use_bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(attention_dim, use_bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(1, use_bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):

        processed_query = self.query_layer(tf.expand_dims(query,1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(tf.math.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = tf.squeeze(energies, -1)
        return energies

    def call(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):

        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = tf.where(mask, alignment, -float("inf"))

        attention_weights = tf.nn.softmax(alignment, axis=1)
        attention_context = tf.matmul(tf.expand_dims(attention_weights, 1), memory)
        attention_context = tf.squeeze(attention_context, 1)

        return attention_context, attention_weights


class Prenet(tf.keras.layers.Layer):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = [LinearNorm( out_size, use_bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)]

    def call(self, x):
        for linear in self.layers:
            x = tf.nn.dropout(tf.nn.relu(linear(x)), 0.5)
        return x


class Postnet(tf.keras.layers.Layer):

    def __init__(self, preprocess_config, model_config):
        super(Postnet, self).__init__()
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        postnet_embedding_dim = model_config["postnet"]["postnet_embedding_dim"]
        postnet_kernel_size = model_config["postnet"]["postnet_kernel_size"]
        postnet_n_convolutions = model_config["postnet"]["postnet_n_convolutions"]

        self.convolutions = []

        self.convolutions.append(
            tf.keras.Sequential([
                ConvNorm(postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, strides=1,
                         dilation_rate=1, w_init_gain='tanh'),
                tf.keras.layers.BatchNormalization()]
        ))

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                tf.keras.Sequential([
                    ConvNorm(postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, strides=1,
                             dilation_rate=1, w_init_gain='tanh'),
                    tf.keras.layers.BatchNormalization()]
            ))

        self.convolutions.append(
            tf.keras.Sequential([
                ConvNorm(n_mel_channels,
                         kernel_size=postnet_kernel_size, strides=1,
                         dilation_rate=1, w_init_gain='linear'),
                tf.keras.layers.BatchNormalization()]
            )
        )
    def call(self, x):

        for i in range(len(self.convolutions) - 1):
            x = tf.nn.dropout(tf.math.tanh(self.convolutions[i](x)), 0.5)
        x = tf.nn.dropout(self.convolutions[-1](x), 0.5)

        return x
