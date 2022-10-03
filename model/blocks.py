import tensorflow as tf


class LinearNorm(tf.keras.layers.Layer):
    def __init__(self, out_dim, use_bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(out_dim, use_bias=use_bias)

    def call(self, x):
        return self.linear_layer(x)


class ConvNorm(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=1, strides=1,
                 dilation_rate=1, use_bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        self.conv = tf.keras.layers.Conv1D(out_channels,
                                    kernel_size=kernel_size, strides=strides,
                                    padding='same', dilation_rate=dilation_rate,
                                    use_bias=use_bias)

    def call(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
