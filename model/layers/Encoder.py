import tensorflow as tf

class EncConvLayer(tf.keras.layers.Layer):
    def __init__(self,
            filters,
            kernel_size,
            dropout_rate) -> None:
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
                filters,
                kernel_size,
                padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(
                rate=dropout_rate)
        self.support_masking = True
    def call(self, x, training=True):
        y = self.conv(x)
        y = self.bn(y, training=training)
        y = tf.nn.relu(y)
        y = self.dropout(y, training=training)
        return y


