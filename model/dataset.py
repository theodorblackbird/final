import tensorflow as tf


def ljspeech_map_func(x):
    x = tf.strings.split(x, sep='|')[1]
    return x

def tv_func(x):
    x = tf.strings.unicode_split(x, 'UTF-8')
    return x
