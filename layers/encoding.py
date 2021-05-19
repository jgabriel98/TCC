from tensorflow.keras.layers import Layer
import tensorflow as tf


class Time2Vector(Layer):
    def __init__(self, seq_len, linear_used_features_shape, **kwargs):
        r"""
        Args:
            seq_len: sequence lenght. The input window.
            linear_used_features_shape: a tuple with the features index that should be used.
        """
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        self.feature_shape = linear_used_features_shape

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        # Convert (batch, seq_len, nfeatures) to (batch, seq_len)
        x = tf.math.reduce_mean(x[:, :, self.feature_shape], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)  # (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # (batch, seq_len, 2)
