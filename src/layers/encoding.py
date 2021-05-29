from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class Time2Vec(Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))

class Time2Vector(Layer):
    def __init__(self, seq_len, linear_used_features_shape, reduceMean=False,  **kwargs):
        r"""
        Args:
            seq_len: sequence lenght. The input window.
            linear_used_features_shape: a tuple with the features index that should be used.
        """
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        self.feature_shape = linear_used_features_shape
        self.reduceMean = reduceMean

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
        def encode(x):
            time_linear = self.weights_linear * x + self.bias_linear
            time_linear = tf.expand_dims(time_linear, axis=-1)  # (batch, seq_len, 1)

            time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
            time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
            return tf.concat([time_linear, time_periodic], axis=-1)  # (batch, seq_len, 2)
        
        if self.reduceMean:
            # Convert (batch, seq_len, nfeatures) to (batch, seq_len)
            x = tf.math.reduce_mean(x[:, :, self.feature_shape], axis=-1)
            return encode(x)
        else:
            encodeds = []
            for x_i in x[:,:, self.feature_shape]:
                encoded = encode(x_i)
                encodeds.append(encoded[0])
                encodeds.append(encoded[1])
            return tf.concat(encodeds, axis=-1)
