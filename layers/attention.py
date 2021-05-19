import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.layers import Dense

import numpy as np


class SingleAttention(Layer):
    __constants__ = ['d_k', 'd_v', 'sqrt_d_k']
    d_k: int
    d_v: int
    sqrt_d_k: float

    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.sqrt_d_k = np.sqrt(d_k)

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape,
                           kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape,
                         kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape,
                           kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/self.sqrt_d_k, attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out


class MultiHeadAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        # outputs with shape (input_shape[0], input_shape[1], 7)
        self.linear = Dense(input_shape[0][-1], input_shape=input_shape[0], kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear
