import tensorflow as tf
from keras.losses import cosine_similarity as keras_cosine_similarity, log_cosh
from keras.losses import mean_absolute_percentage_error as keras_mean_absolute_percentage_error
from keras.metrics import RootMeanSquaredError as keras_RootMeanSquaredError
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorflow.python.keras import backend as K
import numpy as np

def square_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true[:, 1:], y_pred.dtype)
    return math_ops.squared_difference(y_pred, y_true)


def mean_squared_error(y_true, y_pred):
    return K.mean(square_error(y_true, y_pred))

class RootMeanSquaredError(keras_RootMeanSquaredError):
    """Computes root mean squared error metric between `y_true` and `y_pred`.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(RootMeanSquaredError, self).update_state(y_true[:, 1:], y_pred, sample_weight=sample_weight)

def mean_absolute_percentage_error(y_true, y_pred):
    return keras_mean_absolute_percentage_error(y_true[:, 1:], y_pred)

# y_pred|y_true shape = (batches, 2) --> first column is previous step label (needed to calculate cossine), second column is the label indeed

def cosine_similarity(y_true, y_pred):
    y_true_firstColumn = y_true[:, :1]
    y_pred = tf.concat([y_true[:, :1], y_pred], axis=-1)    # junta o valor anterior com o atual previsto
    y_pred = y_pred - y_true_firstColumn + [0.1, 0]       # move o vetor para sair da origem, e faz eixo x = 0.1
    y_true = y_true - y_true_firstColumn + [0.1, 0]

    return keras_cosine_similarity(y_true, y_pred, axis=-1)  # values range is between -1 and 1

# pearson correlation coefficient
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true[:, 1:]
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def custom_loss3(y_true, y_pred):
    cos_sim = cosine_similarity(y_true, y_pred)
    cos_sim = (cos_sim+1.)/2.

    # error = log_cosh(y_true[:,1:], y_pred)
    error = mean_squared_error(y_true, y_pred)
    return error + tf.reduce_mean(cos_sim)

def custom_loss(y_true, y_pred):
    """cacula o mse, e multiplica pelo cosine_similarity. O cosine_similarity possui range de valor entre 1 e 2, atuando como um "penalizador" """
    # tf.print("\ny_true:", y_true.shape, output_stream=sys.stdout)
    # tf.print("\ny_pred:", y_pred.shape, output_stream=sys.stdout)

    se = square_error(y_true, y_pred)  # values range is between 0 and 1
    # movement_score = movement_accuracy(y_true, y_pred)
    cos_sim = cosine_similarity(y_true, y_pred)  # values range is between -1 and 1
    cos_sim = (((cos_sim+1.)/2.)+1)  # now range is between 1 and 2

    # tf.reduce_mean(se*cos_sim**2) * (1.-above_or_below_zero_score(y_true, y_pred))
    return K.mean(se) * K.mean(cos_sim)  # * ((1.-movement_score)**2)

def custom_loss2(y_true, y_pred):
    c = cosine_similarity(y_true, y_pred) #valores entre  -1 e 1 (na pratica entre -1 e 0)
    c = ((c+1.)*2)+1
    # error = log_cosh(y_true[:,1:], y_pred)
    error = mean_squared_error(y_true, y_pred)

    # mse = mean_squared_error(y_true[:,1:], y_pred)
    # mov = movement_hit_or_miss(y_true, y_pred)
    # flipped_mov = (mov *-1) + 1.0
    # # se acertou a direção do movimento
    # hit_mov = mov * (mse)
    # #se errou a direção do movimento
    # miss_mov = flipped_mov * (error)
    # return tf.reduce_mean(hit_mov + miss_mov)

    return error * tf.reduce_mean(c) # ou entao: custom_mse(y_true, y_pred) * tf.reduce_mean(c)

def custom_movement_accuracy(y_true, y_pred):
    error = log_cosh(y_true[:,1:], y_pred)
    return error * ((1-movement_accuracy(y_true, y_pred))**2)



def movement_hit_or_miss(y_true, y_pred):
    movement_true = tf.math.greater(y_true[:, 1], y_true[:, 0])
    movement_pred = tf.math.greater(y_pred[:, -1], y_true[:, 0])
    hits = tf.math.equal(movement_true, movement_pred)
    return tf.cast(hits, tf.float32)


def movement_accuracy(y_true, y_pred):
    return K.mean(movement_hit_or_miss(y_true, y_pred))


def above_or_below_zero_accuracy(y_true, y_pred):
    true_above = tf.math.greater(y_true[:, 1], [0])
    pred_above = tf.math.greater(y_pred[:, -1], [0])
    hits = tf.math.equal(true_above, pred_above)
    return tf.reduce_mean(tf.cast(hits, tf.float32))
