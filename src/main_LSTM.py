# tutorial q explica MUITO bem como usar uma LSTM:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/


from matplotlib import pyplot as plt
from numpy.core.shape_base import block
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.losses import mean_absolute_error
from metrics.custom import custom_loss, above_or_below_zero_accuracy, movement_accuracy, mean_squared_error as custom_mse, movement_hit_or_miss, cosine_similarity as custom_cosine_similarity
from keras import metrics
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import sys
import tensorflow as tf
from data.visualization import plot_data
from data.utils import load_data, add_indicadores_tecnicos, split_data, save_data, save_picture
from data.scaling import normalize, smooth_data_curves

# from layers.attention import MultiHeadAttention, TransformerEncoder
# from layers.encoding import Time2Vector,Time2Vec

from keras.losses import cosine_similarity, mean_squared_error, log_cosh
from keras.layers import LSTM, Dropout, Dense, Bidirectional, concatenate, TimeDistributed, GlobalAveragePooling1D, GlobalMaxPooling1D, LayerNormalization, Conv1D, Add
from keras.models import Sequential, Model

import numpy as np

import os
np.random.seed(1337)  # for reproducibility
tf.random.set_seed(1337)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



coin = 'BTC'
df = load_data(coin, 'bitcoin', event_days_left_lookback=5, storage_folder='./data').iloc[:, :]

prices, prices_scaler = normalize(df.loc[:, 'price'].to_numpy())
variation = df.loc[:, 'price'].pct_change().fillna(0).to_numpy()
# variation, _ = normalize(df.loc[:, 'price'].pct_change().to_numpy())
tweet, _ = normalize(df.loc[:, 'tweet_volume'].to_numpy())
tweet_variation, _ = normalize(df.loc[:, 'tweet_volume'].pct_change().fillna(0).to_numpy())
google_trends, _ = normalize(df.loc[:, 'trend'].to_numpy())
google_trends_variation, _ = normalize(df.loc[:, 'trend'].pct_change().fillna(0).to_numpy())

volume, _ = normalize(df['Volume'].to_numpy())
volume_variation, _ = normalize(df['Volume'].pct_change().to_numpy())
# event_day_count, _ = normalize(df.loc[:, 'days_to_event_happen'].fillna(0).to_numpy())
event_votes, votes_scaler = normalize(df.loc[:, 'event_votes'].fillna(0).to_numpy(), range=(0,1))
event_confidence, conf_scaler = normalize(df.loc[:, 'event_confidence'].fillna(0).to_numpy(), range=(0,1))
event_features = [event_votes, event_confidence]
for i in range(5):
    votes, confidence = df[f'event_in_{i+1}_days_votes'].to_numpy(), df[f'event_in_{i+1}_days_confidence'].to_numpy()
    votes =  votes_scaler.transform(votes.reshape(-1,1)).reshape(-1)
    confidence =  conf_scaler.transform(confidence.reshape(-1,1)).reshape(-1)

    event_features.append(votes)
    event_features.append(confidence)

 

STOCHk_7, _ = normalize(df['STOCHk_7'].fillna(50.0).to_numpy())
STOCHk_14, _ = normalize(df['STOCHk_14'].fillna(50.0).to_numpy())
STOCHk_28, _ = normalize(df['STOCHk_28'].fillna(50.0).to_numpy())
STOCHk_56, _ = normalize(df['STOCHk_56'].fillna(50.0).to_numpy())

RSI_7, _ = normalize(df['RSI_7'].fillna(50.0).to_numpy())
RSI_14, _ = normalize(df['RSI_14'].fillna(50.0).to_numpy())
RSI_28, _ = normalize(df['RSI_28'].fillna(50.0).to_numpy())
RSI_56, _ = normalize(df['RSI_56'].fillna(50.0).to_numpy())

print('Carregou dados do csv')

plot_data([tweet, google_trends, prices, variation],# STOCHk_7, STOCHk_14, STOCHk_28, STOCHk_56, RSI_7, RSI_14, RSI_28, RSI_56], 
legends=['tweet volume', 'google trends', 'price', '%'],# 'K% (7)', 'K% (14)','K% (28)', 'K% (56)', 'RSI (7)', 'RSI (14)', 'RSI (28)', 'RSI (56)'],
          tick=200, verticalLineAt=len(prices)*0.95, labels=df.index.date[::200], blocking=False,
          title="Clique na linha da legenda para mostrar/esconder o dado no gr??fico")

input('press enter to continue (se voc?? veio s?? pra ver dados, melhor fechar ao inv??s de dar enter, pois o c??digo desse script j?? est?? desatualizado)')

# look_behind: passos (dias) anteriores usados para prever o proximo. Eles ser??o tipo "features" do proximo passo
N = 5
foward_days = 1  # quantos dias prever
used_features_names = ['tweet', 'variation', 'trends']
features = [variation, google_trends]
target = variation
# features = [variation, google_trends]
# target = variation

n_samples = len(target) - N - (foward_days-1)

# matriz de dimensao: [n_samples, N, n_features];
features_set = np.empty((n_samples, N, len(features)))  # type: np.ndarray
labels = [None]*n_samples
for i in range(n_samples):
    for j in range(N):
        for ft_idx in range(len(features)):
            feature = features[ft_idx]
            features_set[i, j, ft_idx] = feature[i+j]

    labels[i] = target[N+i-1: N+i+foward_days]
labels = np.array(labels)


# shuffle_mask = np.arange(len(train_X))
# np.random.shuffle(shuffle_mask)
# train_X, train_y = train_X[shuffle_mask], train_y[shuffle_mask]

train_X, test_X = split_data(features_set, ratio=0.93)
train_y, test_y = split_data(labels, ratio=0.93)



from tensorflow_addons.layers import MultiHeadAttention as tf_MultiHeadAttention




print('Dados preparados')
inputs = Input(shape=train_X.shape[1:])
# lstm = LSTM(units=250, return_sequences=True, input_shape=train_X.shape[1:])(inputs)
# lstm = Dropout(0.15)(lstm)

lstm = LSTM(units=450, return_sequences=False, input_shape=train_X.shape[1:])(inputs)
lstm = Dropout(0.15)(lstm)
x = lstm
# time_embedding = TimeDistributed(Time2Vec(kernel_size=1))(lstm)
# x = concatenate([lstm, time_embedding], -1)

# x = tf_MultiHeadAttention(num_heads=4, head_size=96,output_size=250,  dropout=0.15)((x,x))
# x = LayerNormalization(epsilon=1e-6)(Add()([lstm,x]))
# x = Conv1D(filters=128, kernel_size=1, activation='relu')(x)
# x = LayerNormalization(epsilon=1e-6)(Add()([lstm,x]))

# avg_pool = GlobalAveragePooling1D()(x)
# x = GlobalMaxPooling1D()(lstm)
# x = concatenate([avg_pool, max_pool])

x = Dense(units=175, activation='linear')(x)
x = Dropout(0.15)(x)
x = Dense(units=75, activation='linear')(x)
x = Dropout(0.15)(x)
x = Dense(units=foward_days)(x)
modelo = Model(inputs=inputs, outputs=x)


# modelo = Sequential()
# # layer com 100 neuronios.
# modelo.add(LSTM(units=250, return_sequences=False, input_shape=train_X.shape[1:]))
# modelo.add(Dropout(0.15))

# # modelo.add(LSTM(units=250))
# # modelo.add(Dropout(0.15))

# # modelo.add(LSTM(units=250, return_sequences=False))
# # modelo.add(Dropout(0.2))

# modelo.add(Dense(units=175))
# modelo.add(Dropout(0.15))
# modelo.add(Dense(units=25))
# modelo.add(Dropout(0.15))

# modelo.add(Dense(units=foward_days))


# def square_error(y_true, y_pred):
#     y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
#     y_true = math_ops.cast(y_true, y_pred.dtype)
#     return math_ops.squared_difference(y_pred, y_true)

# # y_pred|y_true shape = (batches, 2) --> first column is previous step label (needed to calculate cossine), second column is the label indeed

# def mse_and_cosine_similarity(y_true, y_pred):
#     """cacula o mse, e multiplica pelo cosine_similarity. O cosine_similarity possui range de valor entre 1 e 2, atuando como um "penalizador" """
#     # tf.print("\ny_true:", y_true.shape, output_stream=sys.stdout)

#     se = square_error(y_true[:, 1], y_pred[:, 1])  # values range is between 0 and 1
#     movement_score = movement_accuracy(y_true, y_pred)

#     padding = tf.constant([[0, 0], [0, 1]])
#     y_true_firstColumn = tf.pad(y_true[:, 0:1], padding, 'CONSTANT')
#     y_pred = tf.concat([y_true[:, :1], y_pred[:, 1:]], axis=-1)
#     y_pred = y_pred - y_true_firstColumn + [0.05, 0]
#     y_true = y_true - y_true_firstColumn + [0.05, 0]

#     cos_sim = cosine_similarity(y_true, y_pred, axis=-1)  # values range is between -1 and 1

#     cos_sim = (((cos_sim+1.)/1.)+1.)  # now range is between 1 and 3
#     return tf.reduce_mean(se*cos_sim**2) * (1.-above_or_below_zero_score(y_true, y_pred))


# def movement_accuracy(y_true, y_pred):
#     movement_true = tf.math.greater(y_true[:, 1], y_true[:, 0])
#     movement_pred = tf.math.greater(y_pred[:, 1], y_true[:, 0])
#     hits = tf.math.equal(movement_true, movement_pred)
#     return tf.reduce_mean(tf.cast(hits, tf.float32))


# def above_or_below_zero_score(y_true, y_pred):
#     true_above = tf.math.greater(y_true[:, 1], [0])
#     pred_above = tf.math.greater(y_pred[:, 1], [0])
#     hits = tf.math.equal(true_above, pred_above)
#     return tf.reduce_mean(tf.cast(hits, tf.float32))
def custom(y_true, y_pred):
    # c = custom_cosine_similarity(y_true, y_pred) #valores entre  -1 e 1 (na pratica entre -1 e 0)
    # c = ((c+1.)/0.25)+2 #entre 2 e 10 (na pratica 2 e 6)
    error = log_cosh(y_true[:,1:], y_pred)
    # mse = mean_squared_error(y_true[:,1:], y_pred)
    # mov = movement_hit_or_miss(y_true, y_pred)
    # flipped_mov = (mov *-1) + 1.0
    # # se acertou a dire????o do movimento
    # hit_mov = mov * (mse)
    # #se errou a dire????o do movimento
    # miss_mov = flipped_mov * (error)
    # return tf.reduce_mean(hit_mov + miss_mov)

    return error * ((1-movement_accuracy(y_true, y_pred))**2)#(tf.reduce_mean(c)**2) # ou entao: custom_mse(y_true, y_pred) * tf.reduce_mean(c)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

modelo.compile(optimizer=opt, loss=custom, metrics=['mse', log_cosh, movement_accuracy, above_or_below_zero_accuracy, cosine_similarity])# 'cosine_similarity', 'mean_squared_error'
print(modelo.summary())
print(f'train and test y shape = {train_y.shape} {test_y.shape}')
modelo.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=1, batch_size=64)  # validation_data=(test_X, test_y)
epochs = len(modelo.history.epoch)

print('Modelo treinado')

plt.plot(modelo.history.history['loss'])
plt.plot(modelo.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)

plt.plot(modelo.history.history['mse'])
plt.plot(modelo.history.history['val_mse'])
plt.title('model meanSquaredError')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)

plt.plot(modelo.history.history['movement_accuracy'])
plt.plot(modelo.history.history['val_movement_accuracy'])
plt.title('model movement direction accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)

plt.plot(modelo.history.history['above_or_below_zero_accuracy'])
plt.plot(modelo.history.history['val_above_or_below_zero_accuracy'])
plt.title('price up or down accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=True)


# PREVISAAAAOOO

train_score = modelo.evaluate(x=train_X, y=train_y, batch_size=16)
test_score = modelo.evaluate(x=test_X, y=test_y, batch_size=16)
print('Train Score:', train_score)
print('Test Score:', test_score)
#predictions = scaler.inverse_transform(predictions)


#timestep_size = 1./len(features_set)
testPredict = modelo.predict(test_X)

timestep_size = 0.1
original_vectors = [[timestep_size, test_y[i, -1] - test_y[i-1, -1]] for i in range(1, len(test_y))]
prediction_vectors = [[timestep_size, testPredict[i, -1] - test_y[i-1, -1]] for i in range(1, len(test_y))]
cos_similarity = np.mean(cosine_similarity(original_vectors, prediction_vectors))
print('Cosine similarity %s' % cos_similarity)
mse = mean_squared_error(test_y[:, -1], testPredict[:, -1]).numpy()
print('Mean Square Error %s' % mse)
upDownScore = movement_accuracy(test_y, testPredict).numpy()
print('Movement accuracy %s' % upDownScore)
above_below_zero = above_or_below_zero_accuracy(test_y, testPredict).numpy()
print('Price up or down accuracy: %s' % above_below_zero)


def variation_to_price(base_price, prices, predictions):
    predicted_prices = np.empty_like(predictions)
    # n_samples = len(prices) - N - (foward_days-1)
    last_price = base_price
    for i in range(len(prices)):
        predicted_prices[i] = last_price * (predictions[i] +1)
        last_price = prices[i]
    return predicted_prices


def to_keras_format(features: list, target):
    n_samples = len(target) - N - (foward_days-1)
    # matriz de dimensao: [n_samples, N, n_features];
    features_set = np.empty((n_samples, N, len(features)))  # type: np.ndarray
    labels = [None]*n_samples
    for i in range(n_samples):
        for j in range(N):
            for ft_idx in range(len(features)):
                feature = features[ft_idx]
                features_set[i, j, ft_idx] = feature[i+j]

        labels[i] = target[N+i-1: N+i+foward_days]
    labels = np.array(labels)

    return features_set, labels

pivot = len(train_X)
source_prices, target_prices = to_keras_format([prices], prices)
raw = variation_to_price(target_prices[pivot, 0], target_prices[pivot:, 1], test_y[:, 1])
############ DISPLAYING DATA ############
predictions = testPredict
targets_test = test_y[:, -1:].tolist()
# prediction tem um shape (samples, foward_days), entao vamos pegar previsoes de foward_days dias, a cada partir de pontos com intervalo de foward_days
foward_days_predictions = []
for i in range(1, predictions.shape[0], foward_days):
    foward_days_predictions.append(np.array([None]*(i-1) + targets_test[i-1] +
                                            predictions[i].tolist() +
                                            [None]*(predictions.shape[0]-(i+foward_days))))

# count = len(foward_days_predictions)*foward_days
# if count < predictions.shape[0]:
#     count = predictions.shape[0] - foward_days
#     foward_days_predictions.append(np.array([None]*(count-1) +
#                                             [prices_test[-foward_days-1]] +
#                                             predictions[-1].tolist()))


fig = plot_data([targets_test] + foward_days_predictions, tick=10, legends=['target', 'prediction'],
                colors=['b', 'r'] + ['r']*len(foward_days_predictions), blocking=False)
# save_picture('./resultados', figure=fig, name='1', coin=coin, model=modelo,
#              epochs=epochs, N=N, days=foward_days, features=used_features_names)

# 1 day predictions
fig = plot_data([targets_test, predictions[:, 0]], tick=10, legends=['target', 'prediction'], blocking=False)
# save_picture('./resultados', figure=fig, name='2', coin=coin, model=modelo,
#              epochs=epochs, N=N, days=foward_days, features=used_features_names)

# save_data('./resultados', coin=coin, model=modelo, epochs=epochs, N=N, days=foward_days,
#           features=used_features_names, train_loss=train_score, test_loss=test_score,
#           cosine_similarity=cos_similarity)


input('Enter para sair')
