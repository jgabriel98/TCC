from metrics.custom import custom_loss, above_or_below_zero_accuracy, movement_accuracy, mean_squared_error as custom_mse, cosine_similarity as custom_cosine_similarity
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow_addons.layers import MultiHeadAttention
from keras.layers import Dense, concatenate
from data.utils import load_data, split_data
from data.scaling import normalize
from data.visualization import plot_data

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import cosine_similarity, mean_squared_error

import numpy as np
import pandas_ta as ta
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def add_indicadores_tecnicos(df):
    macd = ta.macd(close=df['price'], fast=50, slow=200, signal=21, min_periods=None, append=True)
    df = pd.concat([df, macd], axis=1)

    df = df.fillna(0)

    bbands = ta.bbands(df['price'], fillna=0)
    df['BB_Middle_Band'] = bbands['BBM_5_2.0']
    df['BB_Upper_Band'] = bbands['BBU_5_2.0']
    df['BB_Lower_Band'] = bbands['BBL_5_2.0']
    return df


coin = 'BTC'
df = load_data(coin, 'bitcoin').iloc[:, :]
df = add_indicadores_tecnicos(df)
prices = normalize(df.loc[:, 'price'].to_numpy())[0]
variation = df.loc[:, 'price (%)'].to_numpy()
tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]
event_day_count = df.loc[:, 'days_to_event_happen'].fillna(0).to_numpy()
event_votes = normalize(df.loc[:, 'event_votes'].fillna(0).to_numpy())[0]
event_confidence = normalize(df.loc[:, 'event_confidence'].fillna(0).to_numpy())[0]
MACD_50_200_21 = normalize(df.loc[:, 'MACD_50_200_21'].to_numpy())[0]
MACDh_50_200_21 = normalize(df.loc[:, 'MACDh_50_200_21'].to_numpy())[0]
MACDs_50_200_21 = normalize(df.loc[:, 'MACDs_50_200_21'].to_numpy())[0]
BB_Middle_Band = normalize(df.loc[:, 'BB_Middle_Band'].to_numpy())[0]
BB_Upper_Band = normalize(df.loc[:, 'BB_Upper_Band'].to_numpy())[0]
BB_Lower_Band = normalize(df.loc[:, 'BB_Lower_Band'].to_numpy())[0]

print('Carregou dados do csv')
plot_data([tweet, google_trends, prices, variation], legends=['tweet volume', 'google trends', 'price', '%'],
          tick=200, verticalLineAt=len(prices)*0.95, labels=df.index.date[::200], blocking=False)
input('press enter to continue')

# look_behind: passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
N = 15
foward_days = 1  # quantos dias prever
used_features_names = ['tweet', 'variation', 'trends']
features = [variation, google_trends]
target = variation

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


train_X, test_X = split_data(features_set, ratio=0.95)
train_y, test_y = split_data(labels, ratio=0.95)

# shuffle_mask = np.arange(len(train_X))
# np.random.shuffle(shuffle_mask)
# train_X, train_y = train_X[shuffle_mask], train_y[shuffle_mask]


class Time2Vec(keras.layers.Layer):
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


class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x


class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, output_size=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)  # sera se esse trem presta mesmo?
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

        self.avg_pool_layer = GlobalAveragePooling1D()
        self.max_pool_layer = GlobalMaxPooling1D()

        # self.concat_layer = concatenate([avg_pool, max_pool])
        self.dense_layer = Dense(64, activation="sigmoid")
        self.dense_final_layer = Dense(output_size, activation="linear")

    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        avg_pool = self.avg_pool_layer(x)
        max_pool = self.max_pool_layer(x)
        concat = concatenate([avg_pool, max_pool])
        x = self.dense_layer(concat)
        x = self.dense_final_layer(x)
        return x  # K.reshape(x, (-1, x.shape[1] * x.shape[2]))  # flat vector of features out


print(f'X shape: {train_X.shape}\ny shape: {train_y.shape}')

opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False)
modelo = ModelTrunk(time2vec_dim=1, num_heads=3, num_layers=4, dropout=0.1, output_size=foward_days)# , metrics=[movement_accuracy, above_or_below_zero_accuracy, custom_mse])
modelo.compile(loss=custom_loss, optimizer=opt)
modelo.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=16, epochs=20)


# PREVISAAAAOOO
testPredict = modelo.predict(test_X)
train_score = modelo.evaluate(x=train_X, y=train_y, batch_size=64)
test_score = modelo.evaluate(x=test_X, y=test_y, batch_size=64)
print('Train Score:', train_score)
print('Test Score:', test_score)
#predictions = scaler.inverse_transform(predictions)

testPredict = modelo.predict(test_X)

#timestep_size = 1./len(features_set)
timestep_size = 0.1
original_vectors = [[timestep_size, test_y[i, 1] - test_y[i-1, 1]] for i in range(1, len(test_y))]
prediction_vectors = [[timestep_size, testPredict[i, -1] - test_y[i-1, 1]] for i in range(1, len(test_y))]

cos_similarity = np.mean(cosine_similarity(original_vectors, prediction_vectors))
print('Cosine similarity %s' % cos_similarity)
mse = mean_squared_error(test_y[:, 1], testPredict).numpy()
print('Mean Square Error %s' % mse)
upDownScore = movement_accuracy(test_y, testPredict).numpy()
print('Movement accuracy %s' % upDownScore)
above_below_zero = above_or_below_zero_accuracy(test_y, testPredict).numpy()
print('Price up or down accuracy: %s' % above_below_zero)


############ DISPLAYING DATA ############
predictions = testPredict
targets_test = test_y[:, 1:].tolist()
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
