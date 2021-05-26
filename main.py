# tutorial q explica MUITO bem como usar uma LSTM:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/


from keras import Input
import tensorflow as tf
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from data.visualization import plot_data
from data.utils import load_data, split_data, save_data, save_picture
from data.scaling import normalize, smooth_data_curves

from layers.attention import MultiHeadAttention, TransformerEncoder
from layers.encoding import Time2Vector

from keras.losses import cosine_similarity, mean_squared_error
from keras.layers import LSTM, Dropout, Dense, Bidirectional, Concatenate
from keras.models import Sequential, Model
#import os
import numpy as np

# np.random.seed(1337)  # for reproducibility
# tf.random.set_seed(1337)
#os.environ['TF_DETERMINISTIC_OPS'] = str(1)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


coin = 'BTC'
df = load_data(coin, 'bitcoin').iloc[:, :]
prices = normalize(df.loc[:, 'price'].to_numpy())[0]
variation = normalize(df.loc[:, 'price (%)'].to_numpy(), range=(-1., 1.))[0]
tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]
event_day_count = normalize(df.loc[:, 'days_to_event_happen'].fillna(0).to_numpy())[0]
event_votes = normalize(df.loc[:, 'event_votes'].fillna(0).to_numpy())[0]
event_confidence = normalize(df.loc[:, 'event_confidence'].fillna(0).to_numpy())[0]

print('Carregou dados do csv')

plot_data([tweet, google_trends, prices, variation], legends=['tweet volume', 'google trends', 'price', '%'],
          tick=200, verticalLineAt=len(prices)*0.95, labels=df.index.date[::200], blocking=False)

input('press enter to continue')

# look_behind: passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
N = 20
foward_days = 1  # quantos dias prever
used_features_names = ['tweet', 'variation', 'trends']
features = [variation, google_trends, tweet]
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

    labels[i] = target[N+i: N+i+foward_days]
labels = np.array(labels)


train_X, test_X = split_data(features_set, ratio=0.9)
train_y, test_y = split_data(labels, ratio=0.9)

# shuffle_mask = np.arange(len(train_X))
# np.random.shuffle(shuffle_mask)
# train_X, train_y = train_X[shuffle_mask], train_y[shuffle_mask]

print('Dados preparados')

# inicializa timeEmbedding e tranformers
time_emb_layer1 = Time2Vector(N, linear_used_features_shape=slice(0, 1), reduceMean=True)
time_emb_layer2 = Time2Vector(N, linear_used_features_shape=slice(1, 2), reduceMean=True)
#time_emb_layer3 = Time2Vector(N, linear_used_features_shape=slice(2, 3), reduceMean=True)
#attn_layer = MultiHeadAttention(d_k=64, d_v=64, n_heads=4)
transf_encoding_layer1 = TransformerEncoder(d_k=64, d_v=64, n_heads=4, ff_dim=256)
transf_encoding_layer2 = TransformerEncoder(d_k=64, d_v=64, n_heads=4, ff_dim=256)
transf_encoding_layer3 = TransformerEncoder(d_k=64, d_v=64, n_heads=4, ff_dim=256)


in_seq = Input(shape=train_X.shape[1:])
time_emb1 = time_emb_layer1(in_seq)
time_emb2 = time_emb_layer2(in_seq)
#time_emb3 = time_emb_layer3(in_seq)
x = Concatenate(axis=-1)([in_seq, time_emb1, time_emb2])

x = transf_encoding_layer1((x, x, x))
x = transf_encoding_layer2((x, x, x))
x = transf_encoding_layer3((x, x, x))

x = Bidirectional(LSTM(units=250, return_sequences=True, input_shape=train_X.shape[1:3]))(x)
x = Dropout(0.15)(x)
x = Bidirectional(LSTM(units=250, return_sequences=False, input_shape=train_X.shape[1:3]))(x)
x = Dropout(0.15)(x)

# x = GlobalAveragePooling1D(data_format='channels_last')(x)
# x = Dropout(0.15)(x)
x = Dense(96, activation='linear')(x)
x = Dropout(0.15)(x)

#x_lstm = GlobalAveragePooling1D(data_format='channels_last')(x_lstm)
# x_lstm = Dense(units=175)(x_lstm)
# x_lstm = Dropout(0.15)(x_lstm)

#x = Concatenate(axis=-1)([x, x_lstm])
x = Dense(units=25)(x)
x = Dropout(0.15)(x)

out = Dense(units=foward_days, activation='linear')(x)

modelo = Model(inputs=in_seq, outputs=out)


def mse_and_cosine_similarity(y_true, y_pred, variables):
    """cacula o mse, e multiplica pelo cosine_similarity. O cosine_similarity possui range de valor entre 1 e 2, atuando como um "penalizador" """
    cos_sim = cosine_similarity(y_true, y_pred)  # values range is between -1 and 1
    mse = mean_squared_error(y_true, y_pred)  # values range is between 0 and 1
    import sys
    tf.print("\ny_true:", y_true.shape, output_stream=sys.stdout)
    tf.print("y_pred:", y_pred.shape, output_stream=sys.stdout)
    tf.print("y_pred:", y_pred[0,:], output_stream=sys.stdout)
    tf.print('counter: ', variables, output_stream=sys.stdout)
    
    cos_sim = (((cos_sim+1.)/4.)+0.5)  # now range is between 0.5 and 1
    return tf.reduce_mean(mse * cos_sim)


opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

modelo.compile(optimizer=opt, loss=mse_and_cosine_similarity)  # 'cosine_similarity', 'mean_squared_error'
print(modelo.summary())

modelo.fit(train_X, train_y, validation_data=(test_X, test_y),
           epochs=700, batch_size=16)  # validation_data=(test_X, test_y)
epochs = len(modelo.history.epoch)

print('Modelo treinado')


# PREVISAAAAOOO
trainPredict = modelo.predict(train_X)
testPredict = modelo.predict(test_X)
train_score = modelo.evaluate(x=train_X, y=train_y, batch_size=64)
test_score = modelo.evaluate(x=test_X, y=test_y, batch_size=64)
print('Train Score:', train_score)
print('Test Score:', test_score)
#predictions = scaler.inverse_transform(predictions)


timestep_size = 1./len(features_set)
original_vectors = [[timestep_size, test_y[i, 0] - test_y[i-1, 0]] for i in range(1, len(test_y))]
prediction_vectors = [[timestep_size, testPredict[i, 0] - test_y[i-1, 0]] for i in range(1, len(test_y))]
cos_similarity = np.mean(cosine_similarity(original_vectors, prediction_vectors))
print('Cosine similarity %s' % cos_similarity)
mse = np.mean(mean_squared_error(original_vectors, prediction_vectors))
print('Mean Square Error %s' % mse)


############ DISPLAYING DATA ############
predictions = testPredict
targets_test = test_y[:, 0].tolist()
# prediction tem um shape (samples, foward_days), entao vamos pegar previsoes de foward_days dias, a cada partir de pontos com intervalo de foward_days
foward_days_predictions = []
for i in range(1, predictions.shape[0], foward_days):
    foward_days_predictions.append(np.array([None]*(i-1) + [targets_test[i-1]] +
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
