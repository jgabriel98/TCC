from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv1D, MaxPooling1D
import sys
import random
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import gc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
import keras
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Queue
from keras import backend as K
from tensorflow.python.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
import pandas_ta as ta


from data.visualization import plot_data
from data.utils import load_data, split_data
from data.scaling import normalize


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# look_behind: passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
N = 15
foward_days = 1  # quantos dias prever

coin = 'BTC'
df = load_data(coin, 'bitcoin').iloc[:, :]

data = df


"""# **Sobre a base de dados**
Como podemos observar, o histórico de preços contém as colunas: data, preço de abertura, preço máximo, preço mínimo, preço de fechamento, ajuste de fechamente e volume, respectivamente. A base de dados contém preços das negociações desde **2016** até **2020**. 
"""


#data['Close'].plot(legend = True, figsize = (18, 6))

"""**2. Preparação dos dados**

Removendo o índice a coluna de data, pois esta não será necessária.

**Inserindo indicadores técnicos**
"""


# aobvdf = ta.aobv(close=data['Close'], volume=data['Volume'], mamode='sma', fast=10, slow=20)
# data['OBV'] = aobvdf['OBV']

macd = ta.macd(close=data['price'], fast=50, slow=200, signal=21, min_periods=None, append=True)
data = pd.concat([data, macd], axis=1)

data = data.fillna(0)

bbands = ta.bbands(data['price'], fillna=0)
data['BB_Middle_Band'] = bbands['BBM_5_2.0']
data['BB_Upper_Band'] = bbands['BBU_5_2.0']
data['BB_Lower_Band'] = bbands['BBL_5_2.0']


# PREDICT_LENGTH = 7
# data["nifty_future_price"] = data["Close"].shift(-PREDICT_LENGTH) # Shift it by 7 days

data.head()
data = data.iloc[221:,:]
prices = normalize(data.loc[:, 'price'].to_numpy())[0]
variation = normalize(data.loc[:, 'price (%)'].to_numpy())[0]
tweet = normalize(data.loc[:, 'tweet_volume'].to_numpy())[0]
google_trends = normalize(data.loc[:, 'trend'].to_numpy())[0]
event_day_count = normalize(data.loc[:, 'days_to_event_happen'].fillna(0).to_numpy())[0]
event_votes = normalize(data.loc[:, 'event_votes'].fillna(0).to_numpy())[0]
event_confidence = normalize(data.loc[:, 'event_confidence'].fillna(0).to_numpy())[0]
MACD_50_200_21 = normalize(data.loc[:, 'MACD_50_200_21'].to_numpy())[0]
MACDh_50_200_21 = normalize(data.loc[:, 'MACDh_50_200_21'].to_numpy())[0]
MACDs_50_200_21 = normalize(data.loc[:, 'MACDs_50_200_21'].to_numpy())[0]
BB_Middle_Band = normalize(data.loc[:, 'BB_Middle_Band'].to_numpy())[0]
BB_Upper_Band = normalize(data.loc[:, 'BB_Upper_Band'].to_numpy())[0]
BB_Lower_Band = normalize(data.loc[:, 'BB_Lower_Band'].to_numpy())[0]

used_features_names = ['tweet', 'variation', 'trends']
features = [variation, MACD_50_200_21, MACDh_50_200_21, MACDs_50_200_21, BB_Middle_Band, BB_Upper_Band, BB_Lower_Band]
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

X_train, X_test = split_data(features_set, ratio=0.90)
y_train, y_test = split_data(labels, ratio=0.90)


# data.dropna(subset=['MACD_50_200_21','MACDh_50_200_21','MACDs_50_200_21'])

# data["Label"] = np.where(data["nifty_future_price"] >= data["Close"],1,0)
# #print(f"df with future column {df[:5]}")
# #dropping 'nifty_future_price'  columns as it is no longer required
# data.drop('nifty_future_price',1,inplace=True) # must be dropped
# data.head()

# try:
#     from dataloader import TokenList, pad_to_longest
#     # for transformer
# except: pass

embed_size = 60


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm:
            return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Transformer():
    def __init__(self, len_limit, embedding_matrix, d_model=embed_size,
                 d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1,
                 share_word_emb=False, **kwargs):
        self.name = 'Transformer'
        self.len_limit = len_limit
        self.src_loc_info = False  # True # sl: fix later
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False,
                            weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix])  # Add Kaggle provided embedding here

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout,
                               word_emb=i_word_emb, pos_emb=pos_emb)

    # def get_pos_seq(self, x):
    #     mask = K.cast(K.not_equal(x, 0), 'int32')
    #     pos = K.cumsum(K.ones_like(x, 'int32'), 1)
    #     return pos * mask

    def compile(self, active_layers=999):
        src_seq_input = Input(shape=(None, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)

        # LSTM before attention layers
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)

        x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        x = Dense(1, activation="sigmoid")(conc)

        self.model = Model(inputs=src_seq_input, outputs=x)
        model.compile(loss='mse', optimizer="adam")


SEQ_LEN = N


def build_model(n_features=1):
    inp = Input(shape=(SEQ_LEN, n_features))

    # pos_emb = Embedding(n_features, 1, trainable=False, weights=[GetPosEncodingMatrix(n_features, 1)])
    # x = pos_emb(inp)

    # LSTM before attention layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=96, d_v=96, dropout=0.1)(x, x, x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="sigmoid")(x)
    x = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss="mse",
        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]),
        # metrics=['binary_accuracy'],
        optimizer="adam")

    # Save entire model to a HDF5 file
    # model.save('my_model.h5')

    return model


#predicted_stock_price_multi_head = multi_head.predict(X_test)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :dataset.shape[1] - 1]
        dataX.append(a)
        dataY.append(dataset[i+look_back, -1])
    return np.array(dataX), np.array(dataY)

# data = data.to_numpy()


look_back = N

# X, Y = create_dataset(data, look_back)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)
BATCH_SIZE = 64
EPOCHS = 50
# y_test = y_test[:, -1]
multi_head = build_model(len(features))
multi_head.summary()
multi_head.fit(X_train, y_train,
               batch_size=BATCH_SIZE,
               epochs=EPOCHS,
               validation_data=(X_test, y_test),
               #callbacks = [checkpoint , lr_reduce]
               )

prediction = multi_head.predict(X_test)
plt.plot(prediction, label='prediction')
plt.plot(y_test, label='true')
plt.show()


def process_data(df):
    SERIES_LENGTH = 30
    sequence = []
    # We want to scale the data except the label part since it is already 0 and 1
    temp = df.loc[:, df.columns != 'Label']
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp = scaler.fit_transform(temp)
    # print(f"temp{temp[:30]}")
    for i in range(len(temp)-SERIES_LENGTH):
        # iloc part is to take last column data i.e. labels
        sequence.append([np.array(temp[i:i+SERIES_LENGTH]), df.iloc[i+SERIES_LENGTH, -1]])

    np.random.shuffle(sequence)

    # Now we will count the sells and buys to balance the data
    # Algorithm : whichever count is less, we will take up the data upto that
    X = []
    y = []

    for seq, label in sequence:
        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)


X, Y = process_data(data)

#scaler = MinMaxScaler(feature_range = (0, 1))
#X = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.20, shuffle=False)

gc.collect()

data.shape

X = data[:, :-1].reshape(54, 30, data.shape[1] - 1)
Y = data[:, -1].reshape(54, 30, 1)

batch_size = 64
nb_epoch = 10

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['binary_accuracy'])
# fit network
history = model.fit(X, Y, epochs=nb_epoch, batch_size=batch_size, validation_split=0.25)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

lstm_prediction = model.predict(X_test)

lstm_prediction = [1 if val > 0.5 else 0 for val in lstm_prediction]

data[-50:]['Close'].reindex().plot(legend=True, figsize=(18, 6))

plt.figure(figsize=(20, 5))
plt.plot(lstm_prediction, label="Prediction")
plt.plot(y_prediction, label="True value")
plt.ylabel("Up or Down")
plt.xlabel("4 hours dataset")
plt.title("Comparison true vs prediction")
plt.legend()
plt.show()

print(classification_report(y_prediction, lstm_prediction))

accr = model.evaluate(X_test, y_test)
accr

"""# Random Forest Classifer"""


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

X_prediction = X_test[-50:]
y_prediction = y_test[-50:]

clf = RandomForestClassifier(random_state=4284, n_estimators=50)
clf.fit(X_train, y_train)

rfc_prediction = clf.predict(X_prediction)

"""# Comparação"""


def benchmark(predictions):
    buy = False
    max_profit = 0
    index = 0
    for _, row in data[-49:].iterrows():
        if buy == False and predictions[index] == 1:
            profit = row['Close']
            buy = True

        if buy == True and predictions[index] == 0:
            max_profit += row['Close'] - profit
            buy = False
        index += 1
    return max_profit


# LSTM Predictions
lstm = benchmark(lstm_prediction)
# Random Forest Classifier Predictions
rfc = benchmark(rfc_prediction)
# Random
random = benchmark(np.random.randint(2, size=50))

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar('LSTM', lstm, color='r')
ax.bar('Random Forest Classifier', rfc, color='b')
ax.bar('Random choice', random, color='g')
ax.set_ylabel('Profit ($)')
ax.set_title('Profit made with the predictions')
plt.show()

print(classification_report(y_prediction, rfc_prediction))
