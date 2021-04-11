# tutorial q explica MUITO bem como usar uma LSTM:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional

from data.scaling import normalize, normalize_behind, normalize_arround
from data.utils import price_to_percentage, load_data, split_data
from data.visualization import plot_data


df = load_data('ltc', 'litecoin').iloc[:,:]
prices = normalize(df.loc[:, 'price'].to_numpy())[0]
variation = df.loc[:, 'variation (%)'].to_numpy()
tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]

print('Carregou dados do csv')

plot_data([tweet, google_trends, prices, variation], legends=['tweet volume', 'google trends', 'price', '%'],
          tick=200,verticalLineAt=len(prices)*0.8, labels=df.index.date[::200], blocking=False)

input('press enter to continue')

# look_behind: passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
N = 40
foward_days = 2  # quantos dias prever
features = [prices, variation]

n_samples = len(prices) -N -(foward_days-1)

# matriz de dimensao: [n_samples, N, n_features];
features_set = np.empty((n_samples, N, len(features)))  # type: np.ndarray
labels = [None]*n_samples
for i in range(n_samples):
    for j in range(N):
        for ft_idx in range(len(features)):
            feature = features[ft_idx]
            features_set[i, j, ft_idx] = feature[i+j]
    
    labels[i] = prices[N+i: N+i+foward_days]
labels = np.array(labels)

train_X, test_X = split_data(features_set)
train_y, test_y = split_data(labels)

print('Dados preparados')


modelo = Sequential()
modelo.add(Bidirectional(LSTM(units=150, return_sequences=False, input_shape=train_X.shape[1:3])))  # layer com 100 neuronios.
# modelo.add(Dropout(0.2))

#modelo.add(Bidirectional(LSTM(units=100, input_shape=(100, len(train_X)))))
# modelo.add(Dropout(0.2))

#modelo.add(LSTM(units=50, return_sequences=True))
# modelo.add(Dropout(0.2))

modelo.add(Dense(units=50))
# modelo.add(Dropout(0.2))

# 1 neuronio só pq só queremos 1 valor na saida
modelo.add(Dense(units=foward_days))  # quantidade de labels/features?


modelo.compile(optimizer='adam', loss='mean_squared_error') #'cosine_similarity', 'mean_squared_error'
modelo.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=400, batch_size=64)

print('Modelo treinado')


# PREVISAAAAOOO
trainPredict = modelo.predict(train_X)
testPredict = modelo.predict(test_X)
train_score = modelo.evaluate(x=train_X, y=train_y, batch_size=64)
test_score = modelo.evaluate(x=test_X, y=test_y, batch_size=64)
print('Train Score: %f' % train_score)
print('Test Score: %f' % test_score)
#predictions = scaler.inverse_transform(predictions)



############ DISPLAYING DATA ############
predictions = testPredict
prices_test = test_y[:,0].tolist()
# prediction tem um shape (samples, foward_days), entao vamos pegar previsoes de foward_days dias, a cada partir de pontos com intervalo de foward_days
foward_days_predictions = []
for i in range(1, predictions.shape[0], foward_days):
    foward_days_predictions.append(np.array([None]*(i-1) + [prices_test[i-1]] +
                                            predictions[i].tolist() +
                                            [None]*(predictions.shape[0]-(i+foward_days))))

count = len(foward_days_predictions)*foward_days
if count < predictions.shape[0]:
    foward_days_predictions += np.array([None]*(count-1) + 
                                        [prices_test[-foward_days]]+
                                        predictions[-1].tolist())

plot_data([prices_test] + foward_days_predictions, tick=10, legends=['actual_price', 'prediction'],
          colors=['b', 'r'] + ['r']*len(foward_days_predictions), blocking=False)
# 1 day predictions
plot_data([prices_test, predictions[:, 0]], tick=10, legends=['actual_price', 'prediction'], blocking=True)



input('Enter para sair')
