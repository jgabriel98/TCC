# tutorial q explica MUITO bem como usar uma LSTM:
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from data.scaling import normalize, normalize_behind, normalize_arround
from data.utils import price_to_percentage, load_data, split_data
from data.visualization import plot_data


df = load_data('btc', 'bitcoin').iloc[:-600,:]
prices = normalize(df.loc[:, 'price'].to_numpy())[0]
prices_train, prices_test = split_data(prices)
variation = df.loc[:, 'variation (%)'].to_numpy()
variation_train, variation_test = split_data(variation)


tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
tweet_train, tweet_test = split_data(tweet)

google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]
trends_train, trends_test = split_data(google_trends)

print('Carregou dados do csv')

plot_data([tweet, google_trends, prices, variation], legends=['tweet volume', 'google trends', 'price', '%'],
          tick=200,verticalLineAt=len(prices_train), labels=df.index.date[::200], blocking=False)

input('press enter to continue')

# look_behind: passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
N = 40
foward_days = 2  # quantos dias prever

features_train = [prices_train, variation_train]

# dados_teste + N dias anteriores
prices_test_input = np.concatenate((prices_train[-N:], prices_test))
variation_test_input = np.concatenate((variation_train[-N:], variation_test))
tweet_test_input = np.concatenate((tweet_train[-N:], tweet_test))
trends_test_input = np.concatenate((trends_train[-N:], trends_test))

test_features_input = [prices_test_input, variation_test_input]

# matriz de dimensao: [n_samples, N, n_features];
features_set = np.empty((len(prices_train)-N-foward_days+1, N, len(features_train)))  # type: np.ndarray
# talvez labels n precise de dimensao tao grande assim
# labels = [np.empty((len(prices)-N, len(features))  # type: np.ndarray
#  preenchendo a matriz de dimensao [n_samples, N, n_features];
for ft_idx in range(len(features_train)):
    feature = features_train[ft_idx]
    for i in range(N, len(feature)-foward_days+1):
        #labels[i-N, ft_idx] = feature[i]
        for j in range(N):
            features_set[i-N, j, ft_idx] = feature[i-N+j]
labels = [prices_train[N+i: -(foward_days-i-1) if i+1 < foward_days else None]
          for i in range(0, foward_days)]
print('Dados preparados')




modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=features_set.shape[1:3]))  # layer com 100 neuronios.
# modelo.add(Dropout(0.2))

modelo.add(LSTM(units=100, return_sequences=True, input_shape=(100, len(features_train))))
# modelo.add(Dropout(0.2))

#modelo.add(LSTM(units=50, return_sequences=True))
# modelo.add(Dropout(0.2))

modelo.add(LSTM(units=100, input_shape=(100, len(features_train))))
# modelo.add(Dropout(0.2))

# 1 neuronio só pq só queremos 1 valor na saida
modelo.add(Dense(units=foward_days))  # quantidade de labels/features?


modelo.compile(optimizer='adam', loss='mean_squared_error') #'cosine_similarity', 'mean_squared_error'
modelo.fit(features_set, labels, epochs=300, batch_size=64)





test_features_set = np.empty((len(prices_test)-foward_days+1, N, len(features_train)))
for ft_idx in range(len(test_features_input)):
    test_feature = test_features_input[ft_idx]
    for i in range(len(prices_test)-foward_days+1):
        for j in range(N):
            test_features_set[i, j, ft_idx] = test_feature[i+j]

#test_labels = prices_test
test_labels = [prices_test[i: -(foward_days-i-1) if i+1 < foward_days else None]
               for i in range(0, foward_days)]

# PREVISAAAAOOO
predictions = modelo.predict(test_features_set)
score = modelo.evaluate(x=test_features_set, y=test_labels, batch_size=384)
#predictions = scaler.inverse_transform(predictions)

print(score)


############ DISPLAYING DATA ############


#plot_data([prices_test, predictions], tick=50, legends=['actual', 'prediction'])

# prediction tem um shape (samples, foward_days), entao vamos pegar previsoes de foward_days dias, a cada partir de pontos com intervalo de foward_days
foward_days_predictions = []
for i in range(1, predictions.shape[0], foward_days):
    foward_days_predictions.append(np.array([None]*(i-1) + [prices_test[i-1]]+
                                            predictions[i].tolist() +
                                            [None]*(predictions.shape[0]-(i+foward_days))))

lastPiece = np.array(([None]*len(foward_days_predictions)*foward_days) + predictions[-1].tolist())
plot_data([prices_test, lastPiece] + foward_days_predictions, tick=10, legends=['actual_price', 'prediction'],
          colors=['b', 'r'] + ['r']*len(foward_days_predictions), blocking=False)

# 1 day predictions
plot_data([prices_test, predictions[:, 0]], tick=10, legends=['actual_price', 'prediction'], blocking=False)
#plot_data([variation_test, predictions[:,1]], tick=50, legends=['actual_variation', 'prediction'], blocking=False)
#plot_data([tweet_volume_test, predictions[:,2]], tick=50, legends=['actual_variation', 'prediction'], blocking=False)


input('Enter para sair')
