import numpy as np

allPrices = [0,1,2,3,4,5,6,7,8,9]
prices = allPrices[:7]
prices_test = allPrices[7:]

features = [prices]
N = 2  # passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo

# matriz de dimensao: [n_samples, N, n_features];
features_set = np.empty((len(prices)-N, N, len(features)))  # type: np.ndarray
# talvez labels n precise de dimensao tao grande assim
# labels = [np.empty((len(prices)-N, len(features))  # type: np.ndarray
#  preenchendo a matriz de dimensao [n_samples, N, n_features];
for ft_idx in range(len(features)):
    feature = features[ft_idx]
    for i in range(N, len(feature)):
        #labels[i-N, ft_idx] = feature[i]
        for j in range(N):
            features_set[i-N, j, ft_idx] = feature[i-N+j]
labels = prices[N:]














prices_test_input = np.concatenate((prices[-N:], prices_test))

test_features_input = [prices_test_input]

test_features = np.empty((len(prices_test), N, len(features)))
for ft_idx in range(len(test_features_input)):
    test_input = test_features_input[ft_idx]
    for i in range(len(prices_test)):
        for j in range(N):
            test_features[i, j, ft_idx] = test_input[i+j]

test_labels = prices_test