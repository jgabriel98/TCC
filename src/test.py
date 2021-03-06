import numpy as np
from data.loaders.coinmarketcalWebScrapper.webScrapper import CoinmarketcalWebScrapper, eventsToTimeSerie



crypto = 'litecoin'
#df = coinmarketcalWebScrapper().get_past_events(crypto)
#df.to_csv('coinmarketcal-%s.csv' % crypto)

import pandas as pd
from datetime import date, timedelta
df = pd.read_csv('coinmarketcal-%s.csv' % crypto, usecols=['added_date', 'event_date','title', 'votes', 'confidence'] ,
                dtype={'votes': int, 'confidence': float},
                parse_dates=['added_date', 'event_date'])
df2 = eventsToTimeSerie(df)


# df3 = eventsToTimeSeries(df, orderby='event_date')

# in_5_days = []
# in_4_days = []
# in_5_days = []
# in_3_days = []
# in_2_days = []
# in_1_day = []



# for row in df3.iterrows():



exit()


allPrices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

features = [allPrices]
N = 3  # passos (dias) anteriores usados para prever o proximo. Eles serão tipo "features" do proximo passo
foward_days = 2

n_samples = len(allPrices) - N - (foward_days-1)
features_set = np.empty((n_samples, N, len(features)))  # type: np.ndarray
labels = [None]*n_samples
for i in range(n_samples):
    for j in range(N):
        for ft_idx in range(len(features)):
            feature = features[ft_idx]
            features_set[i, j, ft_idx] = feature[i+j]

    labels[i] = allPrices[N+i: N+i+foward_days]


train_X, train_y = features_set[:3], labels[:3]
test_X, test_y = features_set[3:], labels[3:]
