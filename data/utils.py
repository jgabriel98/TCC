import os

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from tensorflow.python.keras import layers

from .loaders.bitinfochart import BitinfochartsWebScrapper
from .loaders.googleTrends import GoogleTrends
from .loaders.coinmarketcalWebScrapper.webScrapper import CoinmarketcalWebScrapper, eventsToTimeSerie


def price_to_percentage(data: np.ndarray) -> np.ndarray:
    new_data = np.empty(len(data))
    new_data[0] = .0
    for i in range(1, len(data)):
        diff = data[i] - data[i-1]
        new_data[i] = diff / data[i-1]

    return new_data


def fill_missing_data(data: np.ndarray) -> np.ndarray:
    end = len(data)

    # se o ultimo valor estiver faltando, nao tem como fazer estimativa desse trecho
    if data[-1] == None:
        while data[end-1] == None:
            end -= 1

    i = 1
    while i < end:
        if np.isnan(data[i]) == False:
            i += 1
            continue

        j = i
        while np.isnan(data[j]):
            j += 1

        first_value = data[i-1]
        last_value = data[j]
        step = (last_value-first_value)/(1 + j-i)
        while i < j:
            data[i] = data[i-1]+step
            i += 1

        i += 1
    return data


def load_data(crypto: str, topic: str) -> pd.DataFrame:
    """Carrega os dados.

    Parametros
        ----------
        crypto: str
            abreviação/sigla da cripto moeda. Ex: btc, eth, ada
        topic: str
            topico a pesquisar no google trends e no coinmarketCalendar. Ex: bitcoin, ethereum, cardano
    """
    file_name = 'bitinfochart_data-%s.csv' % crypto.upper()
    df = None
    if not os.path.exists(file_name):
        scrapper = BitinfochartsWebScrapper()
        g_trends = GoogleTrends()
        coinMarketCal = CoinmarketcalWebScrapper()

        df1 = scrapper.get_tweet_volume_data(crypto)
        df1['tweet_volume'] = fill_missing_data(df1.loc[:, 'tweet_volume'].to_numpy())

        start_date = max(df1.index.date[0], date.today() - relativedelta(years=5)
                         )  # limita dados para no máximo ultimos 5 anos
        df2 = g_trends.get_daily_trend(topic, start_d=start_date, end_d=date.today(), verbose=True)
        df2 = df2.drop(columns=['overlap'])

        df3 = scrapper.get_price_data(crypto)
        df3['variation (%)'] = price_to_percentage(df3.loc[:, 'price'].to_numpy())

        df4 = coinMarketCal.get_all_events(crypto=topic)
        df4 = eventsToTimeSerie(df4)
        df4.rename(columns={'days_to_happen': 'days_to_event_happen', 'title': 'event_title',
                            'votes': 'event_votes', 'confidence': 'event_confidence'}, inplace=True)

        df = pd.concat([df1, df2, df3], axis=1, join='inner').astype({'tweet_volume': float, 'trend': float})
        df = pd.merge(left=df, right=df4, on='date', how='left')
        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name, dtype={'tweet_volume': float, 'trend': float},
                         parse_dates=['date']).set_index('date')
    return df


def split_data(data: list, ratio=0.80):
    pivot = int(len(data)*ratio)
    return data[:pivot], data[pivot:]


def __layer_str(layer):
    def __name(layer): return type(layer).__name__
    if __name(layer) == 'InputLayer':
        return ''
    if hasattr(layer, 'layer'):
        return '%s(%s)' % (__name(layer), __layer_str(layer.layer))

    if hasattr(layer, 'units'):
        return '%s(%s)' % (__name(layer), layer.units)
    return '%s(%s)' % (__name(layer), layer.rate)


def save_picture(folder_path: str, figure, name: str, coin: str, model, epochs: int, N: int, days: int, features):
    features_str = '_'.join(features)
    model_str = list(map(__layer_str, model.layers))
    model_str = ' '.join(model_str)

    if not os.path.exists(f'{folder_path}/{coin}/'):
        os.mkdir(f'{folder_path}/{coin}/')
    figure.savefig(f'{folder_path}/{coin}/epoch{epochs}-N{N}-{features_str}-{model_str} - fig {name}.png')


def save_data(folder_path: str, coin: str, model, epochs: int, N: int, days: int, features, train_loss: float, test_loss: float, cosine_similarity: float):
    model_str = list(map(__layer_str, model.layers))
    model_str = ' > '.join(model_str)
    features_str = ','.join(features)
    coin = coin.upper()

    file_name = f'{folder_path}/resultados.csv'
    df = None
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col='id')
    else:
        df = pd.DataFrame()

    data = {
        'coin': coin,
        'N': N,
        'days': days,
        'train loss': train_loss,
        'test loss': test_loss,
        'cos similarity': cosine_similarity,
        'features': features_str,
        'model': model_str,
        'epochs': epochs
    }
    df = df.append(data, ignore_index=True)
    df.index.name = 'id'
    df.to_csv(file_name)
