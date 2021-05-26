import os

import numpy as np
import pandas as pd
import pandas_ta as ta
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


def add_indicadores_tecnicos(df):
    macd = ta.macd(close=df['price'], fast=50, slow=200, signal=21, min_periods=None, append=True)
    df = pd.concat([df, macd], axis=1)

    df = df.fillna(0)

    bbands = ta.bbands(df['price'], fillna=0)
    df['BB_Middle_Band'] = bbands['BBM_5_2.0']
    df['BB_Upper_Band'] = bbands['BBU_5_2.0']
    df['BB_Lower_Band'] = bbands['BBL_5_2.0']
    return df


def eventTimeSeries_to_many(src_df: pd.DataFrame, N: int) -> pd.DataFrame:
    r""" Tranforma uma série temporal de eventos e várias outras séries temporais de eventos que sinalizam a chegada de um evento.
    2(N + 1) séries temporais serão criadas, onde N+1 indicarão a pontuação de um evento e outras N+1 indicação a confiabilidade nesse evento.

    Cada série temporal representa quantos dias faltam para um evento chegar. Por exemplo:
    para 
    - `N=3`
    - `Evento A - anunciado dia 0 - acontece dia 6 - pontuação 2 - confiabilidade 0.95`
    - `Evento B - anunciado dia 5 - acontece dia 8 - pontuação 3 - confiabilidade 0.98`

        as seguintes series temporais serão criadas:

    >>> anuncio_de_evento_score = [2,0,0,0,0,3,0,0,0,0,0]
    >>> evento_falta_3_dia_score= [0,0,0,2,0,3,0,0,0,0,0]
    >>> evento_falta_2_dia_score= [0,0,0,0,2,0,3,0,0,0,0]
    >>> evento_falta_1_dia_score= [0,0,0,0,0,2,0,3,0,0,0]

    >>> anuncio_de_evento_conf = [.95,0,0,0,0,.98,0,0,0,0,0]
    >>> evento_falta_3_dia_conf= [0,0,0,.95,0,.98,0,0,0,0,0]
    >>> evento_falta_2_dia_conf= [0,0,0,0,.95,0,.98,0,0,0,0]
    >>> evento_falta_1_dia_conf= [0,0,0,0,0,.95,0,.98,0,0,0]

    quando eventos se interseccionam, suas pontuações e confiabilidades são somadas|subtraidas do historico
    Args:
        src_df: dataframe fonte
        N: quantos dias (series temporais extras) avisar com antedencedencia da chegada de um evento
    """
    totalTimeSteps = len(src_df.index)
    votes_in_N_days = np.zeros(shape=(N, totalTimeSteps), dtype=int)
    confi_in_N_days = np.zeros(shape=(N, totalTimeSteps), dtype=float)

    # votes_anouncement = np.zeros(shape=(1, tipeSteps), dtype=int)
    # confi_anouncement = np.zeros(shape=(1, tipeSteps), dtype=float)

    timeStep = 0
    for idx, row in src_df.iterrows():
        days_to_happen, votes, confidence = row['days_to_event_happen'], row['event_votes'], row['event_confidence']
        if pd.isnull(votes):
            timeStep += 1
            continue

        diff = days_to_happen - N
        i = timeStep + max(0, diff)  # avança pra futuro, onde esta perto do evento acontecer
        if i >= totalTimeSteps:  # verifica se o evento ta dentro do escopo da série temporal
            timeStep += 1
            continue
        # votes_anouncement[i] = votes
        # confi_anouncement[i] = confidence
        for timeSerie in range(N, 0, -1):
            if timeSerie > days_to_happen:
                continue
            old_v = votes_in_N_days[timeSerie-1, i]
            old_c = confi_in_N_days[timeSerie-1, i]

            # se n ta intuitivo, faz uma regra de 3 dupla(?) no papel
            # score = (v1*c1) + (v2*c2)
            # total_votes = v1 + v2
            # merged_confidence = score / total_votes
            x = (old_v*old_c) + (votes*confidence)
            confidence = x / (old_v + votes)

            votes_in_N_days[timeSerie-1, i] += votes
            confi_in_N_days[timeSerie-1, i] = confidence
            i += 1

        # terminou o "avanço pro futuro". agora volta de onde parou e vai pro proximo dia
        timeStep += 1

    df = src_df.copy()
    for d in range(N):
        df[f'event_in_{d+1}_days_votes'] = votes_in_N_days[d]
        df[f'event_in_{d+1}_days_confidence'] = confi_in_N_days[d]
    # cria apenas as colunas de "n dias até o evento", pois as de anuncio já existem

    return df


def load_data(crypto: str, topic: str, event_days_left_lookback: int = 5) -> pd.DataFrame:
    """Carrega os dados.

    Parametros
        ----------
        crypto: str
            abreviação/sigla da cripto moeda. Ex: btc, eth, ada
        topic: str
            topico a pesquisar no google trends e no coinmarketCalendar. Ex: bitcoin, ethereum, cardano
        event_days_left_lookback: int
            avisar chegada de evento com quantos dias de antecedencia
    """
    file_name = 'bitinfochart_data-%s.csv' % crypto.upper()
    df = None
    if not os.path.exists(file_name):
        scrapper = BitinfochartsWebScrapper()
        # g_trends = GoogleTrends()
        coinMarketCal = CoinmarketcalWebScrapper()

        df1 = scrapper.get_tweet_volume_data(crypto)
        df1['tweet_volume'] = fill_missing_data(df1.loc[:, 'tweet_volume'].to_numpy())

        df2 = scrapper.get_googletrend_data(crypto)
        # start_date = max(df1.index.date[0], date.today() - relativedelta(years=6)
        #                  )  # limita dados para no máximo ultimos 6 anos
        # df2 = g_trends.get_daily_trend(topic, start_d=start_date, end_d=date.today(), verbose=True)
        # df2 = df2.drop(columns=['overlap'])

        df3 = scrapper.get_price_data(crypto)

        df4 = coinMarketCal.get_all_events(crypto=topic)
        df4 = eventsToTimeSerie(df4)
        df4.rename(columns={'days_to_happen': 'days_to_event_happen', 'title': 'event_title',
                            'votes': 'event_votes', 'confidence': 'event_confidence'}, inplace=True)

        df = pd.concat([df1, df2, df3], axis=1, join='inner').astype(
            {'tweet_volume': float, 'trend': float, 'event_votes': int, 'days_to_event_happen': int})
        df = pd.merge(left=df, right=df4, on='date', how='left')

        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name, dtype={'tweet_volume': float, 'trend': float, 'event_votes': pd.Int64Dtype(), 'days_to_event_happen': pd.Int64Dtype()},
                         parse_dates=['date']).set_index('date')

        df['price (%)'] = price_to_percentage(df.loc[:, 'price'].to_numpy())

        df = eventTimeSeries_to_many(df, N=event_days_left_lookback)

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
