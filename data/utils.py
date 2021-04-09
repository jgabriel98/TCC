import os

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from data.loaders.bitinfochart import bitinfochartsWebScrapper
from data.loaders.googleTrends import GoogleTrends

def price_to_percentage(data: np.ndarray) -> np.ndarray:
    new_data = np.empty(len(data))
    new_data[0] = .0
    for i in range(1,len(data)):
        diff = data[i] - data[i-1]
        new_data[i] = diff / data[i-1]
    
    return new_data


def fill_missing_data(data: np.ndarray) -> np.ndarray:
    end = len(data)

    #se o ultimo valor estiver faltando, nao tem como fazer estimativa desse trecho
    if data[-1] == None:
        while data[end-1] == None:
            end-=1
    
    i = 1
    while i < end:
        if np.isnan(data[i]) == False:
            i+=1
            continue

        j=i            
        while np.isnan(data[j]):
            j+=1
        
        first_value = data[i-1]
        last_value = data[j]
        step = (last_value-first_value)/(1 + j-i)
        while i<j:
            data[i] = data[i-1]+step
            i+=1
        
        i+=1
    return data
    

def load_data(crypto: str, topic: str) -> pd.DataFrame:
    """Carrega os dados.

    Parametros
        ----------
        crypto: str
            abreviação/sigla da cripto moeda. Ex: btc, eth, ada
        topic: str
            topico a pesquisar no google trends. Ex: bitcoin, ethereum, cardano
    """
    file_name = 'bitinfochart_data-%s.csv' % crypto.upper()
    df = None
    if not os.path.exists(file_name):
        scrapper = bitinfochartsWebScrapper()
        g_trends = GoogleTrends()
        df1 = scrapper.get_tweet_volume_data(crypto)
        df1['tweet_volume'] = fill_missing_data(df1.loc[:, 'tweet_volume'].to_numpy())
        
        start_date = max(df1.index.date[0], date.today() - relativedelta(years=5) )  #limita dados para no máximo ultimos 5 anos
        df2 = g_trends.get_daily_trend(topic, start_d=start_date, end_d = date.today(), verbose= True)
        df2 = df2.drop(columns=['overlap'])

        df3 = scrapper.get_price_data(crypto)
        df3['variation (%)'] = price_to_percentage(df3.loc[:, 'price'].to_numpy())
        
        df = pd.concat([df1, df2, df3], axis=1, join='inner').astype({'tweet_volume': float, 'trend': float})
        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name, dtype={'tweet_volume': float, 'trend': float}, parse_dates=['date']).set_index('date')
    return df


def split_data(data: list, ratio=0.80):
    pivot = int(len(data)*ratio)
    return data[:pivot], data[pivot:]