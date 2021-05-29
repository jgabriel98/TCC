import pandas as pd

import os
import urllib.request
import json
import datetime as dt

def load_data_from_aplhavantage(crypto='BTC') -> pd.DataFrame:
    """Carrega os dados da API Alphavantage e salva em um csv
    Parametros
    ----------
    crypto : str, default 'BTC' (Bitcoin)

    Retorna
    -------
    DataFrame: Pandas DataFrame
    """
    # ====================== Loading Data from Alpha Vantage ==================================
    api_key = 'C9JVX7BNNZSZQ7UO'
    fiat_currency = 'BRL'

    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=%s&market=%s&outputsize=full&apikey=%s" % (
        crypto, fiat_currency, api_key)
    # Save data to this file
    file_to_save = 'timeseries_data-%s.csv' % crypto

    # If you haven't already saved data,
    # Go ahead and grab the data from the url
    # And store date, low, high, volume, close, open values to a Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Digital Currency Daily)']
            df = pd.DataFrame(
                columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                sufix = ' (%s)' % fiat_currency
                data_row = [date.date(), float(v['3a. low'+sufix]), float(v['2a. high'+sufix]),
                            float(v['4a. close'+sufix]), float(v['1a. open'+sufix])]
                df.loc[-1, :] = data_row
                df.index = df.index + 1
        print('Data saved to : %s' % file_to_save)
        df.to_csv(file_to_save)
        return df

    # If the data is already there, just load it from the CSV
    else:
        print('File already exists. Loading data from CSV')
        return pd.read_csv(file_to_save)