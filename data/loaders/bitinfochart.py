import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


class bitinfochartsWebScrapper(object):
    def __parse_strlist(self, sl):
        clean = re.sub("[\[\],\s]", "", sl)
        splitted = re.split("[\'\"]", clean)
        values_only = [s for s in splitted if s != '']
        return values_only

    def get_tweet_volume_data(self, crypto='btc') -> pd.DataFrame:
        dataList = self.__get_data('tweets', crypto)
        date, tweet = [], []
        for each in dataList:
            if (dataList.index(each) % 2) == 0:
                date.append(each.replace('/', '-'))
            else:
                tweet.append(int(each) if each != 'null' else None)     # se falta o dado, usa o None

        df = pd.DataFrame(list(zip(date, tweet)), columns=["date", "tweet_volume"])
        df['date'] = df['date'].astype('datetime64[ns]')
        return df.set_index('date')

    def get_price_data(self, crypto='btc') -> pd.DataFrame:
        dataList = self.__get_data('price', crypto)
        date, price = [], []
        for each in dataList:
            if (dataList.index(each) % 2) == 0:
                date.append(each.replace('/', '-'))
            else:
                price.append(float(each) if each != 'null' else None)  # se falta o dado do preço, usa None

        df = pd.DataFrame(list(zip(date, price)), columns=["date", "price"])
        df['date'] = df['date'].astype('datetime64[ns]')
        return df.set_index('date')

    def __get_data(self, data_type, crypto):
        crypto = crypto.lower()
        url = 'https://bitinfocharts.com/comparison/%s-%s.html' % (data_type, crypto)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'd = new Dygraph(document.getElementById("container")' in script.string:
                StrList = script.string
                StrList = '[[' + StrList.split('[[')[-1]
                StrList = StrList.split(']]')[0] + ']]'
                StrList = StrList.replace("new Date(", '').replace(')', '')
                dataList = self.__parse_strlist(StrList)

        return dataList
