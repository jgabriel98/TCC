from typing import List

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import pandas as pd
from datetime import date, timedelta

from .EventCardElement import EventCardElement


def eventsToTimeSerie(df: pd.DataFrame, remove_zero_day_lenght_events=False, on_same_added_day='keep_biggest') -> pd.DataFrame:
    df = df.sort_values('added_date')
    timeSerie = []

    current_date = df.iloc[0]['added_date']

    last_row = None
    for i, row in df.iterrows():
        added_date = row['added_date']
        while current_date < added_date:
            timeSerie.append([current_date, None, None, None, None])
            current_date += timedelta(days=1)

        days_to_happen = (row['event_date'] - row['added_date']).days
        if days_to_happen < 0:  # remove eventos "atrasados", isto é, que foram descobertos depois de acontecer
            continue
        if remove_zero_day_lenght_events and days_to_happen == 0:
            continue
         # change 'event_date' column value to 'days_to_happen' = 'event_date - added_date'
        row['event_date'] = days_to_happen

        if (last_row is not None) and (last_row['added_date'] == row['added_date']):
            if on_same_added_day == 'keep_biggest':
                biggest = last_row if last_row['votes'] >= row['votes'] else row
                timeSerie[-1] = list(biggest)
            continue
        else:
            timeSerie.append(list(row))
        current_date += timedelta(days=1)

        last_row = row

    timeSerie = pd.DataFrame(timeSerie, columns=['date', 'days_to_happen', 'title', 'votes', 'confidence'], )
    timeSerie = timeSerie.astype(dtype={'date': 'datetime64[ns]',
                                        'days_to_happen': 'Int64',
                                        'votes': 'Int64'})
    # timeSerie['date'] = timeSerie['date'].astype('datetime64[ns]')
    return timeSerie.set_index('date')


class CoinmarketcalWebScrapper(object):

    def __init__(self, webdriver_path='G:\\Storage\\Repos\\TCC\\data\\loaders\\coinmarketcalWebScrapper\\chromedriver.exe'):
        self.driver = webdriver.Chrome(executable_path=webdriver_path)
        self.wait = WebDriverWait(self.driver, 3)

    def __del__(self):
        self.close()

    def close(self):
        self.driver.quit()

    def _extract_data_from_eventCards(self, data, eventCard_list_element) -> dict:
        for evento in eventCard_list_element:
            evento = EventCardElement(evento)
            data['added_date'].append(evento.added_date)
            data['event_date'].append(evento.event_date)
            data['title'].append(evento.title)
            data['votes'].append(evento.votes)
            data['confidence'].append(evento.confidence_percentage)

        return data

    def get_all_events(self, crypto='bitcoin') -> pd.DataFrame:
        past = self.get_past_events(crypto)
        future = self.get_upcoming_events(crypto)
        return pd.concat([future, past])

    def get_past_events(self, crypto='bitcoin') -> pd.DataFrame:
        url = 'https://coinmarketcal.com/en/coin/%s' % crypto
        self.driver.get(url)
        self.__safeClick((By.CSS_SELECTOR, "a[href=\#past]"))
        self.__show_all_eventCards()

        eventos_elem = self.driver.find_elements_by_css_selector("#past .row.list-card article")
        dados = {'added_date': [], 'event_date': [], 'title': [], 'votes': [], 'confidence': []}
        self._extract_data_from_eventCards(dados, eventos_elem)

        df = pd.DataFrame(dados, columns=['added_date', 'event_date', 'title', 'votes', 'confidence'])
        df['added_date'].astype('datetime64[ns]')
        df['event_date'].astype('datetime64[ns]')

        return df

    def get_upcoming_events(self, crypto='bitcoin') -> pd.DataFrame:
        url = 'https://coinmarketcal.com/en/coin/%s' % crypto
        self.driver.get(url)
        self.__safeClick((By.CSS_SELECTOR, "a[href=\#upcoming]"))
        self.__show_all_eventCards()

        eventos_elem = self.driver.find_elements_by_css_selector("#upcoming .row.list-card article")
        dados = {'added_date': [], 'event_date': [], 'title': [], 'votes': [], 'confidence': []}
        self._extract_data_from_eventCards(dados, eventos_elem)

        df = pd.DataFrame(dados, columns=['added_date', 'event_date', 'title', 'votes', 'confidence'])
        df['added_date'].astype('datetime64[ns]')
        df['event_date'].astype('datetime64[ns]')

        return df

    def __show_all_eventCards(self):
        while True:
            try:
                elem = self.wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, ".active a.load-more:not(.d-none a):not([disabled=disabled])")))
                self.driver.execute_script("arguments[0].scrollIntoView(true);", elem)  # rola até o elemento
                elem.click()  # clica nele pra carregar mais
            except TimeoutException:
                break

        # scrool down, forcing the page to load all data
        page = self.driver.find_element_by_tag_name("body")
        page.send_keys(Keys.CONTROL, Keys.END)

    def __safeClick(self, locator):
        try:
            self.wait.until(EC.element_to_be_clickable(locator)).click()
        except TimeoutException as e:
            self.driver.quit()
            raise e
