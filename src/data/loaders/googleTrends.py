from datetime import datetime, timedelta, date, time
import pandas as pd
import time

from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError


class GoogleTrends:
    """Google Trends data fetcher and manipulator.

    Google Trend API lowers the data resolution for big timestamps:
      hourly when up to 7 days
      daily when up to 90 days
      weekly when up to 5 years
      monthly above 5 years
    And also normalize the values from 0-100 on the timestamp you requested.

    So to fix that, this class makes multiples request (one low resolution with the full timestamp, and many with high resolution with tiny timestamps),
    and scale them, with the overlapping method like specified in https://github.com/e-271/pytrends/blob/master/pytrends/renormalize.py"""

    def __init__(self):
        self.trendreq = TrendReq()

    def get_daily_trend(self, search_term: str, start_d: date, end_d: date, cat=0,
                        geo='', gprop='', delta=269, overlap=100, sleep=0,
                        tz=0, verbose=False) -> pd.DataFrame:
        """Stich and scale consecutive daily trends data between start and end date.
        This function will first download pieces of data and then scale each piece 
        using the overlapped period. 

            Parameters
            ----------
            search_term: str
                the term to search, support only a single keyword, without bracket
            start_d: date
                starting date in string format:YYYY-MM-DD (e.g.2017-02-19)
            end_d: date
                ending date in string format:YYYY-MM-DD (e.g.2017-02-19)
            cat, geo, gprop, sleep: 
                same as defined in pytrends
            delta: int
                The length(days) of each timeframe fragment for fetching google trends data, 
                need to be <269 in order to obtain daily data.
            overlap: int
                The length(days) of the overlap period used for scaling/normalization
            tz: int
                The timezone shift in minute relative to the UTC+0 (google trends default).
                For example, correcting for UTC+8 is 8, and UTC-6 is -6

        """
        topic = self._find_topic_encode(search_term, verbose)
        search_term = topic if topic != None else search_term

        init_end_d = datetime(end_d.year, end_d.month, end_d.day, 23, 59, 59)
        delta = timedelta(days=delta)
        overlap = timedelta(days=overlap)

        itr_d = end_d - delta
        overlap_start = None

        df = pd.DataFrame()
        ol = pd.DataFrame()

        while end_d > start_d:
            tf = itr_d.strftime('%Y-%m-%d')+' '+end_d.strftime('%Y-%m-%d')
            if verbose:
                print('Fetching \''+search_term+'\' for period:'+tf)
            temp = self._fetch_data([search_term], timeframe=tf, cat=cat, geo=geo, gprop=gprop).astype({search_term: float})
            temp.columns.values[0] = tf
            ol_temp = temp.copy()
            ol_temp.iloc[:, :] = None
            if overlap_start is not None:  # not first iteration
                if verbose:
                    print('Normalize by overlapping period:'+overlap_start.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'))
                # normalize using the maximum value of the overlapped period
                y1 = temp.loc[overlap_start:end_d].iloc[:, 0].values.max()
                y2 = df.loc[overlap_start:end_d].iloc[:, -1].values.max()
                coef = y2/y1
                temp = temp * coef
                ol_temp.loc[overlap_start:end_d, :] = 1

            df = pd.concat([df, temp], axis=1)
            ol = pd.concat([ol, ol_temp], axis=1)
            # shift the timeframe for next iteration
            overlap_start = itr_d
            end_d -= (delta-overlap)
            itr_d -= (delta-overlap)
            # in case of short query interval getting banned by server
            time.sleep(sleep)

        df.sort_index(inplace=True)
        ol.sort_index(inplace=True)

        # If the daily trend data is missing the most recent 3-days data, need to complete with hourly data
        if df.index.max() < init_end_d:
            tf = 'now 7-d'
            hourly = self._fetch_data([search_term], timeframe=tf, cat=cat, geo=geo, gprop=gprop).astype({search_term: float})

            # convert hourly data to daily data
            daily = hourly.groupby(hourly.index.date).sum()

            # check whether the first day data is complete (i.e. has 24 hours)
            daily['hours'] = hourly.groupby(hourly.index.date).count()
            if daily.iloc[0].loc['hours'] != 24:
                daily.drop(daily.index[0], inplace=True)
            daily.drop(columns='hours', inplace=True)

            daily.set_index(pd.DatetimeIndex(daily.index), inplace=True)
            daily.columns = [tf]

            ol_temp = daily.copy()
            ol_temp.iloc[:, :] = None
            # find the overlapping date
            intersect = df.index.intersection(daily.index)
            if verbose:
                print('Normalize by overlapping period:'+(intersect.min().strftime('%Y-%m-%d')) +
                      ' '+(intersect.max().strftime('%Y-%m-%d')))
            # scaling use the overlapped today-4 to today-7 data
            coef = df.loc[intersect].iloc[:, 0].max() / daily.loc[intersect].iloc[:, 0].max()
            daily = daily*coef
            ol_temp.loc[intersect, :] = 1

            df = pd.concat([daily, df], axis=1)
            ol = pd.concat([ol_temp, ol], axis=1)

        # taking averages for overlapped period
        df = df.mean(axis=1)
        ol = ol.max(axis=1)
        # merge the two dataframe (trend data and overlap flag)
        df = pd.concat([df, ol], axis=1)
        df.columns = [search_term, 'overlap']
        # Correct the timezone difference
        df.index = df.index + timedelta(minutes=tz*60)
        df = df[start_d:init_end_d]
        # re-normalized to the overall maximum value to have max =100
        #df[search_term] = 100*df[search_term]/df[search_term].max()

        df.index.name = 'date'
        return df.rename(columns={search_term: 'trend'})

    def _find_topic_encode(self, search_term, verbose=False):
        topic_encode = None
        for item in self.trendreq.suggestions(search_term):
            if item['title'].lower() == search_term.lower():
                topic_encode = item['mid']
                if verbose:
                    print('found topic encode for %s: %s - type: %s' % (search_term, topic_encode, item['type']))
                break
        return topic_encode

    def _fetch_data(self, kw_list, timeframe, cat=0, geo='', gprop='') -> pd.DataFrame:
        """Download google trends data. In case of failure, retries (maximum of 3 times)."""
        attempts = 0
        while True:
            try:
                self.trendreq.build_payload(kw_list=kw_list, timeframe=timeframe, cat=cat, geo=geo, gprop=gprop)
            except ResponseError as err:
                print(err)
                print(f'Trying again in {60 + 5 * attempts} seconds.')
                sleep(60 + 5 * attempts)
                attempts += 1
                if attempts > 3:
                    print('Failed after 3 attemps, abort fetching.')
                    break
            else:
                break
        df = self.trendreq.interest_over_time()
        df.drop(columns=['isPartial'], inplace=True)
        return df
