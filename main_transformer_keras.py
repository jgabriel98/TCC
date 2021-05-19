from data.utils import load_data
from data.scaling import normalize

coin = 'BTC'
df = load_data(coin, 'bitcoin').iloc[:, :]
prices = normalize(df.loc[:, 'price'].to_numpy())[0]
variation = df.loc[:, 'variation (%)'].to_numpy()
tweet = normalize(df.loc[:, 'tweet_volume'].to_numpy())[0]
google_trends = normalize(df.loc[:, 'trend'].to_numpy())[0]
event_day_count = df.loc[:, 'days_to_event_happen'].fillna(0).to_numpy()
event_votes = normalize(df.loc[:, 'event_votes'].fillna(0).to_numpy())[0]
event_confidence = normalize(df.loc[:, 'event_confidence'].fillna(0).to_numpy())[0]

