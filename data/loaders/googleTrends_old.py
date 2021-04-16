from pytrends.request import TrendReq
import pytrends
import pandas as pd
from copy import copy
from datetime import date
from dateutil.relativedelta import relativedelta


def get_topic_encode(search_term):
    topic_encode = None
    for item in TrendReq().suggestions(search_term):
        if item['title'].lower() == search_term.lower():
            topic_encode = item['mid']
            print('found topic encode for %s: %s - type: %s'%(search_term,topic_encode, item['type']))
            break
    return topic_encode


def __unormalize(original_min, original_max, data):
    x,y = data.min(), data.max()
    a,b = original_min, original_max
    for i in range(len(data)):
        # normalizando eh assim: data[i] = (data[i]-min_val)/(max_val-min_val), então desnormalizar seria:
        #data[i] = original_min + (data[i] * (original_max+original_min))
        data[i] = ((data[i]-x) / (y-x)) * (b-a) + a


def fix_local_normalized_values(normalized_df, macro_df, macro_itr_s, macro_itr_e):
    # encontra o intervalo na serie temporal macro (que possui o intervalo completo)
    while(normalized_df.index[0] >= macro_df.index[macro_itr_s]):
        macro_itr_s += 1
    macro_itr_s = max(macro_itr_s-1, 0)

    while(normalized_df.index[-1] >= macro_df.index[macro_itr_e]):
        macro_itr_e += 1
        if macro_itr_e == len(macro_df.index):  #se passou do ultimo, para
            break
    macro_itr_e -= 1

    matching_range = macro_df.iloc[macro_itr_s: macro_itr_e, 0]
    # agora podemos desnomalizar os valores, de um intervalo incompleto (3 meses) para o completo
    min_orig, max_orig = matching_range.min(), matching_range.max()
    __unormalize(min_orig, max_orig, normalized_df.iloc[:,0])
     

    macro_itr_s = macro_itr_e
    return normalized_df, macro_itr_s, macro_itr_e


def get_historical_trend_values(search_term, start_time=date(2014, 1, 1), end_time=date.today()):
    # tenta usar o tópico ao invés de uma pesquisa genérica
    topic = get_topic_encode(search_term)
    search_term = topic if topic != None else search_term

    t = TrendReq()
    df = pd.DataFrame()

    t.build_payload([search_term], timeframe='%s %s' % (start_time, end_time))
    # dataframe normalizado no intervalo completo/desejado, porém com granunalidade
    # maior que a desejado, isto é, meses ou semanas ao invés de dias
    macro_df = t.interest_over_time()
    macro_itr_s, macro_itr_e = 0, 0

    # faz varias requisições de 3 meses, pois é o maior timestamp q o google responde com granunalidade de dias
    current_start, current_end = start_time, start_time + relativedelta(months=3)
    while(current_end < end_time):
        s, e = current_start, current_end
        start_s = '%d-%02d-%02d' % (s.year, s.month, s.day)
        end_s = '%d-%02d-%02d' % (e.year, e.month, e.day)

        t.build_payload([search_term], timeframe='%s %s' % (start_s, end_s), geo='', gprop='')
        # dataframe com granunalidade desejada, porém está normalizado num
        # intervalo menor (3 meses), ao invés do intervalo completo.
        normalized_df = t.interest_over_time().astype({search_term: float})
        # converte os valores, normalizados num intervalo menor, para de um intervalo completo
        unormalized_df, macro_itr_s, macro_itr_e = fix_local_normalized_values(normalized_df, macro_df, macro_itr_s, macro_itr_e)
        df = pd.concat([df, unormalized_df])

        current_start = current_end + relativedelta(days=1)
        current_end += relativedelta(months=3)

    # obtém o resto que faltou
    s, e = current_start, end_time
    start_s = '%d-%02d-%02d' % (s.year, s.month, s.day)
    end_s = '%d-%02d-%02d' % (e.year, e.month, e.day)
    t.build_payload([search_term], timeframe='%s %s' % (start_s, end_s), geo='', gprop='')
    unormalized_df = fix_local_normalized_values(t.interest_over_time(),macro_df, macro_itr_s, macro_itr_e)[0]
    df = pd.concat([df, unormalized_df])

    return df.drop('isPartial', axis=1).rename(columns={search_term: 'trend'})
