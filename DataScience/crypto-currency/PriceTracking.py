import os
import numpy as np
import pandas as pd
import pickle
import quandl
import time
from datetime import datetime

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Download {} from Quandl'.format(quandl_id))

    df = quandl.get(quandl_id, returns='pandas')
    df.to_pickle(cache_path)
    print('Cached {} at {}'.format(quandl_id, cache_path))
    return df

btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

# print(btc_usd_price_kraken.head())

# btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
# py.plot([btc_trace])

exchanges=['COINBASE','BITSTAMP','ITBIT']
exchange_data={}
exchange_data['KRAKEN']=btc_usd_price_kraken
for exchange in exchanges:
    exchange_code='BCHARTS/{}USD'.format(exchange)
    btc_exchange_df=get_quandl_data(exchange_code)
    exchange_data[exchange]=btc_exchange_df

def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict={}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
    return pd.DataFrame(series_dict)

btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')

# print(btc_usd_datasets.tail())

def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scaatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col:df[col],label_arr))
    layout = go.Layout(title=title, legend=dict(orientation='h'), xaxis=dict(type='date'), yaxis=dict(title=y_axis_label, showticklabels=not seperate_y_axis, type=scale))
    y_axis_config = dict(overlaying='y', showticklabels=False, type=scale)
    visibility = 'visible'
    if initial_hide: visibility = 'legendonly' #From trace for each serise
    trace_arr=[]
    for index, series in enumerate(series_arr):
        trace = go.Scatter(x=series.index, y=series, name=label_arr[index],visible=visibility)
        #Add seperate axis for the series
        if seperate_y_axis: trace['yaxis']='y{}'.format(index+1)
        layout['yaxis{}'.format(index+1)] = y_axis_config
        trace_arr.append(trace)
    fig=go.Figure(data=trace_arr, layout=layout)
    py.plot(fig)

btc_usd_datasets.replace(0, np.nan, inplace=True)
# df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
# btc_trace = go.Scatter(x=btc_usd_datasets.index, y=btc_usd_datasets['avg_btc_price_usd'])
# py.plot([btc_trace])

def get_json_data(json_url, cache_path):
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    except(OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
    df.to_pickle(cache_path)
    print('Cache {} at {}'.format(json_url, cache_path))

    return df

base_polo_url='https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date=datetime.strptime('2015-01-01','%Y-%m-%d')# get data from the start of 2015
end_date=datetime.now()# up until today
pediod=86400# pull daily data (86,400 seconds per day)


base_coin_market_cap_url = 'https://api.coinmarketcap.com/v1/ticker/{}/?convert=USD'
# print(start_date.time())
# print(end_date.time())


def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url=base_polo_url.format(poloniex_pair,time.mktime(start_date.timetuple()),time.mktime(end_date.timetuple()),pediod)
    data_df=get_json_data(json_url,poloniex_pair)
    data_df=data_df.set_index('date')
    return data_df

altcoins=['ETH','LTC','XRP','ETC','STR','DASH']
altcoin_data={}
for altcoin in altcoins:
    coinpair='BTC_{}'.format(altcoin)
    crypto_price_df=get_crypto_data(coinpair)
    altcoin_data[altcoin]=crypto_price_df

# print(altcoin_data['ETH'].tail())
for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] = altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']

combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')
#add BTC price to the dataframe combined_df
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']

#Chart all of the altocoin prices
df_scatter(combined_df, 'Cryptocurrency Price (USD)', seperate_y_axis=False, y_axis_label='Coin Value(USD)', scale='log')

combined_df_2016 = combined_df[combined_df.index.year == 2016]
# print(combined_df_2016.pct_change().corr(method='pearson'))

def correlation_heatmap(df, title, absolute_bounds=True):
    '''Plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(z=df.corr(method='pearson').as_matrix(),
                         x=df.columns,
                         y=df.columns,
                         colorbar=dict(title='Pearson Coefficient'),)
    layout = go.Layout(title=title)
    if absolute_bounds: heatmap['zmax'] = 1.0
    heatmap['zmin'] = -1.0
    fig = go.Figure(data=[heatmap], layout=layout)
    py.plot(fig)

# correlation_heatmap(combined_df_2016.pct_change(), "Cryptocurrency Correlations in 2016")
