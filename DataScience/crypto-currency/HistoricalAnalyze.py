import numpy as np
import pandas as pd
import pickle
import quandl
import time
from datetime import datetime
from Util import Util

import plotly.offline as py
import plotly.graph_objs as go

class HistoricalAnalyze:
    base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
    start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')  # get data from the start of 2015
    end_date = datetime.now()  # up until today
    pediod = 86400  # pull daily data (86,400 seconds per day)

    def get_quandl_data(self, quandl_id):
        '''Download and cache Quandl dataseries'''
        cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
        try:
            f = open(cache_path, 'rb')
            df = pickle.load(f)
            print('Loaded {} from cache'.format(quandl_id))
        except (OSError, IOError) as e:
            print('Download {} from Quandl'.format(quandl_id))

        quandl.ApiConfig.api_key = 'URHqFyehs55fjNKGE8Ga'
        df = quandl.get(quandl_id, returns='pandas')
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
        return df

    # print(btc_usd_price_kraken.head())

    # btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
    # py.plot([btc_trace])
    def get_exchange_data(self):
        exchanges=['COINBASE','BITSTAMP','ITBIT', 'KRAKEN']
        exchange_data={}
        for exchange in exchanges:
            exchange_code='BCHARTS/{}USD'.format(exchange)
            btc_exchange_df=self.get_quandl_data(exchange_code)
            exchange_data[exchange]=btc_exchange_df
        return exchange_data

    def merge_dfs_on_column(self, dataframes, labels, col):
        '''Merge a single column of each dataframe into a new combined dataframe'''
        series_dict={}
        for index in range(len(dataframes)):
            series_dict[labels[index]] = dataframes[index][col]
        datasets = pd.DataFrame(series_dict)
        datasets.replace(0, np.nan, inplace=True)
        datasets['avg_btc_price_usd'] = datasets.mean(axis=1)
        return datasets

    # btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')

    # print(btc_usd_datasets.tail())

    # btc_usd_datasets.replace(0, np.nan, inplace=True)
    # Util.df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

    # btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
    # btc_trace = go.Scatter(x=btc_usd_datasets.index, y=btc_usd_datasets['avg_btc_price_usd'])
    # py.plot([btc_trace])
    # print(start_date.time())
    # print(end_date.time())

    def get_crypto_data(self, poloniex_pair):
        '''Retrieve cryptocurrency data from poloniex'''
        json_url=HistoricalAnalyze.base_polo_url.format(
            poloniex_pair,
            time.mktime(HistoricalAnalyze.start_date.timetuple()),
            time.mktime(HistoricalAnalyze.end_date.timetuple()),
            HistoricalAnalyze.pediod
        )
        data_df=Util.get_json_data(json_url,poloniex_pair)
        data_df=data_df.set_index('date')
        return data_df

    def correlation_heatmap(self, df, title, absolute_bounds=True):
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

    def run_example(self):
        exchange_data = self.get_exchange_data()
        btc_usd_datasets = self.merge_dfs_on_column(
            list(exchange_data.values()),
            list(exchange_data.keys()),
            'Weighted Price')

        altcoins = ['ETH', 'LTC', 'XRP', 'ETC', 'STR', 'DASH']
        altcoin_data = {}
        for altcoin in altcoins:
            coinpair = 'BTC_{}'.format(altcoin)
            crypto_price_df = self.get_crypto_data(coinpair)
            altcoin_data[altcoin] = crypto_price_df

        # print(altcoin_data['ETH'].tail())
        for altcoin in altcoin_data.keys():
            altcoin_data[altcoin]['price_usd'] = altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets[
                'avg_btc_price_usd']

        combined_df = self.merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')
        # add BTC price to the dataframe combined_df
        combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']

        # Chart all of the altocoin prices
        Util.df_scatter(combined_df, 'Cryptocurrency Price (USD)', seperate_y_axis=False, y_axis_label='Coin Value(USD)',
                   scale='log')

        # combined_df_2016 = combined_df[combined_df.index.year == 2016]
        # print(combined_df_2016.pct_change().corr(method='pearson'))