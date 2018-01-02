import pandas as pd
from datetime import datetime
import urllib3
import simplejson
from Util import Util
import numpy as np
import sys
import json
from multiprocessing.pool import ThreadPool
from binance.client import Client
import matplotlib.pyplot as plt


class CurrentAnalyze:
    api_key = '9Z3gBAmhGPH5lwrucUKXGcJUj6ofeOwwrhmh8fC3LHWBvEbZ8kGynxwb04d2HIrp'
    api_secret = 'bgSXHLtr95K1erKF8DGeUN69q0JfnjWr4jwyzcHDV1PICPkJgjm07SSsFVtfeWSC'
    client = Client(api_key, api_secret)

    ids = ['bitcoin', 'ethereum', 'bitcoin-cash', 'ripple', 'litecoin', 'cardano', 'iota', 'dash', 'nem', 'monero', 'bitcoin-gold', 'eos', 'stellar', 'neo',
           'verge', 'walton']

    symbols = ['XVG', 'XRP', 'WTC', 'IOTA']
    symbol_id_map = {'bitcoin': 'BTC',
                     'ethereum': 'ETH'}

    base_coin_market_cap_url = 'https://api.coinmarketcap.com/v1/ticker/{}/?convert=USD'

    base_binance_url = 'https://api.binance.com'
    candle_stick = '/api/v1/klines?symbol={}&interval={}'
    latest_price = '/api/v3/ticker/price?symbol={}'
    one_day_ticker = '/api/v1/ticker/24hr'

    def get_json_data(self, id):
        url = CurrentAnalyze.base_coin_market_cap_url.format(id)
        return pd.read_json(url)

    def get_latest_price(self, target_currency, base_currency):
        url = CurrentAnalyze.base_coin_market_cap_url.format(base_currency)
        response = urllib3.urlopen(url)
        jsonObj = simplejson.load(response)
        currency_to_usd = float(jsonObj[0]["price_usd"])

        target_url = (CurrentAnalyze.base_binance_url + CurrentAnalyze.latest_price).format(target_currency + CurrentAnalyze.symbol_id_map[base_currency])
        target_obj = simplejson.load(urllib3.urlopen(target_url))
        price_usd = currency_to_usd * float(target_obj['price'])
        target_obj['price_usd'] = price_usd
        target_obj['symbol'] = target_currency
        target_obj['base_currency'] = CurrentAnalyze.symbol_id_map[base_currency]
        return target_obj

    def get_latest_price_df(self, symbol_array, base_currency):
        result = []
        for symbol in symbol_array:
            result.append(self.get_latest_price(symbol, base_currency))
        df = pd.DataFrame(result)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        df.dropna()
        return df

    # interval is 3m, 5m, 1h
    def get_candle_stick(self, symbol, base_currency, interval):
        url = (CurrentAnalyze.base_binance_url + CurrentAnalyze.candle_stick).format(symbol + CurrentAnalyze.symbol_id_map[base_currency], interval)
        print(url)
        try:
            raw_df = pd.read_json(url)
            raw_df[0] = raw_df[0].apply(lambda x: datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M'))
            price_df = raw_df[[0, 3]]
            volumn_df = raw_df[[0, 5]]
            price_df.set_index([0], inplace=True)
            volumn_df.set_index([0], inplace=True)
            price_df.index.names = ['date']
            price_df.columns = [symbol]
            volumn_df.index.names = ['date']
            volumn_df.columns = [symbol]
            price_df_cumreturn = (1 + price_df.pct_change()).cumprod() - 1
            return (price_df_cumreturn, volumn_df)
        except:
            print("Unexpected error:", sys.exc_info()[0])

    def get_candle_sticks(self, symbol_array, base_currency, interval):
        pool = ThreadPool(processes=10)
        results = []
        # start_time = int((datetime.now()-timedelta(days=1)).strftime("%s"))
        # end_time = int(datetime.now().strftime("%s"))
        for symbol in symbol_array:
            async_result = pool.apply_async(
                self.get_candle_stick,
                args=(symbol, base_currency, interval))
            results.append(async_result)
        pool.close()
        pool.join()
        prices = []
        volumns = []
        for res in results:
            try:
                (p, v) = res.get()
                prices.append(p)
                volumns.append(v)
            except:
                print("Exception in handling symbol.")
        price_result = pd.concat(prices)
        volumn_result = pd.concat(volumns)
        return (price_result, volumn_result)

    def get_symbol_list(self):
        url = CurrentAnalyze.base_binance_url + CurrentAnalyze.one_day_ticker
        df = pd.read_json(url)
        extract = lambda s: s[:-3]
        vfunc = np.vectorize(extract)
        x = vfunc(df['symbol'].values)
        tmp = set(x.flat)
        test = [a for a in filter(None, tmp) if not a.isdigit()]
        return test

    def chart_all(self):
        symbolist = self.get_symbol_list()
        print(symbolist)
        Util.df_scatters(self.get_candle_sticks(symbolist, 'ethereum', '1h'), 'cumulative prodct')

    def get_depth_chart(self, symbol, num=0, action='none'):
        print(action)
        scala = 100000000
        orders = CurrentAnalyze.client.get_order_book(symbol=symbol)
        # print(orders['bids'])
        # print(orders['asks'])
        bidsum = 0.0
        asksum = 0.0
        mark = 0.0
        if (action == 'sell'):
            mark = orders['bids'][0][0]
        else:
            mark = orders['asks'][-1][0]

        for bid in orders['bids']:
            bidsum += float(bid[0]) * float(bid[1])
            if num > 0 and action=='sell':
                num -= int(float(bid[1]))
                if num <= 0:
                    mark = bid[0]
            else:
                continue
        for ask in orders['asks']:
            asksum += float(ask[0]) * float(ask[1])
            if num > 0 and action=='buy':
                num -= int(float(ask[1]))
                if num <= 0:
                    mark = ask[0]
            else:
                continue
        bidPd = pd.Series(orders['bids']).to_frame()
        askPd = pd.Series(orders['asks']).to_frame()

        bidDf = bidPd[0].apply(pd.Series)
        bidDf.columns = ['price', 'bid_volumn', 'dummy']
        bidDf['price'] = bidDf['price'].astype(float) * 1000000
        bidDf.set_index(['price'], inplace=True)
        bidDf.drop(columns=['dummy'], inplace=True)
        pd.DataFrame.sort_index(bidDf, inplace=True)

        askDf = askPd[0].apply(pd.Series)
        askDf.columns = ['price', 'ask_volumn', 'dummy']
        askDf['price'] = askDf['price'].astype(float) * 1000000
        askDf.set_index(['price'], inplace=True)
        askDf.drop(columns=['dummy'], inplace=True)
        pd.DataFrame.sort_index(askDf, inplace=True)
        res = pd.concat([bidDf, askDf]).astype(float)

        message = 'depth graph\ntotal buy: {}\ntotal sell: {}\n'.format(bidsum, asksum)
        if action == 'buy':
            message += 'buy limit={}'.format(mark)
        elif action == 'sell':
            message += 'sell limit={}'.format(mark)

        print(message)
        print(orders['bids'])
        plt.plot(res)
        plt.xlabel('price * 10^6')
        plt.ylabel('volumn')
        plt.title(message)
        plt.show()
