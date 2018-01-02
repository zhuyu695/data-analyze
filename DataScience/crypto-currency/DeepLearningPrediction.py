import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

# get market info for bitcoin from the start of 2016 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170101&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# when Volume is equal to '-' convert it to 0
bitcoin_market_info['Volume'].fillna('-')
bitcoin_market_info['Volume'].replace('-', 0).astype('float')
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('float')
bitcoin_market_info.loc[bitcoin_market_info['Volume']<0,'Volume']=0
# look at the first few rows`
# print(bitcoin_market_info.head())

# get market info for ethereum from the start of 2016 to the current day
eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20170101&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
eth_market_info['Volume'].fillna('-')
eth_market_info['Volume'].replace('-', 0).astype('float')
eth_market_info['Volume'] = eth_market_info['Volume'].astype('float')
eth_market_info.loc[eth_market_info['Volume']<0,'Volume']=0
eth_market_info.fillna(0)
# look at the first few rows
# print(eth_market_info.head())

xrp_market_info = pd.read_html("https://coinmarketcap.com/currencies/ripple/historical-data/?start=20170101&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
xrp_market_info = xrp_market_info.assign(Date=pd.to_datetime(xrp_market_info['Date']))
xrp_market_info['Volume'].fillna('-')
xrp_market_info['Volume'] = xrp_market_info['Volume'].replace('-', 0).astype('float')
xrp_market_info.loc[xrp_market_info['Volume']<0,'Volume']=0
xrp_market_info.fillna(0)


currency_map = {'icx': 'icon',
                'xvg': 'verge',
                'xzc': 'zcoin',
                'wabi': 'wabi',
                'iota': 'iota',
                'btc': 'bitcoin'}
target = 'btc'
print(('https://coinmarketcap.com/currencies/{}/historical-data/?start=20170101&end=' + time.strftime("%Y%m%d")).format(
    currency_map.get(target)))
target_prefix = '{}_'.format(target)
target_market_info = pd.read_html(("https://coinmarketcap.com/currencies/{}/historical-data/?start=20170101&end=" + time.strftime("%Y%m%d")).format(
    currency_map.get(target)))[0]
# convert the date string to the correct date format
target_market_info = target_market_info.assign(Date=pd.to_datetime(target_market_info['Date']))
target_market_info['Volume'].fillna('-')
target_market_info['Volume'] = target_market_info['Volume'].replace('-', 0).astype('float')
target_market_info.loc[target_market_info['Volume'] < 0, 'Volume']=0
target_market_info.fillna(0)

bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]
target_market_info.columns = [target_market_info.columns[0]] + [target_prefix + i for i in target_market_info.columns[1:]]
xrp_market_info.columns =[xrp_market_info.columns[0]]+['xrp_'+i for i in xrp_market_info.columns[1:]]

market_info = pd.concat([bitcoin_market_info.set_index('Date'),
                         eth_market_info.set_index('Date'),
                         xrp_market_info.set_index('Date'), target_market_info.set_index('Date')], axis=1, join='inner')
market_info['Date'] = market_info.index
# market_info = pd.merge(eth_market_info, xrp_market_info, on=['Date'])
# print(market_info)
market_info = market_info[market_info['Date']>='2017-11-01']
split_date = '2017-12-20'

for coins in ['bt_', 'eth_', 'xrp_', target_prefix]:
    kwargs = { coins+'close_off_high': lambda x: (2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1),
            coins+'volatility': lambda x: ((x[coins+'High']- x[coins+'Low'])/(x[coins+'Open']))}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_', 'xrp_', target_prefix]
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
model_data = model_data.sort_values(by='Date')
# print(model_data.head())

# we don't need the date columns anymore
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 6
norm_cols = [coin+metric for coin in ['bt_', 'eth_', 'xrp_', target_prefix] for metric in ['Close','Volume']]

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/(temp_set[col].iloc[0]) - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['{}Close'.format(target_prefix)][window_len:].values/training_set['{}Close'.format(target_prefix)][:-window_len].values)-1

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['{}Close'.format(target_prefix)][window_len:].values/test_set['{}Close'.format(target_prefix)][:-window_len].values)-1
# print(LSTM_training_inputs[0])
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

#import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
eth_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['{}Close'.format(target_prefix)][window_len:].values/training_set['{}Close'.format(target_prefix)][:-window_len].values)-1
# train model on data
# note: eth_history contains information on the training error per epoch
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                            epochs=100, batch_size=1, verbose=2, shuffle=True)
# print(eth_history)

month_range = [datetime.date(2017,i+1,1) for i in range(12)] + [datetime.date(2018,i+1,1) for i in range(2)]
month_range_labels = [datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)] + \
                     [datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(2)]

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks(month_range)
ax1.set_xticklabels(month_range_labels)

date = model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)

actual = test_set['{}Close'.format(target_prefix)][window_len:]

ax1.plot(date, actual, label='Actual')

predict = ((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['{}Close'.format(target_prefix)].values[:-window_len])[0]

ax1.plot(date, predict, label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_test_inputs))+1)-\
            (test_set['{}Close'.format(target_prefix)].values[window_len:])/(test_set['{}Close'.format(target_prefix)].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})

print(actual)
print(predict)

plt.show()
# print(pd.concat([date_df, value_df], axis=1))

