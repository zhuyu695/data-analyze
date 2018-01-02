import pickle
import copy
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

class Util:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def df_scatters(dfs, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
        if len(dfs) != 2:
            raise Exception("df size is not equal to 2")
        '''Generate a scaatter plot of the entire dataframe'''
        fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Cumulative Product', 'Volumn'))
        count = 1
        for df in dfs:
            label_arr = copy.copy(list(df))
            series_arr = copy.copy(list(map(lambda col: df[col], label_arr)))
            layout = go.Layout(
                title=title,
                legend=dict(orientation='h'),
                xaxis=dict(type='date'),
                yaxis=dict(title=y_axis_label, showticklabels=not seperate_y_axis, type=scale))
            y_axis_config = copy.copy(dict(overlaying='y', showticklabels=False, type=scale))
            visibility = 'visible'
            if initial_hide: visibility = 'legendonly'  # From trace for each serise
            for index, series in enumerate(series_arr):
                trace = go.Scatter(x=series.index, y=series, name=label_arr[index], visible=visibility)
                # Add seperate axis for the series
                if seperate_y_axis: trace['yaxis'] = 'y{}'.format(index + 1)
                layout['yaxis{}'.format(index + 1)] = y_axis_config
                fig.append_trace(trace, count, 1)
            count += 1
        py.plot(fig)