from env import Environment1
from dqn import train_dqn
import pandas as pd
import numpy as np

from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl

import sys
sys.path.append('./stockdqn')

data = pd.read_csv('./stockdqn/HDFC.NS.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
print(f"Data Min:{data.index.min()} \t Max: {data.index.max()}")

date_split = '2020-01-01'
train = data[:date_split]
test = data[date_split:]
print(f"Train len: {len(train)} \t Test len: {len(test)}")

env = Environment1(train)
print(env.reset())


def plot_train_test(train, test, date_split):

    data = [
        Candlestick(x=train.index, open=train['Open'], high=train['High'],
                    low=train['Low'], close=train['Close'], name='train'),
        Candlestick(x=test.index, open=test['Open'], high=test['High'],
                    low=test['Low'], close=test['Close'], name='test')
    ]
    layout = {
        'shapes': [
             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x',
                 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
        ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper',
                'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper',
                'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)


# plot_train_test(train, test, date_split)

Q, total_losses, total_rewards = train_dqn(Environment1(train))
print("hello")
