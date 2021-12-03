import alpaca_trade_api as tradeapi
import time
import datetime
from datetime import timedelta
from pytz import timezone
tz = timezone('EST')


import numpy as np
import pandas as pd

from yahoo_fin import stock_info as si
import LSTM_pred as nn

api = tradeapi.REST('PK08PSBIZO3XBIEILL9W',
                    'Sl2r4IUVschAd69LsZSfUWBFpCRqUij6ibfYGlfi',
                    'https://paper-api.alpaca.markets')

import logging
logging.basicConfig(filename='./apca_algo.log', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('{} logging started'.format(datetime.datetime.now().strftime("%x %X")))

account = api.get_account()

bp = float(account.buying_power)

# get live price of Apple
x = si.get_live_price("aapl")
print(x)

if nn.future_price > x:
    print('Buy')
else: 
    print('Sell')

expected_profit = (nn.future_price - x)/50 
print(f'expected profit = ${expected_profit}')

if nn.future_price > x:
    api.submit_order('AAPL', 50, 'buy', 'market', 'day')
    print(datetime.datetime.now(tz).strftime("%x %X"), 'buying 50 shares AAPL')
else:
    api.submit_order('AAPL', 50, 'sell', 'market', 'day')
    print(datetime.datetime.now(tz).strftime("%x %X"), 'selling 50 shares AAPL')

expected_profit = (nn.future_price - x)/50 
print(f'expected profit = ${expected_profit}')

time.sleep(3600)    

stocks = ['AAPL']

