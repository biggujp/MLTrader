import ccxt
from config import *

exchange = ccxt.bitget({
    'apiKey': API_KEY,
    'secret': SECRET,
    'password': PASSPHRASE,
    'options': {'defaultType': 'swap'}
})

exchange.set_sandbox_mode(True)

def get_ohlcv(tf):
    return exchange.fetch_ohlcv(SYMBOL, tf, limit=200)

def get_price():
    return exchange.fetch_ticker(SYMBOL)['last']

def place_limit(side, amount, price):
    return exchange.create_limit_order(SYMBOL, side, amount, price)

def place_market(side, amount, params={}):
    return exchange.create_order(SYMBOL, 'market', side, amount, None, params)

def cancel_all():
    try:
        exchange.cancel_all_orders(SYMBOL)
    except:
        pass

def get_positions():
    return exchange.fetch_positions([SYMBOL])