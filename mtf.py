import pandas as pd
import ta
from exchange import get_ohlcv

def get_signal(tf):
    data = get_ohlcv(tf)
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
    
    df['ema'] = ta.trend.ema_indicator(df['c'], 50)
    
    price = df['c'].iloc[-1]
    ema = df['ema'].iloc[-1]
    
    if price > ema:
        return 'long'
    elif price < ema:
        return 'short'
    else:
        return 'neutral'

def get_mtf_signal():
    t1 = get_signal('15m')
    t2 = get_signal('5m')
    
    if t1 == t2:
        return t1
    return 'neutral'