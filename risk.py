from config import *

def calc_size(balance, atr):
    risk = balance * RISK_PER_TRADE
    size = risk / atr
    return max(0.001, min(size, 0.02))

def get_sl_tp(entry, atr, side):
    if side == 'long':
        sl = entry - atr
        tp = entry + atr * 1.5
    else:
        sl = entry + atr
        tp = entry - atr * 1.5
    
    return sl, tp