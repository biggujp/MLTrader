import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from ml_model import *
from config import *

INITIAL_BALANCE = 50

# -----------------------------
# Load Data
# -----------------------------
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = ['ts','open','high','low','close','volume']
    return df

# -----------------------------
# Feature
# -----------------------------
def prepare_df(df):
    df['return'] = df['close'].pct_change()
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['atr'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], 14
    )
    df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

# -----------------------------
# MTF Filter (simplified)
# -----------------------------
def get_trend(df, i):
    ema = df['close'].rolling(50).mean().iloc[i]
    price = df['close'].iloc[i]
    
    if price > ema:
        return 'long'
    else:
        return 'short'

# -----------------------------
# Backtest
# -----------------------------
def backtest(df):
    balance = INITIAL_BALANCE
    equity_curve = []
    
    clf, reg = train(df)
    
    position = None
    
    wins = 0
    losses = 0
    
    for i in range(100, len(df)-1):
        row = df.iloc[:i]
        
        # --- ML ---
        direction, vol = predict(clf, reg, row)
        bias = 'long' if direction == 1 else 'short'
        
        # --- MTF ---
        trend = get_trend(df, i)
        
        if bias != trend:
            equity_curve.append(balance)
            continue
        
        price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        spacing = max(0.5, min(vol, 10))
        
        size = (balance * 0.01) / atr
        
        # --- Entry ---
        if position is None:
            entry = price
            
            if bias == 'long':
                sl = entry - atr
                tp = entry + atr * 1.5
            else:
                sl = entry + atr
                tp = entry - atr * 1.5
            
            position = {
                'side': bias,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'size': size
            }
        
        # --- Exit ---
        if position:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            if position['side'] == 'long':
                if low <= position['sl']:
                    balance -= atr * size
                    losses += 1
                    position = None
                elif high >= position['tp']:
                    balance += atr * 1.5 * size
                    wins += 1
                    position = None
            
            else:
                if high >= position['sl']:
                    balance -= atr * size
                    losses += 1
                    position = None
                elif low <= position['tp']:
                    balance += atr * 1.5 * size
                    wins += 1
                    position = None
        
        equity_curve.append(balance)
    
    return equity_curve, wins, losses

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    df = load_data("data.csv")
    df = prepare_df(df)
    
    equity, wins, losses = backtest(df)
    
    print("Final Balance:", equity[-1])
    print("Wins:", wins, "Losses:", losses)
    
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.show()