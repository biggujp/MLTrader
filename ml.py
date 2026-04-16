# =========================
# FULL SYSTEM: Scanner + ML + Entry + Alert + Chart
# =========================

# INSTALL:
# pip install yfinance pandas numpy xgboost joblib matplotlib mplfinance requests

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import requests
import os
import json
from datetime import date
import joblib

# ================= CONFIG =================
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"
ALERT_FILE = "alerted_today.json"
MODEL_PATH = "model.pkl"

SYMBOLS = ["NVDA","AMD","TSLA","AAPL","MSFT"]

# ================= INDICATORS =================
def ema(series, period):
    return series.ewm(span=period).mean()

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ================= ENTRY SCORE =================
def entry_score(df, spy_df=None):
    df = df.copy()
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['RSI'] = rsi(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    # Trend
    if last['Close'] > last['EMA20']: score += 10
    if last['EMA20'] > last['EMA50']: score += 10
    if last['EMA20'] > df['EMA20'].iloc[-5]: score += 5

    # Pullback
    dist = abs(last['Close'] - last['EMA20']) / last['EMA20']
    if dist < 0.01: score += 20
    elif dist < 0.02: score += 15

    # Reversal
    if last['Close'] > last['Open']: score += 10
    if last['Close'] > prev['High']: score += 10

    # Volume
    vol_avg = df['Volume'].rolling(5).mean().iloc[-1]
    if last['Volume'] > vol_avg * 1.5: score += 15
    elif last['Volume'] > vol_avg: score += 10

    # Breakout
    high20 = df['High'].rolling(20).max().iloc[-2]
    if last['Close'] > high20: score += 10

    # Market filter
    if spy_df is not None:
        spy_ema20 = ema(spy_df['Close'], 20).iloc[-1]
        spy_last = spy_df['Close'].iloc[-1]
        if spy_last > spy_ema20: score += 10
        else: score -= 10

    if last['RSI'] > 75: score -= 10

    return score

# ================= ML =================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ================= SL / TP =================
def find_swing_low(df):
    return df['Low'].rolling(5).min().iloc[-1]

# ================= ALERT STORAGE =================
def load_alerted():
    if not os.path.exists(ALERT_FILE):
        return {}
    return json.load(open(ALERT_FILE))


def save_alerted(data):
    json.dump(data, open(ALERT_FILE, "w"))


def reset_day(alerted):
    today = str(date.today())
    if alerted.get("date") != today:
        return {"date": today, "symbols": []}
    return alerted

# ================= TELEGRAM =================
def send_telegram_photo(path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(path, "rb") as photo:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})

# ================= CHART =================
def plot_chart(symbol, df, score):
    df = df.tail(40)
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)

    last = df.iloc[-1]
    entry = last['Close']

    swing_low = find_swing_low(df)
    sl = min(swing_low, last['EMA20'])
    risk = entry - sl
    tp = entry + risk * 2

    entry_mark = np.full(len(df), np.nan)
    entry_mark[-1] = entry

    sl_line = np.full(len(df), sl)
    tp_line = np.full(len(df), tp)

    apds = [
        mpf.make_addplot(df['EMA20']),
        mpf.make_addplot(df['EMA50']),
        mpf.make_addplot(entry_mark, type='scatter', markersize=200, marker='o'),
        mpf.make_addplot(sl_line),
        mpf.make_addplot(tp_line)
    ]

    filename = f"{symbol}.png"

    mpf.plot(df, type='candle', addplot=apds, volume=True, savefig=filename)

    return filename, entry, sl, tp

# ================= MAIN =================
def run():
    spy = yf.download("SPY", period="3mo")
    model = load_model()

    alerted = reset_day(load_alerted())

    results = []

    for sym in SYMBOLS:
        df = yf.download(sym, period="3mo")

        score = entry_score(df, spy)

        ml_score = 0
        if model is not None:
            features = [[df['Close'].pct_change().iloc[-1]]]
            ml_score = model.predict_proba(features)[0][1]

        final_score = score

        results.append({
            "symbol": sym,
            "score": final_score,
            "price": df['Close'].iloc[-1]
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    for r in results:
        if r['score'] < 80:
            continue

        if r['symbol'] in alerted['symbols']:
            continue

        df = yf.download(r['symbol'], period="3mo")

        filename, entry, sl, tp = plot_chart(r['symbol'], df, r['score'])

        caption = f"""
🔥 ENTRY A+
Symbol: {r['symbol']}
Score: {r['score']}
Entry: {round(entry,2)}
SL: {round(sl,2)}
TP: {round(tp,2)}
"""

        send_telegram_photo(filename, caption)

        os.remove(filename)

        alerted['symbols'].append(r['symbol'])

    save_alerted(alerted)


if __name__ == "__main__":
    run()
