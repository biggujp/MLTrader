# =========================================
# IMPORT
# =========================================
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import ta
import joblib
import os
import pandas_market_calendars as mcal
import time
import sys
from tradingview_screener import Query, col
from xgboost import XGBClassifier
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')


# =========================================
# CHECK MARKET OPEN
# =========================================
def is_us_market_open_today():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty

# =========================================
# CONFIG
# =========================================
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"

MODEL_PATH = "model.pkl"

TICKERS_TRAIN = [
    "AAPL","NVDA","TSLA","AMD","MSFT","META",
    "AMZN","MU","INTC","GOOG","SMCI","JPM","LLY","PLTR"
]

# =========================================
# UTILS
# =========================================
def fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def normalize_ticker(ticker):
    ticker = ticker.strip()
    if "." in ticker:
        ticker = ticker.replace(".", "-")
    return ticker

def safe_download(ticker, period="3mo", interval="1d"):
    ticker = normalize_ticker(ticker)

    for attempt in range(3):
        try:
            if "/" in ticker or ticker == "":
                return pd.DataFrame()

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=10
            )

            if df is None or df.empty or len(df) < 50:
                continue

            df = fix_columns(df)
            df = df.dropna()
            return df

        except Exception as e:
            print(f"Retry {attempt+1} {ticker}: {e}")

    return pd.DataFrame()

# =========================================
# TELEGRAM
# =========================================
def send_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except:
        pass

# =========================================
# FEATURE ENGINEERING
# =========================================
def create_features(df):

    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)

    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['Trend'] = df['Close'] / df['EMA50']

    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close']).average_true_range()

    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    # ✅ FIX: shift(1)
    df['High_20'] = df['High'].rolling(20).max().shift(1)
    df['Breakout'] = df['Close'] / df['High_20']

    df['Momentum'] = df['Close'] / df['Close'].shift(5)

    df['Vol_Avg'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] / df['Vol_Avg']

    df['Range'] = df['High'] - df['Low']
    df['Range_Avg'] = df['Range'].rolling(10).mean()
    df['Tight_Range'] = df['Range'] / df['Range_Avg']

    df['Close_High_Ratio'] = df['Close'] / df['High']

    df['Breakout_Strong'] = df['Close'] > df['High'].shift(1)

    df['EMA20_50'] = df['EMA20'] > df['EMA50']

    df['Breakout_Distance'] = df['Close'] / df['High_20']

    df['Vol_Confirm'] = (
        (df['Volume'] > df['Volume'].shift(1)) &
        (df['Volume'].shift(1) > df['Volume'].shift(2))
    )

    df['EMA20_50'] = df['EMA20_50'].astype(int)
    df['Breakout_Strong'] = df['Breakout_Strong'].astype(int)
    df['Vol_Confirm'] = df['Vol_Confirm'].astype(int)

    return df

# =========================================
# BREAKOUT SCORE
# =========================================
def breakout_score(df):
    last = df.iloc[-1]
    score = 0

    if last['Breakout'] > 1.02:
        score += 3

    if last['Vol_Spike'] > 2:
        score += 3

    if last['EMA20_50']:
        score += 2

    if last['Close_High_Ratio'] > 0.98:
        score += 2

    if last['Breakout_Distance'] < 1.05:
        score += 1

    if (last['High'] - last['Close']) / last['High'] > 0.03:
        score -= 2

    if last['Vol_Confirm']:
        score += 2

    return score

# =========================================
# TARGET (FIXED)
# =========================================
def create_target(df, tp=0.06, sl=0.03, max_days=5):

    targets = []

    for i in range(len(df)):
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+max_days+1]

        if len(future) == 0:
            targets.append(0)
            continue

        hit = 0

        # ✅ FIX: check order (realistic)
        for _, row in future.iterrows():

            if row['Low'] <= entry * (1 - sl):
                hit = 0
                break

            if row['High'] >= entry * (1 + tp):
                hit = 1
                break

        targets.append(hit)

    df['Target'] = targets
    return df

# =========================================
# TRAIN MODEL
# =========================================
def train_model():
    df_all = pd.DataFrame()

    for t in TICKERS_TRAIN:
        df = safe_download(t, period="2y")
        if df.empty:
            continue

        df = create_features(df)
        df = create_target(df)

        df_all = pd.concat([df_all, df])

    # ✅ FIX data clean
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    df_all = df_all.dropna()

    features = [
        'Return_5d','Return_10d','Momentum',
        'Trend','EMA20_50',
        'Breakout','Breakout_Distance',
        'Close_High_Ratio','Breakout_Strong',
        'Vol_Ratio','Vol_Spike','Vol_Confirm',
        'ATR','Tight_Range',
        'RSI'
    ]

    X = df_all[features]
    y = df_all['Target']

    model = XGBClassifier(
        n_estimators=400,      # 🔥 เพิ่ม
        max_depth=6,           # 🔥 ลึกขึ้น
        learning_rate=0.03,    # 🔥 smooth ขึ้น
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X, y)

    joblib.dump((model, features), MODEL_PATH)
    msg = f"""✅ Model trained & saved"""
    print(msg)
    #send_alert(msg)
    return model, features

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        msg = f"""✅ Loaded model"""
        print(msg)
        #send_alert(msg)
        return joblib.load(MODEL_PATH)
    return train_model()

# =========================================
# SCANNER (FIXED RSI)
# =========================================
def scan_market():
    data = (Query()
        .select(
            'name','close','high','RSI',
            'volume','change','relative_volume_10d_calc'
        )
        .where(
            col('exchange').isin(['NASDAQ','NYSE']),
            col('relative_volume_10d_calc') > 1.3,
            col('volume') > 1e6,
            col('change') > 2,
            col('RSI').between(55, 80),   # 🔥 sweet spot (ไม่ overbought เกิน)
            col('close').between(5, 300)
        )
        .limit(100)
        .get_scanner_data()
    )

    return pd.DataFrame(data[1])

# =========================================
# PREDICT
# =========================================
def predict_score(model, features, ticker):

    df = safe_download(ticker, period="3mo")
    if df.empty:
        return 0

    df = create_features(df).dropna()
    if len(df) == 0:
        return 0

    last = df.iloc[-1]

    # 🔥 soften filter
    if last['Close'] < last['EMA50'] * 0.97:
        return 0

    if last['Vol_Spike'] < 1.0:
        return 0

    X = np.array([[last[f] for f in features]])
    ml_score = model.predict_proba(X)[0][1]

    b_score = breakout_score(df) / 10

    # 🔥 balance ใหม่
    final_score = (ml_score * 0.65) + (b_score * 0.35)

    return final_score

# =========================================
# TRADE LEVEL
# =========================================
def calculate_trade_levels(df):

    last = df.iloc[-1]
    entry = last['Close']
    atr = last['ATR']

    sl = entry - (atr * 1.5)
    tp = entry + (atr * 3)

    rr = (tp - entry) / (entry - sl)

    return entry, sl, tp, rr

# =========================================
# VALID TICKER
# =========================================
def is_valid_ticker(ticker):
    if "/" in ticker:
        return False
    if len(ticker) > 6:
        return False
    return True

# =========================================
# DEDUPE PICKS
# =========================================
def dedupe_picks(picks):
    seen = set()
    unique = []

    for p in picks:
        ticker = p[0]

        if ticker not in seen:
            unique.append(p)
            seen.add(ticker)

    return unique    

# =========================================
# MAIN
# =========================================
def main():

    print("🚀 Loading model...")
    #send_alert("🚀 Loading model...")
    model, features = load_model()

    print("📊 Scanning...")
    #send_alert("📊 Scanning...")
    df_scan = scan_market()

    strong = []
    normal = []

    for _, row in df_scan.iterrows():

        ticker = row['name']

        if not is_valid_ticker(ticker):
            continue

        score = predict_score(model, features, ticker)

        if score == 0:
            continue

        df = safe_download(ticker)
        if df.empty:
            continue

        df = create_features(df).dropna()
        if len(df) == 0:
            continue

        entry, sl, tp, rr = calculate_trade_levels(df)

        data = (ticker, score, entry, sl, tp, rr, row['RSI'])

        # 🔥 Tier system
        if score >= 0.75:
            strong.append(data)
        elif score >= 0.6:
            normal.append(data)

    # 🔥 sort
    strong = sorted(strong, key=lambda x: x[1], reverse=True)
    normal = sorted(normal, key=lambda x: x[1], reverse=True)

    # 🔥 combine
    picks = strong[:5]

    if len(picks) < 5:
        picks += normal[:(5 - len(picks))]

    # 🔥 กันซ้ำ
    picks = dedupe_picks(picks)    

    # 🔥 fallback ขั้นสุด (กันไม่มีจริง ๆ)
    if len(picks) < 5:
        print("⚠️ Using extended fallback...")

        all_scores = strong + normal

        if len(all_scores) < 5:
            for _, row in df_scan.iterrows():
                ticker = row['name']

                if not is_valid_ticker(ticker):
                    continue

                score = predict_score(model, features, ticker)

                df = safe_download(ticker)
                if df.empty:
                    continue

                df = create_features(df).dropna()
                if len(df) == 0:
                    continue

                entry, sl, tp, rr = calculate_trade_levels(df)

                all_scores.append((ticker, score, entry, sl, tp, rr, row['RSI']))
                

        all_scores = dedupe_picks(all_scores)
        picks = sorted(all_scores, key=lambda x: x[1], reverse=True)[:5]

    # 🔥 OUTPUT
    for p in picks:
        ticker, score, entry, sl, tp, rr, rsi = p
        tier = "🔥 STRONG" if score >= 0.75 else "👍 NORMAL"
        msg = f"""
{tier}
Symbol: {ticker}
Score: {score:.2f}
RSI: {rsi:.2f}
🎯 Entry: {entry:.2f}
🛑 Stop Loss: {sl:.2f}
🎯 Take Profit: {tp:.2f}
📊 RR: {rr:.2f}
"""
        print(msg)
        #send_alert(msg)

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    if is_us_market_open_today():
        main()
    else:
        msg = f"""⏰ US market is closed today."""
        print(msg)
        #send_alert(msg)
