# =========================================
# IMPORT
# =========================================
from tradingview_screener import Query, col
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import ta
from xgboost import XGBClassifier
import joblib
import os

# =========================================
# CONFIG
# =========================================
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"

MODEL_PATH = "model.pkl"

TICKERS_TRAIN = [
    "AAPL","NVDA","TSLA","AMD","MSFT","META","AMZN",
    "MU","INTC","GOOG","SMCI","JPM","LLY","PLTR"
]

# =========================================
# UTILS
# =========================================
def fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def safe_download(ticker, period="3mo", interval="1d"):
    try:
        if "/" in ticker or ticker.strip() == "":
            return pd.DataFrame()

        if interval in ["1m","5m","15m","30m","60m"]:
            period = "60d"

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df is None or df.empty or len(df) < 50:
            return pd.DataFrame()

        df = fix_columns(df)
        df = df.dropna()

        return df

    except:
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
# FEATURES (Swing Focus)
# =========================================
def create_features(df):
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)

    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['EMA100'] = df['Close'].ewm(span=100).mean()

    df['Trend'] = df['Close'] / df['EMA50']
    df['Strong_Trend'] = df['EMA50'] / df['EMA100']

    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close']
    ).average_true_range()

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] / df['Vol_MA']

    df['High_20'] = df['High'].rolling(20).max()
    df['Breakout'] = df['Close'] / df['High_20']

    df['Momentum'] = df['Close'] / df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10)

    df['Pullback'] = df['Close'] / df['EMA20']

    df['Close_High_Ratio'] = df['Close'] / df['High']

    return df

# =========================================
# BREAKOUT SCORE
# =========================================
def breakout_score(df):
    last = df.iloc[-1]
    score = 0

    if last['Breakout'] > 1.02:
        score += 2

    if last['Vol_Spike'] > 2:
        score += 2

    if last['Strong_Trend'] > 1.02:
        score += 2

    if 0.97 < last['Pullback'] < 1.05:
        score += 1

    if last['Close_High_Ratio'] > 0.98:
        score += 1

    return score

# =========================================
# TARGET
# =========================================
def create_target(df, tp=0.08, sl=0.04, max_days=5):
    targets = []

    for i in range(len(df)):
        entry = df['Close'].iloc[i]
        future = df.iloc[i+1:i+max_days+1]

        if len(future) == 0:
            targets.append(0)
            continue

        hit_tp = (future['High'] >= entry * (1 + tp)).any()
        hit_sl = (future['Low'] <= entry * (1 - sl)).any()

        if hit_tp and not hit_sl:
            targets.append(1)
        else:
            targets.append(0)

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

    df_all = df_all.dropna()

    features = [
        'Return_5d','Return_10d','Trend','ATR',
        'RSI','Breakout','Momentum','Strong_Trend'
    ]

    X = df_all[features]
    y = df_all['Target']

    pos = sum(y == 1)
    neg = sum(y == 0)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight=neg / pos
    )

    model.fit(X, y)

    joblib.dump((model, features), MODEL_PATH)
    print("✅ Model trained")

    return model, features

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model()

# =========================================
# SCANNER
# =========================================
def scan_market():
    data = (Query()
        .select('name','close','volume','relative_volume_10d_calc','RSI','EMA50')
        .where(
            col('exchange').isin(['NASDAQ','NYSE']),
            col('close') >= 1,
            col('relative_volume_10d_calc') > 1.5,
            col('close') > col('EMA50'),
            col('RSI') < 70,
            col('change') > 2
        )
        .order_by('volume', ascending=False)
        .limit(30)
        .get_scanner_data())

    df = pd.DataFrame(data[1])

    # 🔥 filter ticker แปลก
    df = df[df['name'].str.len() <= 5]
    df = df[~df['name'].str.contains(r'[^A-Z]', regex=True)]

    return df

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

    try:
        last = df.iloc[-1]

        X = np.array([[last[f] for f in features]])
        ml_score = model.predict_proba(X)[0][1]

        b_score = breakout_score(df)

        final_score = (ml_score * 0.7) + (b_score * 0.3)

        return final_score

    except:
        return 0

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
# MAIN
# =========================================
def main():
    print("🚀 Loading model...")
    model, features = load_model()

    # 🔥 Market filter (SPY)
    spy = safe_download("SPY")
    spy = create_features(spy)

    if spy.iloc[-1]['Close'] < spy.iloc[-1]['EMA50']:
        print("❌ Market weak - skip")
        return

    print("📊 Scanning...")
    df_scan = scan_market()

    picks = []

    for _, row in df_scan.iterrows():
        ticker = row['name']

        score = predict_score(model, features, ticker)

        if score > 0.7:
            df = safe_download(ticker)
            if df.empty:
                continue

            df = create_features(df).dropna()
            if len(df) == 0:
                continue

            last = df.iloc[-1]

            # 🔥 Trend filter
            if last['EMA20'] < last['EMA50']:
                continue

            # 🔥 RSI filter
            if last['RSI'] > 65:
                continue

            entry, sl, tp, rr = calculate_trade_levels(df)

            # 🔥 RR filter
            if rr < 1.8:
                continue

            picks.append((ticker, score, entry, sl, tp, rr, last['RSI']))

    picks = sorted(picks, key=lambda x: x[1], reverse=True)[:5]

    if not picks:
        print("No strong signals today")
        return

    for p in picks:
        ticker, score, entry, sl, tp, rr, rsi = p

        msg = f"""
🚀 TRADE SETUP
Symbol: {ticker}
Score: {score:.2f}
RSI: {rsi:.1f}

🎯 Entry: {entry:.2f}
🛑 SL: {sl:.2f}
🎯 TP: {tp:.2f}
📊 RR: {rr:.2f}
"""
        print(msg)
        send_alert(msg)

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()