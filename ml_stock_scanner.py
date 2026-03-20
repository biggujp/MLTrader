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
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =========================================
# CONFIG
# =========================================
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"

MODEL_PATH = "model.pkl"

TICKERS_TRAIN = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "MU", "INTC", "GOOG","SMCI", "JPM","LLY","PLTR"]

# =========================================
# UTILS
# =========================================
def fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_download(ticker, period="3mo"):
    try:
        if "/" in ticker:
            return pd.DataFrame()

        df = yf.download(ticker, period=period, interval="1d", progress=False)

        if df.empty:
            return pd.DataFrame()

        return fix_columns(df)
    except:
        return pd.DataFrame()    

# =========================================
# TELEGRAM
# =========================================
def send_alert(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        if res.status_code != 200:
            print("Telegram Error:", res.text)
    except Exception as e:
        print("Telegram Exception:", e)

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

    # 🔥 เพิ่มความแม่น
    df['High_20'] = df['High'].rolling(20).max()
    df['Breakout'] = df['Close'] / df['High_20']
    df['Momentum'] = df['Close'] / df['Close'].shift(5)

    # Volume Spike
    df['Vol_Avg'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] / df['Vol_Avg']

    # Range Compression (สะสมก่อนพุ่ง)
    df['Range'] = df['High'] - df['Low']
    df['Range_Avg'] = df['Range'].rolling(10).mean()
    df['Tight_Range'] = df['Range'] / df['Range_Avg']

    # Close near High (แรงซื้อจริง)
    df['Close_High_Ratio'] = df['Close'] / df['High']

    return df

# =========================================
# ฺBREAKOUT SCORE (สำหรับกรองตัวเทพก่อนเข้าโมเดล)
# =========================================
def breakout_score(df):
    last = df.iloc[-1]

    score = 0

    # 🔥 breakout จริง
    if last['Breakout'] > 1.01:
        score += 2

    # 🔥 volume เข้า
    if last['Vol_Spike'] > 1.5:
        score += 2

    # 🔥 บีบก่อนพุ่ง
    if last['Tight_Range'] < 0.8:
        score += 1

    # 🔥 ปิดใกล้ high
    if last['Close_High_Ratio'] > 0.97:
        score += 1

    return score

# =========================================
# TARGET (TP/SL BASED)
# =========================================
def create_target(df, tp=0.06, sl=0.03, max_days=5):
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
        'Return_5d', 'Return_10d',
        'Trend', 'ATR',
        'Vol_Ratio', 'RSI',
        'Breakout', 'Momentum'
    ]

    X = df_all[features]
    y = df_all['Target']

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X, y)

    joblib.dump((model, features), MODEL_PATH)
    print("✅ Model trained & saved")

    return model, features

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        model, features = joblib.load(MODEL_PATH)
        print("✅ Loaded saved model")
        return model, features
    else:
        return train_model()

# =========================================
# SCANNER
# =========================================
def scan_market():
    data = (Query()
        .select('name', 'close', 'volume', 'relative_volume_10d_calc', 'RSI', 'EMA50')
        .where(
            col('exchange').isin(['NASDAQ', 'NYSE']), 
            col('close') >= 1,            
            col('relative_volume_10d_calc') > 1.5,
            col('close') > col('EMA50'),
            col('RSI') < 70
        )
        .order_by('volume', ascending=False)
        .limit(30)
        .get_scanner_data())

    df = pd.DataFrame(data[1])
    return df

# =========================================
# PREDICT
# =========================================
def predict_score(model, features, ticker):
    df = safe_download(ticker, period="3mo")
    if df.empty:
        return 0

    df = create_features(df)
    df = df.dropna()

    if len(df) == 0:
        return 0

    last = df.iloc[-1]

    # ML score
    X = np.array([[last[f] for f in features]])
    ml_score = model.predict_proba(X)[0][1]

    # Breakout score
    b_score = breakout_score(df)

    # 🔥 รวมคะแนน
    final_score = ml_score + (b_score * 0.1)

    return final_score


def is_valid_ticker(ticker):
    # ❌ ตัด ticker ที่มี /
    if "/" in ticker:
        return False
    return True

# =========================================
# MAIN
# =========================================
def main():
    print("🚀 Loading model...")
    model, features = load_model()

    print("📊 Scanning market...")
    df_scan = scan_market()

    print("🧠 Evaluating...")
    picks = []

    for _, row in df_scan.iterrows():
        ticker = row['name']

        # 🔥 skip ตัวแปลก
        if not is_valid_ticker(ticker):
            continue

        score = predict_score(model, features, ticker)

        if score > 0.8:  # 🔥 filter ตัวเทพ
            picks.append((ticker, score, row['close'], row['RSI']))

    picks = sorted(picks, key=lambda x: x[1], reverse=True)[:5]

    if not picks:
        print("❌ No strong signals today")
        return

    for p in picks:
        ticker, score, price, rsi = p

        msg = f"""
🚀 TOP STOCK (ML)
Symbol: {ticker}
Price: {price}
RSI: {rsi}
Score: {score:.2f}
"""
        print(msg)
        send_alert(msg)

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()
