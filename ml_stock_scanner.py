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

def normalize_ticker(ticker):
    ticker = ticker.strip()
    # 🔥 แก้ . → -
    if "." in ticker:
        ticker = ticker.replace(".", "-")
    return ticker


def safe_download(ticker, period="3mo", interval="1d"):
    ticker = normalize_ticker(ticker)

    for attempt in range(3):  # 🔥 retry 3 ครั้ง
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

    # Breakout Confirm (ทะลุ high เดิมแบบชัด)
    df['Breakout_Strong'] = df['Close'] > df['High'].shift(1)
    # Follow Through (วันถัดไปต้องเขียว)
    df['Follow_Through'] = df['Close'].shift(-1) > df['Close']
    # Trend Strength (EMA alignment)
    df['EMA20_50'] = df['EMA20'] > df['EMA50']

    # Distance from breakout (ไม่ไล่ราคา)
    df['Breakout_Distance'] = df['Close'] / df['High_20']

    df['Vol_Confirm'] = (
    (df['Volume'] > df['Volume'].shift(1)) &
    (df['Volume'].shift(1) > df['Volume'].shift(2)))

    return df

# =========================================
# ฺBREAKOUT SCORE (สำหรับกรองตัวเทพก่อนเข้าโมเดล)
# =========================================
def breakout_score(df):
    last = df.iloc[-1]

    score = 0

    # ✅ breakout จริง (ต้องแรงกว่าเดิม)
    if last['Breakout'] > 1.02:
        score += 3

    # ✅ volume ต้องแรงจริง
    if last['Vol_Spike'] > 2:
        score += 3

    # ✅ trend ต้องขึ้น
    if last['EMA20_50']:
        score += 2

    # ✅ ปิดใกล้ high มากๆ
    if last['Close_High_Ratio'] > 0.98:
        score += 2

    # ✅ ไม่ไล่ราคาเกิน
    if last['Breakout_Distance'] < 1.05:
        score += 1

    # ❌ ตัด breakout หลอก (ไส้เทียนยาว)
    if (last['High'] - last['Close']) / last['High'] > 0.03:
        score -= 2

    if last['Vol_Confirm']:
        score += 2

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
    msg = f"""✅ Model trained & saved"""
    #send_alert(msg)

    return model, features

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        model, features = joblib.load(MODEL_PATH)
        print("✅ Loaded saved model")
        msg = f"""✅ Loaded saved model"""
        #send_alert(msg)
        return model, features
    else:
        return train_model()

# =========================================
# SCANNER (แก้ใหม่)
# =========================================
def scan_market():
    data = (Query()
        .select('name', 'close', 'volume', 'relative_volume_10d_calc', 'RSI', 'EMA50')
        .where(
            col('exchange').isin(['NASDAQ', 'NYSE']),
            col('close') >= 5,

            # 🔥 ผ่อนแล้ว (จาก 2 → 1.2)
            col('relative_volume_10d_calc') > 1.5,

            # 🔥 เอา trend พอประมาณ
            col('close') > col('EMA50'),

            # 🔥 RSI กว้างขึ้น (หา early move)
            col('RSI').between(40, 65),

            # 🔥 ลดความแรง (ไม่ต้อง +3%)
            col('change') > 2
        )
        .order_by('volume', ascending=False)
        .limit(80)   # 🔥 เพิ่มจำนวน
        .get_scanner_data())

    df = pd.DataFrame(data[1])
    return df

# =========================================
# PREDICT (แก้ใหม่)
# =========================================
def predict_score(model, features, ticker):
    df = safe_download(ticker, period="3mo")
    if df.empty:
        return 0

    df = create_features(df).dropna()
    if len(df) == 0:
        return 0

    last = df.iloc[-1]

    # 🔥 ผ่อนเงื่อนไข (ไม่ kill เยอะ)
    if last['Close'] < last['EMA50']:
        return 0

    if last['Vol_Spike'] < 1.1:
        return 0

    # ML score
    X = np.array([[last[f] for f in features]])
    ml_score = model.predict_proba(X)[0][1]

    b_score = breakout_score(df)

    # 🔥 balance ใหม่
    final_score = (ml_score * 0.8) + (b_score * 0.2)

    return final_score

# =========================================
# TRADE LEVELS (สำหรับใส่ใน alert เพื่อให้เห็นภาพก่อนเข้าเทรด)
# =========================================
def calculate_trade_levels(df):
    last = df.iloc[-1]

    entry = last['Close']

    # ใช้ ATR จะแม่นกว่า
    atr = last['ATR']

    stop_loss = entry - (atr * 1.5)
    take_profit = entry + (atr * 3)

    rr = (take_profit - entry) / (entry - stop_loss)

    return entry, stop_loss, take_profit, rr


# =========================================
# VALID TICKER (กรอง ticker ที่มีรูปแบบแปลกๆ เช่นมี / ซึ่งโมเดลไม่ถนัด)
# =========================================
def is_valid_ticker(ticker):
    if "/" in ticker:
        return False

    # 🔥 ตัดหุ้น class แปลก (เช่น BRK.A, TAP.A)
    if "." in ticker:
        return True  # ให้ normalize แทน ไม่ต้องทิ้ง

    if len(ticker) > 6:
        return False

    return True

# =========================================
# MAIN (แก้ threshold)
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

        if not is_valid_ticker(ticker):
            continue

        score = predict_score(model, features, ticker)

        # 🔥 ลดจาก 0.9 → 0.6
        if score > 0.8:
            df = safe_download(ticker)

            if df.empty:
                continue

            df = create_features(df).dropna()
            if len(df) == 0:
                continue

            entry, sl, tp, rr = calculate_trade_levels(df)

            picks.append((ticker, score, entry, sl, tp, rr, row['RSI']))

    # 🔥 เอา top มากขึ้น
    picks = sorted(picks, key=lambda x: x[1], reverse=True)[:10]

    if not picks:
        print("No strong signals today")
        return

    for p in picks:
        ticker, score, entry, sl, tp, rr, rsi = p

        print(f"""
🚀 TRADE SETUP
Symbol: {ticker}
Score: {score:.2f}
RSI: {rsi:.2f}

🎯 Entry: {entry:.2f}
🛑 Stop Loss: {sl:.2f}
🎯 Take Profit: {tp:.2f}
📊 RR: {rr:.2f}
""")

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()
