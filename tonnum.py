import requests
import pandas as pd
import time
from tradingview_screener import Query, Column

# ===== CONFIG =====
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"

TOP_N = 5
MIN_SCORE = 60
SCAN_INTERVAL = 1800

# ===== TELEGRAM =====
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram Error:", e)

# ===== GET DATA =====
def get_data():
    for _ in range(3):
        try:
            q = Query() \
                .select(
                    "name",
                    "close",
                    "volume",
                    "average_volume_30d_calc",
                    "relative_volume_10d_calc",
                    "market_cap_basic",
                    "MACD.macd",
                    "MACD.signal",
                    "EMA20",
                    "EMA50",
                    "change",
                    "exchange"
                ) \
                .where(
                    Column("exchange").isin(["NASDAQ","NYSE","AMEX"]),

                    # ===== รายใหญ่เข้า =====
                    Column("market_cap_basic") > 1_000_000_000,
                    Column("market_cap_basic") < 50_000_000_000,

                    Column("average_volume_30d_calc") > 1_000_000,
                    Column("close") > 5,

                    # ===== base filter =====
                    Column("volume") > 500000
                )\
                .limit(200)

            df = q.get_scanner_data()[1]

            if df is None or df.empty:
                return pd.DataFrame()

            return df

        except Exception as e:
            print("Retry...", e)
            time.sleep(2)

    return pd.DataFrame()

# ===== SCORE =====
def calculate_score(row):
    score = 0

    # ===== Volume =====
    if row["relative_volume_10d_calc"] > 2:
        score += 15

    if row["relative_volume_10d_calc"] > 3:
        score += 5

    if row["volume"] > row["average_volume_30d_calc"] * 2:
        score += 15

    if row["close"] * row["volume"] > 20_000_000:
        score += 10

    # ===== Momentum =====
    if row["MACD.macd"] > row["MACD.signal"]:
        score += 10

    if row["close"] > row["EMA20"]:
        score += 10

    if row["close"] > row["EMA50"]:
        score += 10

    # ===== Breakout Proxy =====
    if row["close"] > row["EMA20"] and row["change"] > 2:
        score += 10

    # ===== Timing =====
    if 2 < row["change"] < 5:
        score += 10

    if row["change"] > 7:
        score -= 10

    # ===== Small Cap =====
    if row["market_cap_basic"] < 1_000_000_000:
        score += 5

    # ===== Institutional Flow =====
    dollar_volume = row["close"] * row["volume"]

    if dollar_volume > 50_000_000:
        score += 15

    if dollar_volume > 100_000_000:
        score += 5

    if row["average_volume_30d_calc"] > 2_000_000:
        score += 10

    # กัน fake breakout
    if row["close"] < 5:
        score -= 20

    if row["market_cap_basic"] < 500_000_000:
        score -= 10

    return score

# ===== SCAN =====
def scan():
    df = get_data()

    if df.empty:
        print("⚠️ No data")
        return df

    df["score"] = df.apply(calculate_score, axis=1)

    df = df[df["score"] >= MIN_SCORE]

    df = df.sort_values(by="score", ascending=False).head(TOP_N)

    return df

# ===== FORMAT =====
def format_alert(df):
    msg = "🚀 TOP SIGNAL (ก่อนวิ่ง)\n\n"

    for _, row in df.iterrows():
        msg += (
            f"{row['name']}\n"
            f"💰 {round(row['close'],2)}$\n"
            f"📊 Score: {row['score']}\n"
            f"🔥 Vol x{round(row['relative_volume_10d_calc'],2)}\n"
            f"📈 {round(row['change'],2)}%\n\n"
        )

    return msg

# ===== RUN =====
def run():
    print("🚀 Bot started...")

    while True:
        try:
            df = scan()

            if not df.empty:
                msg = format_alert(df)
                print(msg)
                send_telegram(msg)
            else:
                print("No strong signal")

        except Exception as e:
            print("ERROR:", e)

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run()