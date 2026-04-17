import requests
import pandas as pd
from tradingview_screener import Query, Column

# ===== CONFIG =====
TELEGRAM_TOKEN = "8559685503:AAGeY-RoyFG7SCKNB4w6nCMe5AeiQ-4mkoY"
CHAT_ID = "-1003805957111"

TOP_N = 5
MIN_SCORE = 70


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})


def get_data():
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
            "high_20d",
            "change"
        ) \
        .where(
            Column("market_cap_basic") < 5_000_000_000,
            Column("volume") > 500000
        ) \
        .limit(200)

    df = q.get_scanner_data(market="america")[1]
    return df


def calculate_score(row):
    score = 0

    # ===== Volume & Flow =====
    if row["relative_volume_10d_calc"] > 2:
        score += 15

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

    # ===== Breakout =====
    if row["close"] > row["high_20d"]:
        score += 15

    # ===== Timing =====
    if 2 < row["change"] < 5:
        score += 10

    return score


def scan():
    df = get_data()

    df["score"] = df.apply(calculate_score, axis=1)

    df = df[df["score"] >= MIN_SCORE]

    df = df.sort_values(by="score", ascending=False).head(TOP_N)

    return df


def format_alert(df):
    msg = "🚀 TOP PRE-MARKET SIGNAL (หุ้นจะวิ่ง)\n\n"

    for _, row in df.iterrows():
        msg += (
            f"{row['name']}\n"
            f"💰 {row['close']}$ | 📊 Score: {row['score']}\n"
            f"🔥 Vol x{round(row['relative_volume_10d_calc'],2)} | +{row['change']}%\n\n"
        )

    return msg


if __name__ == "__main__":
    df = scan()

    if not df.empty:
        msg = format_alert(df)
        send_telegram(msg)
        print(msg)
    else:
        print("No strong signal")