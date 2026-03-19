import ccxt
import pandas as pd
import time

symbol = 'BTC/USDT:USDT'   # เปลี่ยนได้
timeframe = '5m'
limit = 200  # max ต่อครั้ง

exchange = ccxt.bitget({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def fetch_all_data():
    all_data = []
    
    since = exchange.parse8601('2025-01-01T00:00:00Z')  # ย้อนหลัง
    
    while True:
        print("Fetching data from:", exchange.iso8601(since))
        
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        
        if len(ohlcv) == 0:
            break
        
        all_data += ohlcv
        
        # update since → candle ถัดไป
        since = ohlcv[-1][0] + 1
        
        time.sleep(exchange.rateLimit / 1000)
        
        # กัน loop ยาวเกิน (optional)
        if len(all_data) > 10000:
            break
    
    return all_data

def save_csv(data):
    df = pd.DataFrame(data, columns=[
        'timestamp','open','high','low','close','volume'
    ])
    
    df.to_csv("data.csv", index=False)
    print("Saved data.csv")

if __name__ == "__main__":
    data = fetch_all_data()
    save_csv(data)