import time
from exchange import *
from ml_model import *
from mtf import *
from risk import *
from config import *

class MLGridBot:
    def __init__(self):
        self.clf = None
        self.reg = None
    
    def train(self):
        data = get_ohlcv('1m')
        df = prepare(data)
        self.clf, self.reg = train(df)
    
    def get_position(self):
        pos = get_positions()
        for p in pos:
            if float(p['contracts']) > 0:
                return p
        return None
    
    def run(self):
        print("Training model...")
        self.train()
        
        while True:
            try:
                # --- MTF FILTER ---
                mtf_signal = get_mtf_signal()
                
                if mtf_signal == 'neutral':
                    print("No trade condition")
                    time.sleep(30)
                    continue
                
                # --- ML ---
                data = get_ohlcv('1m')
                df = prepare(data)
                
                price = df['c'].iloc[-1]
                atr = df['atr'].iloc[-1]
                
                direction, vol = predict(self.clf, self.reg, df)
                
                spacing = max(0.5, min(vol, 10))
                
                bias = 'long' if direction == 1 else 'short'
                
                # ต้องตรงกับ MTF
                if bias != mtf_signal:
                    print("ML not aligned with MTF")
                    time.sleep(30)
                    continue
                
                print(f"Price {price} | Bias {bias} | Spacing {spacing}")
                
                cancel_all()
                
                size = calc_size(BALANCE, atr)
                
                # --- GRID ---
                orders = 0
                for i in range(1, GRID_LEVELS+1):
                    if orders >= MAX_ORDERS:
                        break
                    
                    if bias == 'long':
                        price_level = price - i * spacing
                        side = 'buy'
                    else:
                        price_level = price + i * spacing
                        side = 'sell'
                    
                    try:
                        place_limit(side, size, price_level)
                        orders += 1
                    except:
                        pass
                
                # --- SL / TP ---
                pos = self.get_position()
                
                if pos:
                    entry = float(pos['entryPrice'])
                    size = float(pos['contracts'])
                    side = pos['side']
                    
                    sl, tp = get_sl_tp(entry, atr, side)
                    
                    try:
                        place_limit(
                            'sell' if side == 'long' else 'buy',
                            size, tp
                        )
                        
                        place_market(
                            'sell' if side == 'long' else 'buy',
                            size,
                            {'stopPrice': sl}
                        )
                    except:
                        pass
                
                time.sleep(30)
            
            except Exception as e:
                print("error:", e)
                time.sleep(10)

if __name__ == "__main__":
    MLGridBot().run()