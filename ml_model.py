import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def prepare(df_raw):
    df = pd.DataFrame(df_raw, columns=['ts','o','h','l','c','v'])
    
    df['return'] = df['c'].pct_change()
    df['rsi'] = ta.momentum.rsi(df['c'], 14)
    df['atr'] = ta.volatility.average_true_range(df['h'], df['l'], df['c'], 14)
    
    df['direction'] = (df['c'].shift(-1) > df['c']).astype(int)
    
    return df.dropna()

def train(df):
    X = df[['return','rsi','atr']]
    
    clf = RandomForestClassifier()
    clf.fit(X, df['direction'])
    
    reg = RandomForestRegressor()
    reg.fit(X, df['atr'])
    
    return clf, reg

def predict(clf, reg, df):
    X = df[['return','rsi','atr']].tail(1)
    
    direction = clf.predict(X)[0]
    vol = reg.predict(X)[0]
    
    return direction, vol