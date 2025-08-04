
import redis
import pandas as pd
import json
import time

def update_features(tick):
    window = json.loads(redis.Redis().get('ohlcv_window') or '[]')
    window.append(tick)
    if len(window) > 50:
        window.pop(0)
    redis.Redis().set('ohlcv_window', json.dumps(window))

    df = pd.DataFrame(window)
    # Example features
    df['sma'] = df['close'].rolling(14).mean().fillna(method='bfill')
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    features = df.iloc[-1][['sma', 'atr']].to_dict()
    redis.Redis().set('latest_features', json.dumps(features))

def stream_ticks():
    import random
    while True:
        tick = {
            'timestamp': time.time(),
            'open': random.random()*100,
            'high': random.random()*100,
            'low': random.random()*100,
            'close': random.random()*100,
            'volume': random.randint(100, 1000)
        }
        update_features(tick)
        time.sleep(1)

if __name__ == "__main__":
    stream_ticks()
