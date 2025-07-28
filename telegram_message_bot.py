from binance.client import Client
import pandas as pd
import ta
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Binance API Key (可為空)
client = Client(api_key='', api_secret='')

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    apiURL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    try:
        response = requests.post(apiURL, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        })
        print(response.text)
    except Exception as e:
        print(e)

def get_top_symbols(limit=100, quote_asset='USDT'):
    tickers = client.get_ticker()
    usdt_pairs = [t for t in tickers if t['symbol'].endswith(quote_asset) and not t['symbol'].endswith('BULLUSDT') and not t['symbol'].endswith('BEARUSDT')]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_pairs[:limit]]

def fetch_klines(symbol, interval, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        return df
    except:
        return None

def check_ma_conditions(df, cross=False):
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    if cross:
        # 檢查是否剛剛出現黃金交叉（ma5剛上穿ma10）
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return prev['ma5'] < prev['ma10'] and curr['ma5'] > curr['ma10']
    else:
        latest = df.iloc[-1]
        return latest['ma5'] > latest['ma10'] > latest['ma20']

def analyze_symbol(symbol):
    timeframes = {
        '1d': False,
        '4h': False,
        '1h': False,
        '15m': False,
    }

    df_1d = fetch_klines(symbol, Client.KLINE_INTERVAL_1DAY)
    df_4h = fetch_klines(symbol, Client.KLINE_INTERVAL_4HOUR)
    df_1h = fetch_klines(symbol, Client.KLINE_INTERVAL_1HOUR)
    df_15m = fetch_klines(symbol, Client.KLINE_INTERVAL_15MINUTE)

    if df_1d is not None and check_ma_conditions(df_1d):
        timeframes['1d'] = True
    if df_4h is not None and check_ma_conditions(df_4h):
        timeframes['4h'] = True
    if df_1h is not None and check_ma_conditions(df_1h):
        timeframes['1h'] = True
    if df_15m is not None and check_ma_conditions(df_15m, cross=True):
        timeframes['15m'] = True

    return all(timeframes.values())

def main():
    passing_symbols = []
    top_symbols = get_top_symbols()

    for symbol in top_symbols:
        print(f"分析 {symbol}...")
        try:
            if analyze_symbol(symbol):
                print(f"{symbol} 通過條件")
                passing_symbols.append(symbol)
        except Exception as e:
            print(f"{symbol} 分析失敗: {e}")
        time.sleep(0.5)  # 防止請求過快

    if passing_symbols:
        message = f"以下幣種符合條件:\n" + "\n".join(passing_symbols)
    else:
        message = "目前無幣種符合條件"

    send_telegram_message(message)

if __name__ == "__main__":
    main()