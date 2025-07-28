from binance.client import Client
import pandas as pd
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

# 負責將分析結果推送到你的 Telegram。
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


#取得成交量最高的 USDT 交易對，過濾掉 BULL/BEAR 等槓桿代幣。
def get_top_symbols(limit=100, quote_asset='USDT'):
    tickers = client.get_ticker()
    usdt_pairs = [
        t for t in tickers if t['symbol'].endswith(quote_asset)
        and not t['symbol'].endswith('BULLUSDT')
        and not t['symbol'].endswith('BEARUSDT')
    ]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_pairs[:limit]]

#用來拉取 K 線數據，轉成 Pandas DataFrame 並處理型別轉換。
def fetch_klines(symbol, interval, limit=200):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
        ])

        # ✅ 把用到的欄位都轉為 float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df
    except:
        return None

#使用的是 Vegas 策略，透過 ema144 和 ema169 構成的範圍作為過濾依據。
def check_vegas_conditions(df):
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df = df.dropna()
    
    if len(df) < 2:
        return False, ""

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    vegas_low = min(curr['ema144'], curr['ema169'])
    vegas_high = max(curr['ema144'], curr['ema169'])

    breakout = prev['close'] < vegas_low and curr['close'] > vegas_high
    bounce = (
        prev['close'] > vegas_high and
        curr['close'] > vegas_high and
        min(curr['low'], prev['low']) <= vegas_high
    )

    if breakout:
        return True, "突破"
    elif bounce:
        return True, "回踩反彈"
    return False, ""

#使用的是 Vegas 策略，透過 ema144 和 ema169 構成的範圍作為過濾依據。
def check_vegas_short_conditions(df):
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df = df.dropna()

    if len(df) < 2:
        return False, ""

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    vegas_low = min(curr['ema144'], curr['ema169'])
    vegas_high = max(curr['ema144'], curr['ema169'])

    breakdown = prev['close'] > vegas_high and curr['close'] < vegas_low
    fail_bounce = (
        prev['close'] < vegas_low and
        curr['close'] < vegas_low and
        max(curr['high'], prev['high']) >= vegas_low
    )

    if breakdown:
        return True, "跌破"
    elif fail_bounce:
        return True, "反彈失敗"
    return False, ""


#對每個幣種進行分析，回傳是否符合多單或空單條件與原因。
def analyze_symbol(symbol):
    result = {
        'symbol': symbol,
        'long': False,
        'short': False,
        'long_reason': "",
        'short_reason': ""
    }

    df_1h = fetch_klines(symbol, Client.KLINE_INTERVAL_1HOUR)  # 修改為1小時

    if df_1h is not None:
        long_pass, long_reason = check_vegas_conditions(df_1h)
        short_pass, short_reason = check_vegas_short_conditions(df_1h)

        if long_pass:
            result['long'] = True
            result['long_reason'] = long_reason

        if short_pass:
            result['short'] = True
            result['short_reason'] = short_reason

    return result

#驅動整個流程，循環處理每個幣種、分析、通知。
def main():
    long_symbols = []
    short_symbols = []
    top_symbols = get_top_symbols()

    for symbol in top_symbols:
        print(f"分析 {symbol}...")
        try:
            result = analyze_symbol(symbol)
            if result['long']:
                print(f"{symbol} 多單訊號 - {result['long_reason']}")
                long_symbols.append(f"{symbol} ({result['long_reason']})")
            if result['short']:
                print(f"{symbol} 空單訊號 - {result['short_reason']}")
                short_symbols.append(f"{symbol} ({result['short_reason']})")
        except Exception as e:
            print(f"{symbol} 分析失敗: {e}")
        time.sleep(0.5)

    message = ""

    if long_symbols:
        message += "📈 *符合 Vegas 多單條件的幣種:*\n" + "\n".join(long_symbols) + "\n\n"
    if short_symbols:
        message += "📉 *符合 Vegas 空單條件的幣種:*\n" + "\n".join(short_symbols)

    if not message:
        message = "❌ 目前無幣種符合 Vegas 多單或空單條件"

    send_telegram_message(message)



if __name__ == "__main__":
    main()