import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------- 參數 ----------------
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
TARGET_WALLET = "0x2078f336fdd260f708bec4a20c82b063274e1b23"

# ---------------- Binance Kline ----------------
def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    time.sleep(0.2)
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_timestamp = int(end_time.timestamp() * 1000)
    remaining = total_limit

    while remaining > 0:
        fetch_limit = min(1000, remaining)
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "endTime": end_timestamp,
            "limit": fetch_limit
        }
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        all_data = data + all_data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

# ---------------- Etherscan API ----------------
def get_wallet_transactions(wallet_address, start_time, end_time, erc20=False, token_list=None):
    wallet_address = wallet_address.lower()
    action = "tokentx" if erc20 else "txlist"

    url = f"https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": action,
        "address": wallet_address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data['status'] != '1':
        return pd.DataFrame(columns=["timestamp","from","to","value","hash","token"])

    tx_list = []
    for tx in data['result']:
        tx_time = pd.to_datetime(int(tx['timeStamp']), unit='s')
        if tx_time < start_time or tx_time > end_time:
            continue
        if erc20:
            token = tx['contractAddress']
            if token_list and token.lower() not in [t.lower() for t in token_list]:
                continue
            value = int(tx['value']) / 10 ** int(tx.get('tokenDecimal', 18))
        else:
            token = None
            value = int(tx['value']) / 1e18
        tx_list.append({
            "timestamp": tx_time,
            "from": tx['from'],
            "to": tx['to'],
            "value": value,
            "token": token,
            "hash": tx['hash']
        })
    df_tx = pd.DataFrame(tx_list)
    if not df_tx.empty:
        df_tx['from'] = df_tx['from'].str.lower()
        df_tx['to'] = df_tx['to'].str.lower()
    return df_tx

# ---------------- 每筆交易都對應 signal ----------------
def map_wallet_to_signals(df_tx, wallet_address):
    if df_tx.empty:
        return pd.DataFrame(columns=["timestamp","direction","value","token","hash"])

    wallet_address = wallet_address.lower()
    df_signals = df_tx.copy()

    # direction: 1 = 收到, -1 = 轉出
    df_signals['direction'] = df_signals.apply(
        lambda row: 1 if row['to'] == wallet_address else -1, axis=1
    )

    return df_signals[["timestamp","direction","value","token","hash"]]

# ---------------- 統一抓取交易信號 ----------------
def get_all_signals(symbol: str, wallet_address=TARGET_WALLET, erc20=False, token_list=None, start_time=None, end_time=None):
    if end_time is None:
        end_time = datetime.utcnow()
    if start_time is None:
        start_time = end_time - timedelta(days=365)  # 預設抓最近一年

    df_tx = get_wallet_transactions(wallet_address, start_time, end_time, erc20=erc20, token_list=token_list)
    df_signals = map_wallet_to_signals(df_tx, wallet_address)
    return df_signals

def attach_kline_to_signals_any_interval(df_signals, symbol="WBTCUSDT", interval="1d", end_time=None):
    if df_signals.empty:
        return pd.DataFrame(columns=["timestamp","signal","value","open","high","low","close","volume"])

    if end_time is None:
        end_time = df_signals['timestamp'].max() + pd.Timedelta(minutes=1)

    # 計算需要抓多少根 K 線
    start_time = df_signals['timestamp'].min()
    interval_mapping = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "8h": 28800,
        "12h": 43200,
        "1d": 86400,
        "3d": 259200,
        "1w": 604800,
        "1M": 2592000
    }
    seconds_per_kline = interval_mapping.get(interval, 86400)
    total_seconds = (end_time - start_time).total_seconds()
    total_limit = int(total_seconds // seconds_per_kline) + 10  # 多抓幾根保險

    # 抓取 K 線
    df_kline = get_binance_kline(symbol, interval, end_time=end_time, total_limit=total_limit)

    if df_kline.empty:
        return pd.DataFrame(columns=["timestamp","signal","value","open","high","low","close","volume"])

    # 對應交易到最近不晚於交易時間的 K 線
    df_kline['timestamp'] = pd.to_datetime(df_kline['timestamp'])
    df_kline = df_kline.sort_values('timestamp')
    df_signals_sorted = df_signals.sort_values('timestamp')
    df_merged = pd.merge_asof(
        df_signals_sorted,
        df_kline,
        left_on='timestamp',
        right_on='timestamp',
        direction='backward'
    )

    # 將 direction 改名為 signal
    df_merged = df_merged.rename(columns={"direction": "signal"})
    df_merged = df_merged[["timestamp","signal","value","open","high","low","close","volume"]]

    return df_merged

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000) -> pd.DataFrame:
    df_all_signals = get_all_signals("WBTCUSDT", wallet_address="0x86b792e6a20c8e8ef56ff4fc92aedcb62dbeefed", erc20=True, token_list=["0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"])
    df_all_signals_kline = attach_kline_to_signals_any_interval(df_all_signals, symbol="WBTCUSDT", interval=interval, end_time=end_time)
    return df_all_signals_kline