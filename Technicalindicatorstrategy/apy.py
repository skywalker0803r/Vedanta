import requests
import pandas as pd
from datetime import datetime, timedelta

def get_token_related_pools(token: str, chain: str = "Ethereum") -> pd.DataFrame:
    url = "https://yields.llama.fi/pools"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()["data"]

    filtered_pools = [
        {
            "project": p["project"],
            "symbol": p["symbol"],
            "pool_id": p["pool"],
            "apy": p["apy"],
            "tvlUsd": p["tvlUsd"]
        }
        for p in data
        if p["chain"] == chain and token.upper() in p["symbol"].split("-")
    ]
    return pd.DataFrame(filtered_pools).sort_values(by='apy', ascending=False).head(1)

def get_binance_kline(symbol: str, interval: str, end_time: datetime, limit: int = 300) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    end_timestamp = int(end_time.timestamp() * 1000)
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "endTime": end_timestamp,
        "limit": limit
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]

def get_yield_history(pool_id: str) -> pd.DataFrame:
    url = f"https://yields.llama.fi/chart/{pool_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # 這裡已經是 datetime 格式
    return df[["timestamp", "apy"]]

def detect_apy_signals(apy_df: pd.DataFrame, short_window: int = 1, long_window: int = 3) -> pd.DataFrame:
    df = apy_df.copy()
    df = df.sort_values("timestamp")
    df["apy_sma_short"] = df["apy"].rolling(window=short_window).mean()
    df["apy_sma_long"] = df["apy"].rolling(window=long_window).mean()

    df["signal"] = 0
    df["prev_short"] = df["apy_sma_short"].shift(1)
    df["prev_long"] = df["apy_sma_long"].shift(1)

    df.loc[(df["apy_sma_short"] > df["apy_sma_long"]) & (df["prev_short"] <= df["prev_long"]), "signal"] = 1
    df.loc[(df["apy_sma_short"] < df["apy_sma_long"]) & (df["prev_short"] >= df["prev_long"]), "signal"] = -1

    # ✅ 確保 timestamp 為 naive datetime
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return df[["timestamp", "signal"]]


def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, n1: int = 1, n2: int = 3) -> pd.DataFrame:
    # Step 1: 抓 Binance K 線
    price_df = get_binance_kline(symbol, interval, end_time, limit)

    # Step 2: 取得相關池子
    token = symbol.replace("USDT", "").replace("BUSD", "")  # 例如 ENAUSDT -> ENA
    pool_df = get_token_related_pools(token)
    if pool_df.empty:
        raise ValueError(f"No related pools found for token: {token}")
    pool_id = pool_df.iloc[0]["pool_id"]

    # Step 3: 抓取該池子的 APY 歷史資料
    apy_df = get_yield_history(pool_id)

    # Step 4: APY 分析產生訊號
    apy_signal_df = detect_apy_signals(apy_df, short_window=n1, long_window=n2)

    print("\n📈 Price Timestamp Info:")
    print("起始時間:", price_df["timestamp"].min())
    print("結束時間:", price_df["timestamp"].max())
    print("筆數:", len(price_df))
    print("價格資料間隔（秒）:", price_df["timestamp"].diff().dropna().dt.total_seconds().mode()[0])

    print("\n📊 APY Timestamp Info:")
    print("起始時間:", apy_df["timestamp"].min())
    print("結束時間:", apy_df["timestamp"].max())
    print("筆數:", len(apy_df))
    print("APY 資料間隔（秒）:", apy_df["timestamp"].diff().dropna().dt.total_seconds().mode()[0])



    # Step 5: 合併價格與訊號資料（以時間對齊）
    merged = pd.merge_asof(
        price_df.sort_values("timestamp"),
        apy_signal_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    print(merged)

    return merged.fillna({"signal": 0})