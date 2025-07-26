import requests
import pandas as pd
from datetime import datetime

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

def get_defillama_tvl(protocol: str) -> pd.DataFrame:
    url = f"https://api.llama.fi/protocol/{protocol}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    tvl_data = data["tvl"]
    df = pd.DataFrame(tvl_data)
    df["timestamp"] = pd.to_datetime(df["date"], unit="s")
    df = df.rename(columns={"totalLiquidityUSD": "tvl"})
    df = df[["timestamp", "tvl"]]
    df["tvl"] = df["tvl"].astype(float)
    return df

def get_chain_tvl(chain_name: str) -> pd.DataFrame:
    """
    從 DefiLlama v2 抓取指定鏈的 TVL 時序資料
    :param chain_name: 鏈名稱，例如 'ethereum', 'arbitrum', 'solana'
    :return: 包含 timestamp 與 tvl 的 DataFrame
    """
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain_name.lower()}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()  # list of dicts with 'date' and 'tvl'

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["date"], unit="s")
    df["tvl"] = df["tvl"].astype(float)
    return df[["timestamp", "tvl"]]

def detect_divergence_signal(df: pd.DataFrame, price_col="close", tvl_col="tvl", window=5) -> pd.DataFrame:
    df = df.copy()

    # 計算 rolling 最大最小
    df["price_max"] = df[price_col].rolling(window).max()
    df["price_min"] = df[price_col].rolling(window).min()
    df["tvl_max"] = df[tvl_col].rolling(window).max()
    df["tvl_min"] = df[tvl_col].rolling(window).min()

    df["signal"] = 0

    # 賣出：價格創高，TVL 沒創高
    cond_sell = (df[price_col] == df["price_max"]) & (df[tvl_col] < df["tvl_max"])

    # 買進：價格創低，TVL 沒創低
    cond_buy = (df[price_col] == df["price_min"]) & (df[tvl_col] > df["tvl_min"])
    df.loc[cond_sell, "signal"] = -1
    df.loc[cond_buy, "signal"] = 1

    # 清理中間欄位
    df.drop(columns=["price_max", "price_min", "tvl_max", "tvl_min"], inplace=True)

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300,
                protocol: str = "aave", window: int = 5) -> pd.DataFrame:
    # 1. 取得價格資料
    price_df = get_binance_kline(symbol, interval, end_time, limit)
    
    # 2. 取得 TVL 資料
    try:
        tvl_df = get_defillama_tvl(protocol)
    except:
        tvl_df = get_chain_tvl(protocol)
    
    # 3. 合併價格與 TVL，對齊最近不晚於價格時間點的 TVL
    merged = pd.merge_asof(price_df.sort_values("timestamp"),
                           tvl_df.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    # 4. 計算背離訊號
    merged = detect_divergence_signal(merged, price_col="close", tvl_col="tvl", window=window)
    return merged[["timestamp", "close", "tvl", "signal"]]

if __name__ == '__main__':
    df_signals = get_signals("BTCUSDT", "1h", datetime.now(), limit=200, protocol="aave", window=5)
    print(df_signals.tail(20))
    print(df_signals["signal"].value_counts())
