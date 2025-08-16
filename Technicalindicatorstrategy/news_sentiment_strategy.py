import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
import time

# =========================
# ✅ 初始化 Gemini API
# =========================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("請設置環境變數 GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# ✅ 使用 Gemini 判斷情緒
# =========================
def analyze_sentiment_with_gemini(title: str) -> str:
    prompt = f"""請判斷下列加密貨幣新聞標題的情緒，只回覆 'positive'、'neutral' 或 'negative'：
新聞標題：{title}"""
    try:
        response = gemini_model.generate_content(prompt)
        sentiment = response.text.strip().lower()
        if sentiment in ["positive", "neutral", "negative"]:
            print(f"[Gemini] {title} → {sentiment}")
            # 呼叫成功後延遲，避免速率限制
            time.sleep(5)
            return sentiment
    except Exception as e:
        print(f"Gemini 分析錯誤：{e}")
    return "neutral"

# =========================
# ✅ 取得新聞情緒資料
# =========================
def get_news_sentiment_from_cryptopanic(days: int, end_time: datetime) -> pd.DataFrame:
    all_news = []
    page = 1
    cutoff = end_time.replace(tzinfo=timezone.utc) - timedelta(days=days)
    url = "https://cryptopanic.com/api/developer/v2/posts/"  # 新版路徑

    while True:
        params = {
            "auth_token": os.getenv('cryptopanic_auth_token'),
            "public": "true",
            "filter": "hot",
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"CryptoPanic API failed: {response.status_code} {response.text}")

        data = response.json().get("results", [])
        if not data:
            break

        for item in data:
            published_at = pd.to_datetime(item["published_at"])
            if published_at > end_time.replace(tzinfo=timezone.utc):
                continue
            if published_at < cutoff:
                continue

            title = item.get("title", "")
            sentiment = analyze_sentiment_with_gemini(title)  # 用 Gemini 判斷
            score = {"positive": 1, "negative": -1, "neutral": 0}.get(sentiment, 0)
            all_news.append({"timestamp": published_at, "score": score})

        page += 1
        time.sleep(1)

    df = pd.DataFrame(all_news)
    if df.empty:
        raise Exception("No news data returned.")

    df["timestamp"] = df["timestamp"].dt.floor("D")
    sentiment_daily = df.groupby("timestamp")["score"].mean().reset_index()
    sentiment_daily = sentiment_daily.rename(columns={"score": "sentiment_score"})

    return sentiment_daily


# =========================
# ✅ 取得 Kline 資料
# =========================
def get_binance_kline(symbol: str, interval: str, end_time: datetime, limit: int = 300) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    end_ts = int(end_time.timestamp() * 1000)
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "endTime": end_ts,
        "limit": limit,
    }
    res = requests.get(base_url, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]

# =========================
# ✅ 合併情緒與價格產生訊號
# =========================
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 30) -> pd.DataFrame:
    price_df = get_binance_kline(symbol, interval, end_time, limit)
    start_time = price_df["timestamp"].iloc[0]
    days_needed = max(30, (end_time.date() - start_time.date()).days + 1)

    sentiment_df = get_news_sentiment_from_cryptopanic(days_needed, end_time)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], utc=True)

    # 對齊資料（backward 合併）
    merged = pd.merge_asof(price_df.sort_values("timestamp"),
                           sentiment_df.sort_values("timestamp"),
                           on="timestamp", direction="backward")

    merged["sentiment_score"] = merged["sentiment_score"].fillna(0)
    merged["sentiment_sma3"] = merged["sentiment_score"].rolling(3).mean()
    merged["sentiment_sma7"] = merged["sentiment_score"].rolling(7).mean()

    # Initialize position tracking
    current_position = 0
    positions = []
    signals = [] # To store the final signal (entry/exit)

    # Iterate through the merged DataFrame to determine position
    for i in range(len(merged)):
        curr_sma3 = merged.loc[i, "sentiment_sma3"]
        curr_sma7 = merged.loc[i, "sentiment_sma7"]

        # Handle NaN values for initial bars where SMA is not yet calculated
        if pd.isna(curr_sma3) or pd.isna(curr_sma7):
            signals.append(0)
            positions.append(0)
            continue

        prev_sma3 = merged.loc[i-1, "sentiment_sma3"] if i > 0 else np.nan
        prev_sma7 = merged.loc[i-1, "sentiment_sma7"] if i > 0 else np.nan

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if SMA3 crosses below SMA7
            if curr_sma3 < curr_sma7 and (pd.isna(prev_sma3) or prev_sma3 >= prev_sma7):
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if SMA3 crosses above SMA7
            if curr_sma3 > curr_sma7 and (pd.isna(prev_sma3) or prev_sma3 <= prev_sma7):
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: SMA3 crosses above SMA7
            if curr_sma3 > curr_sma7 and (pd.isna(prev_sma3) or prev_sma3 <= prev_sma7):
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: SMA3 crosses below SMA7
            elif curr_sma3 < curr_sma7 and (pd.isna(prev_sma3) or prev_sma3 >= prev_sma7):
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    merged["signal"] = signals
    merged["position"] = positions

    return merged[["timestamp", "close", "sentiment_score", "sentiment_sma3", "sentiment_sma7", "signal", "position"]]

# =========================
# ✅ 範例使用
# =========================
if __name__ == "__main__":
    df = get_signals("BTCUSDT", "1h", datetime.utcnow(), limit=100)
    print(df.tail(10))
