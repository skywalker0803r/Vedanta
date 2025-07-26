# Vedanta

Vedanta 是一個簡潔實用的加密貨幣交易策略回測服務。

## 專案結構

- `usage.ipynb`: 一個 Jupyter Notebook，示範如何使用各種交易策略並將回測結果可視化。
- `Backtest/`: 包含核心的回測邏輯。
  - `backtest.py`: 實現回測引擎，計算總回報、最大回撤和勝率等績效指標。
- `Plot/`: 包含繪圖功能。
  - `plot.py`: 將回測結果可視化，包括權益曲線、帶有持倉的價格和交易回報分佈。
- `Technicalindicatorstrategy/`: 基於技術指標的交易策略集合。每個文件實現一個特定的策略，並提供一個生成交易信號的函數。
  - `adx.py`: 平均趨向指標 (ADX) 策略。
  - `boll.py`: 布林通道 (Bollinger Bands) 策略。
  - `cci.py`: 商品通道指標 (CCI) 策略。
  - `ema.py`: 指數移動平均線 (EMA) 交叉策略。
  - `kd.py`: 隨機指標 (KD) 策略。
  - `macd.py`: 移動平均收斂發散 (MACD) 策略。
  - `momentum.py`: 動量策略。
  - `rsi.py`: 相對強弱指數 (RSI) 策略。
  - `sma.py`: 簡單移動平均線 (SMA) 交叉策略。
  - `smartmoney.py`: 使用維加斯通道和船體移動平均線的聰明錢策略。
  - `williams.py`: 威廉指標 (%R) 策略。

## 如何使用

1. **安裝依賴套件:**
   ```bash
   pip install -r requirements.txt
   ```
2. **運行 `usage.ipynb` notebook:**
   此 notebook 展示了如何導入和使用不同的策略、生成信號、運行回測以及繪製結果圖表。

## 依賴套件

- pandas
- numpy
- matplotlib
- requests