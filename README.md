# Vedanta

Vedanta 是一個簡潔實用的加密貨幣交易策略回測與自動化交易服務。

## 專案概述

本專案旨在提供一個全面的解決方案，用於開發、測試和部署加密貨幣交易策略。它結合了強大的回測引擎、豐富的技術指標庫、直觀的結果視覺化工具，以及將策略應用於實際線上自動交易的能力。

## 專案結構

-   `backtest_usage.ipynb`: 示範如何使用回測功能並可視化結果的 Jupyter Notebook。
-   `online_usage.ipynb`: 示範如何使用線上自動交易功能的 Jupyter Notebook。
-   `run.py`: 專案的主要執行入口，展示了如何整合策略進行線上自動交易。
-   `Backtest/`: 包含核心的回測邏輯。
    -   `backtest.py`: 實現回測引擎，計算總回報、最大回撤和勝率等績效指標。
-   `online/`: 包含線上自動交易的相關模組。
    -   `auto_trade.py`: 自動交易的核心邏輯。
    -   `auto_trade_future.py`: 專為期貨交易設計的自動交易邏輯。
-   `Plot/`: 包含繪圖功能。
    -   `plot.py`: 將回測結果可視化，包括權益曲線、帶有持倉的價格和交易回報分佈。
-   `Technicalindicatorstrategy/`: 基於技術指標的交易策略集合。每個文件實現一個特定的策略，並提供一個生成交易信號的函數。
    -   `adx.py`: 平均趨向指標 (ADX) 策略。
    -   `boll.py`: 布林通道 (Bollinger Bands) 策略。
    -   `cci.py`: 商品通道指標 (CCI) 策略。
    -   `ema.py`: 指數移動平均線 (EMA) 交叉策略。
    -   `kd.py`: 隨機指標 (KD) 策略。
    -   `macd.py`: 移動平均收斂發散 (MACD) 策略。
    -   `momentum.py`: 動量策略。
    -   `rsi.py`: 相對強弱指數 (RSI) 策略。
    -   `sma.py`: 簡單移動平均線 (SMA) 交叉策略。
    -   `smartmoney.py`: 使用維加斯通道和船體移動平均線的聰明錢策略。
    -   `williams.py`: 威廉指標 (%R) 策略。

## 如何使用

1.  **安裝依賴套件:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **運行 Jupyter Notebooks:**
    *   打開 `backtest_usage.ipynb` 或 `online_usage.ipynb` 以了解如何導入和使用不同的策略、生成信號、運行回測以及繪製結果圖表。
3.  **配置線上交易 (可選):**
    *   如果需要進行線上自動交易，請根據 `online/` 目錄下的模組和 `run.py` 的範例，配置相關的 API 密鑰和交易參數。

## 依賴套件

-   pandas
-   numpy
-   matplotlib
-   requests
-   ccxt
-   python-dotenv

## 許可證

本專案根據 **GNU General Public License v3.0 (GPL-3.0)** 發布。

這意味著：
*   您可以自由地使用、修改和分發本軟體。
*   如果您分發本軟體的任何部分（無論是否修改），您必須以 GPL-3.0 許可證發布您的作品。
*   您必須提供本許可證的副本以及任何修改的原始碼。

有關完整的許可證條款，請參閱 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)。
