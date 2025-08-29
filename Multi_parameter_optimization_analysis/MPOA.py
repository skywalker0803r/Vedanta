import optuna
import numpy as np
import pandas as pd
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from tqdm import tqdm
import warnings
from datetime import datetime
import multiprocessing
from Backtest.backtest import backtest_signals
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import inspect
import importlib.util
import hiplot as hip
import plotly.graph_objects as go
from numbers import Number  # 用於檢查是否為數值類型

class UniqueParamsSampler(optuna.samplers.TPESampler):
    """自定義採樣器，避免使用已存在的參數組合"""
    def __init__(self, used_params, **kwargs):
        super().__init__(**kwargs)
        self.used_params = used_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        """重寫獨立參數採樣，檢查是否重複"""
        while True:
            value = super().sample_independent(study, trial, param_name, param_distribution)
            # 使用當前試驗參數，避免訪問 best_params
            trial_params = {p: trial.params.get(p, None) for p in trial.params.keys()}
            trial_params[param_name] = value
            param_tuple = tuple(sorted((k, v) for k, v in trial_params.items() if v is not None))
            if param_tuple not in self.used_params or not trial_params:
                return value

    def sample_relative(self, study, trial, search_space):
        """重寫相對參數採樣，檢查是否重複"""
        while True:
            params = super().sample_relative(study, trial, search_space)
            param_tuple = tuple(sorted((k, v) for k, v in params.items()))
            if not param_tuple or param_tuple not in self.used_params:
                self.used_params.add(param_tuple)
                return params

    def sample_relative(self, study, trial, search_space):
        """重寫相對參數採樣，檢查是否重複"""
        while True:
            params = super().sample_relative(study, trial, search_space)
            param_tuple = tuple(sorted((k, v) for k, v in params.items()))
            if not param_tuple or param_tuple not in self.used_params:
                self.used_params.add(param_tuple)
                return params
            
class FlexibleStrategyOptimizer:
    """通用策略參數優化器"""
    
    def __init__(self, strategy_config):
        """
        初始化優化器
        
        Args:
            strategy_config (dict): 策略配置
                {
                    'strategy_module': 策略模組對象,
                    'strategy_function': 策略函數名稱 (str),
                    'fixed_params': 固定參數字典,
                    'optimize_params': 優化參數配置,
                    'target_metrics': 目標指標列表,
                    'backtest_config': 回測配置
                }
        """
        self.strategy_config = strategy_config
        self.validate_config()
        
    def validate_config(self):
        """驗證配置的有效性"""
        required_keys = ['strategy_module', 'strategy_function', 'optimize_params', 'target_metrics']
        for key in required_keys:
            if key not in self.strategy_config:
                raise ValueError(f"Missing required config key: {key}")
                
        # 獲取策略函數
        strategy_module = self.strategy_config['strategy_module']
        strategy_function_name = self.strategy_config['strategy_function']
        
        if not hasattr(strategy_module, strategy_function_name):
            raise ValueError(f"Strategy function '{strategy_function_name}' not found in module")
            
        self.strategy_function = getattr(strategy_module, strategy_function_name)
        
        # 檢查策略函數的參數簽名
        sig = inspect.signature(self.strategy_function)
        self.function_params = list(sig.parameters.keys())
        print(f"檢測到策略函數參數: {self.function_params}")
        
    def trading_strategy(self, trial_params, **kwargs):
        """
        通用交易策略測試函數
        
        Args:
            trial_params (dict): 試驗參數
            **kwargs: 額外參數
        """
        try:
            # 合併固定參數和試驗參數
            all_params = self.strategy_config.get('fixed_params', {}).copy()
            all_params.update(trial_params)
            all_params.update(kwargs)
            
            # 動態調用策略函數
            df_signals = self.strategy_function(**all_params)
            
            # 執行回測
            backtest_config = self.strategy_config.get('backtest_config', {
                'initial_capital': 1000000,
                'fee_rate': 0.0005,
                'leverage': 1,
                'allow_short': True,
                'stop_loss': None,
                'take_profit': None,
                'capital_ratio': 1,
                'max_hold_bars': 10000,
                'delay_entry': False,
                'risk_free_rate': 0.02,
                'interval': '1h'
            })
            
            result = backtest_signals(df_signals.copy(), **backtest_config)
            
            # 提取目標指標
            metrics = result.get('float_type_metrics', {})
            target_metrics = self.strategy_config['target_metrics']
            
            # 返回主要優化指標（第一個指標）
            primary_metric = target_metrics[0]
            primary_value = metrics.get(primary_metric, 0.0)
            
            # 將所有指標存儲在trial的用戶屬性中
            trial_metrics = {}
            for metric in target_metrics:
                value = metrics.get(metric, 0.0)
                trial_metrics[metric] = value if not np.isnan(value) else 0.0
            
            return primary_value if not np.isnan(primary_value) else 0.0, trial_metrics
        except Exception as e:
            print(f"Error in trading_strategy: {e}")
            return 0.0, {}

    def objective(self, trial):
        """Optuna 目標函數"""
        trial_params = {}
        
        # 根據配置設置試驗參數
        for param_name, param_config in self.strategy_config['optimize_params'].items():
            param_type = param_config['type']
            
            if param_type == 'int':
                value = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                value = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', None)
                )
            elif param_type == 'categorical':
                value = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
                
            trial_params[param_name] = value
        
        primary_value, trial_metrics = self.trading_strategy(trial_params)
        
        # 將其他指標存儲為用戶屬性
        for metric_name, metric_value in trial_metrics.items():
            trial.set_user_attr(metric_name, metric_value)
            
        return primary_value

    def run_optimization(self, n_trials=250, n_jobs=-1, study_name=None):
        """運行參數優化，避免使用資料庫中已有的參數組合"""
        print("開始靈活參數優化...")
        
        if study_name is None:
            study_name = f"{self.strategy_config['strategy_function']}_optimization"
            
        storage = f'sqlite:///{study_name}.db'
        
        # 檢查資料庫文件
        import os
        if os.path.exists(study_name + '.db'):
            print(f"檢測到現有資料庫: {study_name}.db")
        else:
            print(f"創建新資料庫: {study_name}.db")
        
        # 確定優化方向
        primary_metric = self.strategy_config['target_metrics'][0]
        direction = 'maximize' if primary_metric in ['Sharpe Ratio', 'Sortino Ratio'] else 'minimize'
        
        # 加載現有試驗的參數組合
        used_params = set()
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            for trial in study.trials:
                if trial.params:
                    param_tuple = tuple(sorted((k, v) for k, v in trial.params.items()))
                    used_params.add(param_tuple)
            print(f"已載入 study，包含 {len(study.trials)} 個試驗，已使用 {len(used_params)} 個參數組合")
        except Exception as e:
            print(f"載入資料庫失敗：{e}")
            print("創建新資料庫...")
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
                study_name=study_name,
                storage=storage
            )
        
        # 使用自定義採樣器
        sampler = UniqueParamsSampler(used_params=used_params, seed=42)
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"已載入 study，包含 {len(study.trials)} 個試驗")
        except Exception as e:
            print(f"載入資料庫失敗：{e}")
            print("創建新資料庫...")
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(),
                study_name=study_name,
                storage=storage
            )
        
        if len(study.trials) < 10:
            print(f"警告：當前試驗數量 ({len(study.trials)}) 較少，建議至少運行 50 次試驗以確保分析可靠性")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.study = study
        self.print_optimization_results()
        
        return study
    
    def print_optimization_results(self):
        """打印優化結果"""
        print(f"\nOptimization completed!")
        print(f"Best {self.strategy_config['target_metrics'][0]}: {self.study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
            
        # 打印所有指標的最佳值
        best_trial = self.study.best_trial
        print(f"\nAll metrics for best trial:")
        for metric in self.strategy_config['target_metrics']:
            if metric in best_trial.user_attrs:
                print(f"  {metric}: {best_trial.user_attrs[metric]:.4f}")

    def analyze_multi_metric_results(self):
        """多指標分析"""
        if not hasattr(self, 'study'):
            print("Please run optimization first!")
            return
            
        trials_df = self.study.trials_dataframe()
        target_metrics = self.strategy_config['target_metrics']
        
        # 添加用戶屬性到 DataFrame
        for metric in target_metrics:
            metric_values = []
            for trial in self.study.trials:
                value = trial.user_attrs.get(metric, np.nan)
                metric_values.append(value)
            trials_df[f'metric_{metric}'] = metric_values
        
        # 多指標相關性分析
        metric_columns = [f'metric_{metric}' for metric in target_metrics]
        metric_data = trials_df[metric_columns].dropna()
        
        if len(metric_data) > 0:
            # 重命名列
            metric_data.columns = target_metrics
            correlation_matrix = metric_data.corr()
            
            # 打印相關性統計
            print("\n多指標相關性分析:")
            for i, metric1 in enumerate(target_metrics):
                for j, metric2 in enumerate(target_metrics[i+1:], i+1):
                    corr = correlation_matrix.loc[metric1, metric2]
                    print(f"  {metric1} vs {metric2}: {corr:.3f}")
        
        return trials_df
    
    

    def plot_hiplot(self, output_html="optimization_results.html"):
        """使用 HiPlot 視覺化優化結果"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return

        trials_data = []
        for trial in self.study.trials:
            trial_data = {}
            # 添加參數
            for param_name in self.strategy_config['optimize_params'].keys():
                value = trial.params.get(param_name, None)
                trial_data[param_name] = value

            # 添加指標
            for metric in self.strategy_config['target_metrics']:
                value = trial.user_attrs.get(metric, None)
                # 檢查是否為數值類型並處理 NaN
                if isinstance(value, Number) and not np.isnan(value):
                    trial_data[metric] = value
                else:
                    trial_data[metric] = None

            # 僅保留完整數據（無 None）
            if all(v is not None for v in trial_data.values()):
                trials_data.append(trial_data)

        if not trials_data:
            print("無有效的數據可用於 HiPlot 視覺化！")
            return

        # 創建 HiPlot 實驗
        exp = hip.Experiment.from_iterable(trials_data)
        exp.display()
        exp.to_html(output_html)
        print(f"HiPlot 視覺化已保存至 {output_html}")

    def analyze_param_metric_correlations(self):
        """分析 optimize_params 之間的相關性與對 target_metrics 的影響"""
        if not hasattr(self, 'study'):
            print("請先運行優化！")
            return None
            
        trials_df = self.study.trials_dataframe()
        print(f"trials_df 欄位: {list(trials_df.columns)}")
        print(f"試驗數量: {len(trials_df)}")
        
        target_metrics = self.strategy_config['target_metrics']
        
        # 添加用戶屬性到 DataFrame，並將 user_attrs_ 轉為 metric_
        for metric in target_metrics:
            metric_values = [trial.user_attrs.get(metric, np.nan) for trial in self.study.trials]
            trials_df[f'metric_{metric}'] = metric_values
        
        # 明確選擇參數和指標欄位
        param_columns = [f'params_{param}' for param in self.strategy_config['optimize_params'].keys()]
        metric_columns = [f'metric_{metric}' for metric in target_metrics]
        numeric_columns = param_columns + metric_columns
        
        # 檢查欄位是否存在
        missing_columns = [col for col in numeric_columns if col not in trials_df.columns]
        if missing_columns:
            print(f"錯誤：以下欄位不存在於 trials_df 中: {missing_columns}")
            return None
        
        # 過濾數值數據並移除 NaN
        trials_df_numeric = trials_df[numeric_columns].dropna()
        print(f"過濾後的數據行數: {len(trials_df_numeric)}")
        
        if len(trials_df_numeric) < 2:
            print("有效數據不足，無法進行相關性分析！")
            return None
        
        # 計算相關性矩陣
        correlation_matrix = trials_df_numeric.corr()
        
        # 重命名列以提高可讀性
        rename_dict = {col: col.replace('params_', '').replace('metric_', '') for col in numeric_columns}
        trials_df_numeric.rename(columns=rename_dict, inplace=True)
        renamed_columns = [rename_dict[col] for col in numeric_columns]
        renamed_param_columns = [rename_dict[col] for col in param_columns]
        renamed_metric_columns = [rename_dict[col] for col in metric_columns]
        correlation_matrix = trials_df_numeric.corr()
        
        # 打印文字表格
        print("\n完整相關性矩陣：")
        print(correlation_matrix.to_string(float_format="%.3f"))
        
        print("\n參數間相關性：")
        param_corr = correlation_matrix.loc[renamed_param_columns, renamed_param_columns]
        print(param_corr.to_string(float_format="%.3f"))
        
        print("\n參數對指標的相關性：")
        param_metric_corr = correlation_matrix.loc[renamed_param_columns, renamed_metric_columns]
        print(param_metric_corr.to_string(float_format="%.3f"))
        
        # 繪製熱圖
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('參數與指標相關性矩陣', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 返回相關性數據
        return {
            'full_correlation': correlation_matrix,
            'param_correlation': param_corr,
            'param_metric_correlation': param_metric_corr
        }
    
    def pareto_frontier_analysis(self):
        """帕雷托前沿分析（適用於三維多目標優化，使用 Plotly 繪圖）"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return
        
        if len(self.strategy_config['target_metrics']) < 3:
            print("三維帕雷托分析需要至少3個指標！")
            return
        
        trials_df = self.analyze_multi_metric_results()
        target_metrics = self.strategy_config['target_metrics']
        
        # 提取前三個指標進行帕雷托分析
        metric1, metric2, metric3 = target_metrics[:3]
        
        # 獲取有效數據
        valid_data = trials_df.dropna(subset=[f'metric_{metric1}', f'metric_{metric2}', f'metric_{metric3}'])
        
        if len(valid_data) == 0:
            print("無有效的帕雷托分析數據！")
            return
        
        # 確保數據為數值類型
        try:
            x = valid_data[f'metric_{metric1}'].astype(float).values
            y = valid_data[f'metric_{metric2}'].astype(float).values
            z = valid_data[f'metric_{metric3}'].astype(float).values
        except ValueError as e:
            print(f"數據轉換錯誤：請確保 {metric1}, {metric2}, {metric3} 的值為數值類型！錯誤詳情：{e}")
            return
        
        # 檢查數據長度一致性
        if not (len(x) == len(y) == len(z)):
            print("錯誤：x, y, z 數據長度不一致！")
            return
        
        # 帕雷托前沿識別（假設三個指標都是越大越好）
        pareto_indices = []
        for i in range(len(x)):
            is_pareto = True
            for j in range(len(x)):
                if i != j:
                    if (x[j] >= x[i] and y[j] >= y[i] and z[j] >= z[i] and
                        (x[j] > x[i] or y[j] > y[i] or z[j] > z[i])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        # 使用 Plotly 繪製三維帕雷托前沿
        try:
            fig = go.Figure()
            
            # 繪製所有試驗點
            fig.add_trace(go.Scatter3d(
                x=x, 
                y=y, 
                z=z,
                mode='markers',
                marker=dict(size=5, color='lightblue', opacity=0.6),
                name='所有試驗'
            ))
            
            if pareto_indices:
                # 繪製帕雷托前沿點
                fig.add_trace(go.Scatter3d(
                    x=x[pareto_indices], 
                    y=y[pareto_indices], 
                    z=z[pareto_indices],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='x', line=dict(color='black', width=1)),
                    name='帕雷托前沿'
                ))
            
            # 設置圖表佈局
            fig.update_layout(
                scene=dict(
                    xaxis_title=metric1,
                    yaxis_title=metric2,
                    zaxis_title=metric3,
                    xaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
                    yaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
                    zaxis=dict(backgroundcolor="white", gridcolor="lightgray")
                ),
                title=dict(
                    text=f'三維帕雷托前沿分析<br>{metric1} vs {metric2} vs {metric3}',
                    font=dict(size=14, family='Arial', color='black'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                margin=dict(l=0, r=0, t=50, b=0),
                scene_aspectmode='cube'  # 使三個軸比例一致
            )
            
            # 顯示圖表
            fig.show()
        
        except Exception as e:
            print(f"繪圖錯誤：{e}")
            return
        
        print(f"\n帕雷托前沿分析：")
        print(f"  總試驗數：{len(valid_data)}")
        print(f"  帕雷托最優解數量：{len(pareto_indices)}")
        
        if pareto_indices:
            print(f"\n前3個帕雷托解：")
            pareto_trials = valid_data.iloc[pareto_indices]
            # 根據主指標排序
            pareto_trials_sorted = pareto_trials.nlargest(3, f'metric_{metric1}')
            
            for idx, (_, trial) in enumerate(pareto_trials_sorted.iterrows()):
                print(f"  解 {idx+1}：")
                for param_name in self.strategy_config['optimize_params'].keys():
                    if f'params_{param_name}' in trial:
                        print(f"    {param_name}: {trial[f'params_{param_name}']}")
                for metric in target_metrics[:3]:
                    print(f"    {metric}: {trial[f'metric_{metric}']:.4f}")
                print()

def create_bbrank_config():
    """創建 bbrank_3 策略的配置"""
    from Technicalindicatorstrategy import bbrank_3
    
    config = {
        'strategy_module': bbrank_3,
        'strategy_function': 'get_signals',
        'fixed_params': {
            'symbol': 'ETHUSDT',
            'interval': '1h',
            'end_time': datetime.now(),
            'limit': 10000,
            'bb_length': 20,
            'mult': 2.0,
            'ATR_period': 20
        },
        'optimize_params': {
            'lookback': {
                'type': 'int',
                'min': 200,
                'max': 500,
                'step': 100
            },
            'rank_th': {
                'type': 'float', 
                'min': 80,
                'max': 95,
                'step': 5
            },
            'ATR_multi_SL': {
                'type': 'float',
                'min': 0.5,
                'max': 2.0,
                'step': 0.1
            }
        },
        'target_metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
        'backtest_config': {
            'initial_capital': 1000000,
            'fee_rate': 0.0005,
            'leverage': 1,
            'allow_short': True,
            'stop_loss': None,
            'take_profit': None,
            'capital_ratio': 1,
            'max_hold_bars': 10000,
            'delay_entry': False,
            'risk_free_rate': 0.02,
            'interval': '1h'
        }
    }
    
    return config

def create_custom_strategy_config(strategy_module, strategy_function, param_config):
    """
    創建自定義策略配置的輔助函數
    
    Args:
        strategy_module: 策略模組對象
        strategy_function: 策略函數名稱
        param_config: 參數配置字典
    """
    
    # 動態檢測策略函數參數
    if hasattr(strategy_module, strategy_function):
        func = getattr(strategy_module, strategy_function)
        sig = inspect.signature(func)
        detected_params = list(sig.parameters.keys())
        print(f"檢測到策略函數 {strategy_function} 的參數: {detected_params}")
    
    config = {
        'strategy_module': strategy_module,
        'strategy_function': strategy_function,
        'fixed_params': param_config.get('fixed_params', {}),
        'optimize_params': param_config.get('optimize_params', {}),
        'target_metrics': param_config.get('target_metrics', ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']),
        'backtest_config': param_config.get('backtest_config', {
            'initial_capital': 1000000,
            'fee_rate': 0.0005,
            'leverage': 1,
            'allow_short': True,
            'stop_loss': None,
            'take_profit': None,
            'capital_ratio': 1,
            'max_hold_bars': 10000,
            'delay_entry': False,
            'risk_free_rate': 0.02,
            'interval': '1h'
        })
    }
    
    return config

def auto_detect_strategy_params(strategy_module, strategy_function):
    """自動檢測策略函數的參數並提供配置建議"""
    if not hasattr(strategy_module, strategy_function):
        print(f"Strategy function '{strategy_function}' not found in module")
        return None
    
    func = getattr(strategy_module, strategy_function)
    sig = inspect.signature(func)
    
    print(f"\n策略函數 {strategy_function} 參數檢測結果:")
    print("=" * 50)
    
    param_suggestions = {}
    
    for param_name, param in sig.parameters.items():
        print(f"參數: {param_name}")
        print(f"  預設值: {param.default}")
        print(f"  註解: {param.annotation}")
        
        # 根據參數名稱提供配置建議
        if 'lookback' in param_name.lower():
            param_suggestions[param_name] = {
                'type': 'int',
                'min': 100,
                'max': 1000,
                'step': 50,
                'suggestion': '回望期間，建議範圍100-1000'
            }
        elif 'threshold' in param_name.lower() or 'th' in param_name.lower():
            param_suggestions[param_name] = {
                'type': 'float',
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'suggestion': '閾值參數，建議範圍0.1-10.0'
            }
        elif 'length' in param_name.lower() or 'period' in param_name.lower():
            param_suggestions[param_name] = {
                'type': 'int', 
                'min': 5,
                'max': 100,
                'step': 5,
                'suggestion': '週期長度，建議範圍5-100'
            }
        elif 'mult' in param_name.lower() or 'multiplier' in param_name.lower():
            param_suggestions[param_name] = {
                'type': 'float',
                'min': 0.5,
                'max': 5.0,
                'step': 0.1,
                'suggestion': '倍數參數，建議範圍0.5-5.0'
            }
        else:
            param_suggestions[param_name] = {
                'type': 'float',
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'suggestion': '通用數值參數'
            }
        
        if param_suggestions[param_name]:
            print(f"  建議配置: {param_suggestions[param_name]['suggestion']}")
        print()
    
    return param_suggestions

def main():
    """主要執行範例"""
    print("靈活策略優化器 - 使用範例")
    print("=" * 50)
    
    # 方式1: 使用預設 bbrank_3 配置
    print("1. 使用預設 bbrank_3 配置:")
    bbrank_config = create_bbrank_config()
    optimizer = FlexibleStrategyOptimizer(bbrank_config)
    study = optimizer.run_optimization(n_trials=25, n_jobs=1, study_name="ex_bbrank3")
    
    # 多指標分析
    trials_df = optimizer.analyze_multi_metric_results()
    
    # 帕雷托前沿分析
    optimizer.pareto_frontier_analysis()
    
    # HiPlot 視覺化
    print("\n2. 繪製 HiPlot 交互式圖表:")
    optimizer.plot_hiplot(output_html="optimization_results.html")
    
    # 相關性分析並以文字表格輸出
    print("\n3. 相關性分析結果：")
    corr_results = optimizer.analyze_param_metric_correlations()
    
    return optimizer, study

def example_custom_strategy():
    """自定義策略使用範例"""
    print("\n2. 自定義策略配置範例:")
    
    custom_config = {
        'fixed_params': {
            'symbol': 'BTCUSDT',
            'interval': '4h',
            'end_time': datetime.now(),
            'limit': 5000
        },
        'optimize_params': {
            'param1': {'type': 'int', 'min': 10, 'max': 100, 'step': 10},
            'param2': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
            'param3': {'type': 'categorical', 'choices': ['option1', 'option2', 'option3']}
        },
        'target_metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
        'backtest_config': {
            'initial_capital': 1000000,
            'fee_rate': 0.001,
            'leverage': 2
        }
    }
    
    print("自定義配置範例已準備完成，可根據實際策略調整參數")
    return custom_config

if __name__ == "__main__":
    optimizer, study = main()
    
    custom_config = example_custom_strategy()
    
    print("\n" + "=" * 80)
    print("使用說明:")
    print("1. 將您的策略模組導入")
    print("2. 使用 auto_detect_strategy_params() 檢測參數")
    print("3. 根據建議配置 optimize_params")
    print("4. 設定 target_metrics 選擇要優化的指標")
    print("5. 運行 FlexibleStrategyOptimizer")
    print("=" * 80)