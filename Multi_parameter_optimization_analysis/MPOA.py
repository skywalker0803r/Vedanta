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
from numbers import Number
from scipy.interpolate import griddata

class UniqueParamsSampler(optuna.samplers.TPESampler):
    """自定義採樣器，避免使用已存在的參數組合"""
    def __init__(self, used_params, **kwargs):
        super().__init__(**kwargs)
        self.used_params = used_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        """重寫獨立參數採樣，檢查是否重複"""
        while True:
            value = super().sample_independent(study, trial, param_name, param_distribution)
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

class FlexibleStrategyOptimizer:
    """通用策略參數優化器 - 增強版"""
    
    def __init__(self, strategy_config):
        self.strategy_config = strategy_config
        self.validate_config()
        self._pp_score_cache = {}
        self.verbose = True  # 添加 verbose 標誌，預設為 True

        # 新增：複合指標配置
        self.composite_config = strategy_config.get('composite_config', {
            'method': 'weighted',  # 'weighted' 或 'geometric'
            'weights': {
                'Sharpe Ratio': 0.4,
                'Sortino Ratio': 0.4,
                'Calmar Ratio': 0.2
            },
            'thresholds': {
                'Sharpe Ratio': 1.0,
                'Sortino Ratio': 1.5,
                'Calmar Ratio': 0.5
            }
        })
        
    def validate_config(self):
        """驗證配置的有效性"""
        required_keys = ['strategy_module', 'strategy_function', 'optimize_params', 'target_metrics']
        for key in required_keys:
            if key not in self.strategy_config:
                raise ValueError(f"Missing required config key: {key}")
                
        strategy_module = self.strategy_config['strategy_module']
        strategy_function_name = self.strategy_config['strategy_function']
        
        if not hasattr(strategy_module, strategy_function_name):
            raise ValueError(f"Strategy function '{strategy_function_name}' not found in module")
            
        self.strategy_function = getattr(strategy_module, strategy_function_name)
        
        sig = inspect.signature(self.strategy_function)
        self.function_params = list(sig.parameters.keys())
        print(f"檢測到策略函數參數: {self.function_params}")
        
    def calculate_composite_metric(self, metrics):
        """計算複合指標"""
        method = self.composite_config['method']
        weights = self.composite_config['weights']

        if method == 'weighted':
            # 加權平均
            composite_score = 0.0
            total_weight = 0.0
            for metric_name, weight in weights.items():
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    composite_score += weight * metrics[metric_name]
                    total_weight += weight
            
            if total_weight > 0:
                return composite_score / total_weight
            else:
                return 0.0
                
        elif method == 'geometric':
            # 幾何平均 (Sharpe * Sortino * Calmar)^(1/3)
            values = []
            for metric_name in weights.keys():
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    # 確保值為正數（對於幾何平均）
                    value = max(metrics[metric_name], 0.001)
                    values.append(value)
            
            if len(values) > 0:
                geometric_mean = np.power(np.prod(values), 1.0/len(values))
                return geometric_mean
            else:
                return 0.0
        
        return 0.0
        
    def trading_strategy(self, trial_params, **kwargs):
        """通用交易策略測試函數"""
        try:
            all_params = self.strategy_config.get('fixed_params', {}).copy()
            all_params.update(trial_params)
            all_params.update(kwargs)
            
            df_signals = self.strategy_function(**all_params)
            
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
            
            metrics = result.get('float_type_metrics', {})
            target_metrics = self.strategy_config['target_metrics']
            
            # 計算複合指標
            composite_value = self.calculate_composite_metric(metrics)
            
            # 將所有指標存儲在trial的用戶屬性中
            trial_metrics = {}
            for metric in target_metrics:
                value = metrics.get(metric, 0.0)
                trial_metrics[metric] = value if not np.isnan(value) else 0.0
            
            # 添加複合指標
            trial_metrics['composite_metric'] = composite_value
            
            return composite_value, trial_metrics
            
        except Exception as e:
            print(f"Error in trading_strategy: {e}")
            return -999.0, {}

    def objective(self, trial):
        """Optuna 目標函數 - 支援單目標與 NSGA-II 多目標"""
        trial_params = {}

        # 根據 config 建議參數
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

        # 執行策略，得到複合指標 + 各個單獨指標
        composite_value, trial_metrics = self.trading_strategy(trial_params)

        # 將指標存成 user_attr
        for metric_name, metric_value in trial_metrics.items():
            trial.set_user_attr(metric_name, metric_value)

        # 判斷是否為 NSGA-II
        optimize_config = self.strategy_config.get('optimize_config', {})
        sampler_type = optimize_config.get('sampler', 'tpe').lower()

        if sampler_type == 'nsga2':
            # NSGA-II 回傳多目標 (Sharpe, Sortino, Calmar)
            return (
                trial_metrics.get('Sharpe Ratio', 0.0),
                trial_metrics.get('Sortino Ratio', 0.0),
                trial_metrics.get('Calmar Ratio', 0.0)
            )
        else:
            # 單目標仍回傳 composite_value
            return composite_value


    def run_optimization(self, n_trials=250, n_jobs=-1, study_name=None):
        """運行參數優化"""
        print("開始靈活參數優化...")
        print(f"複合指標方法: {self.composite_config['method']}")
        if self.composite_config['method'] == 'weighted':
            print(f"權重配置: {self.composite_config['weights']}")
        print(f"門檻條件: {self.composite_config['thresholds']}")

        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        if study_name is None:
            study_name = f"{self.strategy_config['strategy_function']}_optimization"

        storage = f'sqlite:///{study_name}.db'

        import os
        if os.path.exists(study_name + '.db'):
            print(f"檢測到現有資料庫: {study_name}.db")
        else:
            print(f"創建新資料庫: {study_name}.db")

        # 確定優化方向
        optimize_config = self.strategy_config.get('optimize_config', {})
        sampler_type = optimize_config.get('sampler', 'tpe').lower()

        if sampler_type == 'nsga2':
            direction = ["maximize", "maximize", "maximize"]  # 三個目標
        else:
            direction = "maximize"


        # 從 config 讀取 optimize_config
        optimize_config = self.strategy_config.get('optimize_config', {})
        sampler_type = optimize_config.get('sampler', 'tpe').lower()
        seed = optimize_config.get('seed', 42)

        # 建立基礎 sampler
        if sampler_type == 'random':
            base_sampler = optuna.samplers.RandomSampler(seed=seed)
        elif sampler_type == 'cmaes':
            base_sampler = optuna.samplers.CmaEsSampler(seed=seed)
        elif sampler_type == 'nsga2':
            base_sampler = optuna.samplers.NSGAIISampler(seed=seed)
        elif sampler_type == 'qmc':
            base_sampler = optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed)
        else:  # 預設 TPE
            base_sampler = optuna.samplers.TPESampler(seed=seed)

        # 包裝成 UniqueParamsSampler，避免重複
        used_params = set()
        sampler = UniqueParamsSampler(used_params=used_params, seed=seed)

        # 嘗試載入或建立 study
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"已載入 study，包含 {len(study.trials)} 個試驗")
            for trial in study.trials:
                if trial.params:
                    param_tuple = tuple(sorted((k, v) for k, v in trial.params.items()))
                    used_params.add(param_tuple)
        except Exception as e:
            print(f"載入資料庫失敗：{e}，建立新 study")
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(),
                study_name=study_name,
                storage=storage
            )

        if len(study.trials) < 10:
            print(f"警告：當前試驗數量 ({len(study.trials)}) 較少，建議至少運行 50 次試驗以確保分析可靠性")

        # 開始優化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        self.study = study
        self.print_optimization_results()

        print("\n計算加權高原分數...")
        self.calculate_plateau_score()
        self.plot_plateau()
        print("\n生成 optimize_params vs PPScore 熱力圖...")
        self.plot_optimize_params_vs_pp_score()

        return study

    
    def print_optimization_results(self):
        """打印優化結果"""
        print(f"\nOptimization completed!")
        print(f"Best composite metric: {self.study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
            
        best_trial = self.study.best_trial
        print(f"\nAll metrics for best trial:")
        for metric in self.strategy_config['target_metrics']:
            if metric in best_trial.user_attrs:
                print(f"  {metric}: {best_trial.user_attrs[metric]:.4f}")
        
        if 'composite_metric' in best_trial.user_attrs:
            print(f"  Composite Metric: {best_trial.user_attrs['composite_metric']:.4f}")

    def filter_valid_trials(self, trials):
        """根據門檻條件篩選有效試驗"""
        thresholds = self.composite_config['thresholds']
        valid_trials = []
        
        for trial in trials:
            if trial.value is None:
                continue
                
            # 檢查是否滿足所有門檻條件
            meets_criteria = True
            for metric_name, threshold in thresholds.items():
                metric_value = trial.user_attrs.get(metric_name, 0.0)
                if metric_value < threshold:
                    meets_criteria = False
                    break
            
            if meets_criteria:
                valid_trials.append(trial)
        
        return valid_trials

    def calculate_local_plateau_score(self, center_trial, alpha=None, max_radius=2):
        """針對單一 trial 計算 Plateau Score（使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            return None
            
        # 篩選符合門檻條件的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        
        if not valid_trials:
            if self.verbose:
                print("警告：沒有試驗符合門檻條件")
            return None
            
        if self.verbose:
            print(f"符合門檻條件的試驗數量: {len(valid_trials)} / {len(self.study.trials)}")
        
        # 如果沒有指定 alpha，使用符合條件試驗的複合指標平均值
        if alpha is None:
            composite_values = []
            for trial in valid_trials:
                composite_value = trial.user_attrs.get('composite_metric', trial.value)
                if composite_value is not None and composite_value > -999:
                    composite_values.append(composite_value)
            
            if composite_values:
                alpha = np.mean(composite_values)
            else:
                alpha = 0.0
        
        if self.verbose:
            print(f"使用門檻值 alpha = {alpha:.4f}")

        param_names = list(self.strategy_config['optimize_params'].keys())
        param_configs = self.strategy_config['optimize_params']

        # 正規化中心 trial
        norm_center = {}
        for p in param_names:
            conf = param_configs[p]
            if conf['type'] == 'categorical':
                norm_center[p] = center_trial.params[p]
            else:
                minv, maxv = conf['min'], conf['max']
                norm_center[p] = (center_trial.params[p] - minv) / (maxv - minv)

        # 去重 & 正規化 trial（僅使用符合條件的試驗）
        unique_params = {}
        for t in valid_trials:
            param_tuple = tuple(sorted((k, t.params[k]) for k in param_names))
            composite_value = t.user_attrs.get('composite_metric', t.value)
            if param_tuple not in unique_params:
                unique_params[param_tuple] = []
            unique_params[param_tuple].append(composite_value)

        norm_trials = []
        for params, values in unique_params.items():
            norm = {}
            for k, v in params:
                conf = param_configs[k]
                if conf['type'] == 'categorical':
                    norm[k] = v
                else:
                    norm[k] = (v - conf['min']) / (conf['max'] - conf['min'])
            avg_value = np.mean(values)
            norm_trials.append((norm, avg_value))

        # 分層計算
        layer_dict = {r: [] for r in range(max_radius+1)}
        for norm, value in norm_trials:
            dist = 0
            for p in param_names:
                if param_configs[p]['type'] == 'categorical':
                    diff = 0 if norm[p] == norm_center[p] else 1
                else:
                    diff = round(abs(norm[p] - norm_center[p]) * max_radius)
                dist += diff
            if dist <= max_radius:
                layer_dict[dist].append(1 if value > alpha else 0)

        plateau_score = 0.0
        for r in range(max_radius+1):
            if len(layer_dict[r]) == 0:
                continue
            if r == 0:
                score_r = 1
            else:
                score_r = sum(layer_dict[r]) / len(layer_dict[r])
            plateau_score += score_r

        return plateau_score

    def calculate_plateau_score(self, alpha=None, max_radius=2, update_cache=True):
        """全域 Plateau Score（以 best_trial 為中心，使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            return None
            
        # 確保 best_trial 符合門檻條件
        valid_trials = self.filter_valid_trials(self.study.trials)
        if not valid_trials:
            print("警告：沒有試驗符合門檻條件")
            return None
            
        # 從符合條件的試驗中找到最佳試驗
        best_composite_value = -float('inf')
        filtered_best_trial = None
        
        for trial in valid_trials:
            composite_value = trial.user_attrs.get('composite_metric', trial.value)
            if composite_value > best_composite_value:
                best_composite_value = composite_value
                filtered_best_trial = trial
        
        if filtered_best_trial is None:
            print("警告：無法找到符合條件的最佳試驗")
            return None
            
        print(f"使用符合條件的最佳試驗作為中心點，複合指標值: {best_composite_value:.4f}")
        
        return self.calculate_local_plateau_score(filtered_best_trial, alpha=alpha, max_radius=max_radius)

    def analyze_multi_metric_results(self):
        """多指標分析（增強版，包含複合指標）"""
        if not hasattr(self, 'study'):
            print("Please run optimization first!")
            return
            
        trials_df = self.study.trials_dataframe()
        target_metrics = self.strategy_config['target_metrics']
        
        # 添加用戶屬性到 DataFrame
        for metric in target_metrics + ['composite_metric']:
            metric_values = []
            for trial in self.study.trials:
                value = trial.user_attrs.get(metric, np.nan)
                metric_values.append(value)
            trials_df[f'metric_{metric}'] = metric_values
        
        # 篩選符合門檻條件的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        valid_indices = [i for i, trial in enumerate(self.study.trials) if trial in valid_trials]
        
        print(f"符合門檻條件的試驗: {len(valid_trials)} / {len(self.study.trials)}")
        
        if valid_indices:
            filtered_df = trials_df.iloc[valid_indices]
            
            # 多指標相關性分析（僅針對符合條件的試驗）
            metric_columns = [f'metric_{metric}' for metric in target_metrics + ['composite_metric']]
            metric_data = filtered_df[metric_columns].dropna()
            
            if len(metric_data) > 0:
                metric_data.columns = target_metrics + ['composite_metric']
                correlation_matrix = metric_data.corr()
                
                print("\n符合門檻條件試驗的多指標相關性分析:")
                for i, metric1 in enumerate(target_metrics + ['composite_metric']):
                    for j, metric2 in enumerate((target_metrics + ['composite_metric'])[i+1:], i+1):
                        corr = correlation_matrix.loc[metric1, metric2]
                        print(f"  {metric1} vs {metric2}: {corr:.3f}")
        
        return trials_df

    def plot_plateau(self, alpha=None):
        """視覺化高原：距離 vs 指標散點圖（使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return
        
        # 篩選符合門檻條件的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        
        if not valid_trials:
            print("警告：沒有試驗符合門檻條件，無法繪製高原圖")
            return
        
        # 找到符合條件的最佳試驗
        best_composite_value = -float('inf')
        filtered_best_trial = None
        
        for trial in valid_trials:
            composite_value = trial.user_attrs.get('composite_metric', trial.value)
            if composite_value > best_composite_value:
                best_composite_value = composite_value
                filtered_best_trial = trial
        
        if alpha is None:
            composite_values = [trial.user_attrs.get('composite_metric', trial.value) 
                             for trial in valid_trials 
                             if trial.user_attrs.get('composite_metric', trial.value) > -999]
            alpha = np.mean(composite_values) if composite_values else 0
        
        param_names = list(self.strategy_config['optimize_params'].keys())
        param_configs = self.strategy_config['optimize_params']
        norm_best = {}
        for p in param_names:
            conf = param_configs[p]
            if conf['type'] != 'categorical':
                minv = conf['min']
                maxv = conf['max']
                norm_best[p] = (filtered_best_trial.params[p] - minv) / (maxv - minv)
            else:
                norm_best[p] = filtered_best_trial.params[p]
        
        dists = []
        values = []
        for t in valid_trials:
            sq_sum = 0.0
            for p in param_names:
                conf = param_configs[p]
                v = t.params[p]
                if conf['type'] != 'categorical':
                    norm_v = (v - conf['min']) / (conf['max'] - conf['min'])
                    diff = norm_v - norm_best[p]
                else:
                    diff = 0 if v == norm_best[p] else 1
                sq_sum += diff ** 2
            dist = max(np.sqrt(sq_sum), 1e-5)
            dists.append(dist)
            values.append(t.user_attrs.get('composite_metric', t.value))
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        plt.scatter(dists, values, alpha=0.6, label='符合條件的試驗點', color='blue')
        plt.scatter(0, best_composite_value, color='red', s=100, label='最佳點')
        plt.axhline(alpha, color='green', linestyle='--', label=f'門檻 α={alpha:.2f}')
        
        # 添加門檻條件說明
        threshold_text = "門檻條件:\n"
        for metric, threshold in self.composite_config['thresholds'].items():
            threshold_text += f"{metric} > {threshold}\n"
        
        plt.text(0.02, 0.98, threshold_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.xlabel('正規化距離 (到最佳點)')
        plt.ylabel('複合指標值')
        plt.title(f'參數高原視覺化：距離 vs 複合指標\n({self.composite_config["method"]} method)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_optimize_params_vs_pp_score(self, alpha=None, n_bins=20, show_points=True, point_size=50):
        """生成 optimize_params vs PPScore 及 Composite Score 二維熱力圖 (左右並排，使用篩選後的試驗)"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return
            
        # 設置靜默模式以避免重複打印
        original_verbose = self.verbose
        self.verbose = False
        
        valid_trials = self.filter_valid_trials(self.study.trials)
        if not valid_trials:
            print("無有效試驗數據！")
            self.verbose = original_verbose
            return

        pp_scores = []
        composite_scores = []
        for t in valid_trials:
            score = self.calculate_local_plateau_score(t, alpha=alpha)
            if score is not None:
                pp_scores.append((t.params.copy(), score))
                composite_scores.append((t.params.copy(), t.user_attrs.get('composite_metric', t.value)))

        self.verbose = original_verbose  # 恢復原始 verbose 設置

        if not pp_scores or not composite_scores:
            print("無有效 PPScore 或 Composite Score 數據！")
            return

        param_names = list(self.strategy_config['optimize_params'].keys())
        if len(param_names) < 2:
            print("至少需要 2 個 optimize_params")
            return

        # 計算 optimize_params 與 composite_metric 的相關係數
        trials_data = []
        for trial in valid_trials:
            trial_data = {'composite_metric': trial.user_attrs.get('composite_metric', trial.value)}
            for param_name in param_names:
                trial_data[param_name] = trial.params.get(param_name)
            trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data)
        valid_data = trials_df.dropna()

        if len(valid_data) < 2:
            print("有效數據不足，無法計算相關係數！")
            return

        # 選擇與 composite_metric 相關性最高的兩個參數
        correlations = valid_data.corr()['composite_metric'].drop('composite_metric')
        abs_correlations = correlations.abs().sort_values(ascending=False)
        if len(abs_correlations) < 2:
            print("可用的參數不足以選擇兩個軸！")
            return
        x_param, y_param = abs_correlations.index[:2]
        print(f"選擇與 composite_metric 相關性最高的參數：{x_param} (相關係數: {correlations[x_param]:.3f}), {y_param} (相關係數: {correlations[y_param]:.3f})")

        # 提取 PPScore 數據
        x_pp = np.array([p[0][x_param] for p in pp_scores])
        y_pp = np.array([p[0][y_param] for p in pp_scores])
        z_pp = np.array([p[1] for p in pp_scores])

        # 提取 Composite Score 數據
        x_comp = np.array([p[0][x_param] for p in composite_scores])
        y_comp = np.array([p[0][y_param] for p in composite_scores])
        z_comp = np.array([p[1] for p in composite_scores])

        # 裁剪數據以避免極端值
        z_pp_min, z_pp_max = np.percentile(z_pp, 1), np.percentile(z_pp, 99)
        z_pp = np.clip(z_pp, z_pp_min, z_pp_max)
        z_comp_min, z_comp_max = np.percentile(z_comp, 1), np.percentile(z_comp, 99)
        z_comp = np.clip(z_comp, z_comp_min, z_comp_max)

        # 創建左右並排的子圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharex=True, sharey=True)

        # 繪製 PPScore 熱力圖（左側）
        try:
            xi = np.linspace(np.min(x_pp), np.max(x_pp), n_bins)
            yi = np.linspace(np.min(y_pp), np.max(y_pp), n_bins)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x_pp, y_pp), z_pp, (xi, yi), method='linear')
            zi = np.nan_to_num(zi, nan=z_pp_min)
            contour1 = ax1.contourf(xi, yi, zi, levels=20, cmap='Greens', vmin=z_pp_min, vmax=z_pp_max)
            plt.colorbar(contour1, ax=ax1, label='PPScore')
        except ValueError:
            print("PPScore 插值失敗，僅顯示散點圖。")

        if show_points:
            scatter1 = ax1.scatter(x_pp, y_pp, c=z_pp, s=point_size, cmap='Greens', vmin=z_pp_min, vmax=z_pp_max, alpha=0.7)
            if 'contour1' not in locals():
                plt.colorbar(scatter1, ax=ax1, label='PPScore')

        ax1.set_xlabel(x_param)
        ax1.set_ylabel(y_param)
        ax1.set_title(f'{x_param} vs {y_param} 的 PPScore 圖\n(僅符合門檻條件的試驗)')
        ax1.grid(True)

        # 繪製 Composite Score 熱力圖（右側）
        try:
            xi = np.linspace(np.min(x_comp), np.max(x_comp), n_bins)
            yi = np.linspace(np.min(y_comp), np.max(y_comp), n_bins)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x_comp, y_comp), z_comp, (xi, yi), method='linear')
            zi = np.nan_to_num(zi, nan=z_comp_min)
            contour2 = ax2.contourf(xi, yi, zi, levels=20, cmap='Blues', vmin=z_comp_min, vmax=z_comp_max)
            plt.colorbar(contour2, ax=ax2, label='Composite Score')
        except ValueError:
            print("Composite Score 插值失敗，僅顯示散點圖。")

        if show_points:
            scatter2 = ax2.scatter(x_comp, y_comp, c=z_comp, s=point_size, cmap='Blues', vmin=z_comp_min, vmax=z_comp_max, alpha=0.7)
            if 'contour2' not in locals():
                plt.colorbar(scatter2, ax=ax2, label='Composite Score')

        ax2.set_xlabel(x_param)
        ax2.set_title(f'{x_param} vs {y_param} 的 Composite Score 圖\n(僅符合門檻條件的試驗)')
        ax2.grid(True)

        # 調整佈局
        plt.tight_layout()
        plt.show()
        
    def analyze_param_metric_correlations(self):
        """分析 optimize_params 之間的相關性與對 target_metrics 的影響（使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            print("請先運行優化！")
            return None
            
        # 使用篩選後的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        
        if not valid_trials:
            print("沒有符合門檻條件的試驗數據！")
            return None
            
        print(f"使用符合門檻條件的試驗數量: {len(valid_trials)}")
        
        # 創建 DataFrame（僅包含符合條件的試驗）
        trials_data = []
        target_metrics = self.strategy_config['target_metrics']
        
        for trial in valid_trials:
            trial_data = {'trial_number': trial.number, 'value': trial.value}
            
            # 添加參數
            for param_name in self.strategy_config['optimize_params'].keys():
                trial_data[f'params_{param_name}'] = trial.params.get(param_name)
            
            # 添加指標
            for metric in target_metrics + ['composite_metric']:
                trial_data[f'metric_{metric}'] = trial.user_attrs.get(metric, np.nan)
            
            trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data)
        
        param_columns = [f'params_{param}' for param in self.strategy_config['optimize_params'].keys()]
        metric_columns = [f'metric_{metric}' for metric in target_metrics + ['composite_metric']]
        numeric_columns = param_columns + metric_columns
        
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
        print("\n完整相關性矩陣（符合門檻條件的試驗）：")
        print(correlation_matrix.to_string(float_format="%.3f"))
        
        print("\n參數間相關性：")
        param_corr = correlation_matrix.loc[renamed_param_columns, renamed_param_columns]
        print(param_corr.to_string(float_format="%.3f"))
        
        print("\n參數對指標的相關性：")
        param_metric_corr = correlation_matrix.loc[renamed_param_columns, renamed_metric_columns]
        print(param_metric_corr.to_string(float_format="%.3f"))
        
        # 繪製熱圖
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('參數與指標相關性矩陣\n(符合門檻條件的試驗)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'full_correlation': correlation_matrix,
            'param_correlation': param_corr,
            'param_metric_correlation': param_metric_corr,
            'filtered_trials_count': len(valid_trials),
            'total_trials_count': len(self.study.trials)
        }

    def pareto_frontier_analysis(self):
        """帕雷托前沿分析（使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return
        
        if len(self.strategy_config['target_metrics']) < 3:
            print("三維帕雷托分析需要至少3個指標！")
            return
        
        # 使用篩選後的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        if not valid_trials:
            print("沒有符合門檻條件的試驗數據！")
            return
            
        target_metrics = self.strategy_config['target_metrics']
        metric1, metric2, metric3 = target_metrics[:3]
        
        # 收集有效數據
        trials_data = []
        for trial in valid_trials:
            trial_data = {
                'trial_number': trial.number,
                'params': trial.params,  # 儲存參數
                metric1: trial.user_attrs.get(metric1, np.nan),
                metric2: trial.user_attrs.get(metric2, np.nan),
                metric3: trial.user_attrs.get(metric3, np.nan)
            }
            trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data)
        valid_data = trials_df.dropna(subset=[metric1, metric2, metric3])
        
        if len(valid_data) == 0:
            print("無有效的帕雷托分析數據！")
            return
        
        try:
            x = valid_data[metric1].astype(float).values
            y = valid_data[metric2].astype(float).values
            z = valid_data[metric3].astype(float).values
        except ValueError as e:
            print(f"數據轉換錯誤：{e}")
            return
        
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
            
            # 繪製所有符合條件的試驗點
            fig.add_trace(go.Scatter3d(
                x=x, 
                y=y, 
                z=z,
                mode='markers',
                marker=dict(size=5, color='lightblue', opacity=0.6),
                name='符合條件的試驗'
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
                    text=f'三維帕雷托前沿分析（符合門檻條件）<br>{metric1} vs {metric2} vs {metric3}',
                    font=dict(size=14, family='Arial', color='black'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                margin=dict(l=0, r=0, t=50, b=0),
                scene_aspectmode='cube'
            )
            
            fig.show()
        
        except Exception as e:
            print(f"繪圖錯誤：{e}")
            return
        
        print(f"\n帕雷托前沿分析（符合門檻條件）：")
        print(f"  符合條件的試驗數：{len(valid_data)}")
        print(f"  帕雷托最優解數量：{len(pareto_indices)}")
        
        if pareto_indices:
            print(f"\n前3個帕雷托解：")
            pareto_trials = valid_data.iloc[pareto_indices]
            pareto_trials_sorted = pareto_trials.nlargest(3, metric1)
            
            for idx, (_, trial) in enumerate(pareto_trials_sorted.iterrows()):
                print(f"  解 {idx+1}：")
                print(f"    {metric1}: {trial[metric1]:.4f}")
                print(f"    {metric2}: {trial[metric2]:.4f}")
                print(f"    {metric3}: {trial[metric3]:.4f}")
                print(f"    使用的參數：")
                for param_name, param_value in trial['params'].items():
                    print(f"      {param_name}: {param_value}")
                print()

    def plot_hiplot(self, output_html="optimization_results.html"):
        """使用 HiPlot 視覺化優化結果（使用篩選後的試驗）"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return

        # 使用篩選後的試驗
        valid_trials = self.filter_valid_trials(self.study.trials)
        if not valid_trials:
            print("無符合條件的數據可用於 HiPlot 視覺化！")
            return

        trials_data = []
        for trial in valid_trials:
            trial_data = {}
            # 添加參數
            for param_name in self.strategy_config['optimize_params'].keys():
                value = trial.params.get(param_name, None)
                trial_data[param_name] = value

            # 添加指標
            for metric in self.strategy_config['target_metrics'] + ['composite_metric']:
                value = trial.user_attrs.get(metric, None)
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
        print(f"HiPlot 視覺化已保存至 {output_html}（符合門檻條件的試驗）")

    def get_optimization_summary(self):
        """獲取優化摘要統計"""
        if not hasattr(self, 'study'):
            print("請先執行優化！")
            return None
            
        valid_trials = self.filter_valid_trials(self.study.trials)
        
        summary = {
            'total_trials': len(self.study.trials),
            'valid_trials': len(valid_trials),
            'filter_rate': len(valid_trials) / len(self.study.trials) if self.study.trials else 0,
            'composite_method': self.composite_config['method'],
            'weights': self.composite_config['weights'] if self.composite_config['method'] == 'weighted' else None,
            'thresholds': self.composite_config['thresholds']
        }
        
        if valid_trials:
            best_valid_trial = max(valid_trials, 
                                 key=lambda t: t.user_attrs.get('composite_metric', t.value))
            summary['best_composite_value'] = best_valid_trial.user_attrs.get('composite_metric', best_valid_trial.value)
            summary['best_params'] = best_valid_trial.params
            
            # 各指標統計
            for metric in self.strategy_config['target_metrics']:
                values = [t.user_attrs.get(metric, 0) for t in valid_trials if t.user_attrs.get(metric) is not None]
                if values:
                    summary[f'{metric}_stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        return summary

def create_TurtleMACDTimeZoneFilter_config():
    from Technicalindicatorstrategy import TurtleMACDTimeZoneFilter
    config ={
        'strategy_module': TurtleMACDTimeZoneFilter,
        'strategy_function': 'get_signals',
        'fixed_params': {
            'symbol': 'ETHUSDT',
            'interval': '2h',
            'end_time': datetime.now(),
            'limit': 7309
        },
        'optimize_params': {
            'atr_period': {'type': 'int', 'min': 10, 'max': 30,'step': 1},
            'fast_period': {'type': 'int', 'min': 10, 'max': 20,'step': 1},
            'slow_period': {'type': 'int', 'min': 20, 'max': 40,'step': 1},
            'signal_period': {'type': 'int', 'min': 5, 'max': 15,'step': 1},
            'high_low_lookback': {'type': 'int', 'min': 10, 'max': 30,'step': 1},
            'atr_multiplier_sl': {'type': 'float', 'min': 1.0, 'max': 3.0,'step': 0.1}
        },
        'target_metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
        'composite_config': {
            'method': 'weights',  # 或 'geometric'
            'weights': {
                'Sharpe Ratio': 0.4,
                'Sortino Ratio': 0.4,
                'Calmar Ratio': 0.2
            },
            'thresholds': {
                'Sharpe Ratio': 1.0,
                'Sortino Ratio': 1.5,
                'Calmar Ratio': 0.5
            }
        },
        'backtest_config': {
            'initial_capital': 1000000,
            'fee_rate': 0.0005,
            'leverage': 1
        }
    }
    return config

def create_custom_strategy_config(strategy_module, strategy_function, param_config):
    """
    創建自定義策略配置的輔助函數
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
        'optimize_config': {
        'sampler': 'tpe',     # 可選: 'tpe', 'random', 'cmaes', 'nsga2', 'qmc'
        'seed': 42
        },
        'fixed_params': param_config.get('fixed_params', {}),
        'optimize_params': param_config.get('optimize_params', {}),
        'target_metrics': param_config.get('target_metrics', ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']),
        'composite_config': param_config.get('composite_config', {
            'method': 'weighted',
            'weights': {
                'Sharpe Ratio': 0.4,
                'Sortino Ratio': 0.4,
                'Calmar Ratio': 0.2
            },
            'thresholds': {
                'Sharpe Ratio': 1.0,
                'Sortino Ratio': 1.5,
                'Calmar Ratio': 0.5
            }
        }),
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
        if 'threshold' in param_name.lower() or 'th' in param_name.lower():
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

def demo_weighted_optimization():
    """演示加權複合指標優化"""
    print("=" * 60)
    print("演示：加權複合指標優化")
    print("=" * 60)
    
    # 創建加權配置
    weighted_config = create_TurtleMACDTimeZoneFilter_config()
    weighted_config['composite_config']['method'] = 'weighted'
    weighted_config['composite_config']['weights'] = {
        'Sharpe Ratio': 0.5,
        'Sortino Ratio': 0.3,
        'Calmar Ratio': 0.2
    }
    
    print("加權配置:")
    for metric, weight in weighted_config['composite_config']['weights'].items():
        print(f"  {metric}: {weight}")
    
    optimizer = FlexibleStrategyOptimizer(weighted_config)
    study = optimizer.run_optimization(n_trials=50, n_jobs=1, study_name="weighted_TurtleMACDTimeZoneFilter")
    
    return optimizer, study

def demo_geometric_optimization():
    """演示幾何平均複合指標優化"""
    print("=" * 60)
    print("演示：幾何平均複合指標優化")
    print("=" * 60)
    
    # 創建幾何平均配置
    geometric_config = create_TurtleMACDTimeZoneFilter_config()
    geometric_config['composite_config']['method'] = 'geometric'
    
    print("幾何平均配置: (Sharpe × Sortino × Calmar)^(1/3)")
    
    optimizer = FlexibleStrategyOptimizer(geometric_config)
    study = optimizer.run_optimization(n_trials=50, n_jobs=1, study_name="geometric_TurtleMACDTimeZoneFilter")
    
    return optimizer, study

def compare_optimization_methods():
    """比較不同優化方法的結果"""
    print("\n" + "=" * 80)
    print("比較不同複合指標方法")
    print("=" * 80)
    
    # 運行加權優化
    print("\n1. 運行加權優化...")
    weighted_optimizer, weighted_study = demo_weighted_optimization()
    
    # 運行幾何平均優化
    print("\n2. 運行幾何平均優化...")
    geometric_optimizer, geometric_study = demo_geometric_optimization()
    
    # 比較結果
    print("\n3. 結果比較:")
    print("-" * 40)
    
    weighted_summary = weighted_optimizer.get_optimization_summary()
    geometric_summary = geometric_optimizer.get_optimization_summary()
    
    print(f"加權方法:")
    print(f"  符合條件試驗: {weighted_summary['valid_trials']}/{weighted_summary['total_trials']}")
    print(f"  最佳複合指標: {weighted_summary.get('best_composite_value', 'N/A'):.4f}")
    
    print(f"\n幾何平均方法:")
    print(f"  符合條件試驗: {geometric_summary['valid_trials']}/{geometric_summary['total_trials']}")
    print(f"  最佳複合指標: {geometric_summary.get('best_composite_value', 'N/A'):.4f}")
    
    return weighted_optimizer, geometric_optimizer

def main():
    """主要執行範例 - 增強版"""
    print("靈活策略優化器 - 增強版使用範例")
    print("=" * 50)
    
    # 方式1: 使用預設 TurtleMACDTimeZoneFilter 配置（加權方法）
    print("1. 使用加權複合指標配置:")
    bbrank_config = create_TurtleMACDTimeZoneFilter_config()
    optimizer = FlexibleStrategyOptimizer(bbrank_config)
    study = optimizer.run_optimization(n_trials=25, n_jobs=1, study_name="enhanced_TurtleMACDTimeZoneFilter")
    
    # 獲取優化摘要
    summary = optimizer.get_optimization_summary()
    print("\n優化摘要:")
    print(f"總試驗數: {summary['total_trials']}")
    print(f"符合條件試驗數: {summary['valid_trials']}")
    print(f"篩選率: {summary['filter_rate']:.2%}")
    
    # 多指標分析
    trials_df = optimizer.analyze_multi_metric_results()
    
    # 帕雷托前沿分析
    optimizer.pareto_frontier_analysis()
    
    # HiPlot 視覺化
    print("\n2. 繪製 HiPlot 交互式圖表:")
    optimizer.plot_hiplot(output_html="enhanced_optimization_results.html")
    
    # 相關性分析
    print("\n3. 相關性分析結果：")
    corr_results = optimizer.analyze_param_metric_correlations()
    
    return optimizer, study

def example_custom_strategy():
    """自定義策略使用範例"""
    print("\n自定義策略配置範例:")
    
    custom_config = {
         'optimize_config': {
        'sampler': 'tpe',     # 可選: 'tpe', 'random', 'cmaes', 'nsga2', 'qmc'
        'seed': 42
        },
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
        'target_metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
        'composite_config': {
            'method': 'weighted',  # 或 'geometric'
            'weights': {
                'Sharpe Ratio': 0.4,
                'Sortino Ratio': 0.4,
                'Calmar Ratio': 0.2
            },
            'thresholds': {
                'Sharpe Ratio': 1.0,
                'Sortino Ratio': 1.5,
                'Calmar Ratio': 0.5
            }
        },
        'backtest_config': {
            'initial_capital': 1000000,
            'fee_rate': 0.0005,
            'leverage': 1
        }
    }
    
    print("自定義配置範例已準備完成，可根據實際策略調整參數")
    return custom_config

if __name__ == "__main__":
    # 運行主要範例
    optimizer, study = main()
    
    # 比較不同方法
    print("\n" + "=" * 80)
    print("比較加權 vs 幾何平均方法:")
    weighted_opt, geometric_opt = compare_optimization_methods()
    
    # 自定義配置範例
    custom_config = example_custom_strategy()
    
    print("\n" + "=" * 80)
    print("使用說明（增強版）:")
    print("1. 將您的策略模組導入")
    print("2. 使用 auto_detect_strategy_params() 檢測參數")
    print("3. 根據建議配置 optimize_params")
    print("4. 設定 composite_config:")
    print("   - method: 'weighted' 或 'geometric'")
    print("   - weights: 各指標權重（僅加權方法需要）")
    print("   - thresholds: 各指標門檻值")
    print("5. 設定 target_metrics 選擇要優化的指標")
    print("6. 運行 FlexibleStrategyOptimizer")
    print("=" * 80)
