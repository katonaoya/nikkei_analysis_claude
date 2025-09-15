"""
継続的精度改善サイクル
長時間検証による段階的精度向上システム
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContinuousImprovementSystem:
    """継続的改善システム"""
    
    def __init__(self):
        self.df = None
        self.best_score = 0.0
        self.best_config = {}
        self.improvement_history = []
        self.cycle_count = 0
        
        # 結果保存用
        self.results_dir = Path("results/continuous_improvement")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.load_data()
        logger.info("継続的改善システム初期化完了")
    
    def load_data(self):
        """データ読み込み"""
        data_file = Path("data/nikkei225_full_data/nikkei225_full_10years_175stocks_20250831_020101.pkl")
        self.df = pd.read_pickle(data_file)
        
        # 基本前処理
        self.df = self.df.sort_values(['Code', 'Date']).reset_index(drop=True)
        self.df['close_price'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['high_price'] = pd.to_numeric(self.df['High'], errors='coerce') 
        self.df['low_price'] = pd.to_numeric(self.df['Low'], errors='coerce')
        self.df['volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')
        
        self.df['daily_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None)
        self.df['next_day_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
        self.df['target'] = (self.df['next_day_return'] >= 0.01).astype(int)
        
        logger.info(f"データ読み込み完了: {len(self.df):,}レコード, ターゲット分布: {self.df['target'].mean():.1%}")
    
    def create_feature_set_v1(self, df):
        """基本特徴量セット v1"""
        df_features = df.copy()
        
        # 移動平均系
        for window in [5, 10, 20]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
            
        # RSI
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df_features['rsi'] = df_features.groupby('Code')['close_price'].transform(calc_rsi)
        
        # ラグ特徴量
        for lag in [1, 2, 3]:
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
        
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'rsi', 'return_lag'))]
        return df_features, feature_cols
    
    def create_feature_set_v2(self, df):
        """拡張特徴量セット v2"""
        df_features, basic_cols = self.create_feature_set_v1(df)
        
        # ボラティリティ系
        for window in [5, 10, 20]:
            df_features[f'volatility_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
        # モメンタム系
        for period in [5, 10, 20]:
            df_features[f'momentum_{period}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(period, fill_method=None)
            )
            
        # 価格位置
        for window in [10, 20]:
            high_max = df_features.groupby('Code')['high_price'].transform(lambda x: x.rolling(window).max())
            low_min = df_features.groupby('Code')['low_price'].transform(lambda x: x.rolling(window).min())
            df_features[f'price_position_{window}'] = (
                (df_features['close_price'] - low_min) / (high_max - low_min + 1e-8)
            )
            
        enhanced_cols = [col for col in df_features.columns 
                        if col.startswith(('sma_ratio', 'rsi', 'return_lag', 'volatility', 
                                         'momentum', 'price_position'))]
        return df_features, enhanced_cols
    
    def create_feature_set_v3(self, df):
        """高度特徴量セット v3"""
        df_features, v2_cols = self.create_feature_set_v2(df)
        
        # 市場関連
        market_return = df_features.groupby('Date')['daily_return'].mean()
        df_features['market_return'] = df_features['Date'].map(market_return)
        df_features['excess_return'] = df_features['daily_return'] - df_features['market_return']
        
        # ベータ
        for window in [20, 60]:
            df_features[f'beta_{window}'] = df_features.groupby('Code').apply(
                lambda x: x['daily_return'].rolling(window).corr(x['market_return'])
            ).values
            
        # ボリューム分析
        df_features['volume_ma_ratio'] = df_features.groupby('Code').apply(
            lambda x: x['volume'] / x['volume'].rolling(20).mean()
        ).values
        
        # 統計的特徴量
        for window in [10, 20]:
            df_features[f'return_skew_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).skew()
            )
            df_features[f'return_kurt_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).kurt()
            )
            
        # 価格ギャップ
        df_features['gap_up'] = ((df_features['close_price'] / 
                                df_features.groupby('Code')['close_price'].shift(1)) - 1 > 0.02).astype(int)
        df_features['gap_down'] = ((df_features['close_price'] / 
                                  df_features.groupby('Code')['close_price'].shift(1)) - 1 < -0.02).astype(int)
        
        advanced_cols = [col for col in df_features.columns 
                        if col.startswith(('sma_ratio', 'rsi', 'return_lag', 'volatility', 
                                         'momentum', 'price_position', 'excess_return', 
                                         'beta', 'volume_ma_ratio', 'return_skew', 
                                         'return_kurt', 'gap_up', 'gap_down'))]
        return df_features, advanced_cols
    
    def test_models(self, X_train, y_train, X_val, y_val):
        """複数モデルのテスト"""
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, 
                                                  random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=10,
                                             random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=8, random_state=42, verbosity=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=8, random_state=42, verbosity=0),
            'CatBoost': cb.CatBoostClassifier(iterations=200, learning_rate=0.05,
                                            depth=8, random_seed=42, verbose=False),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, 
                                                         learning_rate=0.05,
                                                         max_depth=8, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                
                # 複数閾値で評価
                best_score = 0
                best_threshold = 0.5
                best_predictions = 0
                
                for threshold in np.arange(0.5, 0.85, 0.05):
                    predictions = (proba >= threshold).astype(int)
                    if predictions.sum() > 0:
                        precision = precision_score(y_val, predictions)
                        if precision > best_score:
                            best_score = precision
                            best_threshold = threshold
                            best_predictions = predictions.sum()
                
                results[name] = {
                    'precision': best_score,
                    'threshold': best_threshold,
                    'predictions': best_predictions,
                    'model': model
                }
                
            except Exception as e:
                logger.error(f"{name}でエラー: {e}")
                results[name] = {'precision': 0, 'error': str(e)}
        
        return results
    
    def test_feature_selection(self, X_train, y_train, X_val, y_val):
        """特徴量選択手法のテスト"""
        base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        selectors = {
            'All_Features': None,
            'SelectKBest_f': SelectKBest(f_classif, k=min(20, X_train.shape[1])),
            'SelectKBest_mi': SelectKBest(mutual_info_classif, k=min(20, X_train.shape[1])),
            'RFE': RFE(base_model, n_features_to_select=min(15, X_train.shape[1]), step=2)
        }
        
        results = {}
        
        for name, selector in selectors.items():
            try:
                if selector is None:
                    X_train_selected = X_train
                    X_val_selected = X_val
                else:
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    X_val_selected = selector.transform(X_val)
                
                model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                             random_state=42, n_jobs=-1)
                model.fit(X_train_selected, y_train)
                proba = model.predict_proba(X_val_selected)[:, 1]
                
                best_score = 0
                for threshold in [0.5, 0.6, 0.7]:
                    predictions = (proba >= threshold).astype(int)
                    if predictions.sum() > 0:
                        precision = precision_score(y_val, predictions)
                        if precision > best_score:
                            best_score = precision
                
                results[name] = {
                    'precision': best_score,
                    'features_selected': X_train_selected.shape[1]
                }
                
            except Exception as e:
                logger.error(f"特徴量選択 {name}でエラー: {e}")
                results[name] = {'precision': 0, 'error': str(e)}
        
        return results
    
    def test_preprocessing(self, X_train, y_train, X_val, y_val):
        """前処理手法のテスト"""
        scalers = {
            'RobustScaler': RobustScaler(),
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'NoScaling': None
        }
        
        results = {}
        
        for name, scaler in scalers.items():
            try:
                if scaler is None:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                else:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                
                model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                             random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                proba = model.predict_proba(X_val_scaled)[:, 1]
                
                best_score = 0
                for threshold in [0.5, 0.6, 0.7]:
                    predictions = (proba >= threshold).astype(int)
                    if predictions.sum() > 0:
                        precision = precision_score(y_val, predictions)
                        if precision > best_score:
                            best_score = precision
                
                results[name] = {'precision': best_score}
                
            except Exception as e:
                logger.error(f"前処理 {name}でエラー: {e}")
                results[name] = {'precision': 0, 'error': str(e)}
        
        return results
    
    def run_improvement_cycle(self):
        """改善サイクル実行"""
        self.cycle_count += 1
        cycle_start_time = time.time()
        
        logger.info(f"=== 改善サイクル #{self.cycle_count} 開始 ===")
        
        # 特徴量セット比較
        feature_sets = {
            'v1_basic': self.create_feature_set_v1,
            'v2_enhanced': self.create_feature_set_v2,
            'v3_advanced': self.create_feature_set_v3
        }
        
        best_cycle_score = 0
        best_cycle_config = {}
        
        for set_name, create_func in feature_sets.items():
            logger.info(f"\n特徴量セット {set_name} をテスト中...")
            
            try:
                df_features, feature_cols = create_func(self.df)
                
                X = df_features[feature_cols].fillna(0)
                y = df_features['target']
                
                # 有効データのみ
                valid_mask = ~(y.isna() | X.isna().any(axis=1))
                X, y = X[valid_mask], y[valid_mask]
                
                if len(X) < 10000:
                    logger.warning(f"{set_name}: データ不足")
                    continue
                
                # Train/Validation分割
                split_point = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
                y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
                
                logger.info(f"特徴量数: {len(feature_cols)}, 訓練データ: {len(X_train):,}")
                
                # 1. モデル比較
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model_results = self.test_models(X_train_scaled, y_train, X_val_scaled, y_val)
                
                best_model_name = max(model_results.keys(), 
                                    key=lambda x: model_results[x].get('precision', 0))
                best_model_score = model_results[best_model_name]['precision']
                
                logger.info(f"最高モデル {best_model_name}: {best_model_score:.3f}")
                
                # 2. 特徴量選択テスト
                selection_results = self.test_feature_selection(X_train_scaled, y_train, 
                                                              X_val_scaled, y_val)
                
                best_selection = max(selection_results.keys(),
                                   key=lambda x: selection_results[x].get('precision', 0))
                best_selection_score = selection_results[best_selection]['precision']
                
                logger.info(f"最高特徴量選択 {best_selection}: {best_selection_score:.3f}")
                
                # 3. 前処理テスト
                preprocessing_results = self.test_preprocessing(X_train, y_train, X_val, y_val)
                
                best_preprocessing = max(preprocessing_results.keys(),
                                       key=lambda x: preprocessing_results[x].get('precision', 0))
                best_preprocessing_score = preprocessing_results[best_preprocessing]['precision']
                
                logger.info(f"最高前処理 {best_preprocessing}: {best_preprocessing_score:.3f}")
                
                # このセットの最高スコア
                set_best_score = max(best_model_score, best_selection_score, best_preprocessing_score)
                
                if set_best_score > best_cycle_score:
                    best_cycle_score = set_best_score
                    best_cycle_config = {
                        'feature_set': set_name,
                        'feature_count': len(feature_cols),
                        'best_model': best_model_name,
                        'best_selection': best_selection,
                        'best_preprocessing': best_preprocessing,
                        'score': set_best_score,
                        'model_results': model_results,
                        'selection_results': selection_results,
                        'preprocessing_results': preprocessing_results
                    }
                
            except Exception as e:
                logger.error(f"特徴量セット {set_name} でエラー: {e}")
                continue
        
        # サイクル結果の記録
        cycle_time = time.time() - cycle_start_time
        
        improvement_record = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': cycle_time / 60,
            'best_score': best_cycle_score,
            'config': best_cycle_config,
            'improvement': best_cycle_score - self.best_score
        }
        
        self.improvement_history.append(improvement_record)
        
        # 全体最高記録更新チェック
        if best_cycle_score > self.best_score:
            self.best_score = best_cycle_score
            self.best_config = best_cycle_config.copy()
            logger.info(f"🎉 NEW BEST SCORE: {best_cycle_score:.3f} (改善: +{best_cycle_score - self.best_score:.3f})")
        else:
            logger.info(f"サイクル最高: {best_cycle_score:.3f}, 全体最高: {self.best_score:.3f}")
        
        # 結果保存
        self.save_results()
        
        return improvement_record
    
    def save_results(self):
        """結果保存"""
        # 履歴保存
        history_file = self.results_dir / f"improvement_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        import json
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_score': self.best_score,
                'best_config': self.best_config,
                'cycle_count': self.cycle_count,
                'history': self.improvement_history
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果保存: {history_file}")
    
    def run_continuous_improvement(self, max_cycles=10, target_score=0.85):
        """継続的改善の実行"""
        logger.info(f"=== 継続的改善開始 ===")
        logger.info(f"最大サイクル数: {max_cycles}, 目標精度: {target_score:.1%}")
        
        start_time = time.time()
        
        for cycle in range(max_cycles):
            try:
                cycle_result = self.run_improvement_cycle()
                
                # 目標達成チェック
                if self.best_score >= target_score:
                    logger.info(f"🎯 目標精度 {target_score:.1%} 達成！")
                    break
                
                # 進捗報告
                elapsed_hours = (time.time() - start_time) / 3600
                logger.info(f"経過時間: {elapsed_hours:.1f}時間, 進捗: {cycle+1}/{max_cycles} サイクル")
                
                # 短い休憩
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("ユーザーによる中断")
                break
            except Exception as e:
                logger.error(f"サイクル {cycle+1} でエラー: {e}")
                continue
        
        # 最終結果
        total_time = (time.time() - start_time) / 3600
        
        logger.info(f"\n=== 継続的改善完了 ===")
        logger.info(f"実行時間: {total_time:.1f}時間")
        logger.info(f"実行サイクル数: {self.cycle_count}")
        logger.info(f"最高精度: {self.best_score:.3f}")
        logger.info(f"最適設定: {self.best_config['feature_set']} + {self.best_config['best_model']}")
        
        return {
            'final_score': self.best_score,
            'best_config': self.best_config,
            'total_cycles': self.cycle_count,
            'total_time_hours': total_time,
            'improvement_history': self.improvement_history
        }


def main():
    """メイン実行"""
    print("=== 継続的精度改善システム ===")
    
    try:
        system = ContinuousImprovementSystem()
        
        # 長時間継続的改善を実行
        final_results = system.run_continuous_improvement(
            max_cycles=20,  # 最大20サイクル
            target_score=0.85  # 85%目標
        )
        
        print(f"\n=== 最終結果 ===")
        print(f"最高精度: {final_results['final_score']:.1%}")
        print(f"実行サイクル: {final_results['total_cycles']}")
        print(f"総実行時間: {final_results['total_time_hours']:.1f}時間")
        
        if final_results['final_score'] >= 0.85:
            print("🎉 目標精度85%達成！")
        else:
            print(f"目標まで: {0.85 - final_results['final_score']:.3f}")
            
    except Exception as e:
        logger.error(f"システムエラー: {e}")
        raise


if __name__ == "__main__":
    main()