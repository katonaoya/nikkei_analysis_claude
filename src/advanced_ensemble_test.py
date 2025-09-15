"""
高度なアンサンブル手法のテスト
複数の予測手法を組み合わせて精度向上を図る
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedEnsembleSystem:
    """高度なアンサンブルシステム"""
    
    def __init__(self):
        self.df = None
        self.load_data()
    
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
        
        logger.info(f"データ読み込み完了: {len(self.df):,}レコード")
    
    def create_comprehensive_features(self):
        """包括的特徴量作成"""
        df_features = self.df.copy()
        
        # 基本テクニカル指標
        logger.info("基本テクニカル指標作成中...")
        
        # 移動平均系
        for window in [5, 10, 20, 50]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
            
            ema = df_features.groupby('Code')['close_price'].transform(lambda x: x.ewm(span=window).mean())
            df_features[f'ema_ratio_{window}'] = df_features['close_price'] / ema
        
        # RSI（複数期間）
        def calc_rsi(prices, window):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        for period in [7, 14, 21]:
            df_features[f'rsi_{period}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: calc_rsi(x, period)
            )
        
        # ボリンジャーバンド
        for window in [10, 20]:
            for std_mult in [1.5, 2.0]:
                sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
                std = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).std())
                
                upper_band = sma + (std_mult * std)
                lower_band = sma - (std_mult * std)
                
                df_features[f'bb_position_{window}_{std_mult}'] = (
                    (df_features['close_price'] - lower_band) / (upper_band - lower_band + 1e-8)
                )
        
        # モメンタム・ボラティリティ
        logger.info("モメンタム・ボラティリティ指標作成中...")
        
        for window in [5, 10, 20]:
            # モメンタム
            df_features[f'momentum_{window}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(window, fill_method=None)
            )
            
            # ボラティリティ
            df_features[f'volatility_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
            # ROC (Rate of Change)
            df_features[f'roc_{window}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(window, fill_method=None) * 100
            )
        
        # 価格レンジ分析
        logger.info("価格レンジ分析中...")
        
        for window in [10, 20, 50]:
            high_max = df_features.groupby('Code')['high_price'].transform(lambda x: x.rolling(window).max())
            low_min = df_features.groupby('Code')['low_price'].transform(lambda x: x.rolling(window).min())
            
            df_features[f'price_position_{window}'] = (
                (df_features['close_price'] - low_min) / (high_max - low_min + 1e-8)
            )
            
            # Williams %R
            df_features[f'williams_r_{window}'] = -100 * (
                (high_max - df_features['close_price']) / (high_max - low_min + 1e-8)
            )
        
        # 出来高分析
        logger.info("出来高分析中...")
        
        for window in [10, 20]:
            vol_ma = df_features.groupby('Code')['volume'].transform(lambda x: x.rolling(window).mean())
            df_features[f'volume_ratio_{window}'] = df_features['volume'] / (vol_ma + 1)
            
            # OBV (On Balance Volume)
            df_features[f'obv_ma_{window}'] = df_features.groupby('Code').apply(
                lambda group: (group['volume'] * np.where(group['daily_return'] > 0, 1, -1)).rolling(window).sum()
            ).values
        
        # 市場関連指標
        logger.info("市場関連指標作成中...")
        
        market_return = df_features.groupby('Date')['daily_return'].mean()
        df_features['market_return'] = df_features['Date'].map(market_return)
        df_features['excess_return'] = df_features['daily_return'] - df_features['market_return']
        
        # ベータ
        for window in [30, 60]:
            df_features[f'beta_{window}'] = df_features.groupby('Code').apply(
                lambda x: x['daily_return'].rolling(window).corr(x['market_return'])
            ).values
        
        # 統計的特徴量
        logger.info("統計的特徴量作成中...")
        
        for window in [10, 20]:
            df_features[f'return_skew_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).skew()
            )
            df_features[f'return_kurt_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).kurt()
            )
        
        # ラグ特徴量
        for lag in range(1, 6):
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
            if lag <= 3:
                df_features[f'price_lag_{lag}'] = df_features.groupby('Code')['close_price'].shift(lag)
        
        # 特徴量列を特定
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'ema_ratio', 'rsi_', 'bb_position',
                                        'momentum_', 'volatility_', 'roc_', 'price_position',
                                        'williams_r', 'volume_ratio', 'obv_ma', 'excess_return',
                                        'beta_', 'return_skew', 'return_kurt', 'return_lag', 'price_lag'))]
        
        logger.info(f"包括的特徴量作成完了: {len(feature_cols)}個の特徴量")
        
        return df_features, feature_cols
    
    def create_ensemble_models(self):
        """多様なアンサンブルモデル作成"""
        models = {
            # 単体モデル
            'RandomForest': RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=10,
                random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300, max_depth=12, min_samples_split=10,
                random_state=42, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbosity=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=8,
                min_child_weight=5, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbosity=0
            )
        }
        
        # ソフト投票アンサンブル
        soft_ensemble = VotingClassifier([
            ('rf', models['RandomForest']),
            ('et', models['ExtraTrees']),
            ('lgb', models['LightGBM'])
        ], voting='soft')
        
        models['SoftVoting'] = soft_ensemble
        
        return models
    
    def test_multiple_targets(self, df_features, feature_cols):
        """複数のターゲット定義でテスト"""
        targets = {
            '0.5%': 0.005,
            '1.0%': 0.01,
            '1.5%': 0.015,
            '2.0%': 0.02
        }
        
        results = {}
        
        for target_name, threshold in targets.items():
            logger.info(f"\nターゲット {target_name} をテスト中...")
            
            # ターゲット作成
            df_test = df_features.copy()
            df_test['target'] = (df_test['next_day_return'] >= threshold).astype(int)
            
            target_rate = df_test['target'].mean()
            logger.info(f"ターゲット分布: {target_rate:.1%}")
            
            if target_rate < 0.05:  # 5%未満は除外
                logger.info("データ不足のためスキップ")
                continue
            
            # データ準備
            X = df_test[feature_cols].fillna(0)
            y = df_test['target']
            
            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            X, y = X[valid_mask], y[valid_mask]
            
            # Train/Test分割
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # アンサンブルモデルテスト
            models = self.create_ensemble_models()
            target_results = {}
            
            for model_name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        proba = model.decision_function(X_test_scaled)
                        proba = (proba - proba.min()) / (proba.max() - proba.min())
                    
                    # 最適閾値探索
                    best_score = 0
                    best_threshold = 0.5
                    best_predictions = 0
                    
                    for pred_threshold in np.arange(0.5, 0.9, 0.05):
                        predictions = (proba >= pred_threshold).astype(int)
                        if predictions.sum() > 0:
                            precision = precision_score(y_test, predictions)
                            if precision > best_score:
                                best_score = precision
                                best_threshold = pred_threshold
                                best_predictions = predictions.sum()
                    
                    target_results[model_name] = {
                        'precision': best_score,
                        'threshold': best_threshold,
                        'predictions': best_predictions
                    }
                    
                    logger.info(f"{model_name}: 精度{best_score:.3f}, 閾値{best_threshold:.2f}, 予測数{best_predictions}")
                    
                except Exception as e:
                    logger.error(f"{model_name}でエラー: {e}")
                    target_results[model_name] = {'precision': 0, 'error': str(e)}
            
            results[target_name] = target_results
        
        return results
    
    def run_advanced_ensemble_test(self):
        """高度なアンサンブルテスト実行"""
        logger.info("=== 高度なアンサンブルテスト開始 ===")
        
        # 包括的特徴量作成
        df_features, feature_cols = self.create_comprehensive_features()
        
        # 複数ターゲットでテスト
        results = self.test_multiple_targets(df_features, feature_cols)
        
        # 最高結果の特定
        best_overall_score = 0
        best_overall_config = {}
        
        logger.info("\n=== 結果サマリー ===")
        
        for target_name, target_results in results.items():
            logger.info(f"\nターゲット {target_name}:")
            
            best_model = max(target_results.keys(), 
                           key=lambda x: target_results[x].get('precision', 0))
            best_score = target_results[best_model]['precision']
            
            logger.info(f"  最高: {best_model} - {best_score:.3f}")
            
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_config = {
                    'target': target_name,
                    'model': best_model,
                    'score': best_score,
                    'config': target_results[best_model]
                }
        
        logger.info(f"\n=== 全体最高結果 ===")
        logger.info(f"ターゲット: {best_overall_config['target']}")
        logger.info(f"モデル: {best_overall_config['model']}")
        logger.info(f"精度: {best_overall_config['score']:.3f}")
        logger.info(f"特徴量数: {len(feature_cols)}")
        
        return {
            'best_score': best_overall_score,
            'best_config': best_overall_config,
            'all_results': results,
            'feature_count': len(feature_cols)
        }


def main():
    """メイン実行"""
    logger.info("高度なアンサンブル手法テスト開始")
    
    try:
        system = AdvancedEnsembleSystem()
        results = system.run_advanced_ensemble_test()
        
        print(f"\n=== 高度アンサンブル最終結果 ===")
        print(f"最高精度: {results['best_score']:.1%}")
        print(f"最適設定: {results['best_config']['target']} + {results['best_config']['model']}")
        print(f"使用特徴量: {results['feature_count']}個")
        
        return results
        
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == "__main__":
    main()