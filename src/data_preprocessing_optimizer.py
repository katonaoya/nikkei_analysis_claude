"""
データ前処理最適化システム
外れ値除去、正規化、特徴量変換の効果を詳細検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessingOptimizer:
    """データ前処理最適化"""
    
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
        self.df['target'] = (self.df['next_day_return'] >= 0.01).astype(int)
        
        logger.info(f"データ読み込み完了: {len(self.df):,}レコード")
    
    def create_test_features(self):
        """テスト用特徴量作成"""
        df_features = self.df.copy()
        
        # 基本特徴量
        for window in [10, 20]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
            
            df_features[f'volatility_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
            df_features[f'momentum_{window}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(window, fill_method=None)
            )
        
        # RSI
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df_features['rsi'] = df_features.groupby('Code')['close_price'].transform(calc_rsi)
        
        # 過去リターン
        for lag in [1, 2, 3]:
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
        
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'volatility', 'momentum', 'rsi', 'return_lag'))]
        
        return df_features, feature_cols
    
    def test_outlier_removal_methods(self, df_features, feature_cols):
        """外れ値除去手法のテスト"""
        logger.info("=== 外れ値除去手法テスト ===")
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X_base, y_base = X[valid_mask], y[valid_mask]
        
        logger.info(f"ベースデータ: {len(X_base):,}レコード")
        
        outlier_methods = {
            'No_Removal': None,
            'IsolationForest_5pct': IsolationForest(contamination=0.05, random_state=42),
            'IsolationForest_10pct': IsolationForest(contamination=0.10, random_state=42),
            'Statistical_Z3': 'z_score_3',
            'Statistical_Z2.5': 'z_score_2.5',
            'Quantile_5pct': 'quantile_5',
            'Quantile_1pct': 'quantile_1'
        }
        
        results = {}
        
        for method_name, method in outlier_methods.items():
            logger.info(f"\n{method_name} をテスト中...")
            
            if method is None:
                X_clean, y_clean = X_base.copy(), y_base.copy()
                removed_count = 0
                
            elif hasattr(method, 'fit_predict'):
                # Isolation Forest
                outliers = method.fit_predict(X_base) == -1
                X_clean, y_clean = X_base[~outliers], y_base[~outliers]
                removed_count = outliers.sum()
                
            elif method.startswith('z_score'):
                # Z-score based
                threshold = float(method.split('_')[-1])
                z_scores = np.abs(stats.zscore(X_base, axis=0, nan_policy='omit'))
                outliers = (z_scores > threshold).any(axis=1)
                X_clean, y_clean = X_base[~outliers], y_base[~outliers]
                removed_count = outliers.sum()
                
            elif method.startswith('quantile'):
                # Quantile based
                pct = int(method.split('_')[-1])
                lower_pct = pct / 2
                upper_pct = 100 - pct / 2
                
                lower_bounds = X_base.quantile(lower_pct / 100)
                upper_bounds = X_base.quantile(upper_pct / 100)
                
                outliers = ((X_base < lower_bounds) | (X_base > upper_bounds)).any(axis=1)
                X_clean, y_clean = X_base[~outliers], y_base[~outliers]
                removed_count = outliers.sum()
            
            logger.info(f"除去データ数: {removed_count:,} ({removed_count/len(X_base)*100:.1f}%)")
            logger.info(f"残りデータ数: {len(X_clean):,}")
            
            if len(X_clean) < 10000:
                logger.warning("データ不足のためスキップ")
                continue
            
            # 性能評価
            score = self._evaluate_preprocessing(X_clean, y_clean, method_name)
            
            results[method_name] = {
                'score': score,
                'removed_count': removed_count,
                'remaining_count': len(X_clean),
                'removal_rate': removed_count / len(X_base)
            }
        
        return results
    
    def test_scaling_methods(self, df_features, feature_cols):
        """スケーリング手法のテスト"""
        logger.info("=== スケーリング手法テスト ===")
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X_base, y_base = X[valid_mask], y[valid_mask]
        
        scaling_methods = {
            'No_Scaling': None,
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'),
            'QuantileTransformer': QuantileTransformer(output_distribution='uniform'),
            'QuantileTransformer_Normal': QuantileTransformer(output_distribution='normal')
        }
        
        results = {}
        
        for method_name, scaler in scaling_methods.items():
            logger.info(f"\n{method_name} をテスト中...")
            
            if scaler is None:
                X_scaled = X_base.copy()
            else:
                try:
                    X_scaled = pd.DataFrame(
                        scaler.fit_transform(X_base),
                        columns=X_base.columns,
                        index=X_base.index
                    )
                except Exception as e:
                    logger.error(f"{method_name}でエラー: {e}")
                    continue
            
            # 性能評価
            score = self._evaluate_preprocessing(X_scaled, y_base, method_name)
            
            results[method_name] = {
                'score': score,
                'method': method_name
            }
        
        return results
    
    def test_feature_transformation(self, df_features, feature_cols):
        """特徴量変換のテスト"""
        logger.info("=== 特徴量変換テスト ===")
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X_base, y_base = X[valid_mask], y[valid_mask]
        
        transformation_methods = {
            'Original': 'none',
            'Log_Transform': 'log',
            'Sqrt_Transform': 'sqrt',
            'Box_Cox': 'box_cox',
            'Rank_Transform': 'rank',
            'Winsorize_5pct': 'winsorize_5',
            'Winsorize_1pct': 'winsorize_1'
        }
        
        results = {}
        
        for method_name, transform_type in transformation_methods.items():
            logger.info(f"\n{method_name} をテスト中...")
            
            X_transformed = X_base.copy()
            
            try:
                if transform_type == 'none':
                    pass  # 何もしない
                    
                elif transform_type == 'log':
                    # 対数変換（正値のみ）
                    X_transformed = np.log1p(np.maximum(X_transformed, 0))
                    
                elif transform_type == 'sqrt':
                    # 平方根変換（正値のみ）
                    X_transformed = np.sqrt(np.maximum(X_transformed, 0))
                    
                elif transform_type == 'box_cox':
                    # Box-Cox変換（PowerTransformerで近似）
                    transformer = PowerTransformer(method='box-cox')
                    X_transformed = pd.DataFrame(
                        transformer.fit_transform(np.maximum(X_transformed, 1e-8)),
                        columns=X_transformed.columns,
                        index=X_transformed.index
                    )
                    
                elif transform_type == 'rank':
                    # ランク変換
                    X_transformed = X_transformed.rank(pct=True)
                    
                elif transform_type.startswith('winsorize'):
                    # Winsorizing
                    pct = int(transform_type.split('_')[1])
                    lower_pct = pct / 2 / 100
                    upper_pct = 1 - pct / 2 / 100
                    
                    for col in X_transformed.columns:
                        lower_bound = X_transformed[col].quantile(lower_pct)
                        upper_bound = X_transformed[col].quantile(upper_pct)
                        X_transformed[col] = np.clip(X_transformed[col], lower_bound, upper_bound)
                
                # NaN・Inf値の処理
                X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 性能評価
                score = self._evaluate_preprocessing(X_transformed, y_base, method_name)
                
                results[method_name] = {
                    'score': score,
                    'transform': transform_type
                }
                
            except Exception as e:
                logger.error(f"{method_name}でエラー: {e}")
                results[method_name] = {'score': 0, 'error': str(e)}
        
        return results
    
    def _evaluate_preprocessing(self, X, y, method_name):
        """前処理手法の評価"""
        if len(X) < 5000:
            return 0
        
        # Train/Test分割
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # モデル訓練
        model = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                     random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 予測・評価
        proba = model.predict_proba(X_test)[:, 1]
        
        best_score = 0
        for threshold in [0.5, 0.6, 0.7]:
            predictions = (proba >= threshold).astype(int)
            if predictions.sum() > 0:
                precision = precision_score(y_test, predictions)
                if precision > best_score:
                    best_score = precision
        
        logger.info(f"{method_name}: 精度 {best_score:.3f}")
        return best_score
    
    def run_preprocessing_optimization(self):
        """前処理最適化の実行"""
        logger.info("=== データ前処理最適化開始 ===")
        
        # 特徴量作成
        df_features, feature_cols = self.create_test_features()
        logger.info(f"テスト特徴量: {len(feature_cols)}個")
        
        # 1. 外れ値除去テスト
        outlier_results = self.test_outlier_removal_methods(df_features, feature_cols)
        
        # 2. スケーリングテスト
        scaling_results = self.test_scaling_methods(df_features, feature_cols)
        
        # 3. 特徴量変換テスト
        transform_results = self.test_feature_transformation(df_features, feature_cols)
        
        # 結果分析
        logger.info("\n=== 前処理最適化結果 ===")
        
        # 最高スコアの特定
        all_results = {
            'outlier_removal': outlier_results,
            'scaling': scaling_results,
            'transformation': transform_results
        }
        
        best_overall = 0
        best_config = {}
        
        for category, results in all_results.items():
            logger.info(f"\n{category.upper()}:")
            
            if results:
                best_method = max(results.keys(), key=lambda x: results[x].get('score', 0))
                best_score = results[best_method]['score']
                
                logger.info(f"  最高: {best_method} - {best_score:.3f}")
                
                if best_score > best_overall:
                    best_overall = best_score
                    best_config = {
                        'category': category,
                        'method': best_method,
                        'score': best_score,
                        'details': results[best_method]
                    }
                
                # 全結果表示
                for method, result in results.items():
                    score = result.get('score', 0)
                    logger.info(f"    {method}: {score:.3f}")
        
        logger.info(f"\n=== 最適前処理設定 ===")
        logger.info(f"カテゴリ: {best_config['category']}")
        logger.info(f"手法: {best_config['method']}")
        logger.info(f"精度: {best_config['score']:.3f}")
        
        return {
            'best_overall_score': best_overall,
            'best_config': best_config,
            'all_results': all_results,
            'feature_count': len(feature_cols)
        }


def main():
    """メイン実行"""
    logger.info("データ前処理最適化開始")
    
    try:
        optimizer = DataPreprocessingOptimizer()
        results = optimizer.run_preprocessing_optimization()
        
        print(f"\n=== データ前処理最適化結果 ===")
        print(f"最高精度: {results['best_overall_score']:.1%}")
        print(f"最適手法: {results['best_config']['method']}")
        print(f"カテゴリ: {results['best_config']['category']}")
        
        return results
        
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == "__main__":
    main()