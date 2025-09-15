"""
実用性重視の現実的検証システム
100%精度の原因分析と実運用可能な精度レベルの確立
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticValidator:
    """実用性重視の検証システム"""
    
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
        self.df['daily_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None)
        self.df['next_day_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
        self.df['target'] = (self.df['next_day_return'] >= 0.01).astype(int)
        
        logger.info(f"データ読み込み完了: {len(self.df):,}レコード")
        logger.info(f"ターゲット分布: {self.df['target'].mean():.1%}")
    
    def create_simple_features(self):
        """シンプルで安定した特徴量"""
        df_features = self.df.copy()
        
        # 基本移動平均系（過度に複雑にしない）
        for window in [5, 10, 20]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
        
        # RSI（1つの期間のみ）
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df_features['rsi_14'] = df_features.groupby('Code')['close_price'].transform(calc_rsi)
        
        # ボラティリティ
        df_features['volatility_10'] = df_features.groupby('Code')['daily_return'].transform(
            lambda x: x.rolling(10).std()
        )
        
        # 過去リターン（リーケージを避けるため最小限）
        for lag in [1, 2]:
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
        
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'rsi', 'volatility', 'return_lag'))]
        
        return df_features, feature_cols
    
    def analyze_100_percent_precision(self, df_features, feature_cols):
        """100%精度の原因分析"""
        logger.info("=== 100%精度の原因分析 ===")
        
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        # 有効データのみ
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X, y = X[valid_mask], y[valid_mask]
        
        # 時系列分割（より厳格に）
        tscv = TimeSeriesSplit(n_splits=5, gap=30)  # 30日のギャップを設ける
        
        precisions = []
        prediction_counts = []
        threshold_used = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold+1}/5:")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            logger.info(f"訓練期間: {len(X_train):,}, テスト期間: {len(X_test):,}")
            logger.info(f"テストターゲット分布: {y_test.mean():.1%}")
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル訓練（パラメータを控えめに）
            model = RandomForestClassifier(
                n_estimators=50,  # 少なめに設定
                max_depth=6,      # 浅めに設定
                min_samples_split=20,  # 多めに設定
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # 予測確率
            proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 複数閾値での詳細分析
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
                predictions = (proba >= threshold).astype(int)
                pred_count = predictions.sum()
                
                if pred_count > 0:
                    precision = precision_score(y_test, predictions)
                    recall = recall_score(y_test, predictions)
                    
                    # 混同行列
                    cm = confusion_matrix(y_test, predictions)
                    tn, fp, fn, tp = cm.ravel()
                    
                    logger.info(f"  閾値{threshold}: 精度{precision:.3f}, 再現率{recall:.3f}, 予測数{pred_count}")
                    logger.info(f"    TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}")
                    
                    if threshold == 0.7:  # 基準閾値での記録
                        precisions.append(precision)
                        prediction_counts.append(pred_count)
                        threshold_used.append(threshold)
                    
                    # 100%精度の場合、詳細分析
                    if precision >= 0.999:
                        logger.warning(f"    ⚠️ 100%精度検出 - 予測数わずか{pred_count}件")
                        if pred_count < 10:
                            logger.warning(f"    → 極めて保守的な予測（実用性に疑問）")
        
        # 全体統計
        if precisions:
            avg_precision = np.mean(precisions)
            std_precision = np.std(precisions)
            avg_predictions = np.mean(prediction_counts)
            
            logger.info(f"\n=== Cross-Validation結果 ===")
            logger.info(f"平均精度: {avg_precision:.3f} ± {std_precision:.3f}")
            logger.info(f"平均予測数: {avg_predictions:.1f}件/期間")
            logger.info(f"実用性評価: {'高' if 0.65 <= avg_precision <= 0.85 and avg_predictions >= 10 else '要改善'}")
            
            return {
                'avg_precision': avg_precision,
                'std_precision': std_precision,
                'avg_predictions': avg_predictions,
                'is_realistic': 0.65 <= avg_precision <= 0.85 and avg_predictions >= 10
            }
        
        return {'avg_precision': 0, 'is_realistic': False}
    
    def test_realistic_configurations(self):
        """実用的な設定のテスト"""
        logger.info("=== 実用的設定テスト ===")
        
        df_features, feature_cols = self.create_simple_features()
        
        configs = [
            {
                'name': 'Conservative',
                'n_estimators': 30,
                'max_depth': 4,
                'min_samples_split': 50,
                'threshold': 0.8
            },
            {
                'name': 'Balanced',
                'n_estimators': 50,
                'max_depth': 6,
                'min_samples_split': 20,
                'threshold': 0.7
            },
            {
                'name': 'Aggressive',
                'n_estimators': 100,
                'max_depth': 8,
                'min_samples_split': 10,
                'threshold': 0.6
            }
        ]
        
        results = {}
        
        for config in configs:
            logger.info(f"\n{config['name']} 設定をテスト:")
            
            result = self._test_single_config(df_features, feature_cols, config)
            results[config['name']] = result
            
            logger.info(f"  平均精度: {result['precision']:.3f}")
            logger.info(f"  平均予測数: {result['predictions']:.1f}")
            logger.info(f"  安定性: {result['stability']:.3f}")
        
        return results
    
    def _test_single_config(self, df_features, feature_cols, config):
        """単一設定のテスト"""
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X, y = X[valid_mask], y[valid_mask]
        
        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=3, gap=10)
        
        precisions = []
        predictions_counts = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions = (proba >= config['threshold']).astype(int)
            
            if predictions.sum() > 0:
                precision = precision_score(y_test, predictions)
                precisions.append(precision)
                predictions_counts.append(predictions.sum())
        
        if precisions:
            return {
                'precision': np.mean(precisions),
                'predictions': np.mean(predictions_counts),
                'stability': 1 - np.std(precisions)  # 標準偏差が小さいほど安定
            }
        
        return {'precision': 0, 'predictions': 0, 'stability': 0}
    
    def run_realistic_validation(self):
        """現実的検証の実行"""
        logger.info("=== 現実的検証開始 ===")
        
        # 1. 100%精度の原因分析
        df_features, feature_cols = self.create_simple_features()
        analysis_result = self.analyze_100_percent_precision(df_features, feature_cols)
        
        # 2. 実用的設定のテスト
        config_results = self.test_realistic_configurations()
        
        # 3. 推奨設定の決定
        best_config = max(config_results.keys(), 
                         key=lambda x: config_results[x]['precision'] * config_results[x]['stability'])
        
        logger.info(f"\n=== 現実的検証結果 ===")
        logger.info(f"分析結果: 実用性{'あり' if analysis_result['is_realistic'] else 'なし'}")
        logger.info(f"推奨設定: {best_config}")
        logger.info(f"期待精度: {config_results[best_config]['precision']:.1%}")
        logger.info(f"日次予測数: {config_results[best_config]['predictions']:.0f}件程度")
        
        return {
            'analysis_result': analysis_result,
            'config_results': config_results,
            'recommended_config': best_config,
            'feature_count': len(feature_cols)
        }


def main():
    """メイン実行"""
    logger.info("現実的検証開始")
    
    try:
        validator = RealisticValidator()
        results = validator.run_realistic_validation()
        
        print(f"\n=== 現実的検証完了 ===")
        print(f"推奨設定: {results['recommended_config']}")
        best_config = results['config_results'][results['recommended_config']]
        print(f"期待精度: {best_config['precision']:.1%}")
        print(f"予測数: {best_config['predictions']:.0f}件/日")
        print(f"安定性: {best_config['stability']:.3f}")
        
        if best_config['precision'] >= 0.65:
            print("✅ 実用レベルの性能")
        else:
            print("⚠️ 性能改善が必要")
        
        return results
        
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == "__main__":
    main()