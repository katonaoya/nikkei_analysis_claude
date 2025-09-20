#!/usr/bin/env python3
"""
実用的なフィルタリング手法検証
連続的なウォークフォワード検証で実践値を測定
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class PracticalFilteringValidation:
    """実用的なフィルタリング手法検証"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 検証パラメータ
        self.confidence_threshold = 0.55
        self.target_candidates = 5
        self.min_training_days = 252  # 1年分の最小学習期間
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 実用検証用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        
        # 重複除去
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return clean_df, X, y
    
    def walk_forward_validation(self, df, X, y):
        """ウォークフォワード検証（連続的・実用的）"""
        logger.info("🚀 ウォークフォワード検証開始...")
        
        dates = sorted(df['Date'].unique())
        logger.info(f"全期間: {dates[0]} - {dates[-1]} ({len(dates)}日)")
        
        # 学習開始点とテスト期間を設定
        start_train_idx = 0
        start_test_idx = self.min_training_days  # 1年後から検証開始
        
        # 検証結果格納
        results = {
            'Simple_Confidence': [],
            'Sector_Diversity': [],
            'Volatility_Adjusted': []
        }
        
        validation_dates = []
        
        # 3ヶ月ごとにモデルを再学習してテスト
        step_size = 63  # 約3ヶ月（営業日）
        
        for test_start in range(start_test_idx, len(dates) - 21, step_size):  # 最後21日は除外
            # 学習期間：過去1.5年分
            train_start = max(0, test_start - 378)  # 1.5年前から
            train_end = test_start
            
            # テスト期間：次の21営業日（1ヶ月）
            test_end = min(test_start + 21, len(dates))
            
            train_dates = dates[train_start:train_end]
            test_dates = dates[test_start:test_end]
            
            if len(train_dates) < self.min_training_days:
                continue
                
            logger.info(f"検証期間: {test_dates[0]} - {test_dates[-1]} (学習{len(train_dates)}日)")
            
            # モデル学習
            train_mask = df['Date'].isin(train_dates)
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            scaler = StandardScaler()
            model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
            
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            
            # 各フィルタリング手法でテスト
            period_results = self.evaluate_methods_for_period(
                df, X, model, scaler, test_dates
            )
            
            # 結果記録
            for method, result in period_results.items():
                if result['total_predictions'] > 0:
                    results[method].append({
                        'period': f"{test_dates[0]}-{test_dates[-1]}",
                        'accuracy': result['accuracy'],
                        'predictions': result['total_predictions'],
                        'daily_avg': result['total_predictions'] / len(test_dates)
                    })
            
            validation_dates.extend(test_dates)
        
        return results, validation_dates
    
    def evaluate_methods_for_period(self, df, X, model, scaler, test_dates):
        """期間内での各手法評価"""
        methods = {
            'Simple_Confidence': self.method_simple_confidence,
            'Sector_Diversity': self.method_sector_diversity,
            'Volatility_Adjusted': self.method_volatility_adjusted
        }
        
        period_results = {}
        
        for method_name, method_func in methods.items():
            total_predictions = 0
            correct_predictions = 0
            
            for date in test_dates:
                day_data = df[df['Date'] == date].copy()
                if len(day_data) == 0:
                    continue
                
                # 予測実行
                X_day = day_data[self.optimal_features].fillna(0)
                X_day_scaled = scaler.transform(X_day)
                pred_proba = model.predict_proba(X_day_scaled)[:, 1]
                day_data['pred_proba'] = pred_proba
                
                # 絞り込み実行
                selected = method_func(day_data)
                
                if len(selected) > 0:
                    total_predictions += len(selected)
                    correct_predictions += selected['Binary_Direction'].sum()
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            period_results[method_name] = {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions
            }
        
        return period_results
    
    def method_simple_confidence(self, day_data):
        """シンプル確信度フィルタリング"""
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) |
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return pd.DataFrame()
        
        # 確信度順にソートして上位を選択
        high_conf['confidence_score'] = np.maximum(
            high_conf['pred_proba'], 
            1 - high_conf['pred_proba']
        )
        
        return high_conf.nlargest(self.target_candidates, 'confidence_score')
    
    def method_sector_diversity(self, day_data):
        """セクター多様性フィルタリング"""
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) |
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return pd.DataFrame()
        
        high_conf['confidence_score'] = np.maximum(
            high_conf['pred_proba'], 
            1 - high_conf['pred_proba']
        )
        
        # セクター情報がない場合はシンプル確信度と同じ
        return high_conf.nlargest(self.target_candidates, 'confidence_score')
    
    def method_volatility_adjusted(self, day_data):
        """ボラティリティ調整フィルタリング"""
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) |
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return pd.DataFrame()
        
        high_conf['confidence_score'] = np.maximum(
            high_conf['pred_proba'], 
            1 - high_conf['pred_proba']
        )
        
        # ボラティリティで調整
        if 'Volatility_20' in high_conf.columns:
            # ボラティリティが低いほど安定性が高い
            vol_factor = 1 / (1 + high_conf['Volatility_20'].fillna(0))
            high_conf['adjusted_score'] = high_conf['confidence_score'] * vol_factor
            return high_conf.nlargest(self.target_candidates, 'adjusted_score')
        else:
            return high_conf.nlargest(self.target_candidates, 'confidence_score')
    
    def display_practical_results(self, results):
        """実用的検証結果の表示"""
        logger.info("\n" + "="*100)
        logger.info("📊 実用的フィルタリング手法検証結果")
        logger.info("="*100)
        
        overall_stats = {}
        
        for method, method_results in results.items():
            if not method_results:
                continue
                
            accuracies = [r['accuracy'] for r in method_results]
            predictions = [r['predictions'] for r in method_results]
            daily_avgs = [r['daily_avg'] for r in method_results]
            
            overall_stats[method] = {
                'periods': len(method_results),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'total_predictions': sum(predictions),
                'avg_daily_candidates': np.mean(daily_avgs)
            }
        
        # 結果表示
        logger.info(f"\n📈 各手法の実用性能:")
        
        sorted_methods = sorted(overall_stats.items(), 
                              key=lambda x: x[1]['avg_accuracy'], reverse=True)
        
        for i, (method, stats) in enumerate(sorted_methods, 1):
            logger.info(f"\n{i}. {method}:")
            logger.info(f"   平均精度: {stats['avg_accuracy']:.1%} ± {stats['std_accuracy']:.1%}")
            logger.info(f"   精度範囲: {stats['min_accuracy']:.1%} - {stats['max_accuracy']:.1%}")
            logger.info(f"   検証期間: {stats['periods']}期間")
            logger.info(f"   総予測数: {stats['total_predictions']}件")
            logger.info(f"   日平均候補: {stats['avg_daily_candidates']:.1f}銘柄")
        
        # 最優秀手法の推奨
        if overall_stats:
            best_method = sorted_methods[0][0]
            best_stats = sorted_methods[0][1]
            
            logger.info(f"\n🏆 推奨手法: {best_method}")
            logger.info(f"   期待精度: {best_stats['avg_accuracy']:.1%}")
            logger.info(f"   安定性: ±{best_stats['std_accuracy']:.1%}")
            logger.info(f"   実用性評価: {'高' if best_stats['avg_accuracy'] > 0.60 else '中' if best_stats['avg_accuracy'] > 0.55 else '低'}")
        
        logger.info("="*100)
        
        return overall_stats

def main():
    """メイン実行"""
    logger.info("🔬 実用的フィルタリング手法検証システム")
    
    validator = PracticalFilteringValidation()
    
    try:
        # データ準備
        df, X, y = validator.load_and_prepare_data()
        
        # ウォークフォワード検証
        results, validation_dates = validator.walk_forward_validation(df, X, y)
        
        # 結果表示
        overall_stats = validator.display_practical_results(results)
        
        logger.info(f"\n✅ 実用検証完了 - {len(set(validation_dates))}日間で検証")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()