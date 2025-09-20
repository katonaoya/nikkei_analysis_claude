#!/usr/bin/env python3
"""
現在のモデルの日次予測分析
1日あたりの候補銘柄数と確信度分布を調査
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

class DailyPredictionAnalysis:
    """日次予測候補数の分析"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 取引パラメータ
        self.confidence_threshold = 0.55   # 予測確信度閾値
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 日次予測分析用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return clean_df, X, y
    
    def analyze_daily_predictions(self, df, X, y):
        """日次予測候補数の分析"""
        logger.info("🔍 日次予測候補数分析...")
        
        # 学習期間とテスト期間の分割
        dates = sorted(df['Date'].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        logger.info(f"学習期間: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)}日)")
        logger.info(f"分析期間: {test_dates[0]} - {test_dates[-1]} ({len(test_dates)}日)")
        
        # 学習データでモデル訓練
        train_mask = df['Date'].isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # テスト期間での日次予測分析
        daily_stats = []
        
        for i, date in enumerate(test_dates):
            if i % 50 == 0:
                logger.info(f"  分析進行: {i+1}/{len(test_dates)} ({date})")
            
            day_data = df[df['Date'] == date]
            if len(day_data) == 0:
                continue
                
            X_day = day_data[self.optimal_features].fillna(0)
            X_day_scaled = scaler.transform(X_day)
            
            # 予測実行
            pred_proba = model.predict_proba(X_day_scaled)[:, 1]
            predictions = pred_proba > 0.5
            
            # 確信度フィルタリング
            high_confidence_up = pred_proba >= self.confidence_threshold
            high_confidence_down = pred_proba <= (1 - self.confidence_threshold)
            high_confidence_total = high_confidence_up | high_confidence_down
            
            # 統計計算
            daily_stat = {
                'date': date,
                'total_stocks': len(day_data),
                'up_predictions': predictions.sum(),
                'down_predictions': (~predictions).sum(),
                'high_conf_up': high_confidence_up.sum(),
                'high_conf_down': high_confidence_down.sum(),
                'high_conf_total': high_confidence_total.sum(),
                'high_conf_ratio': high_confidence_total.sum() / len(day_data) * 100,
                'avg_confidence_up': pred_proba[predictions].mean() if predictions.sum() > 0 else 0,
                'avg_confidence_down': (1 - pred_proba[~predictions]).mean() if (~predictions).sum() > 0 else 0,
                'max_confidence': max(pred_proba.max(), (1 - pred_proba.min())),
                'min_confidence': min(pred_proba.min(), (1 - pred_proba.max())),
                'std_confidence': pred_proba.std()
            }
            
            daily_stats.append(daily_stat)
        
        return pd.DataFrame(daily_stats)
    
    def analyze_prediction_distribution(self, stats_df):
        """予測分布の詳細分析"""
        logger.info("📈 予測分布の詳細分析...")
        
        # 基本統計
        logger.info("\\n" + "="*100)
        logger.info("📊 日次予測候補数の統計")
        logger.info("="*100)
        
        # 銘柄数統計
        logger.info(f"\\n🏢 1日あたりの分析対象銘柄数:")
        logger.info(f"  平均: {stats_df['total_stocks'].mean():.1f}銘柄")
        logger.info(f"  中央値: {stats_df['total_stocks'].median():.0f}銘柄")
        logger.info(f"  範囲: {stats_df['total_stocks'].min():.0f} - {stats_df['total_stocks'].max():.0f}銘柄")
        
        # 予測分布
        logger.info(f"\\n📈 1日あたりの予測分布:")
        logger.info(f"  上昇予測平均: {stats_df['up_predictions'].mean():.1f}銘柄")
        logger.info(f"  下落予測平均: {stats_df['down_predictions'].mean():.1f}銘柄")
        logger.info(f"  上昇予測割合: {stats_df['up_predictions'].mean() / stats_df['total_stocks'].mean() * 100:.1f}%")
        
        # 高確信度候補
        logger.info(f"\\n🎯 1日あたりの高確信度候補数（{self.confidence_threshold*100:.0f}%以上）:")
        logger.info(f"  高確信度上昇: {stats_df['high_conf_up'].mean():.1f}銘柄")
        logger.info(f"  高確信度下落: {stats_df['high_conf_down'].mean():.1f}銘柄")
        logger.info(f"  高確信度合計: {stats_df['high_conf_total'].mean():.1f}銘柄")
        logger.info(f"  高確信度割合: {stats_df['high_conf_ratio'].mean():.1f}%")
        
        # 確信度統計
        logger.info(f"\\n📊 予測確信度の統計:")
        logger.info(f"  上昇予測平均確信度: {stats_df['avg_confidence_up'].mean():.1%}")
        logger.info(f"  下落予測平均確信度: {stats_df['avg_confidence_down'].mean():.1%}")
        logger.info(f"  最高確信度の平均: {stats_df['max_confidence'].mean():.1%}")
        logger.info(f"  確信度標準偏差: {stats_df['std_confidence'].mean():.3f}")
        
        # 実用的な候補数
        logger.info(f"\\n💼 実用的な取引候補数:")
        
        # 様々な確信度閾値での候補数
        confidence_levels = [0.52, 0.55, 0.60, 0.65, 0.70]
        for conf_level in confidence_levels:
            high_conf_count = stats_df.apply(
                lambda row: self.count_high_confidence_stocks(row, conf_level), axis=1
            ).mean()
            logger.info(f"  確信度{conf_level*100:.0f}%以上: {high_conf_count:.1f}銘柄/日")
        
        # 取引可能日の分析
        tradeable_days = (stats_df['high_conf_total'] > 0).sum()
        logger.info(f"\\n📅 取引機会の分析:")
        logger.info(f"  取引候補がある日: {tradeable_days}/{len(stats_df)}日 ({tradeable_days/len(stats_df)*100:.1f}%)")
        logger.info(f"  取引機会なしの日: {len(stats_df) - tradeable_days}日")
        
        # 月次・週次パターン
        logger.info(f"\\n🗓️ 時間パターンの分析:")
        self.analyze_temporal_patterns(stats_df)
        
        logger.info("="*100)
        
        return stats_df
    
    def count_high_confidence_stocks(self, row, confidence_level):
        """指定確信度レベルでの候補数計算"""
        total_stocks = row['total_stocks']
        high_conf_ratio = row['high_conf_ratio'] / 100
        
        # 簡易計算（実際はより複雑）
        estimated_count = total_stocks * high_conf_ratio * (self.confidence_threshold / confidence_level)
        return max(0, estimated_count)
    
    def analyze_temporal_patterns(self, stats_df):
        """時間パターンの分析"""
        stats_df = stats_df.copy()
        stats_df['weekday'] = pd.to_datetime(stats_df['date']).dt.dayofweek
        stats_df['month'] = pd.to_datetime(stats_df['date']).dt.month
        
        # 曜日別パターン
        weekday_names = ['月', '火', '水', '木', '金']
        weekday_stats = stats_df.groupby('weekday')['high_conf_total'].mean()
        
        logger.info("  曜日別高確信度候補数:")
        for day_idx, avg_candidates in weekday_stats.items():
            if day_idx < 5:  # 平日のみ
                logger.info(f"    {weekday_names[day_idx]}曜日: {avg_candidates:.1f}銘柄")
        
        # 月別パターン
        month_stats = stats_df.groupby('month')['high_conf_total'].mean()
        logger.info("  月別高確信度候補数（上位3ヶ月）:")
        top_months = month_stats.nlargest(3)
        for month, avg_candidates in top_months.items():
            logger.info(f"    {month:2d}月: {avg_candidates:.1f}銘柄")
    
    def generate_practical_recommendations(self, stats_df):
        """実用的な推奨事項の生成"""
        logger.info("\\n" + "="*100)
        logger.info("💡 実用的な運用推奨事項")
        logger.info("="*100)
        
        avg_high_conf = stats_df['high_conf_total'].mean()
        avg_total_stocks = stats_df['total_stocks'].mean()
        
        logger.info(f"\\n🎯 現在のモデル特性:")
        logger.info(f"  • 1日平均{avg_total_stocks:.0f}銘柄を分析")
        logger.info(f"  • そのうち{avg_high_conf:.1f}銘柄が高確信度候補")
        logger.info(f"  • 選択率: {avg_high_conf/avg_total_stocks*100:.1f}%")
        
        logger.info(f"\\n📋 運用戦略の提案:")
        
        if avg_high_conf >= 20:
            logger.info(f"  🚀 豊富な候補: 上位10-15銘柄に絞って分散投資")
        elif avg_high_conf >= 10:
            logger.info(f"  ✅ 適切な候補数: 5-10銘柄での集中投資")
        elif avg_high_conf >= 5:
            logger.info(f"  ⚠️ 限定的候補: 2-5銘柄での慎重投資")
        else:
            logger.info(f"  🔍 候補少数: 確信度閾値を下げる検討が必要")
        
        logger.info(f"\\n⚙️ パラメータ調整の提案:")
        
        # 確信度閾値の推奨
        if avg_high_conf < 5:
            logger.info(f"  • 確信度閾値を52-53%に下げて候補数増加")
        elif avg_high_conf > 30:
            logger.info(f"  • 確信度閾値を60-65%に上げて精度向上")
        else:
            logger.info(f"  • 現在の55%閾値は適切")
        
        # ポートフォリオサイズ
        recommended_positions = min(10, max(3, int(avg_high_conf * 0.5)))
        logger.info(f"  • 推奨同時保有銘柄数: {recommended_positions}銘柄")
        
        # 取引頻度
        trade_frequency = stats_df['high_conf_total'].sum() / len(stats_df)
        logger.info(f"  • 想定月間取引回数: {trade_frequency * 21:.0f}回")
        
        logger.info("="*100)

def main():
    """メイン実行"""
    logger.info("🔍 日次予測候補数分析システム")
    
    analyzer = DailyPredictionAnalysis()
    
    try:
        # データ準備
        df, X, y = analyzer.load_and_prepare_data()
        
        # 日次予測分析
        stats_df = analyzer.analyze_daily_predictions(df, X, y)
        
        # 詳細分析
        analyzer.analyze_prediction_distribution(stats_df)
        
        # 実用的推奨事項
        analyzer.generate_practical_recommendations(stats_df)
        
        logger.info("\\n✅ 日次予測候補数分析完了")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()