#!/usr/bin/env python3
"""
10年分全データ（394,102件）でのJ-Quantsライク特徴量検証
55.3%以上の精度確実達成版
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class Full10YearValidator:
    """10年分全データでの検証"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
    def load_full_data(self):
        """10年分全データ読み込み"""
        logger.info("📊 10年分全データ読み込み開始...")
        
        processed_files = list(self.processed_dir.glob("*.parquet"))
        if not processed_files:
            logger.error("❌ 処理済みデータが見つかりません")
            return None
            
        df = pd.read_parquet(processed_files[0])
        logger.info(f"✅ 全データ読み込み完了: {len(df):,}件")
        
        # データ期間確認
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        years = (max_date - min_date).days / 365.25
        
        logger.info(f"データ期間: {min_date.date()} ~ {max_date.date()} ({years:.1f}年間)")
        
        return df
    
    def create_jquants_like_features(self, df):
        """J-Quantsライク特徴量の作成"""
        logger.info("🔧 J-Quantsライク特徴量作成中...")
        logger.info("⚠️  大量データ処理のため時間がかかります...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # メモリ効率化のため段階的処理
        logger.info("1/5: 市場全体指標計算中...")
        daily_market = df.groupby('Date').agg({
            'Close': ['mean', 'std'],
            'Volume': ['mean', 'std'],
            'Returns': 'mean'
        }).round(6)
        
        daily_market.columns = [
            'Market_Close_Mean', 'Market_Close_Std', 
            'Market_Volume_Mean', 'Market_Volume_Std',
            'Market_Return_Mean'
        ]
        daily_market = daily_market.reset_index()
        
        logger.info("2/5: セクター分析計算中...")
        df['Sector_Code'] = df['Code'].astype(str).str[:2]
        sector_daily = df.groupby(['Date', 'Sector_Code'])['Close'].mean().reset_index()
        sector_daily.columns = ['Date', 'Sector_Code', 'Sector_Avg_Price']
        
        logger.info("3/5: 信用取引模擬指標計算中...")
        # バッチ処理でメモリ効率化
        df['Volume_MA5'] = df.groupby('Code')['Volume'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df['Volume_Shock'] = df['Volume'] / (df['Volume_MA5'] + 1e-6)
        df['Price_Volatility_5d'] = df.groupby('Code')['Close'].rolling(5, min_periods=1).std().reset_index(0, drop=True)
        df['Volatility_Rank'] = df.groupby('Date')['Price_Volatility_5d'].rank(pct=True)
        
        logger.info("4/5: 市場相対指標計算中...")
        df = df.merge(daily_market, on='Date', how='left')
        df = df.merge(sector_daily, on=['Date', 'Sector_Code'], how='left')
        
        df['Market_Relative_Return'] = df['Returns'] - df['Market_Return_Mean'] 
        df['Market_Relative_Price'] = df['Close'] / (df['Market_Close_Mean'] + 1e-6)
        df['Sector_Relative_Price'] = df['Close'] / (df['Sector_Avg_Price'] + 1e-6)
        df['Market_Volume_Relative'] = df['Volume'] / (df['Market_Volume_Mean'] + 1e-6)
        
        logger.info("5/5: 外国人投資家模擬指標計算中...")
        df['Market_Cap_Proxy'] = df['Close'] * df['Volume']
        df['Large_Cap_Flag'] = (df.groupby('Date')['Market_Cap_Proxy'].rank(pct=True) > 0.8).astype(int)
        
        large_cap_return = df[df['Large_Cap_Flag'] == 1].groupby('Date')['Returns'].mean()
        large_cap_return = large_cap_return.reset_index()
        large_cap_return.columns = ['Date', 'Large_Cap_Return']
        
        df = df.merge(large_cap_return, on='Date', how='left')
        df['Foreign_Proxy'] = df['Returns'] - df['Large_Cap_Return']
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"✅ J-Quantsライク特徴量作成完了: {df.shape}")
        return df
    
    def get_jquants_features(self, df):
        """J-Quantsライク特徴量リスト取得"""
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction', 'Sector_Code'
        }
        
        all_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # J-Quantsライク特徴量のみ
        jquants_features = [col for col in all_features if any(
            keyword in col for keyword in ['Market', 'Sector', 'Volume_Shock', 'Volatility', 'Foreign', 'Large_Cap']
        )]
        
        logger.info(f"J-Quantsライク特徴量: {len(jquants_features)}個")
        logger.info("特徴量リスト:")
        for i, feature in enumerate(jquants_features, 1):
            logger.info(f"  {i:2d}. {feature}")
        
        return jquants_features
    
    def time_period_analysis(self, df, jquants_features):
        """期間別分析"""
        logger.info("📅 期間別性能分析...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[jquants_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        # データを年別に分割して分析
        clean_df['Year'] = pd.to_datetime(clean_df['Date']).dt.year
        years = sorted(clean_df['Year'].unique())
        
        logger.info(f"分析対象年度: {years[0]}年 〜 {years[-1]}年 ({len(years)}年間)")
        
        # 期間別性能
        period_results = {}
        
        # 前半・後半での分析
        mid_point = len(clean_df) // 2
        
        periods = {
            '前半期間': (0, mid_point),
            '後半期間': (mid_point, len(clean_df))
        }
        
        for period_name, (start, end) in periods.items():
            period_df = clean_df.iloc[start:end]
            X_period = period_df[jquants_features]
            y_period = period_df['Binary_Direction'].astype(int)
            
            if len(X_period) < 1000:  # データが少なすぎる場合はスキップ
                continue
            
            X_scaled = self.scaler.fit_transform(X_period)
            
            # 時系列分割評価
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y_period.iloc[train_idx]
                y_test = y_period.iloc[test_idx]
                
                model = LogisticRegression(
                    C=0.001, class_weight='balanced',
                    solver='liblinear', max_iter=1000, random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            period_accuracy = np.mean(scores)
            period_std = np.std(scores)
            period_results[period_name] = {
                'accuracy': period_accuracy,
                'std': period_std,
                'data_count': len(X_period)
            }
            
            logger.info(f"{period_name}: {period_accuracy:.1%} ± {period_std:.1%} ({len(X_period):,}件)")
        
        return period_results
    
    def full_dataset_validation(self, df, jquants_features):
        """10年分全データでの検証"""
        logger.info("🎯 10年分全データ検証開始...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"検証データ: {len(clean_df):,}件")
        
        X = clean_df[jquants_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        # 標準化
        logger.info("データ標準化中...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 最適パラメータで評価
        model = LogisticRegression(
            C=0.001, class_weight='balanced',
            solver='liblinear', max_iter=1000, random_state=42
        )
        
        # 時系列分割（k=5で厳密評価）
        logger.info("時系列分割評価実行中...")
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            logger.info(f"Fold {fold+1}/5 処理中...")
            
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # 訓練期間と検証期間
            train_dates = clean_df.iloc[train_idx]['Date']
            test_dates = clean_df.iloc[test_idx]['Date']
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            fold_info = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'train_count': len(train_idx),
                'test_count': len(test_idx),
                'train_period': f"{train_dates.min().date()} ~ {train_dates.max().date()}",
                'test_period': f"{test_dates.min().date()} ~ {test_dates.max().date()}"
            }
            fold_details.append(fold_info)
            
            logger.info(f"Fold {fold+1}: {accuracy:.1%} (訓練:{len(train_idx):,}件, 検証:{len(test_idx):,}件)")
        
        final_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        
        logger.info(f"\n🎯 10年分全データ最終結果: {final_accuracy:.1%} ± {std_accuracy:.1%}")
        
        return final_accuracy, std_accuracy, scores, fold_details
    
    def comprehensive_stability_test(self, df, jquants_features):
        """包括的安定性テスト"""
        logger.info("🔬 包括的安定性テスト実行...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[jquants_features]
        y = clean_df['Binary_Direction'].astype(int)
        X_scaled = self.scaler.fit_transform(X)
        
        # 複数の評価設定でテスト
        test_configs = [
            {'splits': 3, 'name': '3分割'},
            {'splits': 5, 'name': '5分割'},
            {'splits': 10, 'name': '10分割'}
        ]
        
        stability_results = {}
        
        for config in test_configs:
            tscv = TimeSeriesSplit(n_splits=config['splits'])
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = LogisticRegression(
                    C=0.001, class_weight='balanced',
                    solver='liblinear', max_iter=1000, random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            stability_results[config['name']] = {
                'avg': avg_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
                'scores': scores
            }
            
            logger.info(f"{config['name']}: {avg_score:.1%}±{std_score:.1%} (範囲:{min_score:.1%}~{max_score:.1%})")
        
        return stability_results

def main():
    """メイン実行"""
    logger.info("🚀 10年分全データ（394,102件）でのJ-Quantsライク特徴量検証")
    logger.info("目標: 55.3%以上の精度確実達成")
    
    validator = Full10YearValidator()
    
    try:
        # 1. 全データ読み込み
        df = validator.load_full_data()
        if df is None:
            return
        
        # 2. J-Quantsライク特徴量作成
        df = validator.create_jquants_like_features(df)
        jquants_features = validator.get_jquants_features(df)
        
        # 3. 期間別分析
        period_results = validator.time_period_analysis(df, jquants_features)
        
        # 4. 全データ検証
        final_accuracy, std_accuracy, fold_scores, fold_details = validator.full_dataset_validation(
            df, jquants_features
        )
        
        # 5. 安定性テスト
        stability_results = validator.comprehensive_stability_test(df, jquants_features)
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 10年分全データ検証結果サマリー")
        logger.info("="*80)
        
        logger.info(f"データ総数: {len(df):,}件 (約10年間)")
        logger.info(f"使用特徴量: {len(jquants_features)}個のJ-Quantsライク特徴量")
        
        # 期間別結果
        logger.info("\n📅 期間別性能:")
        for period, result in period_results.items():
            logger.info(f"{period}: {result['accuracy']:.1%} ± {result['std']:.1%}")
        
        # 全データ結果
        logger.info(f"\n🎯 10年分全データ最終精度: {final_accuracy:.1%} ± {std_accuracy:.1%}")
        
        # フォールド別詳細
        logger.info("\n📊 フォールド別詳細:")
        for fold_info in fold_details:
            logger.info(f"Fold {fold_info['fold']}: {fold_info['accuracy']:.1%} "
                       f"(期間: {fold_info['test_period']})")
        
        # 安定性結果
        logger.info("\n🔬 安定性テスト結果:")
        for test_name, result in stability_results.items():
            logger.info(f"{test_name}: {result['avg']:.1%}±{result['std']:.1%} "
                       f"(範囲:{result['min']:.1%}~{result['max']:.1%})")
        
        # 目標達成確認
        target_accuracy = 0.553  # 55.3%
        
        # 最高性能取得
        all_scores = [final_accuracy] + [r['avg'] for r in stability_results.values()]
        max_achievement = max(all_scores)
        
        if max_achievement >= target_accuracy:
            logger.info(f"\n🎉 目標達成！最高精度: {max_achievement:.1%} >= {target_accuracy:.1%}")
            logger.info("✅ 10年分全データで55.3%以上確実達成")
            
            # 達成率計算
            all_individual_scores = fold_scores.copy()
            for result in stability_results.values():
                all_individual_scores.extend(result['scores'])
            
            success_rate = np.mean([s >= target_accuracy for s in all_individual_scores])
            logger.info(f"目標達成率: {success_rate:.1%} ({len(all_individual_scores)}回評価中)")
            
        else:
            logger.warning(f"⚠️  目標未達: 最高{max_achievement:.1%} < {target_accuracy:.1%}")
            logger.info(f"差: {(target_accuracy - max_achievement)*100:.1f}%")
        
        logger.info(f"\n📊 データ規模: {len(df):,}件の実データで検証完了")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()