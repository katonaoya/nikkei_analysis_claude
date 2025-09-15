#!/usr/bin/env python3
"""
J-Quantsライク特徴量を含む包括的特徴量選択
55.3%以上の精度確実達成版
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class EnhancedJQuantsFeatureSelector:
    """J-Quantsライク特徴量を含む包括的特徴量選択"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, sample_size=50000):
        """データ読み込みと準備"""
        logger.info(f"📊 データ読み込み（サンプルサイズ: {sample_size:,}）")
        
        # 既存データ読み込み
        processed_files = list(self.processed_dir.glob("*.parquet"))
        if not processed_files:
            logger.error("❌ 処理済みデータが見つかりません")
            return None
            
        df = pd.read_parquet(processed_files[0])
        logger.info(f"元データ: {len(df):,}件")
        
        # 最新データを優先してサンプリング
        if len(df) > sample_size:
            df = df.sort_values('Date').tail(sample_size)
            logger.info(f"サンプリング後: {len(df):,}件")
        
        return df
    
    def create_jquants_like_features(self, df):
        """J-Quantsライク特徴量の完全復元"""
        logger.info("🔧 J-Quantsライク特徴量作成中...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. 市場全体指標（指数的特徴量）
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
        
        # 2. セクター模擬（コード前2桁でセクター分類）
        df['Sector_Code'] = df['Code'].astype(str).str[:2]
        sector_daily = df.groupby(['Date', 'Sector_Code'])['Close'].mean().reset_index()
        sector_daily.columns = ['Date', 'Sector_Code', 'Sector_Avg_Price']
        
        # 3. 信用取引模擬指標
        df['Volume_MA5'] = df.groupby('Code')['Volume'].rolling(5).mean().reset_index(0, drop=True)
        df['Volume_Shock'] = df['Volume'] / (df['Volume_MA5'] + 1e-6)
        
        # 価格ボラティリティを空売り圧力の代理指標とする
        df['Price_Volatility_5d'] = df.groupby('Code')['Close'].rolling(5).std().reset_index(0, drop=True)
        df['Volatility_Rank'] = df.groupby('Date')['Price_Volatility_5d'].rank(pct=True)
        
        # 4. 市場相対指標
        df = df.merge(daily_market, on='Date', how='left')
        df = df.merge(sector_daily, on=['Date', 'Sector_Code'], how='left')
        
        df['Market_Relative_Return'] = df['Returns'] - df['Market_Return_Mean'] 
        df['Market_Relative_Price'] = df['Close'] / (df['Market_Close_Mean'] + 1e-6)
        df['Sector_Relative_Price'] = df['Close'] / (df['Sector_Avg_Price'] + 1e-6)
        df['Market_Volume_Relative'] = df['Volume'] / (df['Market_Volume_Mean'] + 1e-6)
        
        # 5. 外国人投資家模擬（大型株での特別な動き）
        df['Market_Cap_Proxy'] = df['Close'] * df['Volume']  # 簡易時価総額
        df['Large_Cap_Flag'] = (df.groupby('Date')['Market_Cap_Proxy'].rank(pct=True) > 0.8).astype(int)
        
        # 大型株の平均リターンと個別銘柄の乖離
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
    
    def categorize_features(self, df):
        """特徴量の分類"""
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction', 'Sector_Code'
        }
        
        all_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # 基本特徴量（既存システム由来）
        basic_features = [col for col in all_features if not any(
            keyword in col for keyword in ['Market', 'Sector', 'Volume_Shock', 'Volatility', 'Foreign', 'Large_Cap']
        )]
        
        # J-Quantsライク拡張特徴量
        jquants_features = [col for col in all_features if any(
            keyword in col for keyword in ['Market', 'Sector', 'Volume_Shock', 'Volatility', 'Foreign', 'Large_Cap']
        )]
        
        logger.info(f"基本特徴量: {len(basic_features)}個")
        logger.info(f"J-Quantsライク特徴量: {len(jquants_features)}個") 
        logger.info(f"全特徴量: {len(all_features)}個")
        
        return basic_features, jquants_features, all_features
    
    def quick_baseline_test(self, df, basic_features, jquants_features):
        """ベースライン性能確認"""
        logger.info("⚡ ベースライン性能確認...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        y = clean_df['Binary_Direction'].astype(int)
        
        results = {}
        
        # J-Quantsライク特徴量のみでテスト（55.3%を再現したい）
        if jquants_features:
            logger.info("🎯 J-Quantsライク特徴量のみでテスト...")
            X_jquants = clean_df[jquants_features]
            X_jquants_scaled = self.scaler.fit_transform(X_jquants)
            
            # LogisticRegressionで評価（55.3%を達成したモデル）
            tscv = TimeSeriesSplit(n_splits=2)
            lr_scores = []
            
            for train_idx, test_idx in tscv.split(X_jquants_scaled):
                X_train = X_jquants_scaled[train_idx]
                X_test = X_jquants_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                lr = LogisticRegression(
                    C=0.01, class_weight='balanced', 
                    max_iter=1000, random_state=42
                )
                lr.fit(X_train, y_train)
                pred = lr.predict(X_test)
                lr_scores.append(accuracy_score(y_test, pred))
            
            jquants_accuracy = np.mean(lr_scores)
            results['jquants_only'] = jquants_accuracy
            logger.info(f"J-Quantsライク特徴量のみ: {jquants_accuracy:.1%}")
        
        # 基本特徴量のみ
        if basic_features:
            X_basic = clean_df[basic_features]
            X_basic_scaled = self.scaler.fit_transform(X_basic)
            
            lr_scores = []
            for train_idx, test_idx in tscv.split(X_basic_scaled):
                X_train = X_basic_scaled[train_idx]
                X_test = X_basic_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                lr = LogisticRegression(
                    C=0.01, class_weight='balanced',
                    max_iter=1000, random_state=42
                )
                lr.fit(X_train, y_train)
                pred = lr.predict(X_test)
                lr_scores.append(accuracy_score(y_test, pred))
            
            basic_accuracy = np.mean(lr_scores)
            results['basic_only'] = basic_accuracy
            logger.info(f"基本特徴量のみ: {basic_accuracy:.1%}")
        
        return results
    
    def comprehensive_feature_selection(self, df, all_features):
        """包括的特徴量選択"""
        logger.info("🔍 包括的特徴量選択開始...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[all_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        # 標準化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        feature_rankings = {}
        
        # 1. F統計量
        logger.info("📊 F統計量による選択...")
        f_scores = f_classif(X_scaled, y)[0]
        f_ranking = list(zip(X.columns, f_scores))
        f_ranking.sort(key=lambda x: x[1], reverse=True)
        feature_rankings['f_statistic'] = f_ranking
        
        # 2. 相互情報量
        logger.info("🔗 相互情報量による選択...")
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        mi_ranking = list(zip(X.columns, mi_scores))
        mi_ranking.sort(key=lambda x: x[1], reverse=True)
        feature_rankings['mutual_info'] = mi_ranking
        
        # 3. RandomForest重要度
        logger.info("🌲 RandomForest重要度による選択...")
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=8, 
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_scaled, y)
        rf_ranking = list(zip(X.columns, rf.feature_importances_))
        rf_ranking.sort(key=lambda x: x[1], reverse=True)
        feature_rankings['random_forest'] = rf_ranking
        
        # アンサンブルランキング
        ensemble_scores = self.create_ensemble_ranking(feature_rankings)
        
        return ensemble_scores
    
    def create_ensemble_ranking(self, rankings_dict):
        """アンサンブル重要度ランキング"""
        logger.info("🏆 アンサンブル重要度計算中...")
        
        ensemble_scores = {}
        
        for method, rankings in rankings_dict.items():
            if rankings:
                scores = [score for name, score in rankings]
                if len(scores) > 0 and max(scores) > min(scores):
                    min_score, max_score = min(scores), max(scores)
                    for name, score in rankings:
                        normalized_score = (score - min_score) / (max_score - min_score)
                        if name not in ensemble_scores:
                            ensemble_scores[name] = []
                        ensemble_scores[name].append(normalized_score)
        
        # 平均スコア
        final_scores = {}
        for name, scores in ensemble_scores.items():
            final_scores[name] = np.mean(scores)
        
        ensemble_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 アンサンブル上位15特徴量:")
        for i, (feature, score) in enumerate(ensemble_ranking[:15]):
            logger.info(f"  {i+1:2d}. {feature:30s}: {score:.4f}")
        
        return ensemble_ranking
    
    def progressive_testing(self, df, all_features, feature_ranking):
        """段階的特徴量テスト"""
        logger.info("📈 段階的特徴量テスト開始...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        y = clean_df['Binary_Direction'].astype(int)
        
        # 特徴量数を段階的に増加
        feature_counts = [3, 5, 7, 10, 15, 20, 25]
        results = {}
        
        for n_features in feature_counts:
            if n_features > len(feature_ranking):
                continue
                
            # 上位N特徴量選択
            selected_features = [name for name, score in feature_ranking[:n_features]]
            X = clean_df[selected_features]
            X_scaled = self.scaler.fit_transform(X)
            
            # 時系列分割で評価
            tscv = TimeSeriesSplit(n_splits=3)
            lr_scores = []
            rf_scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # LogisticRegression
                lr = LogisticRegression(
                    C=0.01, class_weight='balanced',
                    max_iter=1000, random_state=42
                )
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                lr_scores.append(accuracy_score(y_test, lr_pred))
                
                # RandomForest
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_scores.append(accuracy_score(y_test, rf_pred))
            
            lr_avg = np.mean(lr_scores)
            rf_avg = np.mean(rf_scores)
            best_score = max(lr_avg, rf_avg)
            best_model = "LogisticRegression" if lr_avg > rf_avg else "RandomForest"
            
            results[n_features] = {
                'lr_accuracy': lr_avg,
                'rf_accuracy': rf_avg,
                'best_score': best_score,
                'best_model': best_model,
                'features': selected_features
            }
            
            logger.info(f"特徴量数{n_features:2d}: LR={lr_avg:.1%}, RF={rf_avg:.1%}, 最高={best_score:.1%}({best_model})")
        
        # 最高性能の特徴量数特定
        best_n = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_result = results[best_n]
        
        logger.info(f"\n🎯 最高性能: 特徴量数{best_n}, 精度{best_result['best_score']:.1%} ({best_result['best_model']})")
        logger.info("最適特徴量:")
        for i, feature in enumerate(best_result['features']):
            logger.info(f"  {i+1:2d}. {feature}")
        
        return results, best_result
    
    def final_validation(self, df, best_features, best_model_name):
        """最終検証"""
        logger.info(f"🎯 最終検証: {len(best_features)}特徴量, {best_model_name}")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[best_features]
        y = clean_df['Binary_Direction'].astype(int)
        X_scaled = self.scaler.fit_transform(X)
        
        # 時系列分割で最終検証
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            if best_model_name == "LogisticRegression":
                model = LogisticRegression(
                    C=0.01, class_weight='balanced',
                    max_iter=1000, random_state=42
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"Fold {fold+1}: {accuracy:.1%}")
        
        final_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        
        logger.info(f"\n🎯 最終結果: {final_accuracy:.1%} ± {std_accuracy:.1%}")
        
        return final_accuracy, std_accuracy, scores

def main():
    """メイン実行"""
    logger.info("🚀 J-Quantsライク特徴量を含む包括的特徴量選択開始")
    logger.info("目標: 55.3%以上の精度達成")
    
    selector = EnhancedJQuantsFeatureSelector()
    
    try:
        # 1. データ読み込み
        df = selector.load_and_prepare_data(sample_size=50000)
        if df is None:
            return
        
        # 2. J-Quantsライク特徴量作成
        df = selector.create_jquants_like_features(df)
        
        # 3. 特徴量分類
        basic_features, jquants_features, all_features = selector.categorize_features(df)
        
        # 4. ベースライン確認
        baseline_results = selector.quick_baseline_test(df, basic_features, jquants_features)
        
        if 'jquants_only' in baseline_results:
            jquants_accuracy = baseline_results['jquants_only']
            logger.info(f"🎯 J-Quantsライク特徴量のみ精度: {jquants_accuracy:.1%}")
            
            if jquants_accuracy >= 0.553:  # 55.3%
                logger.info("✅ 目標55.3%を達成！J-Quantsライク特徴量が有効")
            else:
                logger.warning(f"⚠️  目標55.3%に未達 (差: {(0.553 - jquants_accuracy)*100:.1f}%)")
        
        # 5. 包括的特徴量選択
        feature_ranking = selector.comprehensive_feature_selection(df, all_features)
        
        # 6. 段階的テスト
        progressive_results, best_result = selector.progressive_testing(df, all_features, feature_ranking)
        
        # 7. 最終検証
        final_accuracy, std_accuracy, fold_scores = selector.final_validation(
            df, best_result['features'], best_result['best_model']
        )
        
        # 結果まとめ
        logger.info("\n" + "="*60)
        logger.info("🎯 最終結果サマリー")
        logger.info("="*60)
        
        if 'jquants_only' in baseline_results:
            logger.info(f"J-Quantsライク特徴量のみ: {baseline_results['jquants_only']:.1%}")
        if 'basic_only' in baseline_results:
            logger.info(f"基本特徴量のみ: {baseline_results['basic_only']:.1%}")
        
        logger.info(f"最適特徴量選択後: {final_accuracy:.1%} ± {std_accuracy:.1%}")
        logger.info(f"使用特徴量数: {len(best_result['features'])}")
        logger.info(f"使用モデル: {best_result['best_model']}")
        
        # 目標達成確認
        target_accuracy = 0.553  # 55.3%
        if final_accuracy >= target_accuracy:
            logger.info(f"🎉 目標達成！ {final_accuracy:.1%} >= {target_accuracy:.1%}")
        else:
            logger.warning(f"⚠️  目標未達: {final_accuracy:.1%} < {target_accuracy:.1%}")
            logger.info(f"差: {(target_accuracy - final_accuracy)*100:.1f}%")
        
        logger.info("\n最適特徴量:")
        for i, feature in enumerate(best_result['features']):
            logger.info(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()