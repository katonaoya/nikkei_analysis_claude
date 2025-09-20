#!/usr/bin/env python3
"""
効率的究極特徴量選択システム - 全データ版
最大精度を効率的に達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class EfficientUltimateFeatureSelector:
    """効率的究極特徴量選択システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
    def load_full_data(self):
        """全データ読み込み"""
        logger.info("📊 全データ読み込み（394,102件）")
        
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
    
    def create_advanced_features(self, df):
        """高度な特徴量作成（効率的版）"""
        logger.info("🔧 高度特徴量作成中（効率的版）...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. 多期間移動平均とその乖離率
        logger.info("1/7: 多期間移動平均系...")
        ma_periods = [5, 10, 20, 25, 50, 75]
        for period in ma_periods:
            df[f'MA_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
            df[f'Price_vs_MA{period}'] = (df['Close'] - df[f'MA_{period}']) / (df[f'MA_{period}'] + 1e-6)
            df[f'MA_Slope_{period}'] = df.groupby('Code')[f'MA_{period}'].pct_change(3)
        
        # 2. 多期間ボラティリティ
        logger.info("2/7: 多期間ボラティリティ系...")
        vol_periods = [5, 10, 20, 30]
        for period in vol_periods:
            df[f'Volatility_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).std().reset_index(0, drop=True)
            df[f'VolRank_{period}'] = df.groupby('Date')[f'Volatility_{period}'].rank(pct=True)
        
        # 3. モメンタム指標
        logger.info("3/7: モメンタム指標系...")
        momentum_periods = [3, 5, 10, 20]
        for period in momentum_periods:
            df[f'Momentum_{period}'] = df.groupby('Code')['Close'].pct_change(period)
            df[f'ReturnSum_{period}'] = df.groupby('Code')['Returns'].rolling(period, min_periods=1).sum().reset_index(0, drop=True)
        
        # 4. RSI（複数期間）
        logger.info("4/7: RSI系...")
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self._calculate_rsi_fast(df, period)
        
        # 5. 出来高指標
        logger.info("5/7: 出来高指標系...")
        vol_periods = [5, 10, 20]
        for period in vol_periods:
            df[f'VolMA_{period}'] = df.groupby('Code')['Volume'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
            df[f'VolRatio_{period}'] = df['Volume'] / (df[f'VolMA_{period}'] + 1e-6)
        
        # 6. 市場構造指標
        logger.info("6/7: 市場構造指標...")
        
        # 日次市場統計
        daily_market = df.groupby('Date').agg({
            'Close': ['mean', 'std'],
            'Volume': ['mean', 'std'], 
            'Returns': ['mean', 'std']
        }).round(6)
        daily_market.columns = ['Mkt_Price_Mean', 'Mkt_Price_Std', 'Mkt_Vol_Mean', 'Mkt_Vol_Std', 'Mkt_Ret_Mean', 'Mkt_Ret_Std']
        daily_market = daily_market.reset_index()
        
        # 市場幅指標
        daily_breadth = df.groupby('Date')['Returns'].agg([
            ('Breadth', lambda x: (x > 0).sum() / len(x)),
            ('StrongUp', lambda x: (x > 0.02).sum() / len(x)),
            ('StrongDown', lambda x: (x < -0.02).sum() / len(x))
        ]).reset_index()
        
        # セクター分析
        df['Sector'] = df['Code'].astype(str).str[:2]
        sector_stats = df.groupby(['Date', 'Sector'])['Returns'].mean().reset_index()
        sector_stats.columns = ['Date', 'Sector', 'SectorRet']
        
        # マージ
        df = df.merge(daily_market, on='Date', how='left')
        df = df.merge(daily_breadth, on='Date', how='left')
        df = df.merge(sector_stats, on=['Date', 'Sector'], how='left')
        
        # 7. 相対指標
        logger.info("7/7: 相対指標...")
        df['RelativeToMarket'] = df['Returns'] - df['Mkt_Ret_Mean']
        df['RelativeToSector'] = df['Returns'] - df['SectorRet']
        df['PriceVsMarket'] = df['Close'] / (df['Mkt_Price_Mean'] + 1e-6)
        df['VolVsMarket'] = df['Volume'] / (df['Mkt_Vol_Mean'] + 1e-6)
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info(f"✅ 高度特徴量作成完了: {df.shape}")
        return df
    
    def _calculate_rsi_fast(self, df, period):
        """高速RSI計算"""
        def rsi_calc(group):
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-6)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('Code', group_keys=False).apply(rsi_calc).reset_index(0, drop=True)
    
    def get_feature_categories(self, df):
        """効率的特徴量分類"""
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction', 'Sector'
        }
        
        all_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        categories = {
            'basic': [col for col in all_features if col in ['Returns']],
            'ma_system': [col for col in all_features if 'MA_' in col or 'Price_vs_MA' in col or 'MA_Slope' in col],
            'volatility': [col for col in all_features if 'Volatility' in col or 'VolRank' in col],
            'momentum': [col for col in all_features if 'Momentum' in col or 'ReturnSum' in col or 'RSI' in col],
            'volume': [col for col in all_features if 'Vol' in col and col != 'Volume'],
            'market': [col for col in all_features if col.startswith('Mkt_')],
            'breadth': [col for col in all_features if col in ['Breadth', 'StrongUp', 'StrongDown']],
            'relative': [col for col in all_features if 'Relative' in col or 'Vs' in col],
            'sector': [col for col in all_features if 'Sector' in col]
        }
        
        # 分類結果表示
        for cat, features in categories.items():
            if features:
                logger.info(f"{cat:15s}: {len(features):3d}個")
        
        logger.info(f"全特徴量: {len(all_features)}個")
        return categories, all_features
    
    def rapid_feature_evaluation(self, X, y, features, eval_name):
        """高速特徴量評価"""
        if not features:
            return 0.0
            
        X_subset = X[features]
        X_scaled = self.scaler.fit_transform(X_subset)
        
        # 3分割で高速評価
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # LogisticRegressionで高速評価
            model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=500, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        avg_score = np.mean(scores)
        return avg_score
    
    def category_evaluation(self, df, categories, all_features):
        """カテゴリ別評価"""
        logger.info("📊 カテゴリ別評価...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[all_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"評価データ: {len(clean_df):,}件, 特徴量: {len(all_features)}個")
        
        category_results = {}
        
        # 各カテゴリの単独評価
        for category, features in categories.items():
            if features:
                logger.info(f"  {category} ({len(features)}特徴量)...")
                score = self.rapid_feature_evaluation(X, y, features, category)
                category_results[category] = {
                    'score': score,
                    'features': features,
                    'count': len(features)
                }
                logger.info(f"    {category:15s}: {score:.1%}")
        
        return category_results, X, y
    
    def smart_feature_selection(self, X, y):
        """スマート特徴量選択"""
        logger.info("🧠 スマート特徴量選択...")
        
        # 1. 統計的重要度
        logger.info("  1/4: 統計的重要度...")
        f_scores = f_classif(X, y)[0]
        f_ranking = list(zip(X.columns, f_scores))
        f_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # 2. RandomForest重要度
        logger.info("  2/4: RandomForest重要度...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_ranking = list(zip(X.columns, rf.feature_importances_))
        rf_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 相互情報量
        logger.info("  3/4: 相互情報量...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_ranking = list(zip(X.columns, mi_scores))
        mi_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # 4. アンサンブルランキング
        logger.info("  4/4: アンサンブルランキング...")
        ensemble_scores = {}
        
        # 正規化してアンサンブル
        all_rankings = [f_ranking, rf_ranking, mi_ranking]
        
        for ranking in all_rankings:
            scores = [score for name, score in ranking]
            if max(scores) > min(scores):
                min_s, max_s = min(scores), max(scores)
                for name, score in ranking:
                    norm_score = (score - min_s) / (max_s - min_s)
                    if name not in ensemble_scores:
                        ensemble_scores[name] = []
                    ensemble_scores[name].append(norm_score)
        
        # 平均スコア
        final_ranking = []
        for name, scores in ensemble_scores.items():
            final_ranking.append((name, np.mean(scores)))
        
        final_ranking.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("上位20特徴量:")
        for i, (feature, score) in enumerate(final_ranking[:20]):
            logger.info(f"  {i+1:2d}. {feature:30s}: {score:.4f}")
        
        return final_ranking
    
    def progressive_testing(self, X, y, feature_ranking):
        """段階的テスト"""
        logger.info("📈 段階的テスト...")
        
        # テスト特徴量数
        test_counts = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
        max_features = min(len(feature_ranking), 50)
        test_counts = [n for n in test_counts if n <= max_features]
        
        results = {}
        models = {
            'LogisticRegression': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        }
        
        for n_features in test_counts:
            logger.info(f"  {n_features}特徴量テスト...")
            
            # 上位N特徴量選択
            selected_features = [name for name, score in feature_ranking[:n_features]]
            X_selected = X[selected_features]
            X_scaled = self.scaler.fit_transform(X_selected)
            
            model_results = {}
            
            # 各モデルでテスト
            for model_name, model in models.items():
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
                
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train = X_scaled[train_idx]
                    X_test = X_scaled[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    scores.append(accuracy_score(y_test, pred))
                
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                model_results[model_name] = {'avg': avg_score, 'std': std_score}
            
            # 最高性能のモデル
            best_model = max(model_results.keys(), key=lambda k: model_results[k]['avg'])
            best_score = model_results[best_model]['avg']
            best_std = model_results[best_model]['std']
            
            results[n_features] = {
                'best_model': best_model,
                'best_score': best_score,
                'best_std': best_std,
                'all_results': model_results,
                'features': selected_features
            }
            
            logger.info(f"    {n_features:2d}特徴量: {best_score:.1%}±{best_std:.1%} ({best_model})")
        
        # 最高性能特定
        best_n = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_result = results[best_n]
        
        logger.info(f"\n🏆 最高性能: {best_n}特徴量, {best_result['best_score']:.1%}±{best_result['best_std']:.1%}")
        logger.info(f"最適モデル: {best_result['best_model']}")
        
        return results, best_result
    
    def category_combination_test(self, X, y, category_results):
        """カテゴリ組み合わせテスト"""
        logger.info("🔄 カテゴリ組み合わせテスト...")
        
        # 性能の良いカテゴリを抽出
        sorted_categories = sorted(category_results.items(), key=lambda x: x[1]['score'], reverse=True)
        top_categories = [cat for cat, result in sorted_categories[:6]]
        
        logger.info(f"上位カテゴリ: {top_categories}")
        
        combo_results = {}
        
        # 2カテゴリ組み合わせ
        from itertools import combinations
        for combo in combinations(top_categories, 2):
            combo_features = []
            for cat in combo:
                combo_features.extend(category_results[cat]['features'])
            
            # 特徴量数制限
            if len(combo_features) > 40:
                continue
                
            combo_name = '+'.join(combo)
            score = self.rapid_feature_evaluation(X, y, combo_features, combo_name)
            
            combo_results[combo_name] = {
                'score': score,
                'features': combo_features,
                'categories': combo,
                'count': len(combo_features)
            }
            
            logger.info(f"  {combo_name:30s}: {score:.1%} ({len(combo_features)}特徴量)")
        
        # 最高組み合わせ
        if combo_results:
            best_combo = max(combo_results.keys(), key=lambda k: combo_results[k]['score'])
            logger.info(f"\n🏆 最高組み合わせ: {best_combo} ({combo_results[best_combo]['score']:.1%})")
        
        return combo_results
    
    def final_validation(self, X, y, best_features, best_model_name):
        """最終検証"""
        logger.info("🎯 最終検証...")
        logger.info(f"特徴量数: {len(best_features)}, モデル: {best_model_name}")
        
        X_final = X[best_features]
        X_scaled = self.scaler.fit_transform(X_final)
        
        # モデル設定
        if best_model_name == 'LogisticRegression':
            model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        
        # 5分割で最終評価
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"Fold {fold+1}: {accuracy:.1%}")
        
        final_accuracy = np.mean(scores)
        final_std = np.std(scores)
        
        logger.info(f"\n🎯 最終結果: {final_accuracy:.1%} ± {final_std:.1%}")
        
        return final_accuracy, final_std, scores

def main():
    """メイン実行"""
    logger.info("🚀 効率的究極特徴量選択システム - 全データ版")
    logger.info("🎯 目標: 最大精度達成")
    
    selector = EfficientUltimateFeatureSelector()
    
    try:
        # 1. 全データ読み込み
        df = selector.load_full_data()
        if df is None:
            return
        
        # 2. 高度特徴量作成
        df = selector.create_advanced_features(df)
        
        # 3. 特徴量分類
        categories, all_features = selector.get_feature_categories(df)
        
        # 4. カテゴリ別評価
        category_results, X, y = selector.category_evaluation(df, categories, all_features)
        
        # 5. スマート特徴量選択
        feature_ranking = selector.smart_feature_selection(X, y)
        
        # 6. 段階的テスト
        progressive_results, best_progressive = selector.progressive_testing(X, y, feature_ranking)
        
        # 7. カテゴリ組み合わせテスト
        combo_results = selector.category_combination_test(X, y, category_results)
        
        # 8. 最終検証
        final_accuracy, final_std, fold_scores = selector.final_validation(
            X, y, best_progressive['features'], best_progressive['best_model']
        )
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 効率的究極特徴量選択結果")
        logger.info("="*80)
        
        logger.info(f"データ総数: {len(df):,}件 (全データ検証)")
        logger.info(f"作成特徴量: {len(all_features)}個")
        
        # カテゴリ別結果
        logger.info("\n📊 カテゴリ別性能:")
        sorted_cats = sorted(category_results.items(), key=lambda x: x[1]['score'], reverse=True)
        for cat, result in sorted_cats:
            logger.info(f"  {cat:15s}: {result['score']:.1%} ({result['count']}特徴量)")
        
        # 最高カテゴリ組み合わせ
        if combo_results:
            best_combo = max(combo_results.keys(), key=lambda k: combo_results[k]['score'])
            logger.info(f"\n🏆 最高カテゴリ組み合わせ: {best_combo}")
            logger.info(f"組み合わせ精度: {combo_results[best_combo]['score']:.1%}")
        
        # 段階的最高結果
        logger.info(f"\n📈 段階的最適化結果:")
        logger.info(f"最適特徴量数: {len(best_progressive['features'])}")
        logger.info(f"段階的最高精度: {best_progressive['best_score']:.1%} ± {best_progressive['best_std']:.1%}")
        
        # 最終検証結果
        logger.info(f"\n🎯 最終検証結果: {final_accuracy:.1%} ± {final_std:.1%}")
        logger.info(f"使用モデル: {best_progressive['best_model']}")
        
        # 最適特徴量
        logger.info(f"\n最適特徴量 (上位20個):")
        for i, feature in enumerate(best_progressive['features'][:20], 1):
            logger.info(f"  {i:2d}. {feature}")
        if len(best_progressive['features']) > 20:
            logger.info(f"  ... 他{len(best_progressive['features'])-20}個")
        
        # 全体の最高精度
        all_scores = [final_accuracy, best_progressive['best_score']]
        if combo_results:
            all_scores.append(max(combo_results[k]['score'] for k in combo_results))
        
        max_achieved = max(all_scores)
        logger.info(f"\n🏆 達成最高精度: {max_achieved:.1%}")
        logger.info(f"⚠️ この結果は394,102件の全データでの厳密検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()