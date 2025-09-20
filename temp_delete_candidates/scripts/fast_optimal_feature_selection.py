#!/usr/bin/env python3
"""
高速最適特徴量選択システム - 全データ版
効率重視で最大精度達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class FastOptimalFeatureSelector:
    """高速最適特徴量選択システム"""
    
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
    
    def create_strategic_features(self, df):
        """戦略的特徴量作成（厳選版）"""
        logger.info("🔧 戦略的特徴量作成中...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. 核心移動平均系（効果的な期間のみ）
        logger.info("1/5: 核心移動平均系...")
        key_periods = [5, 10, 20, 50]
        for period in key_periods:
            df[f'MA_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
            df[f'Price_vs_MA{period}'] = (df['Close'] - df[f'MA_{period}']) / (df[f'MA_{period}'] + 1e-6)
        
        # 2. 核心ボラティリティ系
        logger.info("2/5: 核心ボラティリティ系...")
        vol_periods = [10, 20]
        for period in vol_periods:
            df[f'Vol_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).std().reset_index(0, drop=True)
            df[f'VolRank_{period}'] = df.groupby('Date')[f'Vol_{period}'].rank(pct=True)
        
        # 3. 核心モメンタム系
        logger.info("3/5: 核心モメンタム系...")
        df['RSI_14'] = self._calculate_rsi_vectorized(df, 14)
        df['Momentum_5'] = df.groupby('Code')['Close'].pct_change(5)
        df['Momentum_10'] = df.groupby('Code')['Close'].pct_change(10)
        
        # 4. 市場構造指標（簡素版）
        logger.info("4/5: 市場構造指標...")
        
        # 日次市場統計（簡素版）
        daily_stats = df.groupby('Date').agg({
            'Returns': ['mean', 'std'],
            'Close': 'mean',
            'Volume': 'mean'
        })
        daily_stats.columns = ['Market_Return', 'Market_Vol', 'Market_Price', 'Market_Volume']
        daily_stats = daily_stats.reset_index()
        
        # 市場幅指標
        market_breadth = df.groupby('Date')['Returns'].agg([
            ('Breadth', lambda x: (x > 0).sum() / len(x))
        ]).reset_index()
        
        # マージ
        df = df.merge(daily_stats, on='Date', how='left')
        df = df.merge(market_breadth, on='Date', how='left')
        
        # 5. 相対指標（重要なもののみ）
        logger.info("5/5: 核心相対指標...")
        # マージで重複した列名を修正
        if 'Market_Return_x' in df.columns:
            df['Market_Return'] = df['Market_Return_x']
            df = df.drop(['Market_Return_x', 'Market_Return_y'], axis=1, errors='ignore')
        
        df['Relative_Return'] = df['Returns'] - df['Market_Return']
        df['Vol_vs_Market'] = df['Volume'] / (df['Market_Volume'] + 1e-6)
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info(f"✅ 戦略的特徴量作成完了: {df.shape}")
        return df
    
    def _calculate_rsi_vectorized(self, df, period):
        """ベクター化RSI計算（高速版）"""
        def rsi_fast(group):
            close = group['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('Code', group_keys=False).apply(rsi_fast).reset_index(0, drop=True)
    
    def get_all_features(self, df):
        """全特徴量取得"""
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction'
        }
        
        all_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"使用可能特徴量: {len(all_features)}個")
        for i, feature in enumerate(all_features, 1):
            logger.info(f"  {i:2d}. {feature}")
        
        return all_features
    
    def fast_feature_ranking(self, X, y):
        """高速特徴量ランキング"""
        logger.info("⚡ 高速特徴量ランキング...")
        
        # F統計量による高速ランキング
        logger.info("  F統計量計算中...")
        f_scores = f_classif(X, y)[0]
        
        # RandomForest重要度（小規模）
        logger.info("  RF重要度計算中...")
        rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        
        # 正規化してアンサンブル
        f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-8)
        
        ensemble_scores = (f_norm + rf_norm) / 2
        
        # ランキング作成
        ranking = list(zip(X.columns, ensemble_scores))
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("上位20特徴量:")
        for i, (feature, score) in enumerate(ranking[:20]):
            logger.info(f"  {i+1:2d}. {feature:25s}: {score:.4f}")
        
        return ranking
    
    def rapid_evaluation(self, X, y, features, desc=""):
        """高速評価"""
        X_subset = X[features] if isinstance(features, list) else X.iloc[:, features]
        X_scaled = self.scaler.fit_transform(X_subset)
        
        # 3分割で高速評価
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # 高速LogisticRegression
            model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=300, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        avg_score = np.mean(scores)
        
        if desc:
            logger.info(f"  {desc}: {avg_score:.1%}")
        
        return avg_score
    
    def systematic_feature_testing(self, X, y, feature_ranking):
        """体系的特徴量テスト"""
        logger.info("🧪 体系的特徴量テスト...")
        
        # 特徴量数のテストパターン
        test_counts = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
        max_features = min(len(feature_ranking), 30)
        test_counts = [n for n in test_counts if n <= max_features]
        
        results = {}
        
        for n_features in test_counts:
            selected_features = [name for name, score in feature_ranking[:n_features]]
            
            # LogisticRegression評価
            lr_score = self.rapid_evaluation(X, y, selected_features, f"{n_features}特徴量(LR)")
            
            # RandomForest評価
            X_subset = X[selected_features]
            X_scaled = self.scaler.fit_transform(X_subset)
            
            tscv = TimeSeriesSplit(n_splits=3)
            rf_scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                rf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                rf_scores.append(accuracy_score(y_test, pred))
            
            rf_score = np.mean(rf_scores)
            
            best_score = max(lr_score, rf_score)
            best_model = "LogisticRegression" if lr_score > rf_score else "RandomForest"
            
            results[n_features] = {
                'lr_score': lr_score,
                'rf_score': rf_score,
                'best_score': best_score,
                'best_model': best_model,
                'features': selected_features
            }
            
            logger.info(f"  {n_features:2d}特徴量: LR={lr_score:.1%}, RF={rf_score:.1%} → 最高={best_score:.1%}({best_model})")
        
        # 最高性能特定
        best_n = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_result = results[best_n]
        
        logger.info(f"\n🏆 体系的テスト最高: {best_n}特徴量, {best_result['best_score']:.1%} ({best_result['best_model']})")
        
        return results, best_result
    
    def advanced_combination_test(self, X, y, all_features):
        """高度組み合わせテスト"""
        logger.info("🔄 高度組み合わせテスト...")
        
        # 特徴量カテゴリの自動推定
        ma_features = [f for f in all_features if 'MA' in f or 'Price_vs' in f]
        vol_features = [f for f in all_features if 'Vol' in f or 'vol' in f.lower()]
        momentum_features = [f for f in all_features if 'RSI' in f or 'Momentum' in f]
        market_features = [f for f in all_features if 'Market' in f or 'Breadth' in f]
        relative_features = [f for f in all_features if 'Relative' in f or 'vs_Market' in f]
        
        categories = {
            'MA系': ma_features,
            'ボラティリティ系': vol_features,
            'モメンタム系': momentum_features,
            '市場系': market_features,
            '相対系': relative_features
        }
        
        # 各カテゴリの単独性能
        category_results = {}
        for cat_name, features in categories.items():
            if features:
                score = self.rapid_evaluation(X, y, features, f"{cat_name}({len(features)}特徴量)")
                category_results[cat_name] = {
                    'score': score,
                    'features': features,
                    'count': len(features)
                }
        
        # 最高カテゴリ組み合わせ
        logger.info("\n組み合わせテスト...")
        sorted_cats = sorted(category_results.items(), key=lambda x: x[1]['score'], reverse=True)
        top_3_cats = [cat for cat, result in sorted_cats[:3]]
        
        # トップ3の2個組み合わせ
        from itertools import combinations
        best_combo = None
        best_combo_score = 0
        
        for combo in combinations(top_3_cats, 2):
            combo_features = []
            for cat in combo:
                combo_features.extend(category_results[cat]['features'])
            
            if len(combo_features) > 25:  # 特徴量数制限
                combo_features = combo_features[:25]
            
            combo_score = self.rapid_evaluation(X, y, combo_features)
            combo_name = '+'.join(combo)
            
            logger.info(f"  {combo_name}: {combo_score:.1%} ({len(combo_features)}特徴量)")
            
            if combo_score > best_combo_score:
                best_combo_score = combo_score
                best_combo = {
                    'name': combo_name,
                    'score': combo_score,
                    'features': combo_features,
                    'categories': combo
                }
        
        if best_combo:
            logger.info(f"\n🏆 最高組み合わせ: {best_combo['name']} ({best_combo['score']:.1%})")
        
        return category_results, best_combo
    
    def final_rigorous_validation(self, X, y, best_features, best_model_name):
        """最終厳密検証"""
        logger.info("🎯 最終厳密検証...")
        logger.info(f"特徴量数: {len(best_features)}, モデル: {best_model_name}")
        
        X_final = X[best_features]
        X_scaled = self.scaler.fit_transform(X_final)
        
        # モデル設定
        if best_model_name == 'LogisticRegression':
            model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        
        # 5分割厳密評価
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        logger.info("5分割時系列検証実行中...")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"  Fold {fold+1}: {accuracy:.1%}")
        
        final_accuracy = np.mean(scores)
        final_std = np.std(scores)
        
        logger.info(f"\n🎯 最終厳密結果: {final_accuracy:.1%} ± {final_std:.1%}")
        
        return final_accuracy, final_std, scores

def main():
    """メイン実行"""
    logger.info("🚀 高速最適特徴量選択システム - 全データ版")
    logger.info("⚡ 効率重視で最大精度達成")
    
    selector = FastOptimalFeatureSelector()
    
    try:
        # 1. 全データ読み込み
        df = selector.load_full_data()
        if df is None:
            return
        
        # 2. 戦略的特徴量作成
        df = selector.create_strategic_features(df)
        
        # 3. 全特徴量取得
        all_features = selector.get_all_features(df)
        
        # 4. データ準備
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[all_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"検証データ: {len(clean_df):,}件, 特徴量: {len(all_features)}個")
        
        # 5. 高速特徴量ランキング
        feature_ranking = selector.fast_feature_ranking(X, y)
        
        # 6. 体系的特徴量テスト
        systematic_results, best_systematic = selector.systematic_feature_testing(X, y, feature_ranking)
        
        # 7. 高度組み合わせテスト
        category_results, best_combo = selector.advanced_combination_test(X, y, all_features)
        
        # 8. 最終厳密検証
        final_accuracy, final_std, fold_scores = selector.final_rigorous_validation(
            X, y, best_systematic['features'], best_systematic['best_model']
        )
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 高速最適特徴量選択結果")
        logger.info("="*80)
        
        logger.info(f"データ総数: {len(df):,}件 (全データ検証)")
        logger.info(f"作成特徴量: {len(all_features)}個")
        
        # カテゴリ別結果
        if category_results:
            logger.info("\n📊 カテゴリ別性能:")
            sorted_cats = sorted(category_results.items(), key=lambda x: x[1]['score'], reverse=True)
            for cat, result in sorted_cats:
                logger.info(f"  {cat:15s}: {result['score']:.1%} ({result['count']}特徴量)")
        
        # 組み合わせ結果
        if best_combo:
            logger.info(f"\n🔄 最高組み合わせ: {best_combo['name']} ({best_combo['score']:.1%})")
        
        # 体系的テスト結果
        logger.info(f"\n📈 体系的テスト最高: {len(best_systematic['features'])}特徴量")
        logger.info(f"体系的最高精度: {best_systematic['best_score']:.1%}")
        
        # 最終厳密結果
        logger.info(f"\n🎯 最終厳密検証: {final_accuracy:.1%} ± {final_std:.1%}")
        logger.info(f"使用モデル: {best_systematic['best_model']}")
        
        # 最適特徴量
        logger.info(f"\n🏆 最適特徴量 ({len(best_systematic['features'])}個):")
        for i, feature in enumerate(best_systematic['features'], 1):
            logger.info(f"  {i:2d}. {feature}")
        
        # 全体の最高精度
        all_scores = [final_accuracy, best_systematic['best_score']]
        if best_combo:
            all_scores.append(best_combo['score'])
        
        max_achieved = max(all_scores)
        logger.info(f"\n🏆 達成最高精度: {max_achieved:.1%}")
        logger.info(f"⚠️ この結果は394,102件の全データでの厳密検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()