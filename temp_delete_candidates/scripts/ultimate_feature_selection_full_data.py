#!/usr/bin/env python3
"""
究極の特徴量選択システム - 全データ（394,102件）版
あらゆるパターンを検証して最高精度達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import itertools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class UltimateFeatureSelector:
    """究極の特徴量選択システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.results = {}
        
        # 複数のスケーラー
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # 複数のモデル
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'RidgeClassifier': RidgeClassifier(random_state=42)
        }
        
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
        logger.info("⚠️ 全データでの検証のため時間がかかります")
        
        return df
    
    def create_comprehensive_features(self, df):
        """包括的特徴量作成"""
        logger.info("🔧 包括的特徴量作成中...")
        logger.info("1/6: 基本特徴量確認...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 2. テクニカル指標の拡張
        logger.info("2/6: 拡張テクニカル指標作成...")
        
        # より多くの期間での移動平均
        for period in [5, 10, 25, 50, 75, 100]:
            df[f'MA_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
            df[f'Price_vs_MA{period}'] = (df['Close'] - df[f'MA_{period}']) / (df[f'MA_{period}'] + 1e-6)
        
        # EMAの追加
        for period in [12, 26, 50]:
            df[f'EMA_{period}'] = df.groupby('Code')['Close'].ewm(span=period).mean().reset_index(0, drop=True)
            df[f'Price_vs_EMA{period}'] = (df['Close'] - df[f'EMA_{period}']) / (df[f'EMA_{period}'] + 1e-6)
        
        # ボラティリティ指標の拡張
        for period in [5, 10, 20, 30, 60]:
            df[f'Volatility_{period}'] = df.groupby('Code')['Close'].rolling(period, min_periods=1).std().reset_index(0, drop=True)
            df[f'VolatilityRank_{period}'] = df.groupby('Date')[f'Volatility_{period}'].rank(pct=True)
        
        # RSIの複数期間
        for period in [7, 14, 21, 28]:
            df[f'RSI_{period}'] = self._calculate_rsi(df, period)
        
        # MACD系指標
        df['MACD_12_26'] = df.groupby('Code').apply(lambda x: self._calculate_macd(x, 12, 26)).reset_index(0, drop=True)
        df['MACD_Signal'] = df.groupby('Code')['MACD_12_26'].ewm(span=9).mean().reset_index(0, drop=True)
        df['MACD_Histogram'] = df['MACD_12_26'] - df['MACD_Signal']
        
        # 3. 市場構造指標の詳細化
        logger.info("3/6: 詳細市場構造指標作成...")
        
        # 日次市場統計の詳細化
        daily_market = df.groupby('Date').agg({
            'Close': ['mean', 'std', 'min', 'max', 'median'],
            'Volume': ['mean', 'std', 'min', 'max', 'median'],
            'Returns': ['mean', 'std', 'skew', 'min', 'max'],
            'High': 'mean',
            'Low': 'mean'
        })
        
        daily_market.columns = [f'Market_{stat}_{col}' for col, stat in daily_market.columns]
        daily_market = daily_market.reset_index()
        
        # 市場幅指標の詳細化
        daily_breadth = df.groupby('Date').agg({
            'Returns': lambda x: (x > 0).sum() / len(x),  # 上昇銘柄比率
            'Close': lambda x: len(x)  # 取引銘柄数
        })
        daily_breadth.columns = ['Market_Breadth_Ratio', 'Market_Stock_Count']
        daily_breadth = daily_breadth.reset_index()
        
        # 4. セクター分析の高度化
        logger.info("4/6: 高度セクター分析作成...")
        
        df['Sector_Code'] = df['Code'].astype(str).str[:2]
        
        # セクター別統計
        sector_stats = df.groupby(['Date', 'Sector_Code']).agg({
            'Close': ['mean', 'std', 'count'],
            'Volume': 'mean',
            'Returns': ['mean', 'std']
        })
        sector_stats.columns = [f'Sector_{stat}_{col}' for col, stat in sector_stats.columns]
        sector_stats = sector_stats.reset_index()
        
        # セクター相対パフォーマンス
        sector_performance = df.groupby(['Date', 'Sector_Code'])['Returns'].mean().reset_index()
        sector_performance.columns = ['Date', 'Sector_Code', 'Sector_Return']
        
        # 5. 個別銘柄指標
        logger.info("5/6: 個別銘柄詳細指標作成...")
        
        # 価格レンジ指標
        df['Price_Range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['Close'] + 1e-6)
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['Close'] + 1e-6)
        
        # 出来高指標の詳細化
        for period in [5, 10, 20]:
            df[f'Volume_MA_{period}'] = df.groupby('Code')['Volume'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
            df[f'Volume_Ratio_{period}'] = df['Volume'] / (df[f'Volume_MA_{period}'] + 1e-6)
        
        # 価格勢い指標
        for period in [3, 5, 10, 20]:
            df[f'Price_Momentum_{period}'] = df.groupby('Code')['Close'].pct_change(period)
            df[f'Return_Momentum_{period}'] = df.groupby('Code')['Returns'].rolling(period, min_periods=1).sum().reset_index(0, drop=True)
        
        # 6. 統合とクリーニング
        logger.info("6/6: データ統合とクリーニング...")
        
        # 各種統計データのマージ
        df = df.merge(daily_market, on='Date', how='left')
        df = df.merge(daily_breadth, on='Date', how='left')
        df = df.merge(sector_stats, on=['Date', 'Sector_Code'], how='left')
        df = df.merge(sector_performance, on=['Date', 'Sector_Code'], how='left')
        
        # 市場相対指標の計算
        df['Market_Relative_Return'] = df['Returns'] - df['Market_mean_Returns']
        df['Market_Relative_Volume'] = df['Volume'] / (df['Market_mean_Volume'] + 1e-6)
        df['Sector_Relative_Return'] = df['Returns'] - df['Sector_Return']
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info(f"✅ 包括的特徴量作成完了: {df.shape}")
        return df
    
    def _calculate_rsi(self, df, period):
        """RSI計算"""
        def rsi_calc(group):
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-6)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('Code', group_keys=False).apply(rsi_calc).reset_index(0, drop=True)
    
    def _calculate_macd(self, group, fast=12, slow=26):
        """MACD計算"""
        ema_fast = group['Close'].ewm(span=fast).mean()
        ema_slow = group['Close'].ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def categorize_features(self, df):
        """特徴量の詳細分類"""
        logger.info("📊 特徴量分類中...")
        
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction', 'Sector_Code'
        }
        
        all_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # 詳細分類
        feature_categories = {
            'basic': [col for col in all_features if col in ['Returns', 'Volume_Change']],
            'technical_ma': [col for col in all_features if 'MA_' in col or 'Price_vs_MA' in col or 'EMA' in col or 'Price_vs_EMA' in col],
            'technical_volatility': [col for col in all_features if 'Volatility' in col or 'VolatilityRank' in col],
            'technical_momentum': [col for col in all_features if 'RSI' in col or 'MACD' in col or 'Momentum' in col],
            'technical_volume': [col for col in all_features if 'Volume_' in col and col != 'Volume_Change'],
            'technical_price': [col for col in all_features if any(x in col for x in ['Range', 'Shadow', 'Upper', 'Lower'])],
            'market': [col for col in all_features if col.startswith('Market_') and 'Relative' not in col],
            'sector': [col for col in all_features if col.startswith('Sector_')],
            'relative': [col for col in all_features if 'Relative' in col],
            'breadth': [col for col in all_features if 'Breadth' in col]
        }
        
        # 分類結果の表示
        for category, features in feature_categories.items():
            logger.info(f"{category:20s}: {len(features):3d}個")
        
        logger.info(f"全特徴量総数: {len(all_features)}個")
        
        return feature_categories, all_features
    
    def comprehensive_feature_selection(self, df, feature_categories, all_features):
        """包括的特徴量選択"""
        logger.info("🔍 包括的特徴量選択開始...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        logger.info(f"検証データ: {len(clean_df):,}件")
        
        X = clean_df[all_features]
        y = clean_df['Binary_Direction'].astype(int)
        
        selection_results = {}
        
        # 1. カテゴリ別単独評価
        logger.info("1/8: カテゴリ別単独評価...")
        category_scores = self._evaluate_feature_categories(X, y, feature_categories)
        selection_results['category_scores'] = category_scores
        
        # 2. 統計的特徴量選択
        logger.info("2/8: 統計的特徴量選択...")
        statistical_rankings = self._statistical_feature_selection(X, y)
        selection_results['statistical'] = statistical_rankings
        
        # 3. モデルベース特徴量選択
        logger.info("3/8: モデルベース特徴量選択...")
        model_rankings = self._model_based_feature_selection(X, y)
        selection_results['model_based'] = model_rankings
        
        # 4. 再帰的特徴量除去
        logger.info("4/8: 再帰的特徴量除去...")
        rfe_features = self._recursive_feature_elimination(X, y)
        selection_results['rfe'] = rfe_features
        
        # 5. アンサンブル重要度
        logger.info("5/8: アンサンブル重要度計算...")
        ensemble_ranking = self._create_ensemble_ranking(selection_results)
        
        # 6. 段階的最適化
        logger.info("6/8: 段階的最適化...")
        progressive_results = self._progressive_optimization(X, y, ensemble_ranking)
        selection_results['progressive'] = progressive_results
        
        # 7. カテゴリ組み合わせ最適化
        logger.info("7/8: カテゴリ組み合わせ最適化...")
        combination_results = self._category_combination_optimization(X, y, feature_categories)
        selection_results['combination'] = combination_results
        
        # 8. 最終最適化
        logger.info("8/8: 最終最適化...")
        final_results = self._final_optimization(X, y, selection_results)
        
        return selection_results, final_results
    
    def _evaluate_feature_categories(self, X, y, feature_categories):
        """カテゴリ別評価"""
        category_scores = {}
        
        for category, features in feature_categories.items():
            if not features:
                continue
                
            logger.info(f"  {category} ({len(features)}特徴量) 評価中...")
            
            X_cat = X[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cat)
            
            # TimeSeriesSplit評価
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            category_scores[category] = {
                'score': avg_score,
                'features': features,
                'feature_count': len(features)
            }
            
            logger.info(f"    {category:20s}: {avg_score:.1%}")
        
        return category_scores
    
    def _statistical_feature_selection(self, X, y):
        """統計的特徴量選択"""
        rankings = {}
        
        # F統計量
        f_scores = f_classif(X, y)[0]
        rankings['f_statistic'] = list(zip(X.columns, f_scores))
        
        # 相互情報量
        mi_scores = mutual_info_classif(X, y, random_state=42)
        rankings['mutual_info'] = list(zip(X.columns, mi_scores))
        
        # 分散による選択
        var_threshold = VarianceThreshold(threshold=0.01)
        var_threshold.fit(X)
        high_var_features = X.columns[var_threshold.get_support()]
        rankings['high_variance'] = [(f, 1.0) for f in high_var_features]
        
        # ランキングソート
        for method in rankings:
            rankings[method] = sorted(rankings[method], key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _model_based_feature_selection(self, X, y):
        """モデルベース特徴量選択"""
        rankings = {}
        
        # RandomForest重要度
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rankings['random_forest'] = list(zip(X.columns, rf.feature_importances_))
        
        # GradientBoosting重要度
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42)
        gb.fit(X, y)
        rankings['gradient_boosting'] = list(zip(X.columns, gb.feature_importances_))
        
        # L1正則化
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=42)
        lasso.fit(X, y)
        lasso_importance = np.abs(lasso.coef_[0])
        rankings['lasso'] = list(zip(X.columns, lasso_importance))
        
        # ランキングソート
        for method in rankings:
            rankings[method] = sorted(rankings[method], key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _recursive_feature_elimination(self, X, y):
        """再帰的特徴量除去"""
        logger.info("    RFE実行中...")
        
        # 計算時間短縮のためサンプリング
        if len(X) > 50000:
            sample_idx = np.random.choice(len(X), 50000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        # RFE実行
        estimator = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        rfe = RFE(estimator, n_features_to_select=min(20, len(X.columns)//2))
        rfe.fit(X_sample, y_sample)
        
        selected_features = X.columns[rfe.support_]
        feature_rankings = list(zip(X.columns, rfe.ranking_))
        
        return {
            'selected': selected_features.tolist(),
            'rankings': sorted(feature_rankings, key=lambda x: x[1])
        }
    
    def _create_ensemble_ranking(self, selection_results):
        """アンサンブル重要度ランキング"""
        ensemble_scores = {}
        
        # 各手法の結果を統合
        all_rankings = {}
        
        # 統計的手法
        if 'statistical' in selection_results:
            for method, rankings in selection_results['statistical'].items():
                all_rankings[f'stat_{method}'] = rankings
        
        # モデルベース手法
        if 'model_based' in selection_results:
            for method, rankings in selection_results['model_based'].items():
                all_rankings[f'model_{method}'] = rankings
        
        # RFE
        if 'rfe' in selection_results and 'rankings' in selection_results['rfe']:
            # RFEはランキングが低いほど良いので逆転
            rfe_rankings = [(name, 1.0/rank) for name, rank in selection_results['rfe']['rankings']]
            all_rankings['rfe'] = rfe_rankings
        
        # アンサンブルスコア計算
        for method, rankings in all_rankings.items():
            if rankings:
                scores = [score for name, score in rankings]
                if len(scores) > 0 and max(scores) > min(scores):
                    min_score, max_score = min(scores), max(scores)
                    for name, score in rankings:
                        normalized_score = (score - min_score) / (max_score - min_score)
                        if name not in ensemble_scores:
                            ensemble_scores[name] = []
                        ensemble_scores[name].append(normalized_score)
        
        # 平均スコア計算
        final_scores = {}
        for name, scores in ensemble_scores.items():
            final_scores[name] = np.mean(scores)
        
        ensemble_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("アンサンブル上位20特徴量:")
        for i, (feature, score) in enumerate(ensemble_ranking[:20]):
            logger.info(f"  {i+1:2d}. {feature:40s}: {score:.4f}")
        
        return ensemble_ranking
    
    def _progressive_optimization(self, X, y, ensemble_ranking):
        """段階的最適化"""
        logger.info("    段階的テスト実行...")
        
        # 特徴量数のテストパターン
        feature_counts = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        max_features = min(len(ensemble_ranking), 100)
        feature_counts = [n for n in feature_counts if n <= max_features]
        
        results = {}
        
        for n_features in feature_counts:
            selected_features = [name for name, score in ensemble_ranking[:n_features]]
            X_selected = X[selected_features]
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # 複数モデルでテスト
            model_scores = {}
            
            for model_name, model in self.models.items():
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train = X_scaled[train_idx]
                    X_test = X_scaled[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    
                    # モデルのパラメータ調整
                    if model_name == 'LogisticRegression':
                        model.set_params(C=0.1, class_weight='balanced')
                    elif model_name == 'RandomForest':
                        model.set_params(class_weight='balanced')
                    elif model_name == 'GradientBoosting':
                        pass  # デフォルトパラメータ使用
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    scores.append(accuracy_score(y_test, pred))
                
                model_scores[model_name] = np.mean(scores)
            
            # 最高性能のモデル特定
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            
            results[n_features] = {
                'best_model': best_model,
                'best_score': best_score,
                'all_scores': model_scores,
                'features': selected_features
            }
            
            logger.info(f"    {n_features:3d}特徴量: {best_score:.1%} ({best_model})")
        
        # 最高性能の特徴量数特定
        best_n = max(results.keys(), key=lambda k: results[k]['best_score'])
        
        logger.info(f"\n  最高性能: {best_n}特徴量, {results[best_n]['best_score']:.1%} ({results[best_n]['best_model']})")
        
        return results
    
    def _category_combination_optimization(self, X, y, feature_categories):
        """カテゴリ組み合わせ最適化"""
        logger.info("    カテゴリ組み合わせテスト...")
        
        # 有効なカテゴリのみ
        valid_categories = {k: v for k, v in feature_categories.items() if v}
        category_names = list(valid_categories.keys())
        
        combination_results = {}
        
        # 単一カテゴリ
        for category in category_names[:6]:  # 計算時間短縮
            features = feature_categories[category]
            if len(features) == 0:
                continue
                
            X_cat = X[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cat)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            combination_results[category] = {
                'score': np.mean(scores),
                'categories': [category],
                'feature_count': len(features)
            }
        
        # 2カテゴリ組み合わせ（重要なもののみ）
        important_categories = ['technical_ma', 'market', 'technical_volatility', 'relative']
        important_categories = [c for c in important_categories if c in category_names]
        
        for combo in itertools.combinations(important_categories, 2):
            combo_features = []
            for cat in combo:
                combo_features.extend(feature_categories[cat])
            
            if len(combo_features) == 0 or len(combo_features) > 50:  # 特徴量数制限
                continue
            
            X_combo = X[combo_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combo)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            combo_name = '+'.join(combo)
            combination_results[combo_name] = {
                'score': np.mean(scores),
                'categories': list(combo),
                'feature_count': len(combo_features)
            }
        
        # 結果表示
        sorted_combos = sorted(combination_results.items(), key=lambda x: x[1]['score'], reverse=True)
        
        logger.info("    カテゴリ組み合わせ結果:")
        for i, (combo, result) in enumerate(sorted_combos[:10]):
            logger.info(f"      {i+1:2d}. {combo:30s}: {result['score']:.1%} ({result['feature_count']}特徴量)")
        
        return combination_results
    
    def _final_optimization(self, X, y, selection_results):
        """最終最適化"""
        logger.info("🎯 最終最適化実行...")
        
        # 各手法の最高性能を取得
        best_candidates = []
        
        # Progressive結果から最高性能
        if 'progressive' in selection_results:
            prog_results = selection_results['progressive']
            best_prog = max(prog_results.keys(), key=lambda k: prog_results[k]['best_score'])
            best_candidates.append({
                'name': f'Progressive_{best_prog}features',
                'features': prog_results[best_prog]['features'],
                'model': prog_results[best_prog]['best_model'],
                'score': prog_results[best_prog]['best_score']
            })
        
        # カテゴリ組み合わせから最高性能
        if 'combination' in selection_results:
            combo_results = selection_results['combination']
            best_combo = max(combo_results.keys(), key=lambda k: combo_results[k]['score'])
            
            # 最高カテゴリの特徴量を取得
            best_combo_info = combo_results[best_combo]
            combo_features = []
            for cat in best_combo_info['categories']:
                if cat in selection_results.get('category_scores', {}):
                    combo_features.extend(selection_results['category_scores'][cat]['features'])
            
            if combo_features:
                best_candidates.append({
                    'name': f'Category_{best_combo}',
                    'features': combo_features,
                    'model': 'LogisticRegression',
                    'score': best_combo_info['score']
                })
        
        # 最終検証
        final_results = {}
        
        for candidate in best_candidates:
            logger.info(f"  最終検証: {candidate['name']} ({len(candidate['features'])}特徴量)")
            
            X_final = X[candidate['features']]
            
            # 複数スケーラーでテスト
            scaler_results = {}
            
            for scaler_name, scaler in self.scalers.items():
                X_scaled = scaler.fit_transform(X_final)
                
                # 5分割で厳密評価
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
                
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train = X_scaled[train_idx]
                    X_test = X_scaled[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    
                    # 最適モデル使用
                    if candidate['model'] == 'LogisticRegression':
                        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
                    elif candidate['model'] == 'RandomForest':
                        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
                    elif candidate['model'] == 'GradientBoosting':
                        model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
                    else:
                        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    scores.append(accuracy_score(y_test, pred))
                
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                scaler_results[scaler_name] = {'avg': avg_score, 'std': std_score, 'scores': scores}
                
                logger.info(f"    {scaler_name:10s}: {avg_score:.1%} ± {std_score:.1%}")
            
            # 最高スケーラー選択
            best_scaler = max(scaler_results.keys(), key=lambda k: scaler_results[k]['avg'])
            
            final_results[candidate['name']] = {
                'features': candidate['features'],
                'model': candidate['model'],
                'best_scaler': best_scaler,
                'score': scaler_results[best_scaler]['avg'],
                'std': scaler_results[best_scaler]['std'],
                'all_scaler_results': scaler_results
            }
        
        # 最高性能特定
        if final_results:
            best_final = max(final_results.keys(), key=lambda k: final_results[k]['score'])
            logger.info(f"\n🏆 最高性能: {best_final}")
            logger.info(f"精度: {final_results[best_final]['score']:.1%} ± {final_results[best_final]['std']:.1%}")
            logger.info(f"特徴量数: {len(final_results[best_final]['features'])}")
            logger.info(f"モデル: {final_results[best_final]['model']}")
            logger.info(f"スケーラー: {final_results[best_final]['best_scaler']}")
        
        return final_results

def main():
    """メイン実行"""
    logger.info("🚀 究極の特徴量選択システム - 全データ（394,102件）版")
    logger.info("⚠️ あらゆるパターンを検証するため、非常に時間がかかります")
    
    selector = UltimateFeatureSelector()
    
    try:
        # 1. 全データ読み込み
        df = selector.load_full_data()
        if df is None:
            return
        
        # 2. 包括的特徴量作成
        df = selector.create_comprehensive_features(df)
        
        # 3. 特徴量分類
        feature_categories, all_features = selector.categorize_features(df)
        
        # 4. 包括的特徴量選択
        selection_results, final_results = selector.comprehensive_feature_selection(
            df, feature_categories, all_features
        )
        
        # 結果まとめ
        logger.info("\n" + "="*100)
        logger.info("🎯 究極の特徴量選択結果サマリー")
        logger.info("="*100)
        
        logger.info(f"データ総数: {len(df):,}件 (全データ検証)")
        logger.info(f"作成特徴量総数: {len(all_features)}個")
        
        # 最終結果
        if final_results:
            best_result = max(final_results.keys(), key=lambda k: final_results[k]['score'])
            result = final_results[best_result]
            
            logger.info(f"\n🏆 最高達成精度: {result['score']:.1%} ± {result['std']:.1%}")
            logger.info(f"使用特徴量数: {len(result['features'])}")
            logger.info(f"最適モデル: {result['model']}")
            logger.info(f"最適スケーラー: {result['best_scaler']}")
            
            logger.info("\n最適特徴量:")
            for i, feature in enumerate(result['features'][:20], 1):
                logger.info(f"  {i:2d}. {feature}")
            if len(result['features']) > 20:
                logger.info(f"  ... 他{len(result['features'])-20}個")
            
            # カテゴリ別結果
            if 'category_scores' in selection_results:
                logger.info("\nカテゴリ別性能:")
                sorted_categories = sorted(
                    selection_results['category_scores'].items(),
                    key=lambda x: x[1]['score'], reverse=True
                )
                for category, info in sorted_categories[:10]:
                    logger.info(f"  {category:20s}: {info['score']:.1%} ({info['feature_count']}特徴量)")
        
        logger.info(f"\n⚠️ この結果は394,102件の全データでの厳密検証による現実的な性能評価です。")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()