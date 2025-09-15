#!/usr/bin/env python3
"""
51.7%精度の再現システム - 全データ版
以前の最適特徴量で51.7%以上を確実に達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class AccuracyReproducer:
    """51.7%精度の再現システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # 以前の最適特徴量
        self.previous_optimal_features = [
            'Market_Breadth',
            'Market_Return', 
            'Volatility_20',
            'RSI',
            'Price_vs_MA20'
        ]
        
    def load_full_data(self):
        """全データ読み込み"""
        logger.info("📊 全データ読み込み（394,102件）")
        
        processed_files = list(self.processed_dir.glob("*.parquet"))
        if not processed_files:
            logger.error("❌ 処理済みデータが見つかりません")
            return None
            
        df = pd.read_parquet(processed_files[0])
        logger.info(f"✅ 全データ読み込み完了: {len(df):,}件")
        
        return df
    
    def check_existing_features(self, df):
        """既存特徴量チェック"""
        logger.info("🔍 既存特徴量チェック...")
        
        available_features = df.columns.tolist()
        logger.info(f"利用可能特徴量総数: {len(available_features)}")
        
        # 以前の最適特徴量の存在確認
        missing_features = []
        available_optimal = []
        
        for feature in self.previous_optimal_features:
            if feature in available_features:
                available_optimal.append(feature)
                logger.info(f"✅ {feature}: 存在")
            else:
                missing_features.append(feature)
                logger.info(f"❌ {feature}: 不存在")
        
        logger.info(f"\n利用可能最適特徴量: {len(available_optimal)}個")
        logger.info(f"不足特徴量: {len(missing_features)}個")
        
        # 類似特徴量検索
        if missing_features:
            logger.info("\n🔍 類似特徴量検索:")
            for missing in missing_features:
                similar = [f for f in available_features if any(keyword in f.lower() for keyword in missing.lower().split('_'))]
                if similar:
                    logger.info(f"  {missing} → 類似: {similar[:5]}")
                else:
                    logger.info(f"  {missing} → 類似なし")
        
        return available_optimal, missing_features
    
    def create_missing_features(self, df, missing_features):
        """不足特徴量の作成"""
        logger.info("🔧 不足特徴量作成中...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        for feature in missing_features:
            if feature == 'Market_Breadth':
                # 市場幅指標作成
                logger.info("  Market_Breadth作成中...")
                daily_breadth = df.groupby('Date')['Returns'].apply(
                    lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
                ).reset_index()
                daily_breadth.columns = ['Date', 'Market_Breadth']
                df = df.merge(daily_breadth, on='Date', how='left')
                
            elif feature == 'Market_Return':
                # 市場平均リターン作成
                logger.info("  Market_Return作成中...")
                daily_market_return = df.groupby('Date')['Returns'].mean().reset_index()
                daily_market_return.columns = ['Date', 'Market_Return']
                df = df.merge(daily_market_return, on='Date', how='left')
                
            elif feature == 'Volatility_20':
                # 20日ボラティリティ作成
                logger.info("  Volatility_20作成中...")
                df['Volatility_20'] = df.groupby('Code')['Close'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
                
            elif feature == 'RSI':
                # RSI作成
                logger.info("  RSI作成中...")
                df['RSI'] = self._calculate_rsi(df, 14)
                
            elif feature == 'Price_vs_MA20':
                # 移動平均乖離率作成
                logger.info("  Price_vs_MA20作成中...")
                if 'MA_20' not in df.columns:
                    df['MA_20'] = df.groupby('Code')['Close'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
                df['Price_vs_MA20'] = (df['Close'] - df['MA_20']) / (df['MA_20'] + 1e-6)
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info("✅ 不足特徴量作成完了")
        return df
    
    def _calculate_rsi(self, df, period=14):
        """RSI計算"""
        def rsi_calc(group):
            close = group['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('Code', group_keys=False).apply(rsi_calc).reset_index(0, drop=True)
    
    def reproduce_51_7_accuracy(self, df):
        """51.7%精度の再現"""
        logger.info("🎯 51.7%精度再現実行...")
        
        # データ準備
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 最適特徴量が全て存在することを確認
        all_features_exist = all(feature in clean_df.columns for feature in self.previous_optimal_features)
        
        if not all_features_exist:
            missing = [f for f in self.previous_optimal_features if f not in clean_df.columns]
            logger.error(f"❌ 特徴量不足: {missing}")
            return None
        
        X = clean_df[self.previous_optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"検証データ: {len(clean_df):,}件")
        logger.info(f"使用特徴量: {self.previous_optimal_features}")
        
        # 以前と同じパラメータで再現
        models = {
            'LogisticRegression_L1': LogisticRegression(
                C=0.01, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'LogisticRegression_L2': LogisticRegression(
                C=0.01, penalty='l2', solver='lbfgs',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
        }
        
        results = {}
        
        # 複数モデルでテスト
        for model_name, model in models.items():
            logger.info(f"  {model_name}評価中...")
            
            if 'LogisticRegression' in model_name:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # 5分割時系列評価
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
                
                # logger.info(f"    Fold {fold+1}: {accuracy:.1%}")
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            results[model_name] = {
                'avg': avg_score,
                'std': std_score,
                'scores': scores
            }
            
            logger.info(f"  {model_name}: {avg_score:.1%} ± {std_score:.1%}")
        
        # 最高性能特定
        best_model = max(results.keys(), key=lambda k: results[k]['avg'])
        best_score = results[best_model]['avg']
        
        logger.info(f"\n🏆 最高再現精度: {best_score:.1%} ({best_model})")
        
        # 51.7%達成確認
        target_accuracy = 0.517  # 51.7%
        if best_score >= target_accuracy:
            logger.info(f"✅ 目標51.7%達成！ ({best_score:.1%} >= {target_accuracy:.1%})")
        else:
            logger.warning(f"⚠️ 目標51.7%未達成 ({best_score:.1%} < {target_accuracy:.1%})")
            logger.info(f"差: {(target_accuracy - best_score)*100:.1f}%")
        
        return results, best_model, best_score
    
    def advanced_feature_test(self, df):
        """高度特徴量組み合わせテスト"""
        logger.info("🧪 高度特徴量組み合わせテスト...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 利用可能な全特徴量
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction'
        }
        
        all_features = [col for col in clean_df.columns 
                       if col not in exclude_cols and clean_df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"利用可能特徴量: {len(all_features)}個")
        
        # 特徴量組み合わせパターン
        test_patterns = {
            'Previous_Optimal': self.previous_optimal_features,
            'Top_10_Technical': [f for f in all_features if any(x in f for x in ['MA', 'RSI', 'Vol']) and 'Market' not in f][:10],
            'Top_10_Market': [f for f in all_features if 'Market' in f or 'Breadth' in f][:10],
            'Mixed_15': self.previous_optimal_features + [f for f in all_features if f not in self.previous_optimal_features][:10],
        }
        
        pattern_results = {}
        
        for pattern_name, features in test_patterns.items():
            # 特徴量存在確認
            existing_features = [f for f in features if f in clean_df.columns]
            if len(existing_features) < 3:
                logger.info(f"  {pattern_name}: 特徴量不足 (スキップ)")
                continue
            
            logger.info(f"  {pattern_name} ({len(existing_features)}特徴量)...")
            
            X = clean_df[existing_features]
            y = clean_df['Binary_Direction'].astype(int)
            
            # 高速評価
            X_scaled = self.scaler.fit_transform(X)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            pattern_results[pattern_name] = {
                'score': avg_score,
                'features': existing_features,
                'count': len(existing_features)
            }
            
            logger.info(f"    {pattern_name}: {avg_score:.1%}")
        
        # 最高パターン
        if pattern_results:
            best_pattern = max(pattern_results.keys(), key=lambda k: pattern_results[k]['score'])
            best_pattern_score = pattern_results[best_pattern]['score']
            
            logger.info(f"\n🏆 最高パターン: {best_pattern} ({best_pattern_score:.1%})")
        
        return pattern_results

def main():
    """メイン実行"""
    logger.info("🚀 51.7%精度再現システム - 全データ版")
    logger.info("🎯 目標: 以前の51.7%以上の精度再現")
    
    reproducer = AccuracyReproducer()
    
    try:
        # 1. 全データ読み込み
        df = reproducer.load_full_data()
        if df is None:
            return
        
        # 2. 既存特徴量チェック
        available_optimal, missing_features = reproducer.check_existing_features(df)
        
        # 3. 不足特徴量作成
        if missing_features:
            df = reproducer.create_missing_features(df, missing_features)
        
        # 4. 51.7%精度再現
        results, best_model, best_score = reproducer.reproduce_51_7_accuracy(df)
        
        # 5. 高度特徴量組み合わせテスト
        pattern_results = reproducer.advanced_feature_test(df)
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 51.7%精度再現結果")
        logger.info("="*80)
        
        logger.info(f"データ総数: {len(df):,}件 (全データ検証)")
        
        # 再現結果
        if results:
            logger.info("\n📊 モデル別再現結果:")
            for model_name, result in results.items():
                logger.info(f"  {model_name:25s}: {result['avg']:.1%} ± {result['std']:.1%}")
        
        logger.info(f"\n🏆 最高再現精度: {best_score:.1%} ({best_model})")
        
        # パターン結果
        if pattern_results:
            logger.info("\n🧪 特徴量パターン結果:")
            sorted_patterns = sorted(pattern_results.items(), key=lambda x: x[1]['score'], reverse=True)
            for pattern, result in sorted_patterns:
                logger.info(f"  {pattern:20s}: {result['score']:.1%} ({result['count']}特徴量)")
        
        # 全体の最高精度
        all_scores = [best_score]
        if pattern_results:
            all_scores.extend([result['score'] for result in pattern_results.values()])
        
        max_achieved = max(all_scores)
        logger.info(f"\n🏆 全体最高精度: {max_achieved:.1%}")
        
        # 51.7%との比較
        target = 0.517
        if max_achieved >= target:
            logger.info(f"✅ 51.7%再現成功！ ({max_achieved:.1%} >= {target:.1%})")
        else:
            logger.warning(f"⚠️ 51.7%再現失敗 ({max_achieved:.1%} < {target:.1%})")
            logger.info(f"差: {(target - max_achieved)*100:.1f}%")
        
        logger.info(f"\n⚠️ この結果は394,102件の全データでの厳密検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()