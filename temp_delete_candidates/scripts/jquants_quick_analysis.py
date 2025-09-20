#!/usr/bin/env python3
"""
J-Quants高速分析 - データサブセットでの迅速な精度評価
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class JQuantsQuickAnalyzer:
    """J-Quants高速分析"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
    def load_and_sample_data(self, sample_size=50000):
        """データ読み込みとサンプリング"""
        logger.info(f"📊 データ読み込み（サンプルサイズ: {sample_size:,}）")
        
        # 既存の処理済みデータを読み込み
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
    
    def create_jquants_inspired_features(self, df):
        """J-Quantsライクな拡張特徴量作成"""
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
        # 出来高急増を信用取引の代理指標とする
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
        
        logger.info(f"✅ 拡張特徴量作成完了: {df.shape}")
        return df
    
    def quick_evaluation(self, df):
        """高速評価"""
        logger.info("⚡ 高速評価開始...")
        
        if 'Binary_Direction' not in df.columns:
            logger.error("❌ Binary_Directionが見つかりません")
            return None
            
        # 特徴量分類
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
        
        # クリーンデータ
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        logger.info(f"評価データ: {len(clean_df):,}件")
        
        # 評価実行
        results = {}
        
        # 1. 基本特徴量
        if basic_features:
            X_basic = clean_df[basic_features]
            y = clean_df['Binary_Direction']
            results['basic'] = self._fast_evaluate(X_basic, y, "基本特徴量")
        
        # 2. J-Quantsライク特徴量のみ
        if jquants_features:
            X_jquants = clean_df[jquants_features]
            y = clean_df['Binary_Direction']
            results['jquants_only'] = self._fast_evaluate(X_jquants, y, "J-Quantsライク")
        
        # 3. 全特徴量
        X_all = clean_df[all_features]
        y = clean_df['Binary_Direction']
        results['combined'] = self._fast_evaluate(X_all, y, "結合特徴量")
        
        return results
    
    def _fast_evaluate(self, X, y, name):
        """高速評価実行"""
        logger.info(f"🚀 {name}評価中...")
        
        # 高速設定
        tscv = TimeSeriesSplit(n_splits=2)  # 分割数削減
        scaler = StandardScaler()
        
        # 軽量モデル
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                C=0.01, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=500
            )
        }
        
        model_results = {}
        
        for model_name, model in models.items():
            fold_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 前処理
                if 'Logistic' in model_name:
                    X_train_proc = scaler.fit_transform(X_train)
                    X_test_proc = scaler.transform(X_test)
                else:
                    X_train_proc = X_train
                    X_test_proc = X_test
                
                # 学習・予測
                model.fit(X_train_proc, y_train)
                y_pred = model.predict(X_test_proc)
                accuracy = accuracy_score(y_test, y_pred)
                fold_scores.append(accuracy)
            
            avg_score = np.mean(fold_scores)
            model_results[model_name] = {
                'score': avg_score,
                'std': np.std(fold_scores)
            }
            
            logger.info(f"  {model_name}: {avg_score:.3f} ± {np.std(fold_scores):.3f}")
        
        return model_results

def main():
    """メイン実行"""
    try:
        analyzer = JQuantsQuickAnalyzer()
        
        print("⚡ J-Quants高速精度分析")
        print("="*50)
        
        # データ読み込み
        df = analyzer.load_and_sample_data(sample_size=50000)
        if df is None:
            print("❌ データ読み込み失敗")
            return 1
        
        # 拡張特徴量作成
        df_enhanced = analyzer.create_jquants_inspired_features(df)
        
        # 評価実行
        results = analyzer.quick_evaluation(df_enhanced)
        
        if not results:
            print("❌ 評価失敗")
            return 1
        
        # 結果表示
        print("\n" + "="*50)
        print("📋 J-QUANTS高速分析結果")
        print("="*50)
        
        baseline = 0.517  # 既存最高スコア
        best_score = 0
        best_config = ""
        
        for feature_type, models in results.items():
            print(f"\n🔍 {feature_type.upper()}:")
            
            for model_name, result in models.items():
                score = result['score']
                std = result['std']
                improvement = score - baseline
                
                print(f"   {model_name:18s}: {score:.3f} ± {std:.3f} ({improvement:+.3f})")
                
                if score > best_score:
                    best_score = score
                    best_config = f"{feature_type} + {model_name}"
        
        # 最終評価
        total_improvement = best_score - baseline
        
        print(f"\n🏆 最高性能:")
        print(f"   設定: {best_config}")
        print(f"   精度: {best_score:.3f} ({best_score:.1%})")
        print(f"   改善: {total_improvement:+.3f} ({total_improvement:+.1%})")
        
        print(f"\n💡 J-Quantsデータの効果:")
        if total_improvement > 0.01:
            print(f"   ✅ 有意な改善 (+{total_improvement:.1%})")
            print(f"   🚀 J-Quantsデータの追加価値あり")
        elif total_improvement > 0.005:
            print(f"   📈 微細な改善 (+{total_improvement:.1%})")
            print(f"   💡 他のデータソースと組み合わせで効果的")
        else:
            print(f"   ➡️ 限定的な効果 ({total_improvement:+.1%})")
            print(f"   💡 外部データが必要（ニュース・板情報など）")
        
        # 目標達成判定
        if best_score >= 0.53:
            print(f"\n🎉 目標達成! 53%を突破!")
        elif best_score >= 0.525:
            print(f"\n🔥 目標に非常に近い（52.5%以上）")
        elif best_score >= 0.52:
            print(f"\n👍 有意な改善を確認（52%以上）")
        else:
            print(f"\n📈 更なるデータが必要")
        
        print(f"\n📊 推奨次ステップ:")
        if best_score < 0.53:
            print(f"   1. 板情報（kabu API）の追加")
            print(f"   2. ニュース感情分析の導入")
            print(f"   3. セクター特化モデルの検討")
        
        return 0 if total_improvement > 0 else 1
        
    except Exception as e:
        logger.error(f"分析エラー: {e}")
        return 1

if __name__ == "__main__":
    exit(main())