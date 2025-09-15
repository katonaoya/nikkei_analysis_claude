#!/usr/bin/env python3
"""
信頼度閾値別精度テスト
50%, 55%, 60%, 65%, 70%の5パターンで比較
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ConfidenceThresholdTester:
    """信頼度閾値別テストクラス"""
    
    def __init__(self):
        """初期化"""
        self.thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        self.results = {}
        
    def load_and_prepare_data(self):
        """データ読み込みと特徴量生成"""
        logger.info("📥 データ読み込み中...")
        df = pd.read_parquet('data/processed/integrated_with_external.parquet')
        
        # カラム名の調整
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code']
            
        logger.info("🔧 特徴量生成中...")
        features = []
        
        for stock, stock_df in df.groupby('Stock'):
            stock_df = stock_df.sort_values('Date')
            
            # 基本的な価格変化
            stock_df['Return'] = stock_df['close'].pct_change()
            stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
            
            # RSI（14日）
            delta = stock_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1)
            stock_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 移動平均からの乖離
            stock_df['MA5'] = stock_df['close'].rolling(5).mean()
            stock_df['MA20'] = stock_df['close'].rolling(20).mean()
            stock_df['Price_vs_MA5'] = (stock_df['close'] - stock_df['MA5']) / stock_df['MA5']
            stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
            
            # ボラティリティ
            stock_df['Volatility_20'] = stock_df['Return'].rolling(20).std()
            
            # 出来高比率
            stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
            stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
            
            features.append(stock_df)
        
        df = pd.concat(features, ignore_index=True)
        
        # 使用する特徴量（production_config.yamlと同じ）
        feature_cols = ['RSI', 'Price_vs_MA20', 'Volatility_20', 'Price_vs_MA5', 'Volume_Ratio']
        
        return df, feature_cols
    
    def test_threshold(self, df, feature_cols, threshold, test_days=30):
        """特定の閾値でのテスト"""
        logger.info(f"🎯 閾値 {threshold:.0%} のテスト開始...")
        
        # 直近30日でテスト
        df = df.sort_values('Date')
        unique_dates = sorted(df['Date'].unique())
        test_dates = unique_dates[-test_days:]
        
        # 軽量化RandomForestモデル
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=50,
            random_state=42,
            n_jobs=2
        )
        
        daily_results = []
        monthly_stats = {}
        
        for test_date in test_dates:
            # データ分割
            train_data = df[df['Date'] < test_date]
            test_data = df[df['Date'] == test_date]
            
            if len(train_data) < 10000 or len(test_data) < 30:
                continue
            
            # クリーンデータ
            train_clean = train_data.dropna(subset=['Target'] + feature_cols)
            test_clean = test_data.dropna(subset=['Target'] + feature_cols)
            
            if len(train_clean) < 5000 or len(test_clean) < 20:
                continue
            
            X_train = train_clean[feature_cols]
            y_train = train_clean['Target']
            X_test = test_clean[feature_cols]
            y_test = test_clean['Target']
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル学習
            try:
                model.fit(X_train_scaled, y_train)
                
                # 予測確率取得
                pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_clean['pred_proba'] = pred_proba
                
                # 閾値でフィルタリング
                selected = test_clean[test_clean['pred_proba'] >= threshold]
                
                if len(selected) > 0:
                    # 選択された銘柄は全て「上昇」と予測
                    predictions = np.ones(len(selected))
                    actuals = selected['Target'].values
                    
                    # 精度計算
                    precision = sum(actuals) / len(actuals) if len(actuals) > 0 else 0
                    
                    # 月次集計用
                    month_key = test_date.strftime('%Y-%m')
                    if month_key not in monthly_stats:
                        monthly_stats[month_key] = {'predictions': [], 'actuals': []}
                    
                    monthly_stats[month_key]['predictions'].extend(predictions)
                    monthly_stats[month_key]['actuals'].extend(actuals)
                    
                    daily_results.append({
                        'date': test_date,
                        'threshold': threshold,
                        'selected_count': len(selected),
                        'correct_count': sum(actuals),
                        'precision': precision,
                        'avg_confidence': selected['pred_proba'].mean(),
                        'min_confidence': selected['pred_proba'].min(),
                        'max_confidence': selected['pred_proba'].max()
                    })
                    
            except Exception as e:
                continue
        
        # 月次精度計算
        monthly_precisions = {}
        for month, data in monthly_stats.items():
            if len(data['actuals']) > 0:
                monthly_precisions[month] = sum(data['actuals']) / len(data['actuals'])
        
        return daily_results, monthly_precisions
    
    def run_all_threshold_tests(self, df, feature_cols):
        """全閾値パターンのテスト実行"""
        logger.info("🚀 全閾値パターンのテスト開始...")
        
        all_results = {}
        
        for threshold in self.thresholds:
            daily_results, monthly_results = self.test_threshold(df, feature_cols, threshold)
            
            if daily_results:
                # 統計計算
                df_daily = pd.DataFrame(daily_results)
                
                stats = {
                    'threshold': threshold,
                    'total_days': len(df_daily),
                    'avg_precision': df_daily['precision'].mean(),
                    'median_precision': df_daily['precision'].median(),
                    'std_precision': df_daily['precision'].std(),
                    'avg_daily_picks': df_daily['selected_count'].mean(),
                    'total_picks': df_daily['selected_count'].sum(),
                    'total_correct': df_daily['correct_count'].sum(),
                    'overall_precision': df_daily['correct_count'].sum() / df_daily['selected_count'].sum() if df_daily['selected_count'].sum() > 0 else 0,
                    'avg_confidence': df_daily['avg_confidence'].mean(),
                    'days_with_picks': len(df_daily[df_daily['selected_count'] > 0]),
                    'max_daily_picks': df_daily['selected_count'].max(),
                    'monthly_results': monthly_results
                }
                
                all_results[threshold] = stats
                logger.success(f"  閾値 {threshold:.0%}: 全体精度 {stats['overall_precision']:.2%}, 1日平均 {stats['avg_daily_picks']:.1f}銘柄")
            else:
                all_results[threshold] = None
                logger.warning(f"  閾値 {threshold:.0%}: データ不足")
        
        return all_results
    
    def print_comparison_report(self, results):
        """比較レポート出力"""
        print("\n" + "="*100)
        print("📊 信頼度閾値別精度比較レポート")
        print("="*100)
        
        # メイン比較表
        print(f"\n{'閾値':<8} {'全体精度':<10} {'1日平均':<10} {'総選択数':<10} {'的中数':<8} {'取引日数':<10} {'平均信頼度':<10}")
        print("-"*90)
        
        for threshold in self.thresholds:
            if results[threshold]:
                r = results[threshold]
                print(f"{threshold:.0%}      {r['overall_precision']:<10.2%} "
                      f"{r['avg_daily_picks']:<10.1f} {r['total_picks']:<10d} "
                      f"{r['total_correct']:<8d} {r['days_with_picks']:<10d} "
                      f"{r['avg_confidence']:<10.2%}")
            else:
                print(f"{threshold:.0%}      {'データ不足':<10} {'--':<10} {'--':<10} {'--':<8} {'--':<10} {'--':<10}")
        
        print("\n" + "-"*100)
        
        # 詳細統計
        print("\n📈 【詳細統計】")
        for threshold in self.thresholds:
            if results[threshold]:
                r = results[threshold]
                print(f"\n🎯 閾値 {threshold:.0%}:")
                print(f"  • 全体精度: {r['overall_precision']:.2%}")
                print(f"  • 平均精度: {r['avg_precision']:.2%} (±{r['std_precision']:.2%})")
                print(f"  • 中央値: {r['median_precision']:.2%}")
                print(f"  • 1日平均選択数: {r['avg_daily_picks']:.1f}銘柄")
                print(f"  • 最大日次選択数: {r['max_daily_picks']}銘柄")
                print(f"  • 取引発生日: {r['days_with_picks']}/{r['total_days']}日 ({r['days_with_picks']/r['total_days']*100:.1f}%)")
        
        # 月次推移（最も良い結果の閾値）
        best_threshold = max([t for t in self.thresholds if results[t]], 
                           key=lambda t: results[t]['overall_precision'])
        
        if results[best_threshold] and results[best_threshold]['monthly_results']:
            print(f"\n📅 【月次精度推移】(最良閾値 {best_threshold:.0%})")
            monthly = results[best_threshold]['monthly_results']
            for month in sorted(monthly.keys()):
                print(f"  {month}: {monthly[month]:.2%}")
        
        # 推奨事項
        print(f"\n💡 【推奨事項】")
        
        # 60%以上の精度を達成した閾値を探す
        good_thresholds = [(t, r) for t, r in results.items() 
                          if r and r['overall_precision'] >= 0.60]
        
        if good_thresholds:
            best_t, best_r = max(good_thresholds, key=lambda x: x[1]['overall_precision'])
            print(f"✅ 推奨閾値: {best_t:.0%}")
            print(f"   → 精度: {best_r['overall_precision']:.2%}")
            print(f"   → 1日平均: {best_r['avg_daily_picks']:.1f}銘柄")
            print(f"   → 期待リターン: 高精度により安定した収益")
        else:
            # 最良の結果を推奨
            if results[best_threshold]:
                print(f"📍 現状最良: {best_threshold:.0%}")
                print(f"   → 精度: {results[best_threshold]['overall_precision']:.2%}")
                print(f"   → 1日平均: {results[best_threshold]['avg_daily_picks']:.1f}銘柄")
                print(f"   → 改善案: より多様な特徴量の追加、モデルの改良")
        
        print("\n🎯 【運用方針】")
        print("• 高精度を優先: 70%以上の閾値（頻度は低くなるが確実性重視）")
        print("• バランス重視: 60-65%の閾値（精度と頻度のバランス）") 
        print("• 頻度重視: 50-55%の閾値（多くの機会だが精度は劣る）")
        
        print("\n" + "="*100)

def main():
    """メイン実行"""
    tester = ConfidenceThresholdTester()
    
    # データ準備
    df, feature_cols = tester.load_and_prepare_data()
    
    # 全閾値テスト
    results = tester.run_all_threshold_tests(df, feature_cols)
    
    # 比較レポート
    tester.print_comparison_report(results)

if __name__ == "__main__":
    main()