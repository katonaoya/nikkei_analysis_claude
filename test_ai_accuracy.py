#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIシステムの精度テストスクリプト
過去データを使用してバックテストを実施し、予測精度を評価
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class AIAccuracyTester:
    """AI精度テストクラス"""
    
    def __init__(self, config_path="production_config.yaml"):
        """初期化"""
        self.config_path = Path(config_path)
        self.load_config()
        
    def load_config(self):
        """設定ファイル読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.optimal_features = self.config['features']['optimal_features']
        self.confidence_threshold = self.config['system']['confidence_threshold']
        
    def load_data(self):
        """データ読み込み"""
        data_dir = Path(self.config['data']['processed_dir'])
        integrated_file = data_dir / self.config['data']['integrated_file']
        
        if not integrated_file.exists():
            logger.error(f"データファイルが見つかりません: {integrated_file}")
            return None
            
        logger.info(f"📥 データ読み込み: {integrated_file}")
        df = pd.read_parquet(integrated_file)
        
        # 必要な列の処理
        if 'Target' not in df.columns and 'Binary_Direction' in df.columns:
            df['Target'] = df['Binary_Direction']
        if 'Stock' not in df.columns and 'Code' in df.columns:
            df['Stock'] = df['Code']
            
        return df
    
    def run_backtest(self, df, test_days=30):
        """バックテスト実行"""
        logger.info(f"🔄 {test_days}日間のバックテスト開始...")
        
        # 日付でソート
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 利用可能な日付を取得
        unique_dates = sorted(df['Date'].unique())
        
        if len(unique_dates) < test_days + 100:  # 学習用に100日分確保
            logger.error("テストに十分なデータがありません")
            return None
            
        # テスト期間の設定
        test_start_idx = len(unique_dates) - test_days
        test_dates = unique_dates[test_start_idx:]
        
        results = []
        all_predictions = []
        all_actuals = []
        
        for test_date in test_dates:
            # 学習データとテストデータを分割
            train_data = df[df['Date'] < test_date]
            test_data = df[df['Date'] == test_date]
            
            if len(train_data) < 1000 or len(test_data) < 10:
                continue
                
            # 必要な特徴量が全て揃っているデータのみ使用
            train_clean = train_data[['Date', 'Stock', 'Target'] + self.optimal_features].dropna()
            test_clean = test_data[['Date', 'Stock', 'Target'] + self.optimal_features].dropna()
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                continue
                
            # モデル学習
            X_train = train_clean[self.optimal_features]
            y_train = train_clean['Target']
            X_test = test_clean[self.optimal_features]
            y_test = test_clean['Target']
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 予測
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            # 予測確率を取得
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= self.confidence_threshold).astype(int)
            
            # 結果記録
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            
            daily_accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'date': test_date,
                'samples': len(y_test),
                'accuracy': daily_accuracy,
                'predicted_buys': sum(y_pred),
                'actual_ups': sum(y_test)
            })
            
        return results, all_predictions, all_actuals
    
    def calculate_metrics(self, predictions, actuals):
        """評価指標の計算"""
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        f1 = f1_score(actuals, predictions, zero_division=0)
        
        # 混同行列
        cm = confusion_matrix(actuals, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def simulate_trading(self, df, test_days=30):
        """取引シミュレーション"""
        logger.info("💰 取引シミュレーション開始...")
        
        # 日付でソート
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 利用可能な日付を取得
        unique_dates = sorted(df['Date'].unique())
        
        # テスト期間
        test_start_idx = len(unique_dates) - test_days
        test_dates = unique_dates[test_start_idx:]
        
        # 初期資金
        capital = 1000000
        initial_capital = capital
        trades = []
        
        for test_date in test_dates:
            # 学習データとテストデータを分割
            train_data = df[df['Date'] < test_date]
            test_data = df[df['Date'] == test_date]
            
            if len(train_data) < 1000 or len(test_data) < 10:
                continue
                
            # クリーンなデータ
            train_clean = train_data[['Date', 'Stock', 'Close', 'Target'] + self.optimal_features].dropna()
            test_clean = test_data[['Date', 'Stock', 'Close', 'Target'] + self.optimal_features].dropna()
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                continue
                
            # モデル学習と予測
            X_train = train_clean[self.optimal_features]
            y_train = train_clean['Target']
            X_test = test_clean[self.optimal_features]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            # 予測確率を取得
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 信頼度の高い上位5銘柄を選択
            test_clean['confidence'] = y_pred_proba
            test_clean['predicted'] = (y_pred_proba >= self.confidence_threshold).astype(int)
            
            top_picks = test_clean[test_clean['predicted'] == 1].nlargest(5, 'confidence')
            
            if len(top_picks) > 0 and capital > 0:
                # 各銘柄に均等投資
                investment_per_stock = capital * 0.2  # 20%ずつ投資
                
                for _, pick in top_picks.iterrows():
                    if investment_per_stock > 0:
                        # 実際の結果（翌日の動き）
                        actual_direction = pick['Target']
                        
                        # 簡易的な利益計算（14%利確、6%損切）
                        if actual_direction == 1:
                            profit = investment_per_stock * 0.14
                        else:
                            profit = -investment_per_stock * 0.06
                        
                        capital += profit
                        
                        trades.append({
                            'date': test_date,
                            'stock': pick['Stock'],
                            'confidence': pick['confidence'],
                            'investment': investment_per_stock,
                            'actual': actual_direction,
                            'profit': profit,
                            'capital_after': capital
                        })
        
        return trades, initial_capital, capital
    
    def print_results(self, results, metrics, trades):
        """結果表示"""
        print("\n" + "="*80)
        print("🤖 AI株式取引システム - 精度テスト結果")
        print("="*80)
        
        # バックテスト結果
        if results:
            results_df = pd.DataFrame(results)
            
            print("\n📊 【予測精度】")
            print(f"  全体精度: {metrics['accuracy']:.2%}")
            print(f"  適合率（Precision）: {metrics['precision']:.2%}")
            print(f"  再現率（Recall）: {metrics['recall']:.2%}")
            print(f"  F1スコア: {metrics['f1_score']:.2%}")
            
            print(f"\n  混同行列:")
            print(f"    実際↓/予測→  下落予測  上昇予測")
            print(f"    実際下落      {metrics['confusion_matrix'][0,0]:6d}  {metrics['confusion_matrix'][0,1]:6d}")
            print(f"    実際上昇      {metrics['confusion_matrix'][1,0]:6d}  {metrics['confusion_matrix'][1,1]:6d}")
            
            print(f"\n  日次精度:")
            print(f"    平均: {results_df['accuracy'].mean():.2%}")
            print(f"    最高: {results_df['accuracy'].max():.2%}")
            print(f"    最低: {results_df['accuracy'].min():.2%}")
            
            print(f"\n  予測銘柄数:")
            print(f"    1日平均: {results_df['predicted_buys'].mean():.1f}銘柄")
            print(f"    合計: {results_df['predicted_buys'].sum()}銘柄")
        
        # 取引シミュレーション結果
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]
            
            print("\n💰 【取引シミュレーション】")
            print(f"  初期資金: ¥{trades[0]['investment']:,.0f}")
            print(f"  最終資金: ¥{trades[-1]['capital_after']:,.0f}")
            print(f"  損益: ¥{trades[-1]['capital_after'] - 1000000:+,.0f}")
            print(f"  リターン: {((trades[-1]['capital_after'] / 1000000) - 1) * 100:+.2f}%")
            
            print(f"\n  取引統計:")
            print(f"    総取引数: {len(trades_df)}回")
            print(f"    勝ち取引: {len(winning_trades)}回 ({len(winning_trades)/len(trades_df)*100:.1f}%)")
            print(f"    負け取引: {len(losing_trades)}回 ({len(losing_trades)/len(trades_df)*100:.1f}%)")
            
            if len(winning_trades) > 0:
                print(f"    平均利益: ¥{winning_trades['profit'].mean():+,.0f}")
            if len(losing_trades) > 0:
                print(f"    平均損失: ¥{losing_trades['profit'].mean():+,.0f}")
            
            print(f"\n  信頼度統計:")
            print(f"    平均信頼度: {trades_df['confidence'].mean():.2%}")
            print(f"    最高信頼度: {trades_df['confidence'].max():.2%}")
            print(f"    最低信頼度: {trades_df['confidence'].min():.2%}")
        
        print("\n" + "="*80)


def main():
    """メイン実行"""
    tester = AIAccuracyTester()
    
    # データ読み込み
    df = tester.load_data()
    if df is None:
        return
    
    logger.info(f"📊 データ件数: {len(df):,}レコード")
    
    # バックテスト実行（30日間）
    results, predictions, actuals = tester.run_backtest(df, test_days=30)
    
    if results:
        # 評価指標計算
        metrics = tester.calculate_metrics(predictions, actuals)
        
        # 取引シミュレーション
        trades, initial_capital, final_capital = tester.simulate_trading(df, test_days=30)
        
        # 結果表示
        tester.print_results(results, metrics, trades)
        
        # サマリー
        print("\n📈 【総合評価】")
        if metrics['accuracy'] >= 0.55:
            print("  ✅ 予測精度は実用レベルです")
        elif metrics['accuracy'] >= 0.52:
            print("  ⚠️ 予測精度は改善の余地があります")
        else:
            print("  ❌ 予測精度が低いため、モデルの見直しが必要です")
        
        if final_capital > initial_capital:
            print(f"  ✅ シミュレーションでは利益が出ています")
        else:
            print(f"  ❌ シミュレーションでは損失が出ています")
        
        print("\n💡 改善提案:")
        if metrics['precision'] < 0.5:
            print("  • 適合率が低いため、買いシグナルの閾値を上げることを検討")
        if metrics['recall'] < 0.3:
            print("  • 再現率が低いため、より多くの上昇銘柄を捉える特徴量の追加を検討")
        if len(results) > 0:
            avg_picks = sum([r['predicted_buys'] for r in results]) / len(results)
            if avg_picks < 3:
                print("  • 予測銘柄数が少ないため、信頼度閾値の調整を検討")
    else:
        logger.error("バックテストに失敗しました")


if __name__ == "__main__":
    main()