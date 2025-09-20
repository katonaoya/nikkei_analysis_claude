#!/usr/bin/env python3
"""
1年間の長期精度テスト (2024年10月〜2025年9月)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class YearlyAccuracyTester:
    """1年間の精度テストクラス"""
    
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
        
        # カラム名の調整
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code']
            
        # 必要な特徴量の生成
        logger.info("📊 特徴量生成中...")
        df = self.generate_features(df)
        
        return df
    
    def generate_features(self, df):
        """特徴量生成"""
        # グループごとに特徴量を計算
        features = []
        
        for stock, stock_df in df.groupby('Stock'):
            stock_df = stock_df.sort_values('Date')
            
            # 基本的な価格変化
            stock_df['Return'] = stock_df['close'].pct_change()
            stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
            
            # RSI
            delta = stock_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1)
            stock_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 移動平均からの乖離率
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
        return df
    
    def run_yearly_backtest(self, df, start_date='2024-10-01', end_date='2025-09-30'):
        """1年間のバックテスト実行"""
        logger.info(f"🔄 {start_date} から {end_date} までのバックテスト開始...")
        
        # 日付でソート
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 期間フィルタ
        test_start = pd.to_datetime(start_date)
        test_end = pd.to_datetime(end_date)
        
        # テスト期間のデータ
        test_period_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
        unique_test_dates = sorted(test_period_df['Date'].unique())
        
        logger.info(f"テスト期間: {len(unique_test_dates)}営業日")
        
        # 月次集計用
        monthly_results = {}
        all_predictions = []
        all_actuals = []
        daily_results = []
        
        for test_date in unique_test_dates:
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
            
            # 月次集計
            month_key = test_date.strftime('%Y-%m')
            if month_key not in monthly_results:
                monthly_results[month_key] = {
                    'predictions': [],
                    'actuals': [],
                    'dates': []
                }
            
            monthly_results[month_key]['predictions'].extend(y_pred)
            monthly_results[month_key]['actuals'].extend(y_test)
            monthly_results[month_key]['dates'].append(test_date)
            
            daily_results.append({
                'date': test_date,
                'samples': len(y_test),
                'accuracy': daily_accuracy,
                'predicted_buys': sum(y_pred),
                'actual_ups': sum(y_test)
            })
            
        return monthly_results, all_predictions, all_actuals, daily_results
    
    def calculate_metrics(self, predictions, actuals):
        """評価指標の計算"""
        if len(predictions) == 0:
            return None
            
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
            'confusion_matrix': cm,
            'total_predictions': len(predictions),
            'positive_predictions': sum(predictions),
            'actual_positives': sum(actuals)
        }
    
    def print_yearly_results(self, monthly_results, all_predictions, all_actuals, daily_results):
        """年間結果表示"""
        print("\n" + "="*80)
        print("🗓️ 1年間の株価予測精度テスト結果 (2024年10月〜2025年9月)")
        print("="*80)
        
        # 全体の指標
        overall_metrics = self.calculate_metrics(all_predictions, all_actuals)
        
        if overall_metrics:
            print("\n📊 【年間総合成績】")
            print(f"  全体精度: {overall_metrics['accuracy']:.2%}")
            print(f"  適合率（Precision）: {overall_metrics['precision']:.2%}")
            print(f"  再現率（Recall）: {overall_metrics['recall']:.2%}")
            print(f"  F1スコア: {overall_metrics['f1_score']:.2%}")
            print(f"\n  総予測数: {overall_metrics['total_predictions']:,}件")
            print(f"  上昇予測数: {overall_metrics['positive_predictions']:,}件 ({overall_metrics['positive_predictions']/overall_metrics['total_predictions']*100:.1f}%)")
            print(f"  実際の上昇数: {overall_metrics['actual_positives']:,}件")
            
            print(f"\n  混同行列:")
            print(f"    実際↓/予測→  下落予測  上昇予測")
            print(f"    実際下落      {overall_metrics['confusion_matrix'][0,0]:6d}  {overall_metrics['confusion_matrix'][0,1]:6d}")
            print(f"    実際上昇      {overall_metrics['confusion_matrix'][1,0]:6d}  {overall_metrics['confusion_matrix'][1,1]:6d}")
        
        # 月次成績
        print("\n📅 【月次精度推移】")
        print("  月        精度    Prec.   Recall  予測数  上昇予測")
        print("  " + "-"*50)
        
        quarterly_results = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
        
        for month in sorted(monthly_results.keys()):
            month_data = monthly_results[month]
            month_metrics = self.calculate_metrics(
                month_data['predictions'], 
                month_data['actuals']
            )
            
            if month_metrics:
                # 四半期の判定
                month_num = int(month.split('-')[1])
                if month_num in [10, 11, 12]:
                    quarter = 'Q4'
                elif month_num in [1, 2, 3]:
                    quarter = 'Q1'
                elif month_num in [4, 5, 6]:
                    quarter = 'Q2'
                else:
                    quarter = 'Q3'
                
                quarterly_results[quarter].append(month_metrics['accuracy'])
                
                print(f"  {month}  {month_metrics['accuracy']:6.1%}  {month_metrics['precision']:6.1%}  {month_metrics['recall']:6.1%}  {month_metrics['total_predictions']:5d}  {month_metrics['positive_predictions']:5d}")
        
        # 四半期別サマリー
        print("\n📈 【四半期別サマリー】")
        for quarter in ['Q4', 'Q1', 'Q2', 'Q3']:
            if quarterly_results[quarter]:
                q_avg = np.mean(quarterly_results[quarter])
                year = '2024' if quarter == 'Q4' else '2025'
                print(f"  {year} {quarter}: 平均精度 {q_avg:.1%}")
        
        # 統計サマリー
        if daily_results:
            daily_df = pd.DataFrame(daily_results)
            print("\n📊 【日次精度統計】")
            print(f"  平均精度: {daily_df['accuracy'].mean():.2%}")
            print(f"  最高精度: {daily_df['accuracy'].max():.2%}")
            print(f"  最低精度: {daily_df['accuracy'].min():.2%}")
            print(f"  標準偏差: {daily_df['accuracy'].std():.2%}")
            
            # 精度50%以上の日数
            good_days = len(daily_df[daily_df['accuracy'] >= 0.5])
            print(f"\n  精度50%以上の日: {good_days}/{len(daily_df)} ({good_days/len(daily_df)*100:.1f}%)")
            
        print("\n" + "="*80)

def main():
    """メイン実行"""
    tester = YearlyAccuracyTester()
    
    # データ読み込み
    df = tester.load_data()
    if df is None:
        return
    
    logger.info(f"📊 データ件数: {len(df):,}レコード")
    
    # 1年間のバックテスト実行
    monthly_results, predictions, actuals, daily_results = tester.run_yearly_backtest(
        df, 
        start_date='2024-10-01',
        end_date='2025-09-30'
    )
    
    # 結果表示
    tester.print_yearly_results(monthly_results, predictions, actuals, daily_results)

if __name__ == "__main__":
    main()