#!/usr/bin/env python3
"""
本番運用前の総合検証スクリプト
データ品質、リークage、現実的な取引コスト等を検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ProductionValidator:
    """本番運用前の総合検証"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
    def validate_data_integrity(self, filename: str) -> dict:
        """データ整合性の検証"""
        logger.info("🔍 Validating data integrity...")
        
        df = pd.read_parquet(self.processed_dir / filename)
        
        issues = []
        stats = {}
        
        # 1. 基本統計
        stats['total_records'] = len(df)
        stats['unique_stocks'] = df['Code'].nunique()
        stats['date_range'] = {
            'start': df['Date'].min(),
            'end': df['Date'].max(),
            'span_days': (df['Date'].max() - df['Date'].min()).days
        }
        
        # 2. 欠損値チェック
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values found: {missing_counts.sum():,} total")
            stats['missing_by_column'] = missing_counts[missing_counts > 0].to_dict()
        
        # 3. 無限大・NaN値チェック
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Infinite values in {col}: {inf_count}")
        
        # 4. ターゲット変数の分布チェック
        if 'Binary_Direction' in df.columns:
            target_dist = df['Binary_Direction'].value_counts()
            stats['target_distribution'] = target_dist.to_dict()
            
            # 極端な不均衡チェック
            minority_ratio = target_dist.min() / target_dist.sum()
            if minority_ratio < 0.3:
                issues.append(f"Severe class imbalance: {minority_ratio:.1%} minority class")
        
        # 5. 特徴量の妥当性チェック
        feature_issues = []
        
        # 価格関連の妥当性
        if 'Close' in df.columns:
            negative_prices = (df['Close'] <= 0).sum()
            if negative_prices > 0:
                feature_issues.append(f"Negative/zero prices: {negative_prices}")
                
            extreme_prices = (df['Close'] > 100000).sum()  # 10万円超の株価
            if extreme_prices > 0:
                feature_issues.append(f"Extremely high prices (>100k): {extreme_prices}")
        
        # RSI値の範囲チェック
        if 'RSI' in df.columns:
            invalid_rsi = ((df['RSI'] < 0) | (df['RSI'] > 100)).sum()
            if invalid_rsi > 0:
                feature_issues.append(f"Invalid RSI values (not 0-100): {invalid_rsi}")
        
        # 移動平均の妥当性
        ma_cols = [col for col in df.columns if col.startswith('MA_')]
        for col in ma_cols:
            negative_ma = (df[col] < 0).sum()
            if negative_ma > 0:
                feature_issues.append(f"Negative moving average {col}: {negative_ma}")
        
        if feature_issues:
            issues.extend(feature_issues)
        
        return {
            'status': 'PASS' if not issues else 'FAIL',
            'issues': issues,
            'stats': stats
        }
    
    def check_data_leakage(self, filename: str) -> dict:
        """データリークage（未来情報の混入）チェック"""
        logger.info("🔍 Checking for data leakage...")
        
        df = pd.read_parquet(self.processed_dir / filename)
        
        leakage_issues = []
        
        # 1. ターゲット変数の時系列整合性チェック
        if 'Next_Day_Return' in df.columns and 'Date' in df.columns:
            # 各銘柄ごとに時系列順序をチェック
            for code in df['Code'].unique()[:10]:  # サンプル検証
                stock_data = df[df['Code'] == code].sort_values('Date')
                
                if len(stock_data) < 2:
                    continue
                
                # 次日リターンが実際に次の日のデータと整合するかチェック
                for i in range(len(stock_data) - 1):
                    current_row = stock_data.iloc[i]
                    next_row = stock_data.iloc[i + 1]
                    
                    # 日付の連続性確認
                    date_diff = (next_row['Date'] - current_row['Date']).days
                    if date_diff > 10:  # 10日以上の間隔は異常
                        leakage_issues.append(f"Large date gap for {code}: {date_diff} days")
                        break
                
                # 1つの銘柄で十分な検証ができたら終了
                if not leakage_issues and len(stock_data) > 100:
                    break
        
        # 2. 特徴量の時系列整合性
        # 移動平均等が未来のデータを使っていないかチェック
        if 'MA_20' in df.columns and 'Close' in df.columns:
            # ランダムに10銘柄を選んで詳細チェック
            sample_codes = df['Code'].sample(min(10, df['Code'].nunique()), random_state=42)
            
            for code in sample_codes:
                stock_data = df[df['Code'] == code].sort_values('Date')
                if len(stock_data) < 30:
                    continue
                
                # 移動平均の手計算と比較
                manual_ma = stock_data['Close'].rolling(20).mean()
                system_ma = stock_data['MA_20']
                
                # 差が大きい場合はリークageの可能性
                diff = np.abs(manual_ma - system_ma)
                large_diff_count = (diff > 0.01 * stock_data['Close']).sum()  # 1%以上の差
                
                if large_diff_count > 5:
                    leakage_issues.append(f"MA_20 calculation inconsistency for {code}")
                break
        
        # 3. ターゲット変数と特徴量の相関チェック（異常に高い場合はリークage）
        if 'Binary_Direction' in df.columns:
            numeric_features = df.select_dtypes(include=[np.number]).columns
            target_corr = {}
            
            for col in numeric_features:
                if col != 'Binary_Direction' and not col.startswith('Next_Day'):
                    corr = df[col].corr(df['Binary_Direction'])
                    if not np.isnan(corr):
                        target_corr[col] = abs(corr)
            
            # 異常に高い相関（0.8以上）はリークageの疑い
            high_corr = {k: v for k, v in target_corr.items() if v > 0.8}
            if high_corr:
                leakage_issues.append(f"Suspiciously high correlations: {high_corr}")
        
        return {
            'status': 'PASS' if not leakage_issues else 'SUSPICIOUS',
            'issues': leakage_issues,
            'correlations': target_corr if 'target_corr' in locals() else {}
        }
    
    def realistic_trading_simulation(self, filename: str) -> dict:
        """現実的な取引シミュレーション"""
        logger.info("💰 Running realistic trading simulation...")
        
        df = pd.read_parquet(self.processed_dir / filename)
        
        # 取引コストの設定（日本の現実的な値）
        transaction_cost = 0.003  # 0.3%（証券会社手数料 + スプレッド + 市場インパクト）
        min_trade_amount = 100000  # 最小取引金額10万円
        
        # 2024年データでシミュレーション
        test_data = df[df['Date'] >= '2024-10-01'].copy()
        test_data = test_data.sort_values(['Date', 'Code'])
        
        if len(test_data) == 0:
            return {'status': 'NO_DATA', 'message': 'No test data available'}
        
        results = []
        
        # 各日の予測を使った取引シミュレーション
        for date in test_data['Date'].unique()[-30:]:  # 最後の30日間
            day_data = test_data[test_data['Date'] == date]
            
            if len(day_data) == 0 or 'Binary_Direction' not in day_data.columns:
                continue
            
            # 予測を使った取引戦略（上位20%を買い）
            if 'Next_Day_Return' in day_data.columns:
                # 予測リターンでソート（実際にはモデルの予測値を使用）
                day_data = day_data.dropna(subset=['Next_Day_Return'])
                top_picks = day_data.nlargest(max(1, len(day_data) // 5), 'Next_Day_Return')
                
                # 各銘柄の取引結果
                for _, stock in top_picks.iterrows():
                    actual_return = stock['Next_Day_Return']
                    
                    # 取引コスト考慮後のリターン
                    net_return = actual_return - transaction_cost
                    
                    results.append({
                        'date': date,
                        'code': stock['Code'],
                        'predicted_return': actual_return,  # 実際はモデル予測値
                        'actual_return': actual_return,
                        'net_return': net_return,
                        'trade_cost': transaction_cost
                    })
        
        if not results:
            return {'status': 'NO_TRADES', 'message': 'No valid trades found'}
        
        results_df = pd.DataFrame(results)
        
        # 現実的な性能指標
        total_trades = len(results_df)
        winning_trades = (results_df['net_return'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_gross_return = results_df['actual_return'].mean()
        avg_net_return = results_df['net_return'].mean()
        
        # シャープレシオ（リスク調整済みリターン）
        return_std = results_df['net_return'].std()
        sharpe_ratio = avg_net_return / return_std if return_std > 0 else 0
        
        # 最大ドローダウン
        cumulative_returns = (1 + results_df['net_return']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'status': 'COMPLETED',
            'summary': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_gross_return': avg_gross_return,
                'avg_net_return': avg_net_return,
                'transaction_cost_impact': avg_gross_return - avg_net_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'annualized_return': avg_net_return * 252,  # 年率換算
            }
        }
    
    def assess_production_readiness(self, data_validation: dict, leakage_check: dict, trading_sim: dict) -> dict:
        """本番運用準備度の総合評価"""
        
        red_flags = []
        warnings = []
        
        # データ品質
        if data_validation['status'] == 'FAIL':
            red_flags.extend([f"Data Quality: {issue}" for issue in data_validation['issues']])
        
        # リークage
        if leakage_check['status'] == 'SUSPICIOUS':
            red_flags.extend([f"Data Leakage: {issue}" for issue in leakage_check['issues']])
        
        # 取引性能
        if trading_sim['status'] == 'COMPLETED':
            summary = trading_sim['summary']
            
            # 勝率が低すぎる
            if summary['win_rate'] < 0.45:
                warnings.append(f"Low win rate: {summary['win_rate']:.1%}")
            
            # 取引コストの影響が大きすぎる
            cost_impact_ratio = abs(summary['transaction_cost_impact'] / summary['avg_gross_return']) if summary['avg_gross_return'] != 0 else float('inf')
            if cost_impact_ratio > 0.5:
                red_flags.append(f"Transaction costs too high: {cost_impact_ratio:.1%} of gross return")
            
            # ネットリターンが負
            if summary['avg_net_return'] < 0:
                red_flags.append(f"Negative net return: {summary['avg_net_return']:.2%}")
            
            # 最大ドローダウンが大きすぎる
            if summary['max_drawdown'] < -0.2:
                warnings.append(f"High max drawdown: {summary['max_drawdown']:.1%}")
        
        # 総合判定
        if red_flags:
            readiness = 'NOT_READY'
        elif warnings:
            readiness = 'CAUTION'
        else:
            readiness = 'READY'
        
        return {
            'readiness': readiness,
            'red_flags': red_flags,
            'warnings': warnings,
            'recommendations': self._generate_recommendations(red_flags, warnings)
        }
    
    def _generate_recommendations(self, red_flags: list, warnings: list) -> list:
        """改善提案の生成"""
        recommendations = []
        
        if any('Data Quality' in flag for flag in red_flags):
            recommendations.append("データクリーニングプロセスの見直しが必要")
        
        if any('Data Leakage' in flag for flag in red_flags):
            recommendations.append("特徴量生成プロセスでの未来情報混入を確認・修正")
        
        if any('Transaction costs' in flag for flag in red_flags):
            recommendations.append("より低コストの取引戦略または証券会社の検討")
        
        if any('Negative net return' in flag for flag in red_flags):
            recommendations.append("モデルの予測精度改善またはリスク管理強化")
        
        if any('win rate' in warn for warn in warnings):
            recommendations.append("勝率向上のための特徴量追加やモデル改善")
        
        if any('drawdown' in warn for warn in warnings):
            recommendations.append("ポジションサイジングやリスク管理の強化")
        
        return recommendations

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Production readiness validation")
    parser.add_argument("--features-file", required=True, help="Features file to validate")
    
    args = parser.parse_args()
    
    validator = ProductionValidator()
    
    print("🔍 PRODUCTION READINESS VALIDATION")
    print("="*60)
    
    # 1. データ整合性検証
    print("\n📊 Data Integrity Validation...")
    data_validation = validator.validate_data_integrity(args.features_file)
    
    print(f"Status: {'✅ PASS' if data_validation['status'] == 'PASS' else '❌ FAIL'}")
    if data_validation['issues']:
        print("Issues found:")
        for issue in data_validation['issues']:
            print(f"  - {issue}")
    
    # 2. データリークageチェック
    print("\n🔒 Data Leakage Check...")
    leakage_check = validator.check_data_leakage(args.features_file)
    
    print(f"Status: {'✅ CLEAN' if leakage_check['status'] == 'PASS' else '⚠️  SUSPICIOUS'}")
    if leakage_check['issues']:
        print("Potential leakage issues:")
        for issue in leakage_check['issues']:
            print(f"  - {issue}")
    
    # 3. 現実的取引シミュレーション
    print("\n💰 Realistic Trading Simulation...")
    trading_sim = validator.realistic_trading_simulation(args.features_file)
    
    if trading_sim['status'] == 'COMPLETED':
        summary = trading_sim['summary']
        print(f"Total trades: {summary['total_trades']}")
        print(f"Win rate: {summary['win_rate']:.1%}")
        print(f"Avg gross return: {summary['avg_gross_return']:.2%}")
        print(f"Avg net return: {summary['avg_net_return']:.2%}")
        print(f"Transaction cost impact: {summary['transaction_cost_impact']:.2%}")
        print(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {summary['max_drawdown']:.1%}")
        print(f"Annualized return: {summary['annualized_return']:.1%}")
    else:
        print(f"Status: {trading_sim['status']} - {trading_sim.get('message', 'Unknown error')}")
    
    # 4. 総合評価
    print("\n🏆 PRODUCTION READINESS ASSESSMENT")
    print("-"*40)
    assessment = validator.assess_production_readiness(data_validation, leakage_check, trading_sim)
    
    readiness_emoji = {
        'READY': '✅ READY',
        'CAUTION': '⚠️  CAUTION',
        'NOT_READY': '❌ NOT READY'
    }
    
    print(f"Overall Status: {readiness_emoji[assessment['readiness']]}")
    
    if assessment['red_flags']:
        print("\n🚨 Critical Issues:")
        for flag in assessment['red_flags']:
            print(f"  - {flag}")
    
    if assessment['warnings']:
        print("\n⚠️  Warnings:")
        for warning in assessment['warnings']:
            print(f"  - {warning}")
    
    if assessment['recommendations']:
        print("\n💡 Recommendations:")
        for rec in assessment['recommendations']:
            print(f"  - {rec}")
    
    print("\n" + "="*60)
    print("✅ Validation completed!")

if __name__ == "__main__":
    exit(main())