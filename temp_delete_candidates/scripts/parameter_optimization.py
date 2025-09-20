#!/usr/bin/env python3
"""
取引パラメータ最適化システム
保有期間・利確・損切りの全パターンを検証して最大収益を発見
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class ParameterOptimizer:
    """取引パラメータ最適化"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 固定パラメータ
        self.initial_capital = 1000000
        self.confidence_threshold = 0.55
        self.max_positions = 5
        self.transaction_cost_rate = 0.001
        
        # 最適化パラメータ範囲（ユーザー指定）
        self.max_hold_days_range = range(1, 11)  # 1-10日
        self.profit_target_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]  # 1%-15%
        self.stop_loss_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]  # 1%-15%
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 パラメータ最適化用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件")
        
        return clean_df, X, y
    
    def filtered_selection(self, day_data):
        """フィルタリング銘柄選択"""
        if len(day_data) == 0 or 'pred_proba' not in day_data.columns:
            return []
        
        day_data = day_data.copy()
        day_data['abs_confidence'] = np.maximum(day_data['pred_proba'], 1 - day_data['pred_proba'])
        
        high_conf = day_data[day_data['abs_confidence'] >= self.confidence_threshold]
        if len(high_conf) == 0:
            return []
        
        selected = high_conf.nlargest(self.max_positions, 'abs_confidence')
        return selected['Code'].tolist()
    
    def simulate_with_parameters(self, df, X, y, max_hold_days, profit_target, stop_loss):
        """指定パラメータでのシミュレーション"""
        dates = sorted(df['Date'].unique())
        train_end_idx = int(len(dates) * 0.8)
        
        train_dates = dates[:train_end_idx]
        trading_dates = dates[train_end_idx:]
        
        # 初期学習
        train_mask = df['Date'].isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # 運用状態管理
        cash = self.initial_capital
        portfolio = {}
        trades = []
        
        # 簡易シミュレーション（3ヶ月ごとの再学習は省略）
        for current_date in trading_dates:
            current_data = df[df['Date'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # 売却処理
            portfolio, cash, sell_trades = self.process_sells_optimized(
                portfolio, current_data, cash, current_date, max_hold_days, profit_target, stop_loss
            )
            trades.extend(sell_trades)
            
            # 購入処理
            if len(portfolio) < self.max_positions:
                portfolio, cash, buy_trades = self.process_buys_optimized(
                    current_data, portfolio, cash, current_date, model, scaler
                )
                trades.extend(buy_trades)
        
        # 最終評価
        final_value = cash
        current_data = df[df['Date'] == trading_dates[-1]]
        if len(current_data) > 0:
            final_value = self.calculate_total_portfolio_value(portfolio, current_data, cash)
        
        # 結果計算
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        sell_trades_df = pd.DataFrame([t for t in trades if t['action'] == 'SELL'])
        if len(sell_trades_df) > 0:
            win_rate = len(sell_trades_df[sell_trades_df['profit_loss'] > 0]) / len(sell_trades_df)
            total_trades = len(trades)
            total_costs = sum(t.get('cost', 0) for t in trades)
        else:
            win_rate = 0
            total_trades = len(trades)
            total_costs = sum(t.get('cost', 0) for t in trades)
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_costs': total_costs,
            'parameters': {
                'max_hold_days': max_hold_days,
                'profit_target': profit_target,
                'stop_loss': stop_loss
            }
        }
    
    def process_sells_optimized(self, portfolio, current_data, cash, current_date, max_hold_days, profit_target, stop_loss):
        """最適化版売却処理"""
        sells = []
        current_prices = current_data.set_index('Code')['Close']
        
        codes_to_remove = []
        
        for code, position in portfolio.items():
            if code not in current_prices.index:
                continue
                
            current_price = current_prices[code]
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            days_held = (pd.to_datetime(current_date) - pd.to_datetime(position['buy_date'])).days
            profit_rate = (current_price - position['buy_price']) / position['buy_price']
            
            should_sell = False
            sell_reason = ""
            
            # 保有期間経過
            if days_held >= max_hold_days:
                should_sell = True
                sell_reason = "期間満了"
            
            # 損切り
            elif profit_rate <= -stop_loss:
                should_sell = True
                sell_reason = "損切り"
            
            # 利確
            elif profit_rate >= profit_target:
                should_sell = True
                sell_reason = "利確"
            
            if should_sell:
                sell_value = position['shares'] * current_price
                transaction_cost = sell_value * self.transaction_cost_rate
                net_proceeds = sell_value - transaction_cost
                profit_loss = net_proceeds - (position['shares'] * position['buy_price'])
                
                sells.append({
                    'date': current_date,
                    'code': code,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': current_price,
                    'buy_price': position['buy_price'],
                    'value': sell_value,
                    'cost': transaction_cost,
                    'net_proceeds': net_proceeds,
                    'profit_loss': profit_loss,
                    'days_held': days_held,
                    'sell_reason': sell_reason
                })
                
                cash += net_proceeds
                codes_to_remove.append(code)
        
        for code in codes_to_remove:
            del portfolio[code]
        
        return portfolio, cash, sells
    
    def process_buys_optimized(self, current_data, portfolio, cash, current_date, model, scaler):
        """最適化版購入処理"""
        buys = []
        
        # 予測実行
        X_day = current_data[self.optimal_features].fillna(0)
        X_day_scaled = scaler.transform(X_day)
        pred_proba = model.predict_proba(X_day_scaled)[:, 1]
        current_data['pred_proba'] = pred_proba
        
        # フィルタリング選択
        selected_codes = self.filtered_selection(current_data)
        available_codes = [code for code in selected_codes if code not in portfolio]
        
        if len(available_codes) == 0:
            return portfolio, cash, buys
        
        available_positions = self.max_positions - len(portfolio)
        codes_to_buy = available_codes[:available_positions]
        
        if len(codes_to_buy) == 0:
            return portfolio, cash, buys
        
        investment_per_stock = (cash * 0.95) / len(codes_to_buy)
        
        for code in codes_to_buy:
            stock_data = current_data[current_data['Code'] == code].iloc[0]
            buy_price = stock_data['Close']
            
            if pd.isna(buy_price) or buy_price <= 0:
                continue
            
            max_shares = int(investment_per_stock / buy_price)
            if max_shares <= 0:
                continue
            
            buy_value = max_shares * buy_price
            transaction_cost = buy_value * self.transaction_cost_rate
            total_cost = buy_value + transaction_cost
            
            if total_cost > cash:
                continue
            
            portfolio[code] = {
                'shares': max_shares,
                'buy_price': buy_price,
                'buy_date': current_date
            }
            
            buys.append({
                'date': current_date,
                'code': code,
                'action': 'BUY',
                'shares': max_shares,
                'price': buy_price,
                'value': buy_value,
                'cost': transaction_cost,
                'total_cost': total_cost
            })
            
            cash -= total_cost
        
        return portfolio, cash, buys
    
    def calculate_total_portfolio_value(self, portfolio, current_data, cash):
        """ポートフォリオ総評価額計算"""
        total_value = cash
        current_prices = current_data.set_index('Code')['Close']
        
        for code, position in portfolio.items():
            if code in current_prices.index:
                current_price = current_prices[code]
                if not pd.isna(current_price) and current_price > 0:
                    total_value += position['shares'] * current_price
                else:
                    total_value += position['shares'] * position['buy_price']
            else:
                total_value += position['shares'] * position['buy_price']
        
        return total_value
    
    def optimize_parameters(self, df, X, y):
        """パラメータ最適化実行"""
        logger.info("🎯 取引パラメータ最適化開始...")
        
        # 全パラメータ組み合わせ生成
        parameter_combinations = list(product(
            self.max_hold_days_range,
            self.profit_target_range,
            self.stop_loss_range
        ))
        
        # 明らかに損失となる組み合わせを除外
        valid_combinations = []
        for hold, profit, loss in parameter_combinations:
            # 基本条件：利確率 > 損切率
            if profit <= loss:
                continue
            
            # 非効率な組み合わせを除外
            # 1. 損切り率が高すぎる（10%超）かつ利確率が低い（5%未満）
            if loss > 0.10 and profit < 0.05:
                continue
                
            # 2. 損切り率が15%で利確率も15%（リスクが高すぎる）
            if loss >= 0.15 and profit >= 0.15:
                continue
                
            # 3. 保有期間が長い（7日超）かつ損切り率が高い（8%超）
            if hold > 7 and loss > 0.08:
                continue
                
            # 4. 利確率と損切率の差が小さすぎる（1%未満）
            if (profit - loss) < 0.01:
                continue
            
            valid_combinations.append((hold, profit, loss))
        
        logger.info(f"検証パラメータ組み合わせ: {len(valid_combinations)}パターン")
        
        results = []
        
        for i, (max_hold_days, profit_target, stop_loss) in enumerate(valid_combinations):
            if i % 100 == 0:
                logger.info(f"  進行状況: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%)")
            
            try:
                result = self.simulate_with_parameters(df, X, y, max_hold_days, profit_target, stop_loss)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"パラメータ({max_hold_days}, {profit_target:.2f}, {stop_loss:.2f})でエラー: {e}")
                continue
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame([
            {
                'max_hold_days': r['parameters']['max_hold_days'],
                'profit_target': r['parameters']['profit_target'],
                'stop_loss': r['parameters']['stop_loss'],
                'total_return': r['total_return'],
                'final_value': r['final_value'],
                'win_rate': r['win_rate'],
                'total_trades': r['total_trades'],
                'total_costs': r['total_costs']
            }
            for r in results
        ])
        
        return results_df
    
    def display_optimization_results(self, results_df):
        """最適化結果表示"""
        logger.info("\n" + "="*120)
        logger.info("🏆 取引パラメータ最適化結果")
        logger.info("="*120)
        
        # 上位10パターン
        top_10 = results_df.nlargest(10, 'total_return')
        
        logger.info(f"\n📈 総リターン上位10パターン:")
        logger.info("順位 | 保有日数 | 利確率 | 損切率 | 総リターン | 最終評価額 | 勝率   | 取引数")
        logger.info("-" * 80)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            logger.info(
                f"{i:2d}位 | {row['max_hold_days']:4.0f}日  | {row['profit_target']:5.1%} | "
                f"{row['stop_loss']:5.1%} | {row['total_return']:8.2%} | "
                f"¥{row['final_value']:9,.0f} | {row['win_rate']:5.1%} | {row['total_trades']:4.0f}回"
            )
        
        # 最優秀パラメータ
        best = top_10.iloc[0]
        logger.info(f"\n🥇 最優秀パラメータ:")
        logger.info(f"  保有期間: {best['max_hold_days']:.0f}日")
        logger.info(f"  利確閾値: {best['profit_target']:.1%}")
        logger.info(f"  損切閾値: {best['stop_loss']:.1%}")
        logger.info(f"  期待リターン: {best['total_return']:.2%}")
        logger.info(f"  期待最終額: ¥{best['final_value']:,.0f}")
        logger.info(f"  勝率: {best['win_rate']:.1%}")
        
        # 統計分析
        logger.info(f"\n📊 パラメータ別傾向:")
        
        # 保有期間別
        hold_days_stats = results_df.groupby('max_hold_days')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n保有期間別平均リターン:")
        for days, stats in hold_days_stats.iterrows():
            logger.info(f"  {days:2.0f}日: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 利確率別
        profit_stats = results_df.groupby('profit_target')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n利確閾値別平均リターン（上位5位）:")
        for rate, stats in profit_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 損切り率別
        loss_stats = results_df.groupby('stop_loss')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n損切閾値別平均リターン（上位5位）:")
        for rate, stats in loss_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        logger.info("="*120)
        
        return best

def main():
    """メイン実行"""
    logger.info("⚡ 取引パラメータ最適化システム")
    
    optimizer = ParameterOptimizer()
    
    try:
        # データ準備
        df, X, y = optimizer.load_and_prepare_data()
        
        # パラメータ最適化
        results_df = optimizer.optimize_parameters(df, X, y)
        
        # 結果表示
        best_params = optimizer.display_optimization_results(results_df)
        
        logger.info(f"\n✅ パラメータ最適化完了 - {len(results_df)}パターンを検証")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()