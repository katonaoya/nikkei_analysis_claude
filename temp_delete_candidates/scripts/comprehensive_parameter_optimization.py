#!/usr/bin/env python3
"""
完全版パラメータ最適化システム
実データ全使用、全パターン検証で最も正確な最適パラメータを発見
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

class ComprehensiveParameterOptimizer:
    """完全版パラメータ最適化（実データ全使用）"""
    
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
        
        # 完全検証用パラメータ範囲
        self.max_hold_days_range = range(1, 11)  # 1-10日
        self.profit_target_range = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]  # 3%-15%
        self.stop_loss_range = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # 2%-8%
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 完全版最適化用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
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
    
    def complete_simulation(self, df, X, y, max_hold_days, profit_target, stop_loss):
        """完全版シミュレーション（実データ全使用）"""
        dates = sorted(df['Date'].unique())
        train_end_idx = int(len(dates) * 0.8)  # 80%まで学習
        
        train_dates = dates[:train_end_idx]
        trading_dates = dates[train_end_idx:]  # 全取引日を使用
        
        logger.info(f"  学習期間: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)}日)")
        logger.info(f"  取引期間: {trading_dates[0]} - {trading_dates[-1]} ({len(trading_dates)}日)")
        
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
        daily_values = []
        
        # 3ヶ月ごとのモデル再学習
        retrain_interval = 63  # 約3ヶ月
        last_retrain = 0
        
        for i, current_date in enumerate(trading_dates):
            current_data = df[df['Date'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # モデル再学習（3ヶ月ごと）
            if i - last_retrain >= retrain_interval:
                retrain_end = train_end_idx + i
                retrain_start = max(0, retrain_end - int(len(dates) * 0.6))  # 過去60%のデータで学習
                
                retrain_dates = dates[retrain_start:retrain_end]
                retrain_mask = df['Date'].isin(retrain_dates)
                X_retrain = X[retrain_mask]
                y_retrain = y[retrain_mask]
                
                X_retrain_scaled = scaler.fit_transform(X_retrain)
                model.fit(X_retrain_scaled, y_retrain)
                
                last_retrain = i
                logger.info(f"    📚 モデル再学習: {current_date}")
            
            # 売却処理
            portfolio, cash, sell_trades = self.process_sells(
                portfolio, current_data, cash, current_date, max_hold_days, profit_target, stop_loss
            )
            trades.extend(sell_trades)
            
            # 購入処理
            if len(portfolio) < self.max_positions:
                portfolio, cash, buy_trades = self.process_buys(
                    current_data, portfolio, cash, current_date, model, scaler
                )
                trades.extend(buy_trades)
            
            # 日次評価額記録
            total_value = self.calculate_total_portfolio_value(portfolio, current_data, cash)
            daily_values.append({
                'date': current_date,
                'cash': cash,
                'portfolio_value': total_value - cash,
                'total_value': total_value,
                'positions': len(portfolio)
            })
        
        # 最終評価
        final_value = daily_values[-1]['total_value'] if daily_values else cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 詳細統計
        trades_df = pd.DataFrame(trades)
        sell_trades_df = trades_df[trades_df['action'] == 'SELL'] if len(trades_df) > 0 else pd.DataFrame()
        buy_trades_df = trades_df[trades_df['action'] == 'BUY'] if len(trades_df) > 0 else pd.DataFrame()
        
        if len(sell_trades_df) > 0:
            win_rate = len(sell_trades_df[sell_trades_df['profit_loss'] > 0]) / len(sell_trades_df)
            avg_profit = sell_trades_df['profit_loss'].mean()
            total_profit = sell_trades_df['profit_loss'].sum()
            avg_days_held = sell_trades_df['days_held'].mean()
        else:
            win_rate = 0
            avg_profit = 0
            total_profit = 0
            avg_days_held = 0
        
        # 予測精度
        if len(buy_trades_df) > 0 and 'success' in buy_trades_df.columns:
            prediction_accuracy = len(buy_trades_df[buy_trades_df['success'] == True]) / len(buy_trades_df)
        else:
            prediction_accuracy = 0
        
        # ドローダウン計算
        if len(daily_values) > 0:
            daily_df = pd.DataFrame(daily_values)
            daily_df['peak'] = daily_df['total_value'].cummax()
            daily_df['drawdown'] = (daily_df['total_value'] - daily_df['peak']) / daily_df['peak']
            max_drawdown = daily_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        total_costs = sum(t.get('cost', 0) for t in trades)
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades_df),
            'sell_trades': len(sell_trades_df),
            'total_costs': total_costs,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'avg_days_held': avg_days_held,
            'prediction_accuracy': prediction_accuracy,
            'max_drawdown': max_drawdown,
            'trading_days': len(daily_values),
            'parameters': {
                'max_hold_days': max_hold_days,
                'profit_target': profit_target,
                'stop_loss': stop_loss
            }
        }
    
    def process_sells(self, portfolio, current_data, cash, current_date, max_hold_days, profit_target, stop_loss):
        """売却処理"""
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
            
            if days_held >= max_hold_days:
                should_sell = True
                sell_reason = "期間満了"
            elif profit_rate <= -stop_loss:
                should_sell = True
                sell_reason = "損切り"
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
    
    def process_buys(self, current_data, portfolio, cash, current_date, model, scaler):
        """購入処理"""
        buys = []
        
        X_day = current_data[self.optimal_features].fillna(0)
        X_day_scaled = scaler.transform(X_day)
        pred_proba = model.predict_proba(X_day_scaled)[:, 1]
        current_data['pred_proba'] = pred_proba
        
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
            
            # 成功判定用
            actual_success = stock_data['Binary_Direction'] if 'Binary_Direction' in current_data.columns else None
            
            buys.append({
                'date': current_date,
                'code': code,
                'action': 'BUY',
                'shares': max_shares,
                'price': buy_price,
                'value': buy_value,
                'cost': transaction_cost,
                'total_cost': total_cost,
                'confidence': stock_data['pred_proba'],
                'success': actual_success == 1 if actual_success is not None else None
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
    
    def comprehensive_optimization(self, df, X, y):
        """完全版パラメータ最適化"""
        logger.info("🎯 完全版パラメータ最適化開始...")
        
        # 全パラメータ組み合わせ生成（利確 > 損切りのみ）
        parameter_combinations = []
        for hold_days in self.max_hold_days_range:
            for profit_target in self.profit_target_range:
                for stop_loss in self.stop_loss_range:
                    if profit_target > stop_loss:  # 利確 > 損切りの条件
                        parameter_combinations.append((hold_days, profit_target, stop_loss))
        
        logger.info(f"検証パラメータ組み合わせ: {len(parameter_combinations)}パターン")
        logger.info("⏰ 注意: 完全検証のため時間がかかります（推定数時間）")
        
        results = []
        
        for i, (max_hold_days, profit_target, stop_loss) in enumerate(parameter_combinations):
            logger.info(f"  {i+1:3d}/{len(parameter_combinations)}: {max_hold_days}日, {profit_target:.1%}, {stop_loss:.1%}")
            
            try:
                result = self.complete_simulation(df, X, y, max_hold_days, profit_target, stop_loss)
                results.append(result)
                
                # 進捗報告（10%ごと）
                if (i + 1) % (len(parameter_combinations) // 10) == 0:
                    progress = (i + 1) / len(parameter_combinations) * 100
                    logger.info(f"  📊 進捗: {progress:.0f}%完了 (上位結果: {max(r['total_return'] for r in results):.2%})")
                
            except Exception as e:
                logger.error(f"パラメータ({max_hold_days}, {profit_target:.2f}, {stop_loss:.2f})でエラー: {e}")
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
                'prediction_accuracy': r['prediction_accuracy'],
                'avg_profit': r['avg_profit'],
                'max_drawdown': r['max_drawdown'],
                'avg_days_held': r['avg_days_held']
            }
            for r in results
        ])
        
        return results_df
    
    def display_comprehensive_results(self, results_df):
        """完全版結果表示"""
        logger.info("\n" + "="*150)
        logger.info("🏆 完全版パラメータ最適化結果")
        logger.info("="*150)
        
        # 上位20パターン
        top_20 = results_df.nlargest(20, 'total_return')
        
        logger.info(f"\n📈 総リターン上位20パターン:")
        logger.info("順位 | 保有 | 利確 | 損切 | 総リターン | 最終評価額 | 勝率  | 予測精度 | 取引数 | 平均利益 | DD    | 平均保有")
        logger.info("-" * 140)
        
        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            logger.info(
                f"{i:2d}位 | {row['max_hold_days']:2.0f}日 | {row['profit_target']:4.1%} | "
                f"{row['stop_loss']:4.1%} | {row['total_return']:8.2%} | "
                f"¥{row['final_value']:9,.0f} | {row['win_rate']:4.1%} | "
                f"{row['prediction_accuracy']:6.1%} | {row['total_trades']:4.0f}回 | "
                f"¥{row['avg_profit']:6,.0f} | {row['max_drawdown']:5.1%} | {row['avg_days_held']:4.1f}日"
            )
        
        # 最優秀パラメータ
        best = top_20.iloc[0]
        logger.info(f"\n🥇 最優秀パラメータ:")
        logger.info(f"  保有期間: {best['max_hold_days']:.0f}日")
        logger.info(f"  利確閾値: {best['profit_target']:.1%}")
        logger.info(f"  損切閾値: {best['stop_loss']:.1%}")
        logger.info(f"  総リターン: {best['total_return']:.2%}")
        logger.info(f"  最終評価額: ¥{best['final_value']:,.0f}")
        logger.info(f"  勝率: {best['win_rate']:.1%}")
        logger.info(f"  予測精度: {best['prediction_accuracy']:.1%}")
        logger.info(f"  最大DD: {best['max_drawdown']:.1%}")
        
        # 統計分析
        logger.info(f"\n📊 パラメータ別統計:")
        
        # 保有期間別
        hold_stats = results_df.groupby('max_hold_days').agg({
            'total_return': ['mean', 'max', 'std'],
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        logger.info(f"\n保有期間別パフォーマンス:")
        for days in sorted(results_df['max_hold_days'].unique()):
            stats = hold_stats.loc[days]
            logger.info(f"  {days:2.0f}日: 平均{stats[('total_return', 'mean')]:6.2%}/最高{stats[('total_return', 'max')]:6.2%}/σ{stats[('total_return', 'std')]:5.2%}, 勝率{stats[('win_rate', 'mean')]:5.1%}, DD{stats[('max_drawdown', 'mean')]:5.1%}")
        
        # 利確・損切り別上位5位
        profit_stats = results_df.groupby('profit_target')['total_return'].agg(['mean', 'max', 'count']).round(4)
        logger.info(f"\n利確閾値別パフォーマンス（上位5位）:")
        for rate, stats in profit_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%} ({stats['count']}パターン)")
        
        loss_stats = results_df.groupby('stop_loss')['total_return'].agg(['mean', 'max', 'count']).round(4)
        logger.info(f"\n損切閾値別パフォーマンス（上位5位）:")
        for rate, stats in loss_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%} ({stats['count']}パターン)")
        
        logger.info("="*150)
        
        return best, results_df

def main():
    """メイン実行"""
    logger.info("💎 完全版取引パラメータ最適化システム（実データ全使用）")
    
    optimizer = ComprehensiveParameterOptimizer()
    
    try:
        # データ準備
        df, X, y = optimizer.load_and_prepare_data()
        
        # 完全版最適化
        results_df = optimizer.comprehensive_optimization(df, X, y)
        
        # 結果表示
        best_params, full_results = optimizer.display_comprehensive_results(results_df)
        
        # 結果保存
        results_file = Path("optimization_results.csv")
        full_results.to_csv(results_file, index=False)
        logger.info(f"📁 結果保存: {results_file}")
        
        logger.info(f"\n✅ 完全版パラメータ最適化完了 - {len(full_results)}パターンを検証")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()