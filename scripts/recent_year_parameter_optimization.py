#!/usr/bin/env python3
"""
直近1年間検証期間での詳細パラメータ最適化
利確・損切り1%から1%刻みで全パターン並行処理検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class RecentYearParameterOptimizer:
    """直近1年間での詳細パラメータ最適化"""
    
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
        
        # 詳細検証用パラメータ範囲（1%刻み）
        self.max_hold_days_range = range(1, 11)  # 1-10日
        self.profit_target_range = [i/100 for i in range(1, 21)]  # 1%-20% (1%刻み)
        self.stop_loss_range = [i/100 for i in range(1, 16)]  # 1%-15% (1%刻み)
        
        # データをクラス変数として保持（並行処理用）
        self.df = None
        self.X = None
        self.y = None
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 直近1年検証用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        # クラス変数に保存
        self.df = clean_df
        self.X = X
        self.y = y
        
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
    
    def single_parameter_simulation(self, params):
        """単一パラメータでのシミュレーション（直近1年間版）"""
        max_hold_days, profit_target, stop_loss = params
        
        try:
            df = self.df
            X = self.X
            y = self.y
            
            dates = sorted(df['Date'].unique())
            
            # 直近1年間のテスト期間設定
            test_days = 252  # 1年 = 252営業日
            trading_dates = dates[-test_days:]
            
            # 学習期間は直近1年前までの全期間（大幅に拡張）
            train_end_date = trading_dates[0]
            train_dates = [d for d in dates if d < train_end_date]
            
            # 学習期間が短すぎる場合の調整（最低3年分は確保）
            if len(train_dates) < 756:  # 3年 = 756営業日
                train_dates = dates[:max(756, len(dates) - test_days)]
            
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
            retrain_interval = 63
            last_retrain = 0
            
            for i, current_date in enumerate(trading_dates):
                current_data = df[df['Date'] == current_date].copy()
                
                if len(current_data) == 0:
                    continue
                
                # モデル再学習
                if i - last_retrain >= retrain_interval:
                    retrain_end_date = current_date
                    retrain_start_date = dates[max(0, dates.index(retrain_end_date) - 378)]  # 1.5年分
                    
                    retrain_dates = [d for d in dates if retrain_start_date <= d < retrain_end_date]
                    retrain_mask = df['Date'].isin(retrain_dates)
                    X_retrain = X[retrain_mask]
                    y_retrain = y[retrain_mask]
                    
                    if len(X_retrain) > 100:
                        X_retrain_scaled = scaler.fit_transform(X_retrain)
                        model.fit(X_retrain_scaled, y_retrain)
                        last_retrain = i
                
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
                    'total_value': total_value
                })
            
            # 結果計算
            final_value = daily_values[-1]['total_value'] if daily_values else cash
            total_return = (final_value - self.initial_capital) / self.initial_capital
            annual_return = total_return  # 直近1年間なのでそのまま年率
            
            # 統計計算
            trades_df = pd.DataFrame(trades)
            sell_trades_df = trades_df[trades_df['action'] == 'SELL'] if len(trades_df) > 0 else pd.DataFrame()
            buy_trades_df = trades_df[trades_df['action'] == 'BUY'] if len(trades_df) > 0 else pd.DataFrame()
            
            if len(sell_trades_df) > 0:
                win_rate = len(sell_trades_df[sell_trades_df['profit_loss'] > 0]) / len(sell_trades_df)
                avg_profit = sell_trades_df['profit_loss'].mean()
                avg_days_held = sell_trades_df['days_held'].mean()
            else:
                win_rate = 0
                avg_profit = 0
                avg_days_held = 0
            
            # 予測精度
            if len(buy_trades_df) > 0 and 'success' in buy_trades_df.columns:
                prediction_accuracy = len(buy_trades_df[buy_trades_df['success'] == True]) / len(buy_trades_df)
            else:
                prediction_accuracy = 0
            
            # ドローダウン
            if len(daily_values) > 0:
                daily_df = pd.DataFrame(daily_values)
                daily_df['peak'] = daily_df['total_value'].cummax()
                daily_df['drawdown'] = (daily_df['total_value'] - daily_df['peak']) / daily_df['peak']
                max_drawdown = daily_df['drawdown'].min()
            else:
                max_drawdown = 0
            
            return {
                'max_hold_days': max_hold_days,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'total_return': total_return,
                'annual_return': annual_return,
                'final_value': final_value,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'prediction_accuracy': prediction_accuracy,
                'avg_profit': avg_profit,
                'max_drawdown': max_drawdown,
                'avg_days_held': avg_days_held
            }
        
        except Exception as e:
            return None
    
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
    
    def parallel_optimization(self, df, X, y):
        """並行処理版パラメータ最適化（直近1年）"""
        logger.info("⚡ 直近1年間での詳細パラメータ最適化開始...")
        
        # 全パラメータ組み合わせ生成（利確 > 損切りのみ）
        parameter_combinations = []
        for hold_days in self.max_hold_days_range:
            for profit_target in self.profit_target_range:
                for stop_loss in self.stop_loss_range:
                    if profit_target > stop_loss:  # 利確 > 損切りの条件
                        parameter_combinations.append((hold_days, profit_target, stop_loss))
        
        n_cores = min(cpu_count(), 8)
        logger.info(f"検証パラメータ: {len(parameter_combinations)}パターン")
        logger.info(f"使用CPU cores: {n_cores}")
        logger.info(f"検証期間: 直近1年間（252営業日）")
        logger.info("⏰ 1%刻みの詳細検証により高精度な最適化")
        
        # 並行処理実行
        with Pool(processes=n_cores) as pool:
            results = []
            
            for i, result in enumerate(pool.imap(self.single_parameter_simulation, parameter_combinations)):
                if result is not None:
                    results.append(result)
                
                # 進捗報告
                if (i + 1) % 100 == 0 or i == len(parameter_combinations) - 1:
                    progress = (i + 1) / len(parameter_combinations) * 100
                    current_best = max(results, key=lambda x: x['annual_return'])['annual_return'] if results else 0
                    logger.info(f"  📊 進捗: {i+1}/{len(parameter_combinations)} ({progress:.1f}%) - 現在最高年率: {current_best:.2%}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def display_recent_year_results(self, results_df):
        """直近1年検証結果表示"""
        logger.info("\n" + "="*150)
        logger.info("🏆 直近1年間詳細パラメータ最適化結果")
        logger.info("="*150)
        
        # 上位30パターン
        top_30 = results_df.nlargest(30, 'annual_return')
        
        logger.info(f"\n📈 年率リターン上位30パターン:")
        logger.info("順位 | 保有 | 利確 | 損切 | 年率リターン | 最終評価額 | 勝率  | 予測精度 | 取引数 | 平均利益 | DD    | 平均保有")
        logger.info("-" * 140)
        
        for i, (_, row) in enumerate(top_30.iterrows(), 1):
            logger.info(
                f"{i:2d}位 | {row['max_hold_days']:2.0f}日 | {row['profit_target']:4.1%} | "
                f"{row['stop_loss']:4.1%} | {row['annual_return']:10.2%} | "
                f"¥{row['final_value']:9,.0f} | {row['win_rate']:4.1%} | "
                f"{row['prediction_accuracy']:6.1%} | {row['total_trades']:4.0f}回 | "
                f"¥{row['avg_profit']:6,.0f} | {row['max_drawdown']:5.1%} | {row['avg_days_held']:4.1f}日"
            )
        
        # 最優秀パラメータ
        best = top_30.iloc[0]
        logger.info(f"\n🥇 直近1年間での最優秀パラメータ:")
        logger.info(f"  保有期間: {best['max_hold_days']:.0f}日")
        logger.info(f"  利確閾値: {best['profit_target']:.0%}")
        logger.info(f"  損切閾値: {best['stop_loss']:.0%}")
        logger.info(f"  年率リターン: {best['annual_return']:.2%}")
        logger.info(f"  最終評価額: ¥{best['final_value']:,.0f}")
        logger.info(f"  勝率: {best['win_rate']:.1%}")
        logger.info(f"  予測精度: {best['prediction_accuracy']:.1%}")
        logger.info(f"  最大DD: {best['max_drawdown']:.1%}")
        
        # 保有期間別分析
        logger.info(f"\n📊 保有期間別パフォーマンス（上位5位）:")
        hold_stats = results_df.groupby('max_hold_days')['annual_return'].agg(['mean', 'max']).round(4)
        top_hold_days = hold_stats.nlargest(5, 'max')
        
        for days, stats in top_hold_days.iterrows():
            logger.info(f"  {days:2.0f}日: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 利確率別分析
        logger.info(f"\n📊 利確閾値別パフォーマンス（上位5位）:")
        profit_stats = results_df.groupby('profit_target')['annual_return'].agg(['mean', 'max']).round(4)
        top_profit_rates = profit_stats.nlargest(5, 'max')
        
        for rate, stats in top_profit_rates.iterrows():
            logger.info(f"  {rate:5.0%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 損切り率別分析
        logger.info(f"\n📊 損切閾値別パフォーマンス（上位5位）:")
        loss_stats = results_df.groupby('stop_loss')['annual_return'].agg(['mean', 'max']).round(4)
        top_loss_rates = loss_stats.nlargest(5, 'max')
        
        for rate, stats in top_loss_rates.iterrows():
            logger.info(f"  {rate:5.0%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        logger.info("="*150)
        
        return best, results_df

def main():
    """メイン実行"""
    logger.info("🚀 直近1年間詳細パラメータ最適化システム（1%刻み）")
    
    optimizer = RecentYearParameterOptimizer()
    
    try:
        # データ準備
        df, X, y = optimizer.load_and_prepare_data()
        
        # 並行処理最適化
        results_df = optimizer.parallel_optimization(df, X, y)
        
        # 結果表示
        best_params, full_results = optimizer.display_recent_year_results(results_df)
        
        # 結果保存
        results_file = Path("recent_year_optimization_results.csv")
        full_results.to_csv(results_file, index=False)
        logger.info(f"📁 結果保存: {results_file}")
        
        logger.info(f"\n✅ 直近1年間詳細最適化完了 - {len(full_results)}パターンを検証")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()