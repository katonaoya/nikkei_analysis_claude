#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
運用パラメータ最適化システム
設定ファイルベースでパラメータ最適化を実行し、結果を自動で設定ファイルに反映
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProductionParameterOptimizer:
    """運用向けパラメータ最適化クラス"""
    
    def __init__(self, config_path="production_config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_data_paths()
        
    def load_config(self):
        """設定ファイル読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 設定値を変数に展開
        self.initial_capital = self.config['system']['initial_capital']
        self.max_positions = self.config['system']['max_positions']
        self.confidence_threshold = self.config['system']['confidence_threshold']
        self.transaction_cost_rate = self.config['system']['transaction_cost_rate']
        self.optimal_features = self.config['features']['optimal_features']
        
        logger.info(f"✅ 設定ファイル読み込み完了: {self.config_path}")
        
    def setup_data_paths(self):
        """データパス設定"""
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / self.config['data']['processed_dir'].replace('data/', '')
        self.integrated_file = self.processed_dir / self.config['data']['integrated_file']
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 データ準備中...")
        
        df = pd.read_parquet(self.integrated_file)
        
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
    
    def run_optimization(self, validation_period="recent_1year"):
        """パラメータ最適化実行"""
        logger.info(f"🚀 パラメータ最適化開始 - 検証期間: {validation_period}")
        
        # データ準備
        self.load_and_prepare_data()
        
        # パラメータ範囲設定
        config_opt = self.config['optimization']
        hold_days_range = range(config_opt['hold_days_range'][0], config_opt['hold_days_range'][1] + 1)
        
        profit_min, profit_max, profit_step = config_opt['profit_target_range']
        profit_targets = [round(i * profit_step + profit_min, 3) for i in range(int((profit_max - profit_min) / profit_step) + 1)]
        
        loss_min, loss_max, loss_step = config_opt['stop_loss_range']
        stop_losses = [round(i * loss_step + loss_min, 3) for i in range(int((loss_max - loss_min) / loss_step) + 1)]
        
        # パラメータ組み合わせ生成（利確 > 損切りの条件）
        parameter_combinations = [
            (hold, profit, stop) 
            for hold in hold_days_range
            for profit in profit_targets
            for stop in stop_losses
            if profit > stop
        ]
        
        # 検証期間設定
        test_days = self.config['optimization']['validation_periods'][validation_period]
        
        logger.info(f"📊 検証パターン数: {len(parameter_combinations)}")
        logger.info(f"📅 検証期間: {test_days}営業日")
        
        # 並行処理実行
        n_cores = min(cpu_count(), config_opt['max_cpu_cores'])
        logger.info(f"⚡ 使用CPUコア数: {n_cores}")
        
        with Pool(processes=n_cores) as pool:
            results = []
            for i, result in enumerate(pool.imap(
                self.single_parameter_simulation_wrapper, 
                [(params, test_days) for params in parameter_combinations]
            )):
                if result is not None:
                    results.append(result)
                
                # 進捗報告
                if (i + 1) % config_opt['progress_report_interval'] == 0 or i == len(parameter_combinations) - 1:
                    progress = (i + 1) / len(parameter_combinations) * 100
                    current_best = max(results, key=lambda x: x['annual_return'])['annual_return'] if results else 0
                    logger.info(f"  📊 進捗: {i+1}/{len(parameter_combinations)} ({progress:.1f}%) - 現在最高年率: {current_best:.2%}")
        
        results_df = pd.DataFrame(results)
        
        # 結果表示と設定ファイル更新
        self.display_and_update_results(results_df, validation_period)
        
        return results_df
    
    def single_parameter_simulation_wrapper(self, args):
        """シングルパラメータシミュレーション（マルチプロセス用ラッパー）"""
        params, test_days = args
        return self.single_parameter_simulation(params, test_days)
    
    def single_parameter_simulation(self, params, test_days):
        """単一パラメータでのシミュレーション"""
        max_hold_days, profit_target, stop_loss = params
        
        try:
            df = self.df
            X = self.X
            y = self.y
            
            dates = sorted(df['Date'].unique())
            
            # 検証期間設定
            trading_dates = dates[-test_days:]
            train_end_date = trading_dates[0]
            train_dates = [d for d in dates if d < train_end_date]
            
            # 学習期間の最低保証
            if len(train_dates) < 756:  # 3年分
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
            annual_return = total_return  # 検証期間に応じて調整が必要な場合は修正
            
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
    
    def display_and_update_results(self, results_df, validation_period):
        """結果表示と設定ファイル更新"""
        logger.info("\n" + "="*150)
        logger.info("🏆 パラメータ最適化結果")
        logger.info("="*150)
        
        # 上位10パターン表示
        top_10 = results_df.nlargest(10, 'annual_return')
        
        logger.info(f"\n📈 年率リターン上位10パターン:")
        logger.info("順位 | 保有 | 利確 | 損切 | 年率リターン | 最終評価額 | 勝率  | 予測精度 | 取引数 | 平均利益 | DD")
        logger.info("-" * 110)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            logger.info(
                f"{i:2d}位 | {row['max_hold_days']:2.0f}日 | {row['profit_target']:4.1%} | "
                f"{row['stop_loss']:4.1%} | {row['annual_return']:10.2%} | "
                f"¥{row['final_value']:9,.0f} | {row['win_rate']:4.1%} | "
                f"{row['prediction_accuracy']:6.1%} | {row['total_trades']:4.0f}回 | "
                f"¥{row['avg_profit']:6,.0f} | {row['max_drawdown']:5.1%}"
            )
        
        # 最優秀パラメータ
        best = top_10.iloc[0]
        logger.info(f"\n🥇 最優秀パラメータ:")
        logger.info(f"  保有期間: {best['max_hold_days']:.0f}日")
        logger.info(f"  利確閾値: {best['profit_target']:.0%}")
        logger.info(f"  損切閾値: {best['stop_loss']:.0%}")
        logger.info(f"  年率リターン: {best['annual_return']:.2%}")
        logger.info(f"  最終評価額: ¥{best['final_value']:,.0f}")
        
        # 設定ファイル更新
        self.update_config_with_best_params(best, validation_period)
        
        logger.info("="*150)
        
        return best, results_df
    
    def update_config_with_best_params(self, best_params, validation_period):
        """最優秀パラメータで設定ファイル更新"""
        self.config['optimal_params']['hold_days'] = int(best_params['max_hold_days'])
        self.config['optimal_params']['profit_target'] = float(best_params['profit_target'])
        self.config['optimal_params']['stop_loss'] = float(best_params['stop_loss'])
        self.config['optimal_params']['annual_return'] = float(best_params['annual_return'])
        self.config['optimal_params']['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        # ファイル保存
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ 設定ファイル更新完了: {self.config_path}")
        logger.info(f"🔄 新しいパラメータが運用システムに反映されました")


def main():
    """メイン実行関数"""
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python production_optimizer.py optimize [期間]")
        print("    期間オプション: recent_1year, recent_6months, recent_3months")
        print("    例: python production_optimizer.py optimize recent_1year")
        return
    
    command = sys.argv[1]
    
    if command == "optimize":
        validation_period = sys.argv[2] if len(sys.argv) > 2 else "recent_1year"
        
        optimizer = ProductionParameterOptimizer()
        results_df = optimizer.run_optimization(validation_period)
        
        # 結果をCSVで保存
        output_file = f"production_optimization_results_{validation_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"📁 結果保存: {output_file}")
        
        print(f"\n✅ パラメータ最適化完了!")
        print(f"📊 検証パターン数: {len(results_df)}")
        print(f"📁 詳細結果: {output_file}")
        print(f"⚙️  設定ファイル更新: production_config.yaml")
    else:
        print(f"不明なコマンド: {command}")


if __name__ == "__main__":
    main()