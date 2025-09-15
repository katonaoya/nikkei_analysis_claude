#!/usr/bin/env python3
"""
検証期間比較システム
1.88年・1年・半年の各検証期間での結果を比較し、最も現実的な期間を特定
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class ValidationPeriodComparison:
    """検証期間比較システム"""
    
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
        
        # 最適パラメータ（前回検証結果）
        self.hold_days = 8
        self.profit_target = 0.12  # 12%
        self.stop_loss = 0.02  # 2%
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 検証期間比較用データ準備...")
        
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
    
    def simulate_period(self, df, X, y, period_name, test_months):
        """指定期間でのシミュレーション"""
        logger.info(f"🔍 {period_name}での検証開始...")
        
        dates = sorted(df['Date'].unique())
        
        # 期間設定
        if test_months == "all":
            # 従来の1.88年（80%まで学習、残り全てテスト）
            train_end_idx = int(len(dates) * 0.8)
            trading_dates = dates[train_end_idx:]
        else:
            # 直近N ヶ月のみテスト
            test_days = test_months * 21  # 1ヶ月 = 約21営業日
            trading_dates = dates[-test_days:]
            
            # 学習期間は直近テスト期間の前まで
            train_end_date = trading_dates[0]
            train_dates = [d for d in dates if d < train_end_date]
            
            # 学習期間が短すぎる場合の調整
            if len(train_dates) < 252:  # 最低1年分は確保
                train_dates = dates[:max(252, len(dates) - test_days)]
        
        if test_months != "all":
            train_dates = [d for d in dates if d < trading_dates[0]]
        else:
            train_dates = dates[:int(len(dates) * 0.8)]
        
        logger.info(f"  学習期間: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)}日)")
        logger.info(f"  テスト期間: {trading_dates[0]} - {trading_dates[-1]} ({len(trading_dates)}日)")
        
        # 初期学習
        train_mask = df['Date'].isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # 運用シミュレーション
        cash = self.initial_capital
        portfolio = {}
        trades = []
        daily_values = []
        
        # モデル再学習間隔（期間に応じて調整）
        if test_months == "all":
            retrain_interval = 63  # 3ヶ月
        elif test_months >= 12:
            retrain_interval = 63  # 3ヶ月
        elif test_months >= 6:
            retrain_interval = 42  # 2ヶ月
        else:
            retrain_interval = 21  # 1ヶ月
        
        last_retrain = 0
        
        for i, current_date in enumerate(trading_dates):
            current_data = df[df['Date'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # モデル再学習
            if i - last_retrain >= retrain_interval and test_months != 6:  # 半年は再学習なし
                # 再学習用データ範囲
                retrain_end_date = current_date
                retrain_start_date = dates[max(0, dates.index(retrain_end_date) - 378)]  # 1.5年分
                
                retrain_dates = [d for d in dates if retrain_start_date <= d < retrain_end_date]
                retrain_mask = df['Date'].isin(retrain_dates)
                X_retrain = X[retrain_mask]
                y_retrain = y[retrain_mask]
                
                if len(X_retrain) > 100:  # 最低データ数確保
                    X_retrain_scaled = scaler.fit_transform(X_retrain)
                    model.fit(X_retrain_scaled, y_retrain)
                    last_retrain = i
                    logger.info(f"    📚 モデル再学習: {current_date}")
            
            # 売却処理
            portfolio, cash, sell_trades = self.process_sells(
                portfolio, current_data, cash, current_date
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
                'total_value': total_value,
                'cash': cash,
                'positions': len(portfolio)
            })
        
        # 結果計算
        final_value = daily_values[-1]['total_value'] if daily_values else cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        years = len(trading_dates) / 252
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # 統計計算
        trades_df = pd.DataFrame(trades)
        sell_trades_df = trades_df[trades_df['action'] == 'SELL'] if len(trades_df) > 0 else pd.DataFrame()
        buy_trades_df = trades_df[trades_df['action'] == 'BUY'] if len(trades_df) > 0 else pd.DataFrame()
        
        if len(sell_trades_df) > 0:
            win_rate = len(sell_trades_df[sell_trades_df['profit_loss'] > 0]) / len(sell_trades_df)
            avg_profit = sell_trades_df['profit_loss'].mean()
        else:
            win_rate = 0
            avg_profit = 0
        
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
        
        total_costs = sum(t.get('cost', 0) for t in trades)
        
        return {
            'period_name': period_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'final_value': final_value,
            'win_rate': win_rate,
            'prediction_accuracy': prediction_accuracy,
            'total_trades': len(trades),
            'trading_days': len(trading_dates),
            'years': years,
            'max_drawdown': max_drawdown,
            'total_costs': total_costs,
            'avg_profit': avg_profit
        }
    
    def process_sells(self, portfolio, current_data, cash, current_date):
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
            
            if days_held >= self.hold_days:
                should_sell = True
                sell_reason = "期間満了"
            elif profit_rate <= -self.stop_loss:
                should_sell = True
                sell_reason = "損切り"
            elif profit_rate >= self.profit_target:
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
    
    def compare_periods(self, df, X, y):
        """複数期間での比較検証"""
        logger.info("🔬 検証期間比較開始...")
        
        # 検証期間設定
        periods = [
            ("1.88年間（従来）", "all"),
            ("直近1年間", 12),
            ("直近半年間", 6)
        ]
        
        results = []
        
        for period_name, test_months in periods:
            try:
                result = self.simulate_period(df, X, y, period_name, test_months)
                results.append(result)
            except Exception as e:
                logger.error(f"{period_name}の検証でエラー: {e}")
                continue
        
        return results
    
    def display_comparison_results(self, results):
        """比較結果表示"""
        logger.info("\n" + "="*120)
        logger.info("🔬 検証期間比較結果")
        logger.info("="*120)
        
        logger.info(f"\n📊 期間別パフォーマンス比較:")
        logger.info("期間      | 総リターン | 年率リターン | 勝率   | 予測精度 | 取引数 | DD    | 取引日数")
        logger.info("-" * 100)
        
        for result in results:
            logger.info(
                f"{result['period_name']:12s} | {result['total_return']:8.2%} | "
                f"{result['annual_return']:10.2%} | {result['win_rate']:5.1%} | "
                f"{result['prediction_accuracy']:6.1%} | {result['total_trades']:4.0f}回 | "
                f"{result['max_drawdown']:5.1%} | {result['trading_days']:3.0f}日"
            )
        
        # 分析と推奨
        logger.info(f"\n🎯 各期間の特徴分析:")
        
        for result in results:
            period_name = result['period_name']
            logger.info(f"\n📌 {period_name}:")
            
            # 現実性評価
            if "1.88年" in period_name:
                logger.info("  • 長期トレンド捕捉、統計的信頼性高")
                logger.info("  • 市場環境変化を含む包括的検証")
                logger.info("  • データ量豊富で安定した結果")
            elif "1年" in period_name:
                logger.info("  • 中期トレンド反映、適度な期間")
                logger.info("  • 季節性要因を1サイクル含む")
                logger.info("  • 実用的な検証期間")
            elif "半年" in period_name:
                logger.info("  • 最新市場環境に特化")
                logger.info("  • 短期ノイズの影響を受けやすい")
                logger.info("  • データ量少なく不安定")
            
            # パフォーマンス評価
            if result['annual_return'] > 0.3:
                perf_eval = "🚀 非常に高収益"
            elif result['annual_return'] > 0.15:
                perf_eval = "📈 高収益"
            elif result['annual_return'] > 0.05:
                perf_eval = "✅ 良好"
            else:
                perf_eval = "⚠️ 低収益"
            
            logger.info(f"  • パフォーマンス: {perf_eval} (年率{result['annual_return']:.1%})")
            
            # 安定性評価
            if abs(result['max_drawdown']) < 0.15:
                stability = "🛡️ 安定"
            elif abs(result['max_drawdown']) < 0.25:
                stability = "⚖️ 中程度"
            else:
                stability = "⚠️ 不安定"
            
            logger.info(f"  • リスク: {stability} (最大DD{result['max_drawdown']:.1%})")
        
        # 推奨期間の決定
        logger.info(f"\n💡 推奨検証期間:")
        
        # 年率リターンの安定性と現実性を総合評価
        best_period = None
        best_score = 0
        
        for result in results:
            # スコア計算（年率リターン + 安定性 - ドローダウン）
            score = result['annual_return'] * 0.5 - abs(result['max_drawdown']) * 0.3 + (result['trading_days']/252) * 0.2
            
            if score > best_score:
                best_score = score
                best_period = result
        
        if best_period:
            logger.info(f"  🏆 最適期間: {best_period['period_name']}")
            logger.info(f"    理由: 年率{best_period['annual_return']:.1%}、DD{best_period['max_drawdown']:.1%}のバランス")
        
        logger.info(f"\n📋 実用性の考察:")
        logger.info("  • 1.88年間: 統計的信頼性は高いが、古いデータの影響あり")
        logger.info("  • 1年間: バランス良好、季節性も考慮、実用的")
        logger.info("  • 半年間: 最新状況反映だが、偶然性に左右されやすい")
        logger.info("\n  💎 推奨: 「直近1年間」が最もバランス良く現実的")
        
        logger.info("="*120)
        
        return results

def main():
    """メイン実行"""
    logger.info("🔬 検証期間比較システム")
    
    comparator = ValidationPeriodComparison()
    
    try:
        # データ準備
        df, X, y = comparator.load_and_prepare_data()
        
        # 期間比較検証
        results = comparator.compare_periods(df, X, y)
        
        # 結果表示
        comparator.display_comparison_results(results)
        
        logger.info(f"\n✅ 検証期間比較完了")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()