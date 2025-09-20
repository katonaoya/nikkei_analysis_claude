#!/usr/bin/env python3
"""
超短期取引シミュレーション
1日保有、1%利確・1%損切りでの運用結果を検証
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

class UltraShortTermSimulation:
    """超短期取引シミュレーション"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 超短期取引パラメータ
        self.initial_capital = 1000000
        self.confidence_threshold = 0.55
        self.max_positions = 5
        self.transaction_cost_rate = 0.001
        
        # 超短期設定
        self.hold_days = 1  # 最大1日保有
        self.profit_target = 0.01  # 1%利確
        self.stop_loss = 0.01  # 1%損切り
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 超短期取引シミュレーション用データ準備...")
        
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
    
    def ultra_short_simulation(self, df, X, y):
        """超短期取引シミュレーション"""
        logger.info("⚡ 超短期取引シミュレーション開始...")
        logger.info(f"設定: {self.hold_days}日保有, {self.profit_target*100:.0f}%利確, {self.stop_loss*100:.0f}%損切り")
        
        dates = sorted(df['Date'].unique())
        train_end_idx = int(len(dates) * 0.8)
        
        train_dates = dates[:train_end_idx]
        trading_dates = dates[train_end_idx:]
        
        logger.info(f"学習期間: {train_dates[0]} - {train_dates[-1]}")
        logger.info(f"取引期間: {trading_dates[0]} - {trading_dates[-1]}")
        
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
                retrain_end = train_end_idx + i
                retrain_start = max(0, retrain_end - int(len(dates) * 0.6))
                
                retrain_dates = dates[retrain_start:retrain_end]
                retrain_mask = df['Date'].isin(retrain_dates)
                X_retrain = X[retrain_mask]
                y_retrain = y[retrain_mask]
                
                X_retrain_scaled = scaler.fit_transform(X_retrain)
                model.fit(X_retrain_scaled, y_retrain)
                
                last_retrain = i
                logger.info(f"  📚 モデル再学習: {current_date}")
            
            # 売却処理（超短期設定）
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
                'cash': cash,
                'portfolio_value': total_value - cash,
                'total_value': total_value,
                'positions': len(portfolio)
            })
        
        return trades, daily_values, portfolio, cash
    
    def process_sells(self, portfolio, current_data, cash, current_date):
        """売却処理（超短期版）"""
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
            
            # 1日経過で強制売却
            if days_held >= self.hold_days:
                should_sell = True
                sell_reason = "期間満了"
            
            # 1%損切り
            elif profit_rate <= -self.stop_loss:
                should_sell = True
                sell_reason = "損切り"
            
            # 1%利確
            elif profit_rate >= self.profit_target:
                should_sell = True
                sell_reason = "利確"
            
            if should_sell:
                sell_value = position['shares'] * current_price
                transaction_cost = sell_value * self.transaction_cost_rate
                net_proceeds = sell_value - transaction_cost
                profit_loss = net_proceeds - (position['shares'] * position['buy_price'])
                profit_rate_actual = profit_loss / (position['shares'] * position['buy_price'])
                
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
                    'profit_rate': profit_rate_actual,
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
            
            # 成功判定
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
    
    def analyze_results(self, trades, daily_values, final_cash):
        """結果分析"""
        logger.info("📊 超短期取引結果分析...")
        
        trades_df = pd.DataFrame(trades)
        daily_df = pd.DataFrame(daily_values)
        
        # 売却取引のみで分析
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        
        # 基本統計
        final_value = daily_df['total_value'].iloc[-1] if len(daily_df) > 0 else final_cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        trading_days = len(daily_df)
        years = trading_days / 252 if trading_days > 0 else 1
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1
        
        # 勝率計算
        successful_sells = sell_trades[sell_trades['profit_loss'] > 0] if len(sell_trades) > 0 else pd.DataFrame()
        win_rate = len(successful_sells) / len(sell_trades) if len(sell_trades) > 0 else 0
        
        # 予測精度計算
        buy_success = buy_trades[buy_trades['success'] == True] if len(buy_trades) > 0 else pd.DataFrame()
        prediction_accuracy = len(buy_success) / len(buy_trades) if len(buy_trades) > 0 else 0
        
        # ドローダウン計算
        if len(daily_df) > 0:
            daily_df['peak'] = daily_df['total_value'].cummax()
            daily_df['drawdown'] = (daily_df['total_value'] - daily_df['peak']) / daily_df['peak']
            max_drawdown = daily_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # 売却理由分析
        if len(sell_trades) > 0:
            reason_counts = sell_trades['sell_reason'].value_counts()
            logger.info(f"\n📋 売却理由分析:")
            for reason, count in reason_counts.items():
                pct = count / len(sell_trades) * 100
                logger.info(f"  {reason}: {count}回 ({pct:.1f}%)")
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'prediction_accuracy': prediction_accuracy,
            'trading_days': trading_days,
            'years': years,
            'total_costs': sell_trades['cost'].sum() + buy_trades['cost'].sum() if len(sell_trades) > 0 and len(buy_trades) > 0 else 0,
            'avg_profit_per_trade': sell_trades['profit_loss'].mean() if len(sell_trades) > 0 else 0,
            'avg_days_held': sell_trades['days_held'].mean() if len(sell_trades) > 0 else 0
        }
    
    def display_results(self, results):
        """結果表示"""
        logger.info("\n" + "="*100)
        logger.info("⚡ 超短期取引シミュレーション結果")
        logger.info("="*100)
        
        logger.info(f"\n⚙️ 取引設定:")
        logger.info(f"  保有期間        : 最大{self.hold_days}日")
        logger.info(f"  利確閾値        : {self.profit_target*100:.0f}%")
        logger.info(f"  損切閾値        : {self.stop_loss*100:.0f}%")
        logger.info(f"  取引コスト      : {self.transaction_cost_rate*100:.1f}%")
        
        logger.info(f"\n📊 運用パフォーマンス:")
        logger.info(f"  初期資本        : ¥{results['initial_capital']:,.0f}")
        logger.info(f"  最終評価額      : ¥{results['final_value']:,.0f}")
        logger.info(f"  総リターン      : {results['total_return']:+.2%}")
        logger.info(f"  年率リターン    : {results['annual_return']:+.2%}")
        logger.info(f"  最大ドローダウン: {results['max_drawdown']:.2%}")
        logger.info(f"  運用期間        : {results['years']:.2f}年")
        
        logger.info(f"\n📈 取引統計:")
        logger.info(f"  総取引数        : {results['total_trades']}回")
        logger.info(f"  買い取引        : {results['buy_trades']}回")
        logger.info(f"  売り取引        : {results['sell_trades']}回")
        logger.info(f"  勝率（売却のみ）: {results['win_rate']:.1%}")
        logger.info(f"  取引コスト総額  : ¥{results['total_costs']:,.0f}")
        logger.info(f"  コスト比率      : {results['total_costs']/results['initial_capital']:.2%}")
        logger.info(f"  平均保有日数    : {results['avg_days_held']:.1f}日")
        logger.info(f"  1取引平均利益   : ¥{results['avg_profit_per_trade']:,.0f}")
        
        logger.info(f"\n🎯 超短期取引効果:")
        logger.info(f"  予測精度        : {results['prediction_accuracy']:.1%}")
        logger.info(f"  最大同時保有    : {self.max_positions}銘柄")
        logger.info(f"  確信度閾値      : {self.confidence_threshold*100:.0f}%")
        
        # 総合評価
        if results['total_return'] > 0.5:
            evaluation = "🚀 優秀"
        elif results['total_return'] > 0.2:
            evaluation = "📈 良好"
        elif results['total_return'] > 0:
            evaluation = "📈 プラス"
        else:
            evaluation = "📉 マイナス"
        
        logger.info(f"\n⚖️ 総合評価: {evaluation}")
        logger.info("="*100)

def main():
    """メイン実行"""
    logger.info("⚡ 超短期取引（1日保有・1%利確損切り）シミュレーション")
    
    simulator = UltraShortTermSimulation()
    
    try:
        # データ準備
        df, X, y = simulator.load_and_prepare_data()
        
        # 超短期シミュレーション
        trades, daily_values, final_portfolio, final_cash = simulator.ultra_short_simulation(df, X, y)
        
        # 結果分析
        results = simulator.analyze_results(trades, daily_values, final_cash)
        
        # 結果表示
        simulator.display_results(results)
        
        logger.info(f"\n✅ 超短期取引シミュレーション完了")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()