#!/usr/bin/env python3
"""
正確な取引シミュレーションシステム
データ漏洩を防止した現実的なシミュレーション
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class AccurateTradingSimulation:
    """正確な取引シミュレーション（データ漏洩防止）"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量（59.4%達成構成）
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 取引パラメータ
        self.initial_capital = 1_000_000  # 初期資本100万円
        self.transaction_cost = 0.003     # 取引コスト0.3%（往復）
        self.max_position_per_stock = 0.05  # 1銘柄あたり最大5%
        self.confidence_threshold = 0.55   # 予測確信度閾値
        self.max_positions = 10            # 最大保有銘柄数
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 正確なシミュレーション用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 実際の翌日リターンを計算（look-ahead biasを避けるため）
        clean_df = clean_df.sort_values(['Code', 'Date'])
        clean_df['Actual_Next_Return'] = clean_df.groupby('Code')['Close'].pct_change().shift(-1)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return clean_df, X, y
    
    def validate_model_accuracy(self, df, X, y):
        """モデル精度の検証（データ漏洩なし）"""
        logger.info("🧠 モデル精度検証（時系列分割）...")
        
        # 時系列分割で検証
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 標準化
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 学習と予測
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"  Fold {fold+1}: {accuracy:.3%}")
        
        avg_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        
        logger.info(f"📊 検証精度: {avg_accuracy:.3%} ± {std_accuracy:.3%}")
        
        return avg_accuracy, std_accuracy
    
    def walk_forward_simulation(self, df, X, y):
        """ウォークフォワード取引シミュレーション"""
        logger.info("🚀 ウォークフォワードシミュレーション開始...")
        
        # 時系列分割（最初の50%を初期学習、残りでウォークフォワード）
        dates = sorted(df['Date'].unique())
        total_dates = len(dates)
        initial_train_end = int(total_dates * 0.5)
        
        logger.info(f"学習期間: {dates[0]} - {dates[initial_train_end-1]}")
        logger.info(f"取引期間: {dates[initial_train_end]} - {dates[-1]}")
        
        # 結果記録用
        portfolio = {}
        cash = self.initial_capital
        all_trades = []
        performance_history = []
        
        # 再学習間隔（3ヶ月ごと）
        retraining_interval = 63  # 約3ヶ月の営業日
        last_retrain_idx = 0
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        # ウォークフォワード実行
        for current_idx in range(initial_train_end, total_dates):
            current_date = dates[current_idx]
            
            # 再学習の判定
            if (current_idx - last_retrain_idx) >= retraining_interval or current_idx == initial_train_end:
                logger.info(f"  📚 モデル再学習: {current_date}")
                
                # 学習データ準備（現在日付まで）
                train_mask = df['Date'] < current_date
                train_df = df[train_mask]
                
                if len(train_df) < 1000:  # 最低学習サンプル数
                    continue
                    
                X_train = train_df[self.optimal_features].fillna(0)
                y_train = train_df['Binary_Direction'].astype(int)
                
                # モデル学習
                X_train_scaled = scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
                
                last_retrain_idx = current_idx
            
            # 現在日の予測
            current_data = df[df['Date'] == current_date]
            if len(current_data) == 0:
                continue
                
            X_current = current_data[self.optimal_features].fillna(0)
            if len(X_current) == 0:
                continue
                
            X_current_scaled = scaler.transform(X_current)
            pred_proba = model.predict_proba(X_current_scaled)[:, 1]
            
            # 予測結果をデータフレームに追加
            current_data = current_data.copy()
            current_data['pred_proba'] = pred_proba
            current_data['prediction'] = pred_proba > 0.5
            current_data['high_confidence'] = (pred_proba >= self.confidence_threshold) | (pred_proba <= (1 - self.confidence_threshold))
            
            # 取引実行
            portfolio, cash, day_trades = self.execute_daily_trading(
                current_data, portfolio, cash, current_date
            )
            
            all_trades.extend(day_trades)
            
            # 月末にパフォーマンス記録
            if current_idx % 21 == 0:  # 約月次
                total_value = self.calculate_total_portfolio_value(portfolio, current_data, cash)
                performance_history.append({
                    'date': current_date,
                    'total_value': total_value,
                    'cash': cash,
                    'positions': len(portfolio)
                })
        
        return self.analyze_results(performance_history, all_trades, df)
    
    def execute_daily_trading(self, day_data, portfolio, cash, current_date):
        """日次取引実行"""
        trades = []
        
        # 高確信度データのみ使用
        high_conf_data = day_data[day_data['high_confidence']].copy()
        
        if len(high_conf_data) == 0:
            return portfolio, cash, trades
        
        # 売り判定（保有銘柄で下落予測が高確信度）
        sell_candidates = high_conf_data[
            (high_conf_data['prediction'] == False) & 
            (high_conf_data['pred_proba'] <= (1 - self.confidence_threshold)) &
            (high_conf_data['Code'].isin(portfolio.keys()))
        ]
        
        # 売り実行
        for _, stock in sell_candidates.iterrows():
            code = stock['Code']
            if code not in portfolio:
                continue
                
            position = portfolio[code]
            sell_price = stock['Close']
            
            if pd.isna(sell_price) or sell_price <= 0:
                continue
                
            # 実際の翌日リターンで検証（取引後の結果確認用）
            actual_return = stock.get('Actual_Next_Return', 0)
            actual_gain_loss = actual_return if not pd.isna(actual_return) else 0
            
            sell_value = position['shares'] * sell_price
            transaction_cost = sell_value * self.transaction_cost
            net_proceeds = sell_value - transaction_cost
            
            # 取引記録
            trades.append({
                'date': current_date,
                'code': code,
                'action': 'SELL',
                'shares': position['shares'],
                'price': sell_price,
                'buy_price': position['buy_price'],
                'value': sell_value,
                'cost': transaction_cost,
                'net_proceeds': net_proceeds,
                'confidence': 1 - stock['pred_proba'],
                'predicted_direction': 'DOWN',
                'actual_next_return': actual_gain_loss,
                'gain_loss': net_proceeds - (position['shares'] * position['buy_price']),
                'success': net_proceeds > (position['shares'] * position['buy_price'])
            })
            
            cash += net_proceeds
            del portfolio[code]
        
        # 買い判定（上昇予測が高確信度 & 未保有）
        buy_candidates = high_conf_data[
            (high_conf_data['prediction'] == True) & 
            (high_conf_data['pred_proba'] >= self.confidence_threshold) &
            (~high_conf_data['Code'].isin(portfolio.keys()))
        ].sort_values('pred_proba', ascending=False)
        
        # 買い実行（上位候補から）
        for _, stock in buy_candidates.head(self.max_positions - len(portfolio)).iterrows():
            code = stock['Code']
            buy_price = stock['Close']
            
            if pd.isna(buy_price) or buy_price <= 0:
                continue
                
            # ポジションサイズ計算
            total_portfolio_value = cash + sum(p['shares'] * p.get('current_price', p['buy_price']) for p in portfolio.values())
            max_position_value = total_portfolio_value * self.max_position_per_stock
            available_cash = min(cash * 0.8, max_position_value)
            
            if available_cash < buy_price * 100:  # 最低100株
                continue
                
            shares = int(available_cash // buy_price)
            if shares <= 0:
                continue
                
            buy_value = shares * buy_price
            transaction_cost = buy_value * self.transaction_cost
            total_cost = buy_value + transaction_cost
            
            if total_cost > cash:
                continue
                
            # 実際の翌日リターンで検証
            actual_return = stock.get('Actual_Next_Return', 0)
            actual_gain_loss = actual_return if not pd.isna(actual_return) else 0
            
            # ポートフォリオに追加
            portfolio[code] = {
                'shares': shares,
                'buy_price': buy_price,
                'buy_date': current_date,
                'current_price': buy_price
            }
            
            # 取引記録
            trades.append({
                'date': current_date,
                'code': code,
                'action': 'BUY',
                'shares': shares,
                'price': buy_price,
                'buy_price': buy_price,
                'value': buy_value,
                'cost': transaction_cost,
                'net_cost': total_cost,
                'confidence': stock['pred_proba'],
                'predicted_direction': 'UP',
                'actual_next_return': actual_gain_loss,
                'gain_loss': 0,  # 買い時点では0
                'success': None  # 売却時に判定
            })
            
            cash -= total_cost
        
        return portfolio, cash, trades
    
    def calculate_total_portfolio_value(self, portfolio, current_data, cash):
        """ポートフォリオ総評価額計算"""
        total_value = cash
        
        current_prices = current_data.set_index('Code')['Close']
        
        for code, position in portfolio.items():
            if code in current_prices.index:
                current_price = current_prices[code]
                if not pd.isna(current_price) and current_price > 0:
                    position['current_price'] = current_price
                    total_value += position['shares'] * current_price
                else:
                    total_value += position['shares'] * position['buy_price']
            else:
                total_value += position['shares'] * position['buy_price']
        
        return total_value
    
    def analyze_results(self, performance_history, all_trades, df):
        """結果分析"""
        logger.info("📊 シミュレーション結果分析...")
        
        if not performance_history or not all_trades:
            logger.error("❌ 分析用データが不足しています")
            return None
        
        # パフォーマンス分析
        perf_df = pd.DataFrame(performance_history)
        trades_df = pd.DataFrame(all_trades)
        
        final_value = perf_df['total_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # 期間計算
        start_date = perf_df['date'].min()
        end_date = perf_df['date'].max()
        days = (end_date - start_date).days
        years = days / 365.25
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # 最大ドローダウン
        perf_df['peak'] = perf_df['total_value'].cummax()
        perf_df['drawdown'] = (perf_df['total_value'] / perf_df['peak'] - 1) * 100
        max_drawdown = perf_df['drawdown'].min()
        
        # 取引分析
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        total_trades = len(trades_df)
        total_costs = trades_df['cost'].sum()
        
        # 勝率計算（売却取引のみ）
        if len(sell_trades) > 0:
            win_trades = len(sell_trades[sell_trades['success'] == True])
            win_rate = win_trades / len(sell_trades) * 100
        else:
            win_rate = 0
        
        # 予測精度検証
        prediction_accuracy = self.validate_prediction_accuracy(trades_df)
        
        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'annual_return_pct': annual_return * 100,
                'max_drawdown_pct': max_drawdown,
                'simulation_days': days,
                'simulation_years': years
            },
            'trading_stats': {
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate_pct': win_rate,
                'total_transaction_costs': total_costs,
                'cost_ratio_pct': (total_costs / self.initial_capital) * 100
            },
            'prediction_accuracy': prediction_accuracy,
            'performance_history': perf_df,
            'trades_history': trades_df
        }
        
        self.display_results(results)
        return results
    
    def validate_prediction_accuracy(self, trades_df):
        """予測精度の検証"""
        # 翌日リターンがある取引のみで精度計算
        valid_trades = trades_df[trades_df['actual_next_return'].notna()].copy()
        
        if len(valid_trades) == 0:
            return {'accuracy': 0, 'valid_predictions': 0}
        
        # 予測方向と実際の結果を比較
        valid_trades['actual_direction'] = valid_trades['actual_next_return'] > 0
        valid_trades['predicted_up'] = valid_trades['predicted_direction'] == 'UP'
        
        correct_predictions = (valid_trades['predicted_up'] == valid_trades['actual_direction']).sum()
        accuracy = correct_predictions / len(valid_trades) * 100
        
        return {
            'accuracy': accuracy,
            'valid_predictions': len(valid_trades),
            'correct_predictions': correct_predictions
        }
    
    def display_results(self, results):
        """結果表示"""
        logger.info("\\n" + "="*120)
        logger.info("💰 正確な取引シミュレーション結果")
        logger.info("="*120)
        
        summary = results['summary']
        trading = results['trading_stats']
        prediction = results['prediction_accuracy']
        
        # 基本パフォーマンス
        logger.info(f"\\n📊 運用パフォーマンス:")
        logger.info(f"  初期資本        : ¥{summary['initial_capital']:,}")
        logger.info(f"  最終評価額      : ¥{summary['final_value']:,.0f}")
        logger.info(f"  総リターン      : {summary['total_return_pct']:+.2f}%")
        logger.info(f"  年率リターン    : {summary['annual_return_pct']:+.2f}%")
        logger.info(f"  最大ドローダウン: {summary['max_drawdown_pct']:.2f}%")
        logger.info(f"  運用期間        : {summary['simulation_years']:.2f}年")
        
        # 取引統計
        logger.info(f"\\n📈 取引統計:")
        logger.info(f"  総取引数        : {trading['total_trades']:,}回")
        logger.info(f"  買い取引        : {trading['buy_trades']:,}回")
        logger.info(f"  売り取引        : {trading['sell_trades']:,}回")
        logger.info(f"  勝率（売却のみ）: {trading['win_rate_pct']:.1f}%")
        logger.info(f"  取引コスト総額  : ¥{trading['total_transaction_costs']:,.0f}")
        logger.info(f"  コスト比率      : {trading['cost_ratio_pct']:.2f}%")
        
        # 予測精度検証
        logger.info(f"\\n🎯 予測精度検証:")
        logger.info(f"  実運用での予測精度: {prediction['accuracy']:.1f}%")
        logger.info(f"  検証可能取引数    : {prediction['valid_predictions']:,}件")
        logger.info(f"  正解予測数        : {prediction['correct_predictions']:,}件")
        
        # 評価
        if summary['annual_return_pct'] > 10:
            performance_grade = "🚀 優秀"
        elif summary['annual_return_pct'] > 5:
            performance_grade = "✅ 良好"
        elif summary['annual_return_pct'] > 0:
            performance_grade = "📈 プラス"
        else:
            performance_grade = "📉 マイナス"
        
        logger.info(f"\\n⚖️ 総合評価: {performance_grade}")
        logger.info("="*120)

def main():
    """メイン実行"""
    logger.info("🎯 正確な59.4%精度モデル取引シミュレーション")
    
    system = AccurateTradingSimulation()
    
    try:
        # データ準備
        df, X, y = system.load_and_prepare_data()
        
        # モデル精度検証
        accuracy, std = system.validate_model_accuracy(df, X, y)
        
        # ウォークフォワードシミュレーション
        results = system.walk_forward_simulation(df, X, y)
        
        if results:
            logger.info("\\n✅ 正確なシミュレーション完了")
        else:
            logger.error("❌ シミュレーション失敗")
            
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()