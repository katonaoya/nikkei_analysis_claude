#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Precision System V3 利確/損切り戦略最適化システム
78.5%精度を活用した最適な利確・損切り・保有期間の包括的検証
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedV3ProfitLossOptimizer:
    """Enhanced V3 利確/損切り戦略最適化システム"""
    
    def __init__(self):
        """初期化"""
        self.system_accuracy = 0.785  # Enhanced V3精度
        self.initial_capital = 1_000_000
        self.max_positions = 3  # Enhanced V3の推奨銘柄数
        self.commission_rate = 0.001  # 0.1%手数料
        self.slippage_rate = 0.0005  # 0.05%スリッページ
        
        # 検証パラメータ範囲（より細かく設定）
        self.profit_targets = np.arange(0.01, 0.20, 0.005)  # 1%-20% (0.5%刻み)
        self.stop_losses = np.arange(0.005, 0.15, 0.005)    # 0.5%-15% (0.5%刻み)
        self.holding_periods = range(1, 21)                  # 1-20日
        
        # データ保存ディレクトリ
        self.results_dir = Path("profit_loss_optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced V3 利確/損切り最適化システム初期化完了")
        logger.info(f"パラメータ範囲: 利確{len(self.profit_targets)}種, 損切{len(self.stop_losses)}種, 保有{len(self.holding_periods)}種")
        logger.info(f"予想検証数: {len(self.profit_targets) * len(self.stop_losses) * len(self.holding_periods):,}パターン")
    
    def _find_latest_stock_file(self) -> str:
        """最新の株価データファイルを取得"""
        import glob
        
        patterns = [
            "data/processed/nikkei225_complete_*.parquet",
            "data/real_jquants_data/nikkei225_real_data_*.pkl",
            "data/processed/nikkei225_*.parquet"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    file_time = os.path.getmtime(file)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file
                except:
                    continue
        
        if latest_file is None:
            latest_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
            logger.warning(f"最新株価ファイルが見つからないため、固定ファイルを使用: {latest_file}")
        
        return latest_file
    
    def _find_latest_external_file(self) -> str:
        """最新の外部データファイルを取得"""
        import glob
        
        patterns = [
            "data/external_extended/external_integrated_*.parquet",
            "data/processed/enhanced_integrated_data.parquet",
            "data/processed/external_*.parquet"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    file_time = os.path.getmtime(file)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file
                except:
                    continue
        
        if latest_file is None:
            latest_file = "data/external_extended/external_integrated_10years_20250909_231815.parquet"
            logger.warning(f"最新外部データファイルが見つからないため、固定ファイルを使用: {latest_file}")
        
        return latest_file
    
    def load_historical_data(self):
        """Enhanced V3対応の履歴データ読み込み"""
        logger.info("📊 Enhanced V3用履歴データ読み込み開始...")
        
        # 実際のデータファイルパス（Enhanced V3システム用）
        try:
            # 動的に最新ファイルを取得
            data_file = self._find_latest_stock_file()
            df = pd.read_parquet(data_file)
            
            # 外部指標データも統合（Enhanced V3の特徴）
            external_file = self._find_latest_external_file()
            external_df = pd.read_parquet(external_file)
            
            # データ統合
            df['Date'] = pd.to_datetime(df['Date'])
            external_df['Date'] = pd.to_datetime(external_df['Date'])
            
            # 外部データとマージ
            integrated_df = pd.merge(df, external_df, on='Date', how='left')
            
            # 前方補完で欠損値処理
            integrated_df = integrated_df.fillna(method='ffill').fillna(method='bfill')
            
        except Exception as e:
            logger.warning(f"統合データ読み込み失敗: {e}")
            # フォールバック: 基本的な株価データのみ使用
            integrated_df = self.generate_realistic_data()
        
        # 基本的な特徴量エンジニアリング
        integrated_df = self.engineer_features(integrated_df)
        
        # Enhanced V3予測確率をシミュレート（実際の運用では保存済みモデルを使用）
        integrated_df = self.simulate_enhanced_v3_predictions(integrated_df)
        
        logger.info(f"データ読み込み完了: {len(integrated_df):,}件, {integrated_df['Code'].nunique()}銘柄")
        logger.info(f"期間: {integrated_df['Date'].min()} 〜 {integrated_df['Date'].max()}")
        
        return integrated_df
    
    def generate_realistic_data(self):
        """現実的なテストデータ生成（データファイルが見つからない場合）"""
        logger.info("📊 テスト用リアリスティックデータ生成...")
        
        # 現在日付から動的に期間設定
        current_date = datetime.now()
        start_date = datetime(2020, 1, 1)
        end_date = current_date
        dates = pd.date_range(start_date, end_date, freq='D')
        business_days = [d for d in dates if d.weekday() < 5]  # 平日のみ
        
        codes = [1000 + i for i in range(225)]  # 日経225っぽいコード
        
        data = []
        np.random.seed(42)  # 再現性確保
        
        for code in codes:
            initial_price = np.random.uniform(500, 10000)
            price = initial_price
            
            for date in business_days:
                # リアルな価格変動（日次ボラティリティ約2%）
                daily_return = np.random.normal(0.0005, 0.02)
                price = max(price * (1 + daily_return), 1)
                
                # ボリューム
                volume = np.random.lognormal(12, 1.5)
                
                # OHLC生成
                daily_vol = abs(daily_return) * 0.5
                high = price * (1 + np.random.uniform(0, daily_vol))
                low = price * (1 - np.random.uniform(0, daily_vol))
                close = price
                
                data.append({
                    'Date': date,
                    'Code': code,
                    'Open': price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': int(volume)
                })
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """Enhanced V3用特徴量エンジニアリング"""
        logger.info("🔧 Enhanced V3用特徴量エンジニアリング...")
        
        # 銘柄別に処理
        enhanced_df = []
        
        for code in df['Code'].unique():
            code_df = df[df['Code'] == code].sort_values('Date').copy()
            
            if len(code_df) < 50:  # 最低限のデータが必要
                continue
            
            # 基本的な技術指標
            code_df['Returns'] = code_df['Close'].pct_change()
            code_df['Volume_MA_20'] = code_df['Volume'].rolling(20).mean()
            
            # 移動平均
            for window in [5, 20, 60]:
                code_df[f'MA_{window}'] = code_df['Close'].rolling(window).mean()
                code_df[f'MA_{window}_ratio'] = code_df['Close'] / code_df[f'MA_{window}']
            
            # ボラティリティ
            for window in [5, 20]:
                code_df[f'Volatility_{window}'] = code_df['Returns'].rolling(window).std()
            
            # RSI
            for window in [14, 21]:
                delta = code_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                code_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = code_df['Close'].ewm(span=12).mean()
            exp2 = code_df['Close'].ewm(span=26).mean()
            code_df['MACD'] = exp1 - exp2
            code_df['MACD_signal'] = code_df['MACD'].ewm(span=9).mean()
            
            # 目的変数作成（Enhanced V3準拠）
            code_df['Next_High'] = code_df['High'].shift(-1)
            code_df['Prev_Close'] = code_df['Close'].shift(1)
            code_df['Target'] = (code_df['Next_High'] / code_df['Prev_Close'] > 1.01).astype(int)
            
            enhanced_df.append(code_df)
        
        result_df = pd.concat(enhanced_df, ignore_index=True)
        result_df = result_df.dropna()
        
        logger.info(f"特徴量エンジニアリング完了: {len(result_df):,}件")
        return result_df
    
    def simulate_enhanced_v3_predictions(self, df):
        """Enhanced V3予測確率シミュレート"""
        logger.info("🎯 Enhanced V3予測確率シミュレーション...")
        
        # Enhanced V3の精度特性を反映
        np.random.seed(42)
        
        # 実際のTargetに基づいてリアルな予測確率を生成
        predictions = []
        for _, row in df.iterrows():
            target = row['Target']
            
            if target == 1:  # 実際に上昇する場合
                # 78.5%精度を反映：正解時は高確率、誤答時は低確率
                if np.random.random() < self.system_accuracy:
                    pred_prob = np.random.beta(7, 2)  # 高確率寄り
                else:
                    pred_prob = np.random.beta(2, 5)  # 低確率寄り
            else:  # 実際に上昇しない場合
                if np.random.random() < self.system_accuracy:
                    pred_prob = np.random.beta(2, 7)  # 低確率寄り（正解）
                else:
                    pred_prob = np.random.beta(5, 2)  # 高確率寄り（誤答）
            
            predictions.append(pred_prob)
        
        df['pred_proba'] = predictions
        
        # 予測精度確認
        high_conf_mask = df['pred_proba'] >= 0.5
        if len(df[high_conf_mask]) > 0:
            actual_accuracy = df[high_conf_mask]['Target'].mean()
            logger.info(f"シミュレート精度: {actual_accuracy:.1%} (目標: {self.system_accuracy:.1%})")
        
        return df
    
    def simulate_trading_strategy(self, df, profit_target, stop_loss, max_holding_days):
        """個別戦略のトレーディングシミュレーション"""
        
        # データを日付でソート
        df_sorted = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        unique_dates = sorted(df_sorted['Date'].unique())
        
        # ポートフォリオ状態
        cash = self.initial_capital
        positions = {}  # {code: {'shares': int, 'entry_price': float, 'entry_date': datetime, 'pred_prob': float}}
        trade_log = []
        daily_portfolio_values = []
        
        for current_date in unique_dates[60:]:  # 最初の60日は特徴量計算用
            current_data = df_sorted[df_sorted['Date'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # 既存ポジションの処理（売却判定）
            positions_to_close = []
            for code, position in positions.items():
                code_data = current_data[current_data['Code'] == code]
                if len(code_data) == 0:
                    continue
                
                current_price = code_data['Close'].iloc[0]
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                holding_days = (current_date - entry_date).days
                
                # 利益率計算
                profit_rate = (current_price - entry_price) / entry_price
                
                # 売却判定
                sell_reason = None
                if holding_days >= max_holding_days:
                    sell_reason = "期間満了"
                elif profit_rate >= profit_target:
                    sell_reason = "利確"
                elif profit_rate <= -stop_loss:
                    sell_reason = "損切り"
                
                if sell_reason:
                    # 売却実行
                    shares = position['shares']
                    gross_proceeds = shares * current_price
                    commission = gross_proceeds * self.commission_rate
                    slippage = gross_proceeds * self.slippage_rate
                    net_proceeds = gross_proceeds - commission - slippage
                    
                    profit_loss = net_proceeds - (shares * entry_price)
                    profit_loss_pct = profit_loss / (shares * entry_price)
                    
                    trade_log.append({
                        'date': current_date,
                        'code': code,
                        'action': 'SELL',
                        'shares': shares,
                        'price': current_price,
                        'entry_price': entry_price,
                        'holding_days': holding_days,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct,
                        'sell_reason': sell_reason,
                        'pred_prob': position['pred_prob']
                    })
                    
                    cash += net_proceeds
                    positions_to_close.append(code)
            
            # 売却したポジション削除
            for code in positions_to_close:
                del positions[code]
            
            # 新規購入判定
            if len(positions) < self.max_positions:
                # Enhanced V3方式：上位確率の銘柄を選択
                available_slots = self.max_positions - len(positions)
                
                # 既に保有していない銘柄で高確率のものを選択
                available_data = current_data[~current_data['Code'].isin(positions.keys())]
                if len(available_data) > 0:
                    # 予測確率上位を選択
                    top_candidates = available_data.nlargest(available_slots, 'pred_proba')
                    
                    # 投資金額計算
                    available_cash = cash * 0.95  # 95%投資
                    investment_per_stock = available_cash / len(top_candidates) if len(top_candidates) > 0 else 0
                    
                    for _, stock in top_candidates.iterrows():
                        if cash < investment_per_stock:
                            break
                        
                        code = stock['Code']
                        price = stock['Close']
                        pred_prob = stock['pred_proba']
                        
                        # 最低投資額チェック
                        if investment_per_stock < 10000:  # 最低1万円
                            continue
                        
                        # 株数計算
                        shares = int(investment_per_stock / price)
                        if shares == 0:
                            continue
                        
                        # 実際のコスト計算
                        gross_cost = shares * price
                        commission = gross_cost * self.commission_rate
                        slippage = gross_cost * self.slippage_rate
                        total_cost = gross_cost + commission + slippage
                        
                        if total_cost <= cash:
                            # 購入実行
                            positions[code] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': current_date,
                                'pred_prob': pred_prob
                            }
                            
                            trade_log.append({
                                'date': current_date,
                                'code': code,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                                'pred_prob': pred_prob
                            })
                            
                            cash -= total_cost
            
            # 日次ポートフォリオ価値計算
            portfolio_value = cash
            for code, position in positions.items():
                code_data = current_data[current_data['Code'] == code]
                if len(code_data) > 0:
                    current_price = code_data['Close'].iloc[0]
                    portfolio_value += position['shares'] * current_price
                else:
                    portfolio_value += position['shares'] * position['entry_price']  # 価格不明時は簿価
            
            daily_portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_count': len(positions)
            })
        
        # 戦略パフォーマンス計算
        return self.calculate_strategy_performance(
            trade_log, daily_portfolio_values, profit_target, stop_loss, max_holding_days
        )
    
    def calculate_strategy_performance(self, trade_log, daily_values, profit_target, stop_loss, max_holding_days):
        """戦略パフォーマンス計算"""
        
        if len(daily_values) == 0:
            return {
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'max_holding_days': max_holding_days,
                'total_return': 0,
                'total_return_pct': 0,
                'final_value': self.initial_capital,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_return_per_trade': 0,
                'avg_holding_days': 0,
                'profit_factor': 0
            }
        
        # 基本的な収益指標
        final_value = daily_values[-1]['portfolio_value']
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # ドローダウン計算
        portfolio_values = [v['portfolio_value'] for v in daily_values]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 日次リターン
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # シャープレシオ
        if len(daily_returns) > 1:
            excess_return = np.mean(daily_returns) - (0.01 / 252)  # リスクフリーレート1%
            sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 取引分析
        sell_trades = [t for t in trade_log if t['action'] == 'SELL']
        if len(sell_trades) > 0:
            wins = [t for t in sell_trades if t['profit_loss'] > 0]
            losses = [t for t in sell_trades if t['profit_loss'] <= 0]
            
            win_rate = len(wins) / len(sell_trades)
            avg_return_per_trade = np.mean([t['profit_loss'] for t in sell_trades])
            avg_holding_days = np.mean([t['holding_days'] for t in sell_trades])
            
            total_wins = sum(t['profit_loss'] for t in wins) if wins else 0
            total_losses = sum(abs(t['profit_loss']) for t in losses) if losses else 0.01
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_return_per_trade = 0
            avg_holding_days = 0
            profit_factor = 0
        
        return {
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'max_holding_days': max_holding_days,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(sell_trades),
            'avg_return_per_trade': avg_return_per_trade,
            'avg_holding_days': avg_holding_days,
            'profit_factor': profit_factor
        }
    
    def run_single_optimization(self, params):
        """単一パラメータ組み合わせの最適化"""
        df, profit_target, stop_loss, max_holding_days = params
        
        # 制約条件チェック
        if profit_target <= stop_loss:
            return None
        
        try:
            result = self.simulate_trading_strategy(df, profit_target, stop_loss, max_holding_days)
            return result
        except Exception as e:
            logger.error(f"シミュレーションエラー (利確:{profit_target:.1%}, 損切:{stop_loss:.1%}, 保有:{max_holding_days}日): {e}")
            return None
    
    def run_comprehensive_optimization(self, df):
        """包括的最適化実行"""
        logger.info("🚀 Enhanced V3 包括的利確/損切り最適化開始...")
        
        # パラメータ組み合わせ生成（利確 > 損切りの制約付き）
        param_combinations = []
        for profit_target in self.profit_targets:
            for stop_loss in self.stop_losses:
                for holding_days in self.holding_periods:
                    if profit_target > stop_loss:  # 制約条件
                        param_combinations.append((df, profit_target, stop_loss, holding_days))
        
        logger.info(f"検証パラメータ組み合わせ: {len(param_combinations):,}パターン")
        logger.info("⏰ 注意: 全パターン検証のため時間がかかります（推定1-3時間）")
        
        # 並列実行でパフォーマンス向上
        cpu_count = min(mp.cpu_count(), 8)  # 最大8プロセス
        logger.info(f"並列実行: {cpu_count}プロセス使用")
        
        results = []
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            # バッチ処理で進捗確認
            batch_size = 100
            for i in range(0, len(param_combinations), batch_size):
                batch = param_combinations[i:i+batch_size]
                batch_results = list(executor.map(self.run_single_optimization, batch))
                
                # None結果を除外
                batch_results = [r for r in batch_results if r is not None]
                results.extend(batch_results)
                
                # 進捗報告
                progress = min((i + batch_size) / len(param_combinations) * 100, 100)
                if len(results) > 0:
                    best_return = max(r['total_return_pct'] for r in results)
                    logger.info(f"進捗: {progress:.1f}% ({len(results)}結果, 最高リターン: {best_return:.2%})")
        
        logger.info(f"🎉 最適化完了: {len(results):,}パターン検証完了")
        
        return pd.DataFrame(results)
    
    def analyze_and_visualize_results(self, results_df):
        """結果分析と可視化"""
        logger.info("📊 結果分析と可視化開始...")
        
        if len(results_df) == 0:
            logger.error("分析する結果がありません")
            return None
        
        # 基本統計
        print("\n" + "="*100)
        print("🏆 Enhanced Precision System V3 利確/損切り戦略最適化結果")
        print("="*100)
        
        # TOP20結果
        top_20 = results_df.nlargest(20, 'total_return_pct')
        
        print(f"\n📈 総リターン上位20戦略:")
        print("順位 | 利確  | 損切  | 保有日 | 総リターン | 最終価値    | 勝率   | シャープ | DD    | 取引数 | 平均保有日")
        print("-" * 110)
        
        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            print(f"{i:2d}位 | {row['profit_target']:4.1%} | {row['stop_loss']:4.1%} | "
                  f"{row['max_holding_days']:2.0f}日   | {row['total_return_pct']:8.2%} | "
                  f"¥{row['final_value']:9,.0f} | {row['win_rate']:5.1%} | "
                  f"{row['sharpe_ratio']:6.2f} | {row['max_drawdown']:5.1%} | "
                  f"{row['total_trades']:4.0f}回 | {row['avg_holding_days']:6.1f}日")
        
        # 最優秀戦略
        best_strategy = top_20.iloc[0]
        print(f"\n🥇 最優秀戦略:")
        print(f"  利確閾値: {best_strategy['profit_target']:.1%}")
        print(f"  損切閾値: {best_strategy['stop_loss']:.1%}")
        print(f"  最大保有日数: {best_strategy['max_holding_days']:.0f}日")
        print(f"  総リターン: {best_strategy['total_return_pct']:.2%}")
        print(f"  最終評価額: ¥{best_strategy['final_value']:,.0f}")
        print(f"  勝率: {best_strategy['win_rate']:.1%}")
        print(f"  シャープレシオ: {best_strategy['sharpe_ratio']:.2f}")
        print(f"  最大ドローダウン: {best_strategy['max_drawdown']:.1%}")
        print(f"  平均取引リターン: ¥{best_strategy['avg_return_per_trade']:,.0f}")
        
        # リスク調整後リターン上位
        results_df['risk_adjusted_return'] = results_df['total_return_pct'] / (abs(results_df['max_drawdown']) + 0.01)
        top_risk_adjusted = results_df.nlargest(10, 'risk_adjusted_return')
        
        print(f"\n💎 リスク調整後リターン上位10戦略:")
        print("順位 | 利確  | 損切  | 保有日 | リスク調整 | 総リターン | 最大DD | 勝率   | 取引数")
        print("-" * 85)
        
        for i, (_, row) in enumerate(top_risk_adjusted.iterrows(), 1):
            print(f"{i:2d}位 | {row['profit_target']:4.1%} | {row['stop_loss']:4.1%} | "
                  f"{row['max_holding_days']:2.0f}日   | {row['risk_adjusted_return']:8.2f} | "
                  f"{row['total_return_pct']:8.2%} | {row['max_drawdown']:6.1%} | "
                  f"{row['win_rate']:5.1%} | {row['total_trades']:4.0f}回")
        
        # 統計分析
        self.print_statistical_analysis(results_df)
        
        # 可視化
        self.create_visualizations(results_df)
        
        return best_strategy, top_20
    
    def print_statistical_analysis(self, results_df):
        """統計分析結果表示"""
        print(f"\n📊 パラメータ別統計分析:")
        
        # 利確閾値別分析
        profit_stats = results_df.groupby('profit_target').agg({
            'total_return_pct': ['mean', 'max', 'std', 'count'],
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        print(f"\n利確閾値別パフォーマンス（上位10位）:")
        print("利確率 | 平均リターン | 最高リターン | 標準偏差 | 平均勝率 | 平均DD | パターン数")
        print("-" * 80)
        
        top_profit_targets = profit_stats.sort_values(('total_return_pct', 'mean'), ascending=False).head(10)
        for profit_target, stats in top_profit_targets.iterrows():
            print(f"{profit_target:5.1%} | {stats[('total_return_pct', 'mean')]:9.2%} | "
                  f"{stats[('total_return_pct', 'max')]:9.2%} | {stats[('total_return_pct', 'std')]:7.2%} | "
                  f"{stats[('win_rate', 'mean')]:6.1%} | {stats[('max_drawdown', 'mean')]:6.1%} | "
                  f"{int(stats[('total_return_pct', 'count')]):4d}個")
        
        # 損切閾値別分析
        loss_stats = results_df.groupby('stop_loss').agg({
            'total_return_pct': ['mean', 'max', 'std'],
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        
        print(f"\n損切閾値別パフォーマンス（上位10位）:")
        print("損切率 | 平均リターン | 最高リターン | 標準偏差 | 平均勝率 | 平均DD")
        print("-" * 70)
        
        top_stop_losses = loss_stats.sort_values(('total_return_pct', 'mean'), ascending=False).head(10)
        for stop_loss, stats in top_stop_losses.iterrows():
            print(f"{stop_loss:5.1%} | {stats[('total_return_pct', 'mean')]:9.2%} | "
                  f"{stats[('total_return_pct', 'max')]:9.2%} | {stats[('total_return_pct', 'std')]:7.2%} | "
                  f"{stats[('win_rate', 'mean')]:6.1%} | {stats[('max_drawdown', 'mean')]:6.1%}")
        
        # 保有期間別分析
        holding_stats = results_df.groupby('max_holding_days').agg({
            'total_return_pct': ['mean', 'max', 'std'],
            'win_rate': 'mean',
            'avg_holding_days': 'mean'
        }).round(4)
        
        print(f"\n保有期間別パフォーマンス（上位10位）:")
        print("最大保有 | 平均リターン | 最高リターン | 標準偏差 | 平均勝率 | 実平均保有")
        print("-" * 72)
        
        top_holdings = holding_stats.sort_values(('total_return_pct', 'mean'), ascending=False).head(10)
        for holding_days, stats in top_holdings.iterrows():
            print(f"{holding_days:6.0f}日 | {stats[('total_return_pct', 'mean')]:9.2%} | "
                  f"{stats[('total_return_pct', 'max')]:9.2%} | {stats[('total_return_pct', 'std')]:7.2%} | "
                  f"{stats[('win_rate', 'mean')]:6.1%} | {stats[('avg_holding_days', 'mean')]:8.1f}日")
    
    def create_visualizations(self, results_df):
        """可視化作成"""
        logger.info("📊 可視化作成...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced V3 利確/損切り戦略最適化結果', fontsize=16, fontweight='bold')
        
        # 1. 利確 vs 損切 ヒートマップ（総リターン）
        pivot_return = results_df.pivot_table(
            values='total_return_pct', 
            index='profit_target', 
            columns='stop_loss', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_return, annot=False, cmap='RdYlGn', center=0, 
                    ax=axes[0, 0], cbar_kws={'format': '%.1%'})
        axes[0, 0].set_title('利確 vs 損切 (平均総リターン)')
        axes[0, 0].set_xlabel('損切閾値')
        axes[0, 0].set_ylabel('利確閾値')
        
        # 2. 保有期間別リターン分布
        results_df.boxplot(column='total_return_pct', by='max_holding_days', ax=axes[0, 1])
        axes[0, 1].set_title('保有期間別リターン分布')
        axes[0, 1].set_xlabel('最大保有日数')
        axes[0, 1].set_ylabel('総リターン')
        
        # 3. リターン vs リスク散布図
        scatter = axes[0, 2].scatter(abs(results_df['max_drawdown']), results_df['total_return_pct'], 
                                   c=results_df['sharpe_ratio'], cmap='viridis', alpha=0.6)
        axes[0, 2].set_xlabel('最大ドローダウン (絶対値)')
        axes[0, 2].set_ylabel('総リターン')
        axes[0, 2].set_title('リターン vs リスク (色:シャープレシオ)')
        plt.colorbar(scatter, ax=axes[0, 2])
        
        # 4. 勝率 vs リターン
        axes[1, 0].scatter(results_df['win_rate'], results_df['total_return_pct'], alpha=0.6)
        axes[1, 0].set_xlabel('勝率')
        axes[1, 0].set_ylabel('総リターン')
        axes[1, 0].set_title('勝率 vs 総リターン')
        
        # 5. 取引数 vs 平均リターン
        axes[1, 1].scatter(results_df['total_trades'], results_df['avg_return_per_trade'], alpha=0.6)
        axes[1, 1].set_xlabel('総取引数')
        axes[1, 1].set_ylabel('平均取引リターン')
        axes[1, 1].set_title('取引数 vs 平均取引リターン')
        
        # 6. 上位戦略のパラメータ分布
        top_50 = results_df.nlargest(50, 'total_return_pct')
        axes[1, 2].hist([top_50['profit_target'], top_50['stop_loss']], 
                       bins=15, alpha=0.7, label=['利確閾値', '損切閾値'])
        axes[1, 2].set_xlabel('閾値')
        axes[1, 2].set_ylabel('頻度')
        axes[1, 2].set_title('上位50戦略のパラメータ分布')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # 保存
        viz_file = self.results_dir / f"optimization_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        logger.info(f"可視化保存: {viz_file}")
        
        plt.show()
    
    def save_results(self, results_df, best_strategy):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 全結果CSV保存
        results_file = self.results_dir / f"enhanced_v3_optimization_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # 最優秀戦略詳細保存
        best_strategy_file = self.results_dir / f"best_strategy_{timestamp}.json"
        import json
        with open(best_strategy_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_strategy': best_strategy.to_dict(),
                'optimization_date': timestamp,
                'system_accuracy': self.system_accuracy,
                'total_patterns_tested': len(results_df)
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"結果保存完了:")
        logger.info(f"  全結果: {results_file}")
        logger.info(f"  最優秀戦略: {best_strategy_file}")
        
        return results_file, best_strategy_file
    
    def run_full_optimization(self):
        """完全最適化実行"""
        logger.info("💎 Enhanced Precision System V3 完全利確/損切り最適化開始!")
        
        start_time = datetime.now()
        
        try:
            # 1. データ読み込み
            df = self.load_historical_data()
            
            # 2. 包括的最適化実行
            results_df = self.run_comprehensive_optimization(df)
            
            # 3. 結果分析・可視化
            best_strategy, top_strategies = self.analyze_and_visualize_results(results_df)
            
            # 4. 結果保存
            results_file, best_strategy_file = self.save_results(results_df, best_strategy)
            
            # 完了報告
            elapsed_time = datetime.now() - start_time
            logger.info(f"\n🎉 Enhanced V3 利確/損切り最適化完了!")
            logger.info(f"実行時間: {elapsed_time}")
            logger.info(f"検証パターン数: {len(results_df):,}")
            logger.info(f"最優秀戦略リターン: {best_strategy['total_return_pct']:.2%}")
            logger.info(f"結果ファイル: {results_file}")
            
            return {
                'best_strategy': best_strategy,
                'top_strategies': top_strategies,
                'results_df': results_df,
                'results_file': results_file,
                'execution_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """メイン実行"""
    optimizer = EnhancedV3ProfitLossOptimizer()
    results = optimizer.run_full_optimization()
    
    if results:
        print(f"\n✅ Enhanced Precision System V3 利確/損切り戦略最適化完了!")
        print(f"🏆 最優秀戦略: 利確{results['best_strategy']['profit_target']:.1%}, "
              f"損切{results['best_strategy']['stop_loss']:.1%}, "
              f"保有{results['best_strategy']['max_holding_days']:.0f}日")
        print(f"💰 期待リターン: {results['best_strategy']['total_return_pct']:.2%}")
    else:
        print(f"\n❌ 最適化に失敗しました")

if __name__ == "__main__":
    main()