#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Precision System V3 利確/損切り戦略最適化システム
78.5%精度を活用した最適な利確・損切り・保有期間の包括的検証
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from systems.enhanced_precision_system_v3 import EnhancedPrecisionSystemV3

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 並列ワーカーで共有するデータ
_worker_df = None
_worker_config = None
_worker_date_groups = None


def _init_worker_shared(df, config):
    """サブプロセス用にデータと設定を初期化"""
    global _worker_df, _worker_config, _worker_date_groups
    _worker_df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
    _worker_config = config

    grouped = _worker_df.groupby('Date', sort=True)
    indices = grouped.indices
    _worker_date_groups = [(date, indices[date]) for date in sorted(indices.keys())]


def _simulate_trading_strategy(df, profit_target, stop_loss, max_holding_days, config):
    """利確/損切り戦略シミュレーション（共有設定版）"""
    initial_capital = config['initial_capital']
    max_positions = config['max_positions']
    commission_rate = config['commission_rate']
    slippage_rate = config['slippage_rate']

    if _worker_df is not None:
        data_source = _worker_df
        date_groups = _worker_date_groups
    else:
        data_source = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        group = data_source.groupby('Date', sort=True)
        date_groups = [(date, idx) for date, idx in sorted(group.indices.items(), key=lambda x: x[0])]

    cash = initial_capital
    positions = {}
    trade_log = []
    daily_portfolio_values = []

    for idx, (current_date, row_idx) in enumerate(date_groups):
        if idx < 60:
            continue

        current_data = data_source.iloc[row_idx].copy()
        if len(current_data) == 0:
            continue

        positions_to_close = []
        for code, position in positions.items():
            code_data = current_data[current_data['Code'] == code]
            if len(code_data) == 0:
                continue

            current_price = code_data['Close'].iloc[0]
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            holding_days = (current_date - entry_date).days

            profit_rate = (current_price - entry_price) / entry_price

            sell_reason = None
            if holding_days >= max_holding_days:
                sell_reason = "期間満了"
            elif profit_rate >= profit_target:
                sell_reason = "利確"
            elif profit_rate <= -stop_loss:
                sell_reason = "損切り"

            if sell_reason:
                shares = position['shares']
                gross_proceeds = shares * current_price
                commission = gross_proceeds * commission_rate
                slippage = gross_proceeds * slippage_rate
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

        for code in positions_to_close:
            del positions[code]

        if len(positions) < max_positions:
            available_slots = max_positions - len(positions)
            available_data = current_data[~current_data['Code'].isin(positions.keys())]
            if len(available_data) > 0:
                top_candidates = available_data.nlargest(available_slots, 'pred_proba')

                available_cash = cash * 0.95
                investment_per_stock = available_cash / len(top_candidates) if len(top_candidates) > 0 else 0

                for _, stock in top_candidates.iterrows():
                    if cash < investment_per_stock:
                        break

                    code = stock['Code']
                    price = stock['Close']
                    pred_prob = stock['pred_proba']

                    if investment_per_stock < 10000:
                        continue

                    shares = int(investment_per_stock / price)
                    if shares == 0:
                        continue

                    gross_cost = shares * price
                    commission = gross_cost * commission_rate
                    slippage = gross_cost * slippage_rate
                    total_cost = gross_cost + commission + slippage

                    if total_cost <= cash:
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

        portfolio_value = cash
        for code, position in positions.items():
            code_data = current_data[current_data['Code'] == code]
            if len(code_data) > 0:
                current_price = code_data['Close'].iloc[0]
                portfolio_value += position['shares'] * current_price
            else:
                portfolio_value += position['shares'] * position['entry_price']

        daily_portfolio_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions_count': len(positions)
        })

    return _calculate_strategy_performance(
        trade_log,
        daily_portfolio_values,
        profit_target,
        stop_loss,
        max_holding_days,
        config
    )


def _calculate_strategy_performance(trade_log, daily_values, profit_target, stop_loss, max_holding_days, config):
    initial_capital = config['initial_capital']

    if len(daily_values) == 0:
        return {
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'max_holding_days': max_holding_days,
            'total_return': 0,
            'total_return_pct': 0,
            'final_value': initial_capital,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_return_per_trade': 0,
            'avg_holding_days': 0,
            'profit_factor': 0
        }

    final_value = daily_values[-1]['portfolio_value']
    total_return = final_value - initial_capital
    total_return_pct = total_return / initial_capital

    portfolio_values = [v['portfolio_value'] for v in daily_values]
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak
    max_drawdown = np.min(drawdown)

    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return)

    if len(daily_returns) > 1:
        excess_return = np.mean(daily_returns) - (0.01 / 252)
        sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    else:
        sharpe_ratio = 0

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


def _process_combo(combo):
    profit_target, stop_loss, max_holding_days = combo
    if profit_target <= stop_loss:
        return None
    try:
        return _simulate_trading_strategy(_worker_df, profit_target, stop_loss, max_holding_days, _worker_config)
    except Exception as e:
        logger.error(
            "シミュレーションエラー (利確:%s, 損切:%s, 保有:%s日): %s",
            f"{profit_target:.1%}",
            f"{stop_loss:.1%}",
            max_holding_days,
            e
        )
        return None

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
        self.holding_periods = [1, 2, 3]                     # 最大保有日数は1〜3日に固定

        # シミュレーション用に事前計算した日次データ
        self.simulation_slices = []

        # モデル・スケーラー等を読み込み
        (
            self.model,
            self.scaler,
            self.selector,
            self.feature_cols,
            self.model_path
        ) = self._load_latest_model()

        # データ保存ディレクトリ
        self.results_dir = Path("profit_loss_optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Enhanced V3 利確/損切り最適化システム初期化完了")
        logger.info(f"パラメータ範囲: 利確{len(self.profit_targets)}種, 損切{len(self.stop_losses)}種, 保有{len(self.holding_periods)}種")
        logger.info(f"予想検証数: {len(self.profit_targets) * len(self.stop_losses) * len(self.holding_periods):,}パターン")
        logger.info(f"使用モデル: {self.model_path}")

    def _load_latest_model(self):
        """最新の学習済みモデルを読み込む"""
        model_paths = sorted(
            glob.glob('models/enhanced_v3/enhanced_model_v3_*.joblib'),
            key=os.path.getmtime
        )
        if not model_paths:
            raise FileNotFoundError('models/enhanced_v3/ 以下に学習済みモデルが見つかりません。')

        latest_path = model_paths[-1]
        model_data = joblib.load(latest_path)
        required_keys = {'model', 'scaler', 'selector', 'feature_cols'}
        if not required_keys.issubset(model_data.keys()):
            raise RuntimeError(f"モデルファイルに必要な情報が含まれていません: {latest_path}")

        return (
            model_data['model'],
            model_data.get('scaler'),
            model_data.get('selector'),
            model_data['feature_cols'],
            latest_path
        )

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
                except Exception:
                    continue

        if latest_file is None:
            raise FileNotFoundError("最新の株価データが見つかりません。データ取得処理を確認してください。")

        return latest_file
    
    def load_historical_data(self):
        """Enhanced V3対応の履歴データ読み込み"""
        logger.info("📊 Enhanced V3用履歴データ読み込み開始...")

        system = EnhancedPrecisionSystemV3()

        try:
            raw_df = system.load_and_integrate_data()
            feature_df = system.create_enhanced_features(raw_df)
        except Exception as e:
            logger.error(f"特徴量生成に失敗しました: {e}")
            raise RuntimeError("実データの前処理に失敗したため、最適化を中断します。")

        feature_df = feature_df.sort_values(['Date', 'Code']).reset_index(drop=True)

        missing_cols = [col for col in self.feature_cols if col not in feature_df.columns]
        if missing_cols:
            raise RuntimeError(f"学習時の特徴量が見つかりません: {missing_cols}")

        X = feature_df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)

        if self.selector is not None:
            X_transformed = self.selector.transform(X.values)
        else:
            X_transformed = X.values

        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)

        pred_proba = self.model.predict_proba(X_transformed)[:, 1]

        prediction_df = feature_df[['Date', 'Code', 'Close']].copy()
        prediction_df['pred_proba'] = pred_proba.astype('float32')

        # 日次上位5銘柄に絞り込み
        prediction_df = (
            prediction_df.groupby('Date', group_keys=False)
            .apply(lambda g: g.nlargest(5, 'pred_proba'))
            .reset_index(drop=True)
        )

        prediction_df['Close'] = prediction_df['Close'].astype('float32')

        self.simulation_slices = self._prepare_simulation_slices(prediction_df)

        logger.info(
            "データ読み込み完了: %s件, %s営業日, %s銘柄",
            f"{len(prediction_df):,}",
            prediction_df['Date'].nunique(),
            prediction_df['Code'].nunique()
        )
        logger.info(f"期間: {prediction_df['Date'].min()} 〜 {prediction_df['Date'].max()}")

        return prediction_df

    def _prepare_simulation_slices(self, df: pd.DataFrame):
        """日次ごとの銘柄配列を事前計算してメモリ効率を高める"""
        logger.info("🧮 シミュレーション用データを構築中...")
        grouped = df.groupby('Date', sort=True)
        slices = []
        for date, group in grouped:
            codes = group['Code'].to_numpy(dtype=np.int32, copy=True)
            close = group['Close'].to_numpy(dtype=np.float32, copy=True)
            proba = group['pred_proba'].to_numpy(dtype=np.float32, copy=True)
            slices.append({
                'date': date,
                'codes': codes,
                'close': close,
                'proba': proba
            })
        logger.info(f"🧮 シミュレーション用データ件数: {len(slices)}日")
        return slices

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
    
    def simulate_trading_strategy(self, profit_target, stop_loss, max_holding_days):
        """個別利確/損切り戦略のシミュレーション"""

        if not self.simulation_slices:
            raise RuntimeError("シミュレーション用データが準備されていません。")

        cash = self.initial_capital
        positions = {}
        trade_log = []
        daily_portfolio_values = []

        for idx in range(len(self.simulation_slices)):
            if idx < 60:
                continue  # 初期化期間

            slice_data = self.simulation_slices[idx]
            current_date = slice_data['date']
            codes = slice_data['codes']
            closes = slice_data['close']
            probas = slice_data['proba']

            if codes.size == 0:
                continue

            code_to_idx = {int(code): i for i, code in enumerate(codes)}

            # 既存ポジションの売却判定
            positions_to_close = []
            for code, position in positions.items():
                idx_in_day = code_to_idx.get(code)
                if idx_in_day is None:
                    continue

                current_price = float(closes[idx_in_day])
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                holding_days = (current_date - entry_date).days

                profit_rate = (current_price - entry_price) / entry_price

                sell_reason = None
                if holding_days >= max_holding_days:
                    sell_reason = "期間満了"
                elif profit_rate >= profit_target:
                    sell_reason = "利確"
                elif profit_rate <= -stop_loss:
                    sell_reason = "損切り"

                if sell_reason:
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

            for code in positions_to_close:
                del positions[code]

            # 新規購入判定
            if len(positions) < self.max_positions:
                available_slots = self.max_positions - len(positions)
                if available_slots > 0:
                    held_codes = np.array(list(positions.keys()), dtype=np.int32) if positions else np.array([], dtype=np.int32)
                    if held_codes.size:
                        available_mask = ~np.isin(codes, held_codes)
                    else:
                        available_mask = np.ones_like(codes, dtype=bool)

                    available_indices = np.flatnonzero(available_mask)
                    if available_indices.size > 0:
                        sorted_idx = available_indices[np.argsort(probas[available_indices])[::-1]]
                        top_indices = sorted_idx[:available_slots]

                        available_cash = cash * 0.95
                        if top_indices.size > 0:
                            investment_per_stock = available_cash / top_indices.size
                            for idx_candidate in top_indices:
                                price = float(closes[idx_candidate])
                                if investment_per_stock < 10000 or price <= 0:
                                    continue

                                shares = int(investment_per_stock / price)
                                if shares == 0:
                                    continue

                                gross_cost = shares * price
                                commission = gross_cost * self.commission_rate
                                slippage = gross_cost * self.slippage_rate
                                total_cost = gross_cost + commission + slippage

                                if total_cost <= cash:
                                    code = int(codes[idx_candidate])
                                    pred_prob = float(probas[idx_candidate])

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

            # 日次ポートフォリオ価値
            portfolio_value = cash
            for code, position in positions.items():
                idx_in_day = code_to_idx.get(code)
                if idx_in_day is not None:
                    current_price = float(closes[idx_in_day])
                else:
                    current_price = position['entry_price']
                portfolio_value += position['shares'] * current_price

            daily_portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_count': len(positions)
            })

        return self.calculate_strategy_performance(
            trade_log,
            daily_portfolio_values,
            profit_target,
            stop_loss,
            max_holding_days
        )

    def calculate_strategy_performance(self, trade_log, daily_values, profit_target, stop_loss, max_holding_days):
        """シミュレーション結果の集計"""

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

        final_value = daily_values[-1]['portfolio_value']
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        portfolio_values = np.array([v['portfolio_value'] for v in daily_values], dtype=np.float64)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([])
        if daily_returns.size > 1 and np.std(daily_returns) > 0:
            excess_return = np.mean(daily_returns) - (0.01 / 252)
            sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        sell_trades = [t for t in trade_log if t['action'] == 'SELL']
        if sell_trades:
            wins = [t for t in sell_trades if t['profit_loss'] > 0]
            losses = [t for t in sell_trades if t['profit_loss'] <= 0]

            win_rate = len(wins) / len(sell_trades)
            avg_return_per_trade = float(np.mean([t['profit_loss'] for t in sell_trades]))
            avg_holding_days = float(np.mean([t['holding_days'] for t in sell_trades]))

            total_wins = sum(t['profit_loss'] for t in wins) if wins else 0.0
            total_losses = sum(abs(t['profit_loss']) for t in losses) if losses else 0.01
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        else:
            win_rate = 0.0
            avg_return_per_trade = 0.0
            avg_holding_days = 0.0
            profit_factor = 0.0

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
    
    def run_comprehensive_optimization(self):
        """包括的最適化実行"""
        logger.info("🚀 Enhanced V3 包括的利確/損切り最適化開始...")

        combos = [
            (profit_target, stop_loss, holding_days)
            for holding_days in self.holding_periods
            for profit_target in self.profit_targets
            for stop_loss in self.stop_losses
            if profit_target > stop_loss
        ]

        total_patterns = len(combos)
        logger.info(f"検証パラメータ組み合わせ: {total_patterns:,}パターン")

        if total_patterns == 0:
            logger.warning("検証対象となるパターンが存在しません")
            return pd.DataFrame()

        results = []
        start_time = datetime.now()
        last_log_time = start_time
        progress_interval = timedelta(seconds=30)

        for idx, (profit_target, stop_loss, holding_days) in enumerate(combos, start=1):
            try:
                result = self.simulate_trading_strategy(profit_target, stop_loss, holding_days)
                results.append(result)
            except Exception as e:
                logger.error(
                    "シミュレーションエラー (利確:%s, 損切:%s, 保有:%s日): %s",
                    f"{profit_target:.1%}",
                    f"{stop_loss:.1%}",
                    holding_days,
                    e
                )

            now = datetime.now()
            if idx == total_patterns or now - last_log_time >= progress_interval:
                progress_pct = idx / total_patterns * 100
                elapsed = now - start_time
                best_return = max((r['total_return_pct'] for r in results), default=None)
                logger.info(
                    "進捗: %d/%d (%.1f%%) | 経過時間: %s | 有効結果: %d件 | 最高リターン: %s",
                    idx,
                    total_patterns,
                    progress_pct,
                    str(elapsed).split('.')[0],
                    len(results),
                    f"{best_return * 100:.2f}%" if best_return is not None else "N/A"
                )
                last_log_time = now

        elapsed = datetime.now() - start_time
        logger.info(
            "🎉 最適化完了: %d件の有効結果 | 総処理時間: %s",
            len(results),
            str(elapsed).split('.')[0]
        )

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
        
        heat = sns.heatmap(
            pivot_return,
            annot=False,
            cmap='RdYlGn',
            center=0,
            ax=axes[0, 0],
            cbar_kws={'format': '%.1f'}
        )
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
            results_df = self.run_comprehensive_optimization()
            
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
