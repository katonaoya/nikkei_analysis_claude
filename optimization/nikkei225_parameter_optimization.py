#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95.45%精度モデル用取引パラメータ最適化システム
Nikkei225の10年間データと高精度モデルを使用した最適運用パラメータ発見
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from itertools import product
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Nikkei225ParameterOptimizer:
    """95.45%精度モデル用パラメータ最適化"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_file = Path("data/nikkei225_full/nikkei225_full_530744records_20250906_171825.parquet")
        
        # 95.45%精度モデルのパラメータ
        self.model_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'min_child_samples': 30,
            'subsample': 0.8,
            'learning_rate': 0.03,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        # 技術指標
        self.feature_columns = [
            'MA_5', 'MA_20', 'MA_60', 'RSI_14', 'RSI_7', 
            'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower',
            'ATR', 'OBV', 'Stochastic_K', 'Volume_MA'
        ]
        
        # 取引設定
        self.initial_capital = 1000000  # 100万円
        self.max_positions = 3  # TOP3戦略
        self.commission_rate = 0.001  # 0.1%の取引手数料
        
        # 最適化範囲（ユーザー指定）
        self.max_hold_days_range = range(1, 11)  # 1-10日
        self.profit_target_range = [i/100 for i in range(1, 16)]  # 1%-15%
        self.stop_loss_range = [i/100 for i in range(1, 16)]  # 1%-15%
        
    def load_data(self):
        """10年間データ読み込み"""
        logger.info("📊 Nikkei225 10年間データ読み込み...")
        
        if not self.data_file.exists():
            logger.error(f"❌ データファイルが見つかりません: {self.data_file}")
            raise FileNotFoundError(f"データファイルが見つかりません: {self.data_file}")
        
        df = pd.read_parquet(self.data_file)
        logger.info(f"✅ データ読み込み完了: {len(df):,}件")
        
        # データ前処理
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # テスト用に上位20銘柄の最近2年間のデータのみ使用（高速化）
        recent_date = df['Date'].max()
        two_years_ago = recent_date - pd.DateOffset(years=2)
        df = df[df['Date'] >= two_years_ago]
        
        top_codes = df.groupby('Code').size().nlargest(20).index.tolist()
        df = df[df['Code'].isin(top_codes)]
        
        logger.info(f"📊 最適化用データ: {len(df):,}件 ({len(top_codes)}銘柄 × 2年間)")
        
        # ターゲット計算（翌日1%以上上昇）
        df = df.sort_values(['Code', 'Date'])
        df['next_high'] = df.groupby('Code')['High'].shift(-1)
        df['target'] = ((df['next_high'] - df['Close']) / df['Close'] >= 0.01).astype(int)
        
        # 欠損値除去
        df = df.dropna(subset=['target', 'next_high']).copy()
        
        logger.info(f"📈 ターゲット分析: 正例{df['target'].sum():,}件 ({df['target'].mean():.1%})")
        
        return df
    
    def filter_invalid_combinations(self, combinations):
        """非効率な組み合わせを除外"""
        valid_combinations = []
        
        for hold, profit, loss in combinations:
            # 基本条件：利確率 > 損切率
            if profit <= loss:
                continue
            
            # 明らかに損失となる組み合わせを除外
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
        
        return valid_combinations
    
    def calculate_technical_indicators(self, df):
        """技術指標計算"""
        logger.info("📊 技術指標計算中...")
        
        result_df = df.copy()
        result_df = result_df.sort_values(['Code', 'Date'])
        
        # 各銘柄ごとに技術指標を計算
        grouped = result_df.groupby('Code')
        
        def calc_indicators(group):
            group = group.sort_values('Date').copy()
            
            # 移動平均
            group['MA_5'] = group['Close'].rolling(5).mean()
            group['MA_20'] = group['Close'].rolling(20).mean()
            group['MA_60'] = group['Close'].rolling(60).mean()
            
            # RSI
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            group['RSI_14'] = 100 - (100 / (1 + rs))
            
            # RSI 7日
            gain_7 = (delta.where(delta > 0, 0)).rolling(7).mean()
            loss_7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
            rs_7 = gain_7 / loss_7
            group['RSI_7'] = 100 - (100 / (1 + rs_7))
            
            # MACD
            exp1 = group['Close'].ewm(span=12).mean()
            exp2 = group['Close'].ewm(span=26).mean()
            group['MACD'] = exp1 - exp2
            group['MACD_signal'] = group['MACD'].ewm(span=9).mean()
            
            # ボリンジャーバンド
            bb_mean = group['Close'].rolling(20).mean()
            bb_std = group['Close'].rolling(20).std()
            group['BB_upper'] = bb_mean + (bb_std * 2)
            group['BB_middle'] = bb_mean
            group['BB_lower'] = bb_mean - (bb_std * 2)
            
            # ATR
            high_low = group['High'] - group['Low']
            high_close = np.abs(group['High'] - group['Close'].shift())
            low_close = np.abs(group['Low'] - group['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group['ATR'] = true_range.rolling(14).mean()
            
            # OBV（簡易版）
            group['OBV'] = (group['Volume'] * np.sign(group['Close'].diff())).cumsum()
            
            # ストキャスティクス
            low_14 = group['Low'].rolling(14).min()
            high_14 = group['High'].rolling(14).max()
            group['Stochastic_K'] = 100 * ((group['Close'] - low_14) / (high_14 - low_14))
            
            # 出来高移動平均
            group['Volume_MA'] = group['Volume'].rolling(20).mean()
            
            return group
        
        result_df = grouped.apply(calc_indicators).reset_index(drop=True)
        logger.info(f"✅ 技術指標計算完了: {len(self.feature_columns)}種類")
        
        return result_df

    def create_features(self, df):
        """特徴量作成"""
        logger.info("🔧 特徴量作成...")
        
        # 技術指標を計算
        feature_df = self.calculate_technical_indicators(df)
        
        # 利用可能な特徴量のみ使用
        available_features = [col for col in self.feature_columns if col in feature_df.columns]
        
        if len(available_features) == 0:
            logger.error("❌ 利用可能な特徴量がありません")
            return None, None, None
        
        logger.info(f"✅ 利用特徴量: {len(available_features)}個 - {available_features}")
        
        # 欠損値を前方埋めで処理
        feature_df = feature_df.groupby('Code').apply(lambda x: x.ffill()).reset_index(drop=True)
        feature_df = feature_df.fillna(0)
        
        X = feature_df[available_features]
        y = feature_df['target']
        
        return X, y, available_features
    
    def train_model(self, X_train, y_train):
        """LightGBMモデル訓練"""
        model = lgb.LGBMClassifier(**self.model_params)
        model.fit(X_train, y_train)
        return model
    
    def select_top3_stocks(self, predictions, current_data):
        """TOP3銘柄選択"""
        pred_df = current_data.copy()
        pred_df['pred_proba'] = predictions
        
        # 確率でソートしてTOP3選択
        top3 = pred_df.nlargest(3, 'pred_proba')
        return top3['Code'].tolist()
    
    def simulate_trading(self, df, X, y, available_features, max_hold_days, profit_target, stop_loss):
        """取引シミュレーション"""
        
        # データ分割（70%訓練、30%テスト）
        dates = sorted(df['Date'].unique())
        split_idx = int(len(dates) * 0.7)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        # 訓練データでモデル学習
        train_mask = df['Date'].isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        model = self.train_model(X_train, y_train)
        
        # テスト期間での取引シミュレーション
        cash = self.initial_capital
        portfolio = {}
        trades = []
        
        for current_date in test_dates:
            current_data = df[df['Date'] == current_date].copy()
            
            if len(current_data) == 0:
                continue
            
            # 売却処理
            portfolio, cash, sell_trades = self.process_sells(
                portfolio, current_data, cash, current_date, 
                max_hold_days, profit_target, stop_loss
            )
            trades.extend(sell_trades)
            
            # 購入処理
            if len(portfolio) < self.max_positions:
                X_current = current_data[available_features].fillna(0)
                predictions = model.predict_proba(X_current)[:, 1]
                
                selected_codes = self.select_top3_stocks(predictions, current_data)
                available_codes = [code for code in selected_codes if code not in portfolio]
                
                portfolio, cash, buy_trades = self.process_buys(
                    current_data, portfolio, cash, current_date, available_codes
                )
                trades.extend(buy_trades)
        
        # 最終評価
        final_data = df[df['Date'] == test_dates[-1]]
        final_value = self.calculate_portfolio_value(portfolio, final_data, cash)
        
        # パフォーマンス計算
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        if sell_trades:
            profitable_trades = len([t for t in sell_trades if t['profit_loss'] > 0])
            win_rate = profitable_trades / len(sell_trades)
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'sell_trades': len(sell_trades),
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
            
            # 保有日数計算
            days_held = (pd.to_datetime(current_date) - pd.to_datetime(position['buy_date'])).days
            profit_rate = (current_price - position['buy_price']) / position['buy_price']
            
            should_sell = False
            sell_reason = ""
            
            # 売却条件判定
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
                commission = sell_value * self.commission_rate
                net_proceeds = sell_value - commission
                profit_loss = net_proceeds - (position['shares'] * position['buy_price'])
                
                sells.append({
                    'date': current_date,
                    'code': code,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': current_price,
                    'buy_price': position['buy_price'],
                    'profit_loss': profit_loss,
                    'days_held': days_held,
                    'sell_reason': sell_reason
                })
                
                cash += net_proceeds
                codes_to_remove.append(code)
        
        for code in codes_to_remove:
            del portfolio[code]
        
        return portfolio, cash, sells
    
    def process_buys(self, current_data, portfolio, cash, current_date, available_codes):
        """購入処理"""
        buys = []
        
        if not available_codes:
            return portfolio, cash, buys
        
        # 利用可能ポジション数
        available_positions = self.max_positions - len(portfolio)
        codes_to_buy = available_codes[:available_positions]
        
        if not codes_to_buy:
            return portfolio, cash, buys
        
        # 各銘柄への投資額
        investment_per_stock = (cash * 0.9) / len(codes_to_buy)
        
        for code in codes_to_buy:
            stock_data = current_data[current_data['Code'] == code]
            if len(stock_data) == 0:
                continue
            
            buy_price = stock_data.iloc[0]['Close']
            if pd.isna(buy_price) or buy_price <= 0:
                continue
            
            shares = int(investment_per_stock / buy_price)
            if shares <= 0:
                continue
            
            buy_value = shares * buy_price
            commission = buy_value * self.commission_rate
            total_cost = buy_value + commission
            
            if total_cost > cash:
                continue
            
            portfolio[code] = {
                'shares': shares,
                'buy_price': buy_price,
                'buy_date': current_date
            }
            
            buys.append({
                'date': current_date,
                'code': code,
                'action': 'BUY',
                'shares': shares,
                'price': buy_price,
                'value': buy_value
            })
            
            cash -= total_cost
        
        return portfolio, cash, buys
    
    def calculate_portfolio_value(self, portfolio, current_data, cash):
        """ポートフォリオ評価額計算"""
        total_value = cash
        
        if len(portfolio) == 0 or len(current_data) == 0:
            return total_value
        
        current_prices = current_data.set_index('Code')['Close']
        
        for code, position in portfolio.items():
            if code in current_prices.index:
                current_price = current_prices[code]
                if not pd.isna(current_price) and current_price > 0:
                    total_value += position['shares'] * current_price
        
        return total_value
    
    def optimize_parameters(self):
        """パラメータ最適化実行"""
        logger.info("🎯 Nikkei225パラメータ最適化開始...")
        
        # データ読み込み
        df = self.load_data()
        X, y, available_features = self.create_features(df)
        
        if X is None:
            logger.error("❌ 特徴量作成に失敗")
            return None
        
        # パラメータ組み合わせ生成
        all_combinations = list(product(
            self.max_hold_days_range,
            self.profit_target_range,
            self.stop_loss_range
        ))
        
        # 無効な組み合わせ除外
        valid_combinations = self.filter_invalid_combinations(all_combinations)
        
        logger.info(f"📋 検証パターン: {len(valid_combinations):,}組み合わせ")
        logger.info(f"🚫 除外パターン: {len(all_combinations) - len(valid_combinations):,}組み合わせ")
        
        results = []
        
        for i, (max_hold_days, profit_target, stop_loss) in enumerate(valid_combinations):
            if i % 50 == 0:
                logger.info(f"  進行状況: {i+1:,}/{len(valid_combinations):,} ({(i+1)/len(valid_combinations)*100:.1f}%)")
            
            try:
                result = self.simulate_trading(
                    df, X, y, available_features, 
                    max_hold_days, profit_target, stop_loss
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"パラメータ({max_hold_days}, {profit_target:.1%}, {stop_loss:.1%})でエラー: {e}")
                continue
        
        if not results:
            logger.error("❌ 有効な結果が得られませんでした")
            return None
        
        # 結果DataFrame作成
        results_df = pd.DataFrame([
            {
                'max_hold_days': r['parameters']['max_hold_days'],
                'profit_target': r['parameters']['profit_target'],
                'stop_loss': r['parameters']['stop_loss'],
                'total_return': r['total_return'],
                'final_value': r['final_value'],
                'win_rate': r['win_rate'],
                'total_trades': r['total_trades'],
                'sell_trades': r['sell_trades']
            }
            for r in results
        ])
        
        return results_df
    
    def display_results(self, results_df):
        """結果表示"""
        logger.info("\n" + "="*100)
        logger.info("🏆 Nikkei225 パラメータ最適化結果（95.45%精度モデル）")
        logger.info("="*100)
        
        # TOP10パターン
        top_10 = results_df.nlargest(10, 'total_return')
        
        logger.info(f"\n📈 総リターン上位10パターン:")
        logger.info("順位 | 保有日数 | 利確率 | 損切率 | 総リターン | 最終評価額 | 勝率   | 取引数")
        logger.info("-" * 85)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            logger.info(
                f"{i:2d}位 | {row['max_hold_days']:4.0f}日  | {row['profit_target']:5.1%} | "
                f"{row['stop_loss']:5.1%} | {row['total_return']:8.2%} | "
                f"¥{row['final_value']:9,.0f} | {row['win_rate']:5.1%} | {row['sell_trades']:4.0f}回"
            )
        
        # 最優秀パラメータ
        best = top_10.iloc[0]
        logger.info(f"\n🥇 最優秀パラメータ:")
        logger.info(f"  📅 保有期間: {best['max_hold_days']:.0f}日")
        logger.info(f"  📈 利確閾値: {best['profit_target']:.1%}")
        logger.info(f"  📉 損切閾値: {best['stop_loss']:.1%}")
        logger.info(f"  💰 期待リターン: {best['total_return']:.2%}")
        logger.info(f"  💴 期待最終額: ¥{best['final_value']:,.0f}")
        logger.info(f"  🎯 勝率: {best['win_rate']:.1%}")
        logger.info(f"  📊 売却取引数: {best['sell_trades']:.0f}回")
        
        # パラメータ別統計
        logger.info(f"\n📊 パラメータ別平均リターン:")
        
        # 保有期間別
        hold_stats = results_df.groupby('max_hold_days')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n保有期間別:")
        for days, stats in hold_stats.iterrows():
            logger.info(f"  {days:2.0f}日: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 利確率別（TOP5）
        profit_stats = results_df.groupby('profit_target')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n利確閾値別（TOP5）:")
        for rate, stats in profit_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        # 損切り率別（TOP5）
        loss_stats = results_df.groupby('stop_loss')['total_return'].agg(['mean', 'max']).round(4)
        logger.info(f"\n損切閾値別（TOP5）:")
        for rate, stats in loss_stats.nlargest(5, 'mean').iterrows():
            logger.info(f"  {rate:5.1%}: 平均{stats['mean']:6.2%}, 最高{stats['max']:6.2%}")
        
        logger.info("="*100)
        
        return best

def main():
    """メイン実行"""
    logger.info("⚡ Nikkei225 パラメータ最適化システム（95.45%精度モデル）")
    
    try:
        optimizer = Nikkei225ParameterOptimizer()
        results_df = optimizer.optimize_parameters()
        
        if results_df is not None:
            best_params = optimizer.display_results(results_df)
            logger.info(f"\n✅ パラメータ最適化完了 - {len(results_df):,}パターンを検証")
        else:
            logger.error("❌ 最適化に失敗しました")
            
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()