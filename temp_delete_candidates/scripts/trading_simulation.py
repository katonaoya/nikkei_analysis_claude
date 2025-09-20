#!/usr/bin/env python3
"""
51.7%精度での実取引シミュレーション
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingSimulation:
    """現実的な取引シミュレーション"""
    
    def __init__(self, initial_capital=1000000):  # 100万円
        self.initial_capital = initial_capital
        self.transaction_cost = 0.003  # 0.3%
        self.win_rate = 0.517  # 51.7%
        self.trade_frequency = 0.464  # 46.4%の日で取引
        
    def simulate_daily_trading(self, days=252):
        """1年間のデイトレーディングシミュレーション"""
        print("📊 51.7%精度での1年間取引シミュレーション")
        print("="*50)
        
        # シミュレーション設定
        capital = self.initial_capital
        daily_returns = []
        trade_days = []
        cumulative_returns = []
        win_count = 0
        loss_count = 0
        total_trades = 0
        
        # 日本市場の現実的な値動き設定
        avg_price_move = 0.025  # 平均2.5%の値動き
        volatility = 0.015      # 1.5%のボラティリティ
        
        for day in range(days):
            # 取引判断（46.4%の確率で取引）
            if np.random.random() < self.trade_frequency:
                total_trades += 1
                
                # 価格変動をシミュレート
                price_change = np.random.normal(avg_price_move, volatility)
                
                # 予測判断（51.7%の確率で正解）
                prediction_correct = np.random.random() < self.win_rate
                
                # ポジションサイズ（資金の10%で取引）
                position_size = capital * 0.1
                
                if prediction_correct:
                    # 予測的中：価格変動分の利益
                    gross_return = position_size * abs(price_change)
                    net_return = gross_return * (1 - self.transaction_cost)
                    capital += net_return
                    daily_returns.append(net_return / position_size)
                    win_count += 1
                    
                else:
                    # 予測外れ：価格変動分の損失
                    gross_loss = position_size * abs(price_change)
                    net_loss = gross_loss * (1 + self.transaction_cost)
                    capital -= net_loss
                    daily_returns.append(-net_loss / position_size)
                    loss_count += 1
                
                trade_days.append(day)
                cumulative_returns.append((capital - self.initial_capital) / self.initial_capital)
            
            else:
                # 取引しない日
                daily_returns.append(0)
                cumulative_returns.append((capital - self.initial_capital) / self.initial_capital)
        
        return {
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'actual_win_rate': win_count / total_trades if total_trades > 0 else 0,
            'trade_days': trade_days
        }
    
    def conservative_simulation(self, days=252):
        """保守的シミュレーション（より現実的）"""
        print("\n💼 保守的シミュレーション（現実的設定）")
        print("="*50)
        
        capital = self.initial_capital
        monthly_results = []
        
        # より保守的な設定
        avg_gain_per_trade = 0.015  # 1.5%/回
        avg_loss_per_trade = 0.018  # 1.8%/回（損切り）
        
        for month in range(12):
            monthly_trades = int(days * self.trade_frequency / 12)
            month_start_capital = capital
            
            for trade in range(monthly_trades):
                position_size = capital * 0.05  # より保守的な5%
                
                if np.random.random() < self.win_rate:
                    # 勝ち
                    profit = position_size * avg_gain_per_trade * (1 - self.transaction_cost)
                    capital += profit
                else:
                    # 負け
                    loss = position_size * avg_loss_per_trade * (1 + self.transaction_cost)
                    capital -= loss
            
            monthly_return = (capital - month_start_capital) / month_start_capital
            monthly_results.append({
                'month': month + 1,
                'capital': capital,
                'monthly_return': monthly_return,
                'cumulative_return': (capital - self.initial_capital) / self.initial_capital
            })
        
        return {
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'monthly_results': monthly_results
        }
    
    def risk_analysis(self, simulation_runs=1000):
        """リスク分析（モンテカルロ）"""
        print("\n⚠️  リスク分析（1000回シミュレーション）")
        print("="*50)
        
        final_returns = []
        
        for run in range(simulation_runs):
            result = self.simulate_daily_trading(days=252)
            final_returns.append(result['total_return'])
        
        returns_array = np.array(final_returns)
        
        return {
            'mean_return': np.mean(returns_array),
            'median_return': np.median(returns_array),
            'std_return': np.std(returns_array),
            'win_probability': np.mean(returns_array > 0),
            'percentiles': {
                'p5': np.percentile(returns_array, 5),
                'p25': np.percentile(returns_array, 25),
                'p75': np.percentile(returns_array, 75),
                'p95': np.percentile(returns_array, 95)
            },
            'max_loss': np.min(returns_array),
            'max_gain': np.max(returns_array)
        }

def main():
    """シミュレーション実行"""
    sim = TradingSimulation(initial_capital=1000000)
    
    # 1. 基本シミュレーション
    basic_result = sim.simulate_daily_trading()
    
    print(f"\n📈 基本シミュレーション結果:")
    print(f"   初期資金:     {sim.initial_capital:,}円")
    print(f"   最終資金:     {basic_result['final_capital']:,.0f}円")
    print(f"   総利益率:     {basic_result['total_return']:.2%}")
    print(f"   年間取引数:   {basic_result['total_trades']}回")
    print(f"   実際勝率:     {basic_result['actual_win_rate']:.1%}")
    print(f"   勝ち:        {basic_result['win_count']}回")
    print(f"   負け:        {basic_result['loss_count']}回")
    
    # 2. 保守的シミュレーション
    conservative_result = sim.conservative_simulation()
    
    print(f"\n📊 保守的シミュレーション結果:")
    print(f"   最終資金:     {conservative_result['final_capital']:,.0f}円")
    print(f"   年間収益率:   {conservative_result['total_return']:.2%}")
    
    print(f"\n📅 月別推移:")
    for month_data in conservative_result['monthly_results'][:6]:
        print(f"   {month_data['month']:2d}月: {month_data['cumulative_return']:+.1%} "
              f"(残高: {month_data['capital']:,.0f}円)")
    
    # 3. リスク分析
    risk_result = sim.risk_analysis(simulation_runs=100)  # 高速化
    
    print(f"\n⚠️  リスク分析結果 (100回平均):")
    print(f"   平均収益率:   {risk_result['mean_return']:.2%}")
    print(f"   中央値:       {risk_result['median_return']:.2%}")
    print(f"   標準偏差:     {risk_result['std_return']:.2%}")
    print(f"   勝率:         {risk_result['win_probability']:.1%}")
    
    print(f"\n📊 収益率分布:")
    print(f"   最悪ケース(5%):  {risk_result['percentiles']['p5']:.1%}")
    print(f"   下位25%:         {risk_result['percentiles']['p25']:.1%}")
    print(f"   上位25%:         {risk_result['percentiles']['p75']:.1%}")
    print(f"   最良ケース(5%):  {risk_result['percentiles']['p95']:.1%}")
    
    # 4. 実用性評価
    print(f"\n💡 実用性評価:")
    if risk_result['mean_return'] > 0.05:
        print(f"   ✅ 年率5%以上期待可能")
        print(f"   ✅ 銀行預金より有利")
    elif risk_result['mean_return'] > 0:
        print(f"   ⚠️  僅かにプラス期待値")
        print(f"   💡 リスク管理が重要")
    else:
        print(f"   ❌ 期待値マイナス")
        print(f"   💡 運用は推奨されません")
    
    print(f"\n🎯 結論:")
    if risk_result['win_probability'] > 0.6:
        print(f"   🚀 運用推奨（60%以上の確率で利益）")
    elif risk_result['win_probability'] > 0.5:
        print(f"   ⚖️  慎重な運用（50%以上の確率で利益）")
    else:
        print(f"   🚫 運用非推奨（損失リスク高い）")
    
    return {
        'basic': basic_result,
        'conservative': conservative_result,
        'risk': risk_result
    }

if __name__ == "__main__":
    results = main()