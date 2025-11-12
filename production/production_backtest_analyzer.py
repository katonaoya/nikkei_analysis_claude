#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Reportsバックテスト分析システム
最適化パラメータ（保有10日・利確7%・損切5%）での実際の収益を算出
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProductionBacktestAnalyzer:
    """Production Reportsのバックテスト分析"""
    
    def __init__(self):
        self.reports_dir = Path("production_reports")
        self.data_dir = Path("data")
        self.results_dir = Path("results/production_backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 最適パラメータ（並列最適化結果より）
        self.optimal_params = {
            'holding_days': 10,
            'profit_target': 0.07,  # 7%
            'stop_loss': 0.05,      # 5%
            'initial_capital': 1000000,  # 100万円
            'max_positions': 5,          # 最大同時保有数
            'position_size': 200000      # 1銘柄あたり20万円
        }
        
        # 株価データ読み込み
        self.price_data = self._load_price_data()
        
    def _load_price_data(self) -> pd.DataFrame:
        """株価データを読み込み"""
        logger.info("📊 株価データ読み込み中...")
        
        parquet_files = list(self.data_dir.rglob("*nikkei225*.parquet"))
        if not parquet_files:
            logger.error("株価データファイルが見つかりません")
            return pd.DataFrame()
        
        # 最新のファイルを使用
        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        # 日付変換
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Code', 'Date'])
        
        logger.info(f"✅ 株価データ読み込み完了: {len(df):,}件")
        return df
    
    def parse_report_file(self, file_path: Path) -> Optional[Dict]:
        """レポートファイルを解析して推奨銘柄を抽出"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 日付を抽出（ファイル名から）
            date_match = re.search(r'(\d{8})', file_path.name)
            if not date_match:
                return None
            
            date_str = date_match.group(1)
            report_date = datetime.strptime(date_str, '%Y%m%d')
            
            # TOP3推奨銘柄を抽出
            recommendations = []
            
            # パターン1: 【極高信頼度】や【高信頼度】のある形式
            pattern1 = r'### \d+\. 【.*?】(.*?) \((\d+)\).*?\n- \*\*現在価格\*\*: ([\d,]+)円.*?\n- \*\*上昇確率\*\*: ([\d.]+)%'
            matches1 = re.findall(pattern1, content, re.DOTALL)
            
            for match in matches1:
                company_name = match[0].strip()
                raw_code = match[1].strip()
                code = raw_code if len(raw_code) == 5 else f"{raw_code}0"
                price = float(match[2].replace(',', ''))
                probability = float(match[3])
                
                recommendations.append({
                    'company_name': company_name,
                    'code': code,
                    'price': price,
                    'probability': probability
                })
            
            # パターン2: その他注目銘柄テーブル
            table_pattern = r'\| (\d+) \| (\d+) \| (.*?) \| ([\d,]+) \| ([\d.]+)% \|'
            table_matches = re.findall(table_pattern, content)
            
            for match in table_matches:
                rank = int(match[0])
                if rank <= 6:  # TOP6まで取得
                    raw_code = match[1].strip()
                    code = raw_code if len(raw_code) == 5 else f"{raw_code}0"
                    company_name = match[2].strip()
                    price = float(match[3].replace(',', ''))
                    probability = float(match[4])

                    # 重複チェック
                    if not any(rec['code'] == code for rec in recommendations):
                        recommendations.append({
                            'company_name': company_name,
                            'code': code,
                            'price': price,
                            'probability': probability
                        })
            
            # 確率順でソート、TOP5を選択
            recommendations.sort(key=lambda x: x['probability'], reverse=True)
            top_recommendations = recommendations[:5]
            
            return {
                'date': report_date,
                'recommendations': top_recommendations
            }
            
        except Exception as e:
            logger.warning(f"レポート解析エラー {file_path.name}: {e}")
            return None
    
    def get_stock_price_data(self, code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """指定銘柄の株価データを取得"""
        if self.price_data.empty:
            return pd.DataFrame()
        
        code_data = self.price_data[
            (self.price_data['Code'] == int(code)) & 
            (self.price_data['Date'] >= start_date) & 
            (self.price_data['Date'] <= end_date)
        ].copy()

        return code_data.sort_values('Date')
    
    def simulate_trade(self, code: str, entry_price: float, entry_date: datetime) -> Dict:
        """個別取引をシミュレーション"""
        # 取引期間設定（土日考慮して2倍の期間を設定）
        end_date = entry_date + timedelta(days=self.optimal_params['holding_days'] * 2)
        
        # 株価データ取得
        price_data = self.get_stock_price_data(code, entry_date, end_date)
        
        if price_data.empty:
            return {
                'result': 'no_data',
                'return_rate': 0,
                'holding_days': 0,
                'exit_price': entry_price,
                'exit_reason': 'データなし'
            }
        
        # 利確・損切価格設定
        profit_target_price = entry_price * (1 + self.optimal_params['profit_target'])
        stop_loss_price = entry_price * (1 - self.optimal_params['stop_loss'])
        
        # 日次価格チェック
        trading_days = 0
        for _, row in price_data.iterrows():
            trading_days += 1
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # 利確チェック（高値で利確）
            if high_price >= profit_target_price:
                return_rate = (profit_target_price - entry_price) / entry_price
                return {
                    'result': 'profit',
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': profit_target_price,
                    'exit_reason': f'利確(+{return_rate:.1%})'
                }
            
            # 損切チェック（安値で損切）
            if low_price <= stop_loss_price:
                return_rate = (stop_loss_price - entry_price) / entry_price
                return {
                    'result': 'loss',
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': stop_loss_price,
                    'exit_reason': f'損切({return_rate:.1%})'
                }
            
            # 最大保有期間到達
            if trading_days >= self.optimal_params['holding_days']:
                return_rate = (close_price - entry_price) / entry_price
                result = 'profit' if return_rate > 0 else 'loss' if return_rate < 0 else 'flat'
                return {
                    'result': result,
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': close_price,
                    'exit_reason': f'期間満了({return_rate:.1%})'
                }
        
        # データ不足の場合
        return {
            'result': 'insufficient_data',
            'return_rate': 0,
            'holding_days': trading_days,
            'exit_price': entry_price,
            'exit_reason': 'データ不足'
        }
    
    def run_backtest_analysis(self) -> Dict:
        """全レポートのバックテスト分析を実行"""
        logger.info("🚀 Production Reportsバックテスト分析開始...")
        
        # レポートファイル一覧取得
        report_files = sorted(list(self.reports_dir.glob("*.md")))
        logger.info(f"📋 分析対象レポート数: {len(report_files)}件")
        
        # 取引結果保存用
        all_trades = []
        daily_performance = []
        
        for report_file in report_files:
            logger.info(f"📊 解析中: {report_file.name}")
            
            # レポート解析
            report_data = self.parse_report_file(report_file)
            if not report_data or not report_data['recommendations']:
                logger.warning(f"レポート解析失敗またはデータなし: {report_file.name}")
                continue
            
            logger.info(f"推奨銘柄数: {len(report_data['recommendations'])}")
            
            # 日次取引結果
            daily_trades = []
            daily_return = 0
            
            # 各推奨銘柄で取引シミュレーション
            for i, rec in enumerate(report_data['recommendations']):
                if i >= self.optimal_params['max_positions']:  # 最大同時保有数制限
                    break
                
                # 取引実行
                trade_result = self.simulate_trade(
                    rec['code'], 
                    rec['price'], 
                    report_data['date']
                )
                logger.info(f"  {rec['company_name']} ({rec['code']}): {trade_result['exit_reason']}")
                
                # 取引記録
                trade_record = {
                    'date': report_data['date'].strftime('%Y-%m-%d'),
                    'code': rec['code'],
                    'company_name': rec['company_name'],
                    'entry_price': rec['price'],
                    'exit_price': trade_result['exit_price'],
                    'return_rate': trade_result['return_rate'],
                    'holding_days': trade_result['holding_days'],
                    'result': trade_result['result'],
                    'exit_reason': trade_result['exit_reason'],
                    'position_size': self.optimal_params['position_size'],
                    'profit_loss': self.optimal_params['position_size'] * trade_result['return_rate'],
                    'prediction_probability': rec['probability']
                }
                
                all_trades.append(trade_record)
                daily_trades.append(trade_record)
                daily_return += trade_result['return_rate']
            
            # 日次パフォーマンス記録
            if daily_trades:
                avg_daily_return = daily_return / len(daily_trades)
                daily_profit = sum(trade['profit_loss'] for trade in daily_trades)
                
                daily_performance.append({
                    'date': report_data['date'].strftime('%Y-%m-%d'),
                    'num_trades': len(daily_trades),
                    'avg_return': avg_daily_return,
                    'daily_profit': daily_profit
                })
        
        # 分析結果集計
        analysis_result = self._analyze_results(all_trades, daily_performance)
        
        # 結果保存
        self._save_results(analysis_result, all_trades, daily_performance)
        
        logger.info("✅ バックテスト分析完了")
        return analysis_result
    
    def _analyze_results(self, all_trades: List[Dict], daily_performance: List[Dict]) -> Dict:
        """取引結果を分析"""
        logger.info("📊 結果分析中...")
        
        if not all_trades:
            return {}
        
        df_trades = pd.DataFrame(all_trades)
        df_daily = pd.DataFrame(daily_performance)
        
        # 基本統計
        total_trades = len(all_trades)
        profitable_trades = len(df_trades[df_trades['return_rate'] > 0])
        loss_trades = len(df_trades[df_trades['return_rate'] < 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 収益統計
        total_profit = df_trades['profit_loss'].sum()
        avg_return_per_trade = df_trades['return_rate'].mean()
        avg_holding_days = df_trades['holding_days'].mean()
        
        # 最終資産
        final_capital = self.optimal_params['initial_capital'] + total_profit
        total_return_rate = total_profit / self.optimal_params['initial_capital']
        
        # 銘柄別パフォーマンス
        stock_stats = df_trades.groupby(['code', 'company_name']).agg({
            'profit_loss': ['sum', 'count'],
            'return_rate': 'mean',
            'result': lambda x: (x == 'profit').sum() / len(x)
        }).round(3)
        stock_stats = stock_stats.sort_values(('profit_loss', 'sum'), ascending=False)
        
        return {
            'summary': {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'loss_trades': loss_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_return_per_trade': avg_return_per_trade,
                'avg_holding_days': avg_holding_days,
                'initial_capital': self.optimal_params['initial_capital'],
                'final_capital': final_capital,
                'total_return_rate': total_return_rate,
                'analysis_period': f"{df_trades['date'].min()} ~ {df_trades['date'].max()}"
            },
            'stock_stats': stock_stats,
            'daily_performance': df_daily
        }
    
    def _save_results(self, analysis: Dict, all_trades: List[Dict], daily_performance: List[Dict]):
        """結果をファイルに保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV出力
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(self.results_dir / f'production_backtest_trades_{timestamp}.csv', 
                           index=False, encoding='utf-8-sig')
        
        if daily_performance:
            daily_df = pd.DataFrame(daily_performance)
            daily_df.to_csv(self.results_dir / f'production_backtest_daily_{timestamp}.csv', 
                          index=False, encoding='utf-8-sig')
        
        # 分析結果JSON（DataFrameは除外）
        analysis_json = {
            'summary': analysis['summary']
        }
        
        with open(self.results_dir / f'production_backtest_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_json, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 結果保存: {self.results_dir}/")
    
    def print_analysis_report(self, analysis: Dict):
        """分析結果をコンソールに出力"""
        if not analysis or 'summary' not in analysis:
            logger.error("分析結果が空です")
            return
        
        s = analysis['summary']
        
        print(f"""
📈 Production Reports バックテスト結果
==============================================

💰 収益サマリー:
  📅 分析期間: {s['analysis_period']}
  💴 初期資金: ¥{s['initial_capital']:,}
  💵 最終資産: ¥{s['final_capital']:,.0f}
  📈 総利益: ¥{s['total_profit']:+,.0f}
  📊 総利益率: {s['total_return_rate']:+.2%}

📊 取引統計:
  🔢 総取引数: {s['total_trades']}回
  ✅ 利益取引: {s['profitable_trades']}回
  ❌ 損失取引: {s['loss_trades']}回
  🎯 勝率: {s['win_rate']:.1%}
  📈 平均リターン/取引: {s['avg_return_per_trade']:+.2%}
  📅 平均保有日数: {s['avg_holding_days']:.1f}日

⚙️  運用パラメータ:
  📅 保有期間: {self.optimal_params['holding_days']}日
  📈 利確設定: +{self.optimal_params['profit_target']:.1%}
  📉 損切設定: -{self.optimal_params['stop_loss']:.1%}
  💰 1銘柄投資額: ¥{self.optimal_params['position_size']:,}
  🔢 最大同時保有: {self.optimal_params['max_positions']}銘柄

==============================================
""")
        
        # TOP5収益銘柄表示
        if 'stock_stats' in analysis and not analysis['stock_stats'].empty:
            print("🏆 TOP5収益銘柄:")
            top_stocks = analysis['stock_stats'].head(5)
            for i, ((code, name), row) in enumerate(top_stocks.iterrows(), 1):
                total_profit = row[('profit_loss', 'sum')]
                trade_count = int(row[('profit_loss', 'count')])
                avg_return = row[('return_rate', 'mean')]
                win_rate = row[('result', '<lambda>')]
                print(f"  {i}位: {name} ({code}) - 利益¥{total_profit:+,.0f} ({trade_count}回, 平均{avg_return:+.1%}, 勝率{win_rate:.1%})")
            print()

def main():
    analyzer = ProductionBacktestAnalyzer()
    analysis = analyzer.run_backtest_analysis()
    analyzer.print_analysis_report(analysis)

if __name__ == "__main__":
    main()
