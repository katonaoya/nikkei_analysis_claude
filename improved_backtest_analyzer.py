#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版Production Reportsバックテスト分析システム
- より保守的なパラメータ設定
- データマッピングの改善
- 詳細な分析結果
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class ImprovedBacktestAnalyzer:
    """改良版バックテスト分析システム"""
    
    def __init__(self):
        self.reports_dir = Path("production_reports")
        self.data_dir = Path("data")
        self.results_dir = Path("results/improved_backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 改良版パラメータ（より保守的）
        self.params = {
            'holding_days': 10,
            'profit_target': 0.05,     # 7% → 5%
            'stop_loss': 0.08,         # 5% → 8%
            'initial_capital': 1000000,
            'max_positions': 10,       # 5 → 10銘柄
            'position_size': 100000,   # 20万 → 10万円
            'min_probability': 0.80    # 80%以上のみ取引
        }
        
        # 株価データと銘柄マッピング
        self.price_data = self._load_price_data()
        self.code_mapping = self._create_code_mapping()
        
    def _load_price_data(self) -> pd.DataFrame:
        """株価データを読み込み"""
        logger.info("📊 株価データ読み込み中...")
        
        parquet_files = list(self.data_dir.rglob("*nikkei225*.parquet"))
        if not parquet_files:
            logger.error("株価データファイルが見つかりません")
            return pd.DataFrame()
        
        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Code', 'Date'])
        
        logger.info(f"✅ 株価データ読み込み完了: {len(df):,}件 (期間: {df['Date'].min().date()} ~ {df['Date'].max().date()})")
        return df
    
    def _create_code_mapping(self) -> Dict[str, str]:
        """4桁コードと5桁コードのマッピングを作成"""
        if self.price_data.empty:
            return {}
        
        # 銘柄名CSVファイルを読み込み
        code_file = self.data_dir / "nikkei225_codes.csv"
        if not code_file.exists():
            logger.warning("銘柄コードファイルが見つかりません")
            return {}
        
        codes_df = pd.read_csv(code_file)
        mapping = {}
        
        # 株価データから実際に存在するコードを取得
        existing_codes = set(self.price_data['Code'].unique())
        
        for _, row in codes_df.iterrows():
            code_4digit = str(row['code']).zfill(4)
            code_5digit = code_4digit + '0'
            
            if code_5digit in existing_codes:
                mapping[code_4digit] = code_5digit
                
        logger.info(f"✅ コードマッピング作成完了: {len(mapping)}銘柄")
        return mapping
    
    def parse_report_file(self, file_path: Path) -> Optional[Dict]:
        """レポートファイルを解析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 日付抽出
            date_match = re.search(r'(\d{8})', file_path.name)
            if not date_match:
                return None
            
            date_str = date_match.group(1)
            report_date = datetime.strptime(date_str, '%Y%m%d')
            
            recommendations = []
            
            # パターン1: 推奨銘柄セクション
            pattern1 = r'### \d+\. 【.*?】(.*?) \((\d+)\).*?\n- \*\*現在価格\*\*: ([\d,]+)円.*?\n- \*\*上昇確率\*\*: ([\d.]+)%'
            matches1 = re.findall(pattern1, content, re.DOTALL)
            
            for match in matches1:
                company_name = match[0].strip()
                code = match[1].zfill(4)  # 4桁に統一
                price = float(match[2].replace(',', ''))
                probability = float(match[3])
                
                recommendations.append({
                    'company_name': company_name,
                    'code': code,
                    'price': price,
                    'probability': probability
                })
            
            # パターン2: テーブル形式
            table_pattern = r'\| (\d+) \| (\d+) \| (.*?) \| ([\d,]+) \| ([\d.]+)% \|'
            table_matches = re.findall(table_pattern, content)
            
            for match in table_matches:
                rank = int(match[0])
                if rank <= 10:  # TOP10まで取得
                    code = match[1].zfill(4)
                    company_name = match[2].strip()
                    price = float(match[3].replace(',', ''))
                    probability = float(match[4])
                    
                    if not any(rec['code'] == code for rec in recommendations):
                        recommendations.append({
                            'company_name': company_name,
                            'code': code,
                            'price': price,
                            'probability': probability
                        })
            
            # 確率フィルタ適用 + TOP10選択
            filtered = [r for r in recommendations if r['probability'] >= self.params['min_probability']]
            filtered.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'date': report_date,
                'recommendations': filtered[:10]
            }
            
        except Exception as e:
            logger.warning(f"レポート解析エラー {file_path.name}: {e}")
            return None
    
    def get_stock_price_data(self, code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """株価データを取得（改良版マッピング）"""
        if self.price_data.empty or code not in self.code_mapping:
            return pd.DataFrame()
        
        mapped_code = self.code_mapping[code]
        
        code_data = self.price_data[
            (self.price_data['Code'] == mapped_code) & 
            (self.price_data['Date'] >= start_date) & 
            (self.price_data['Date'] <= end_date)
        ].copy()
        
        return code_data.sort_values('Date')
    
    def simulate_trade_enhanced(self, code: str, entry_price: float, entry_date: datetime) -> Dict:
        """改良版取引シミュレーション"""
        # より長い期間を設定（土日祝日考慮）
        end_date = entry_date + timedelta(days=self.params['holding_days'] * 3)
        
        price_data = self.get_stock_price_data(code, entry_date, end_date)
        
        if price_data.empty:
            return {
                'result': 'no_data',
                'return_rate': 0,
                'holding_days': 0,
                'exit_price': entry_price,
                'exit_reason': 'データなし',
                'max_gain': 0,
                'max_loss': 0
            }
        
        # 価格設定
        profit_target_price = entry_price * (1 + self.params['profit_target'])
        stop_loss_price = entry_price * (1 - self.params['stop_loss'])
        
        trading_days = 0
        max_gain = 0
        max_loss = 0
        
        for _, row in price_data.iterrows():
            trading_days += 1
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # 最大含み益/含み損を記録
            day_max_gain = (high_price - entry_price) / entry_price
            day_max_loss = (low_price - entry_price) / entry_price
            max_gain = max(max_gain, day_max_gain)
            max_loss = min(max_loss, day_max_loss)
            
            # 利確チェック
            if high_price >= profit_target_price:
                return_rate = (profit_target_price - entry_price) / entry_price
                return {
                    'result': 'profit',
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': profit_target_price,
                    'exit_reason': f'利確(+{return_rate:.1%})',
                    'max_gain': max_gain,
                    'max_loss': max_loss
                }
            
            # 損切チェック
            if low_price <= stop_loss_price:
                return_rate = (stop_loss_price - entry_price) / entry_price
                return {
                    'result': 'loss',
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': stop_loss_price,
                    'exit_reason': f'損切({return_rate:.1%})',
                    'max_gain': max_gain,
                    'max_loss': max_loss
                }
            
            # 最大保有期間到達
            if trading_days >= self.params['holding_days']:
                return_rate = (close_price - entry_price) / entry_price
                result = 'profit' if return_rate > 0 else 'loss' if return_rate < 0 else 'flat'
                return {
                    'result': result,
                    'return_rate': return_rate,
                    'holding_days': trading_days,
                    'exit_price': close_price,
                    'exit_reason': f'期間満了({return_rate:.1%})',
                    'max_gain': max_gain,
                    'max_loss': max_loss
                }
        
        # データ不足
        if trading_days > 0:
            last_close = price_data.iloc[-1]['Close']
            return_rate = (last_close - entry_price) / entry_price
            result = 'profit' if return_rate > 0 else 'loss' if return_rate < 0 else 'flat'
            return {
                'result': result,
                'return_rate': return_rate,
                'holding_days': trading_days,
                'exit_price': last_close,
                'exit_reason': f'データ終了({return_rate:.1%})',
                'max_gain': max_gain,
                'max_loss': max_loss
            }
        
        return {
            'result': 'insufficient_data',
            'return_rate': 0,
            'holding_days': 0,
            'exit_price': entry_price,
            'exit_reason': 'データ不足',
            'max_gain': 0,
            'max_loss': 0
        }
    
    def run_improved_analysis(self) -> Dict:
        """改良版分析を実行"""
        logger.info("🚀 改良版バックテスト分析開始...")
        
        report_files = sorted(list(self.reports_dir.glob("*.md")))
        logger.info(f"📋 分析対象レポート数: {len(report_files)}件")
        
        all_trades = []
        daily_performance = []
        capital = self.params['initial_capital']
        successful_reports = 0
        
        for report_file in report_files:
            logger.info(f"📊 解析中: {report_file.name}")
            
            report_data = self.parse_report_file(report_file)
            if not report_data or not report_data['recommendations']:
                logger.warning(f"  推奨銘柄なし（確率{self.params['min_probability']:.0%}未満）")
                continue
                
            successful_reports += 1
            logger.info(f"  推奨銘柄数: {len(report_data['recommendations'])}銘柄（確率{self.params['min_probability']:.0%}以上）")
            
            daily_trades = []
            daily_profit = 0
            
            for i, rec in enumerate(report_data['recommendations']):
                if i >= self.params['max_positions']:
                    break
                
                trade_result = self.simulate_trade_enhanced(
                    rec['code'], 
                    rec['price'], 
                    report_data['date']
                )
                
                logger.info(f"    {rec['company_name']} ({rec['code']}, {rec['probability']:.1f}%): {trade_result['exit_reason']}")
                
                profit_loss = self.params['position_size'] * trade_result['return_rate']
                daily_profit += profit_loss
                
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
                    'position_size': self.params['position_size'],
                    'profit_loss': profit_loss,
                    'prediction_probability': rec['probability'],
                    'max_gain': trade_result['max_gain'],
                    'max_loss': trade_result['max_loss']
                }
                
                all_trades.append(trade_record)
                daily_trades.append(trade_record)
            
            if daily_trades:
                avg_return = sum(t['return_rate'] for t in daily_trades) / len(daily_trades)
                daily_performance.append({
                    'date': report_data['date'].strftime('%Y-%m-%d'),
                    'num_trades': len(daily_trades),
                    'avg_return': avg_return,
                    'daily_profit': daily_profit,
                    'capital': capital + daily_profit
                })
                
                capital += daily_profit
                logger.info(f"  日次損益: ¥{daily_profit:+,.0f} (累計: ¥{capital:,.0f})")
        
        logger.info(f"✅ 有効レポート数: {successful_reports}/{len(report_files)}")
        
        # 分析結果
        analysis_result = self._analyze_improved_results(all_trades, daily_performance)
        self._save_improved_results(analysis_result, all_trades, daily_performance)
        
        return analysis_result
    
    def _analyze_improved_results(self, all_trades: List[Dict], daily_performance: List[Dict]) -> Dict:
        """改良版結果分析"""
        if not all_trades:
            return {}
        
        df_trades = pd.DataFrame(all_trades)
        
        # 基本統計
        total_trades = len(all_trades)
        profitable_trades = len(df_trades[df_trades['return_rate'] > 0])
        loss_trades = len(df_trades[df_trades['return_rate'] < 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # データ可用性
        data_available = len(df_trades[df_trades['result'] != 'no_data'])
        data_coverage = data_available / total_trades if total_trades > 0 else 0
        
        # 収益統計
        total_profit = df_trades['profit_loss'].sum()
        avg_return = df_trades['return_rate'].mean()
        avg_holding_days = df_trades['holding_days'].mean()
        
        final_capital = self.params['initial_capital'] + total_profit
        total_return_rate = total_profit / self.params['initial_capital']
        
        # 結果別統計
        result_stats = df_trades['result'].value_counts().to_dict()
        
        # 銘柄別パフォーマンス
        stock_stats = df_trades.groupby(['code', 'company_name']).agg({
            'profit_loss': ['sum', 'count', 'mean'],
            'return_rate': ['mean', 'std'],
            'result': lambda x: (x == 'profit').sum() / len(x),
            'max_gain': 'max',
            'max_loss': 'min'
        }).round(4)
        stock_stats = stock_stats.sort_values(('profit_loss', 'sum'), ascending=False)
        
        return {
            'summary': {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'loss_trades': loss_trades,
                'win_rate': win_rate,
                'data_coverage': data_coverage,
                'total_profit': total_profit,
                'avg_return_per_trade': avg_return,
                'avg_holding_days': avg_holding_days,
                'initial_capital': self.params['initial_capital'],
                'final_capital': final_capital,
                'total_return_rate': total_return_rate,
                'analysis_period': f"{df_trades['date'].min()} ~ {df_trades['date'].max()}",
                'result_breakdown': result_stats
            },
            'stock_stats': stock_stats,
            'daily_performance': daily_performance,
            'parameters_used': self.params
        }
    
    def _save_improved_results(self, analysis: Dict, all_trades: List[Dict], daily_performance: List[Dict]):
        """改良版結果保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(self.results_dir / f'improved_backtest_trades_{timestamp}.csv', 
                           index=False, encoding='utf-8-sig')
        
        if daily_performance:
            daily_df = pd.DataFrame(daily_performance)
            daily_df.to_csv(self.results_dir / f'improved_backtest_daily_{timestamp}.csv', 
                          index=False, encoding='utf-8-sig')
        
        # サマリーのみJSON保存
        summary_json = {
            'summary': analysis.get('summary', {}),
            'parameters': analysis.get('parameters_used', {})
        }
        
        with open(self.results_dir / f'improved_backtest_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 改良版結果保存: {self.results_dir}/")
    
    def print_improved_report(self, analysis: Dict):
        """改良版レポート出力"""
        if not analysis or 'summary' not in analysis:
            logger.error("分析結果が空です")
            return
        
        s = analysis['summary']
        p = analysis.get('parameters_used', self.params)
        
        print(f"""
📈 改良版Production Reports バックテスト結果
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
  📊 データカバー率: {s['data_coverage']:.1%}

⚙️  改良パラメータ:
  📅 保有期間: {p['holding_days']}日
  📈 利確設定: +{p['profit_target']:.1%}
  📉 損切設定: -{p['stop_loss']:.1%}
  💰 1銘柄投資額: ¥{p['position_size']:,}
  🔢 最大同時保有: {p['max_positions']}銘柄
  🎯 最小予測確率: {p['min_probability']:.0%}

📋 結果内訳:
""")
        
        # 結果内訳表示
        for result, count in s.get('result_breakdown', {}).items():
            result_jp = {
                'profit': '利確',
                'loss': '損切',
                'no_data': 'データなし',
                'insufficient_data': 'データ不足',
                'flat': '変化なし'
            }.get(result, result)
            print(f"  {result_jp}: {count}回")
        
        print()
        
        # TOP5銘柄表示
        if 'stock_stats' in analysis and not analysis['stock_stats'].empty:
            print("🏆 TOP5収益銘柄:")
            top_stocks = analysis['stock_stats'].head(5)
            for i, ((code, name), row) in enumerate(top_stocks.iterrows(), 1):
                total_profit = row[('profit_loss', 'sum')]
                trade_count = int(row[('profit_loss', 'count')])
                avg_return = row[('return_rate', 'mean')]
                win_rate = row[('result', '<lambda>')]
                max_gain = row[('max_gain', 'max')]
                print(f"  {i}位: {name} ({code})")
                print(f"      利益¥{total_profit:+,.0f} ({trade_count}回, 平均{avg_return:+.1%}, 勝率{win_rate:.1%}, 最大含み益{max_gain:+.1%})")
            print()
        
        print("==============================================")

def main():
    analyzer = ImprovedBacktestAnalyzer()
    analysis = analyzer.run_improved_analysis()
    analyzer.print_improved_report(analysis)

if __name__ == "__main__":
    main()