#!/usr/bin/env python3
"""
Enhanced Backtesting with Enhanced J-Quants Data
==================================================
推奨銘柄データが豊富なenhanced_jquantsデータを使用してバックテスト再実行
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBacktestAnalyzer:
    """Enhanced J-Quantsデータを使用したバックテスト解析"""
    
    def __init__(self):
        self.reports_dir = Path("./production_reports")
        self.data_dir = Path("./data")
        self.results_dir = Path("./results/enhanced_backtest")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 改良されたパラメータ
        self.params = {
            'holding_days': 10,
            'profit_target': 0.05,     # 5%
            'stop_loss': 0.08,         # 8%
            'min_probability': 0.80    # 80%以上のみ取引
        }
        
        logger.info(f"📊 バックテストパラメータ:")
        logger.info(f"   保有期間: {self.params['holding_days']}日")
        logger.info(f"   利確目標: {self.params['profit_target']*100:.1f}%")
        logger.info(f"   損切り: {self.params['stop_loss']*100:.1f}%")
        logger.info(f"   最小確率: {self.params['min_probability']*100:.1f}%")
        
        # Enhanced J-Quantsデータを読み込み
        self.price_data = self._load_enhanced_data()
        
    def _load_enhanced_data(self) -> pd.DataFrame:
        """Enhanced J-Quantsデータを読み込み"""
        logger.info("🚀 Enhanced J-Quantsデータ読み込み中...")
        
        enhanced_files = list(self.data_dir.rglob("enhanced_jquants*.parquet"))
        if not enhanced_files:
            logger.error("Enhanced J-Quantsデータファイルが見つかりません")
            return pd.DataFrame()
        
        latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str)  # 文字列として扱う
        df = df.sort_values(['Code', 'Date'])
        
        unique_codes = df['Code'].unique()
        logger.info(f"✅ Enhanced J-Quantsデータ読み込み完了: {len(df):,}件")
        logger.info(f"   銘柄数: {len(unique_codes)}")
        logger.info(f"   期間: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
        logger.info(f"   銘柄例: {sorted(unique_codes)[:10]}")
        
        return df

    def parse_report_file(self, report_file: Path) -> list:
        """レポートファイルから推奨銘柄を抽出"""
        logger.info(f"📋 レポート解析中: {report_file.name}")
        
        try:
            content = report_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"レポート読み込みエラー: {e}")
            return []
        
        recommendations = []
        
        # 推奨銘柄パターン（改良版）
        patterns = [
            r'### \d+\.\s*【.*?】\s*(.+?)\s*\((\d{4})\).*?\n.*?上昇確率.*?(\d+\.\d+)%',
            r'【.*?】\s*(.+?)\s*\((\d{4})\).*?上昇確率.*?(\d+\.\d+)%',
            r'\|\s*\d+\s*\|\s*(\d{4})\s*\|\s*(.+?)\s*\|.*?\|\s*(\d+\.\d+)%',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match) == 3:
                    if pattern.startswith(r'\|'):  # テーブル形式
                        code, name, probability = match
                    else:  # 通常形式
                        name, code, probability = match
                    
                    try:
                        prob_value = float(probability)
                        if prob_value >= self.params['min_probability'] * 100:  # 80%以上のみ
                            recommendations.append({
                                'code': code.strip(),
                                'name': name.strip(),
                                'probability': prob_value
                            })
                    except ValueError:
                        continue
        
        # 重複除去
        unique_recs = []
        seen_codes = set()
        for rec in recommendations:
            if rec['code'] not in seen_codes:
                unique_recs.append(rec)
                seen_codes.add(rec['code'])
        
        logger.info(f"   抽出された推奨銘柄: {len(unique_recs)}件 (80%以上)")
        for rec in unique_recs:
            logger.info(f"     {rec['code']} ({rec['name']}) - {rec['probability']:.1f}%")
        
        return unique_recs

    def get_stock_price_data(self, code: str, date: datetime) -> pd.DataFrame:
        """指定銘柄の価格データを取得"""
        if self.price_data.empty:
            return pd.DataFrame()
        
        # 文字列として照合
        stock_data = self.price_data[self.price_data['Code'] == code].copy()
        if stock_data.empty:
            return pd.DataFrame()
        
        # 指定日以降のデータを取得
        stock_data = stock_data[stock_data['Date'] >= date].sort_values('Date')
        
        return stock_data

    def simulate_trade_enhanced(self, code: str, name: str, entry_date: datetime, probability: float) -> dict:
        """Enhanced取引シミュレーション"""
        stock_data = self.get_stock_price_data(code, entry_date)
        
        if stock_data.empty:
            return {
                'code': code,
                'name': name,
                'status': 'データなし',
                'entry_date': entry_date.date(),
                'entry_price': 0,
                'exit_date': None,
                'exit_price': 0,
                'return': 0,
                'days_held': 0,
                'exit_reason': 'データなし',
                'probability': probability
            }
        
        # エントリー価格（翌日の始値）
        entry_row = stock_data.iloc[0]
        entry_price = entry_row['Open']
        
        if pd.isna(entry_price) or entry_price <= 0:
            return {
                'code': code,
                'name': name,
                'status': 'エントリー不可',
                'entry_date': entry_date.date(),
                'entry_price': 0,
                'exit_date': None,
                'exit_price': 0,
                'return': 0,
                'days_held': 0,
                'exit_reason': 'エントリー価格なし',
                'probability': probability
            }
        
        # 取引シミュレーション
        profit_target = entry_price * (1 + self.params['profit_target'])
        stop_loss = entry_price * (1 - self.params['stop_loss'])
        
        for i, (_, row) in enumerate(stock_data.iterrows(), 1):
            current_date = row['Date']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # 利確チェック
            if pd.notna(high_price) and high_price >= profit_target:
                return {
                    'code': code,
                    'name': name,
                    'status': '利確',
                    'entry_date': entry_date.date(),
                    'entry_price': entry_price,
                    'exit_date': current_date.date(),
                    'exit_price': profit_target,
                    'return': self.params['profit_target'],
                    'days_held': i,
                    'exit_reason': '利確達成',
                    'probability': probability
                }
            
            # 損切りチェック
            if pd.notna(low_price) and low_price <= stop_loss:
                return {
                    'code': code,
                    'name': name,
                    'status': '損切り',
                    'entry_date': entry_date.date(),
                    'entry_price': entry_price,
                    'exit_date': current_date.date(),
                    'exit_price': stop_loss,
                    'return': -self.params['stop_loss'],
                    'days_held': i,
                    'exit_reason': '損切り執行',
                    'probability': probability
                }
            
            # 保有期間チェック
            if i >= self.params['holding_days']:
                return {
                    'code': code,
                    'name': name,
                    'status': '期間満了',
                    'entry_date': entry_date.date(),
                    'entry_price': entry_price,
                    'exit_date': current_date.date(),
                    'exit_price': close_price,
                    'return': (close_price - entry_price) / entry_price if pd.notna(close_price) and close_price > 0 else 0,
                    'days_held': i,
                    'exit_reason': '保有期間満了',
                    'probability': probability
                }
        
        # データ不足で終了
        last_row = stock_data.iloc[-1]
        return {
            'code': code,
            'name': name,
            'status': 'データ不足',
            'entry_date': entry_date.date(),
            'entry_price': entry_price,
            'exit_date': last_row['Date'].date(),
            'exit_price': last_row['Close'],
            'return': (last_row['Close'] - entry_price) / entry_price if pd.notna(last_row['Close']) and last_row['Close'] > 0 else 0,
            'days_held': len(stock_data),
            'exit_reason': 'データ不足',
            'probability': probability
        }

    def run_enhanced_analysis(self):
        """Enhanced解析実行"""
        logger.info("🎯 Enhanced バックテスト開始")
        
        # レポートファイル取得
        report_files = sorted(list(self.reports_dir.glob("*.md")))
        if not report_files:
            logger.error("レポートファイルが見つかりません")
            return
        
        logger.info(f"📁 レポート数: {len(report_files)}")
        
        all_trades = []
        processed_reports = 0
        
        for report_file in report_files:
            # レポート日付抽出
            date_match = re.search(r'(\d{8})', report_file.name)
            if not date_match:
                logger.warning(f"日付抽出失敗: {report_file.name}")
                continue
            
            try:
                report_date = datetime.strptime(date_match.group(1), "%Y%m%d")
            except ValueError:
                logger.warning(f"日付解析失敗: {report_file.name}")
                continue
            
            # 推奨銘柄抽出
            recommendations = self.parse_report_file(report_file)
            if not recommendations:
                logger.info(f"推奨銘柄なし: {report_file.name}")
                continue
            
            # 各推奨銘柄の取引シミュレーション
            entry_date = report_date + timedelta(days=1)  # 翌日エントリー
            
            for rec in recommendations:
                trade_result = self.simulate_trade_enhanced(
                    rec['code'], rec['name'], entry_date, rec['probability']
                )
                trade_result['report_file'] = report_file.name
                trade_result['report_date'] = report_date.date()
                all_trades.append(trade_result)
            
            processed_reports += 1
        
        logger.info(f"✅ 処理完了: {processed_reports}レポート, {len(all_trades)}取引")
        
        # 結果分析
        analysis = self._analyze_enhanced_results(all_trades)
        
        # 結果保存
        self._save_enhanced_results(all_trades, analysis)
        
        # レポート出力
        self.print_enhanced_report(analysis, all_trades)
        
        return analysis, all_trades

    def _analyze_enhanced_results(self, trades: list) -> dict:
        """Enhanced結果分析"""
        if not trades:
            return {'error': '分析対象データなし'}
        
        trades_df = pd.DataFrame(trades)
        
        # 基本統計
        total_trades = len(trades)
        successful_trades = trades_df[trades_df['status'].isin(['利確', '損切り', '期間満了', 'データ不足'])].copy()
        data_available = trades_df[trades_df['status'] != 'データなし']
        
        if len(data_available) == 0:
            return {
                'error': 'データ利用可能な取引なし',
                'total_trades': total_trades,
                'data_available': 0,
                'data_coverage': 0
            }
        
        # 収益計算
        returns = data_available['return'].astype(float)
        total_return = returns.sum()
        avg_return = returns.mean()
        
        # 状況別統計
        status_counts = trades_df['status'].value_counts().to_dict()
        
        # 銘柄別統計
        profitable_codes = data_available[data_available['return'] > 0]['code'].unique()
        
        return {
            'total_trades': total_trades,
            'data_available': len(data_available),
            'data_coverage': len(data_available) / total_trades,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'profitable_trades': len(data_available[data_available['return'] > 0]),
            'loss_trades': len(data_available[data_available['return'] < 0]),
            'status_breakdown': status_counts,
            'profitable_codes': list(profitable_codes),
            'summary': {
                'profit_percentage': total_return * 100,
                'win_rate': len(data_available[data_available['return'] > 0]) / len(data_available) if len(data_available) > 0 else 0,
                'avg_return_percentage': avg_return * 100
            }
        }

    def _save_enhanced_results(self, trades: list, analysis: dict):
        """Enhanced結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 取引詳細CSV
        trades_df = pd.DataFrame(trades)
        trades_csv = self.results_dir / f"enhanced_backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
        
        # 分析結果JSON（シンプル版）
        analysis_json = self.results_dir / f"enhanced_backtest_analysis_{timestamp}.json"
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump({'summary': analysis.get('summary', {})}, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 結果保存完了:")
        logger.info(f"   取引詳細: {trades_csv}")
        logger.info(f"   分析結果: {analysis_json}")

    def print_enhanced_report(self, analysis: dict, trades: list):
        """Enhanced結果レポート出力"""
        print("\n" + "="*80)
        print("🚀 ENHANCED BACKTEST RESULTS - Enhanced J-Quants Data")
        print("="*80)
        
        if 'error' in analysis:
            print(f"❌ エラー: {analysis['error']}")
            return
        
        # サマリー
        summary = analysis.get('summary', {})
        print(f"\n📊 総合結果:")
        print(f"   総利益率: {summary.get('profit_percentage', 0):+.2f}%")
        print(f"   勝率: {summary.get('win_rate', 0)*100:.1f}%")
        print(f"   平均リターン: {summary.get('avg_return_percentage', 0):+.2f}%")
        
        print(f"\n📈 取引統計:")
        print(f"   総取引数: {analysis['total_trades']}")
        print(f"   データ利用可能: {analysis['data_available']} ({analysis['data_coverage']*100:.1f}%)")
        print(f"   利益取引: {analysis['profitable_trades']}")
        print(f"   損失取引: {analysis['loss_trades']}")
        
        # ステータス内訳
        print(f"\n📋 取引結果内訳:")
        for status, count in analysis['status_breakdown'].items():
            percentage = count / analysis['total_trades'] * 100
            print(f"   {status}: {count}件 ({percentage:.1f}%)")
        
        # 利益を上げた銘柄
        if analysis['profitable_codes']:
            print(f"\n💰 利益銘柄 ({len(analysis['profitable_codes'])}銘柄):")
            print(f"   {', '.join(analysis['profitable_codes'][:10])}")
            if len(analysis['profitable_codes']) > 10:
                print(f"   他 {len(analysis['profitable_codes'])-10}銘柄...")

def main():
    """メイン関数"""
    analyzer = EnhancedBacktestAnalyzer()
    analysis, trades = analyzer.run_enhanced_analysis()

if __name__ == "__main__":
    main()