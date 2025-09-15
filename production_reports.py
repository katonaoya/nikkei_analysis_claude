#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
運用レポート生成システム
売買推奨と保有銘柄管理のレポートを生成
"""

import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProductionReportGenerator:
    """運用レポート生成クラス"""
    
    def __init__(self, config_path="production_config.yaml"):
        self.config_path = Path(config_path)
        
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """設定ファイル読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # パラメータ展開
        self.optimal_params = self.config['optimal_params']
        self.initial_capital = self.config['system']['initial_capital']
        self.max_positions = self.config['system']['max_positions']
        self.transaction_cost_rate = self.config['system']['transaction_cost_rate']
        
        logger.info("✅ 設定ファイル読み込み完了")
        
    def setup_directories(self):
        """ディレクトリ設定"""
        self.reports_dir = Path(self.config['reports']['output_dir'])
        self.reports_dir.mkdir(exist_ok=True)
        
        # 日付別サブディレクトリ
        report_date = datetime.now().strftime('%Y%m%d')
        self.daily_reports_dir = self.reports_dir / report_date
        self.daily_reports_dir.mkdir(exist_ok=True)
        
    def generate_buy_recommendations(self, predictions: Dict, current_prices: Dict) -> Dict:
        """購入推奨レポート生成"""
        logger.info("📊 購入推奨レポート生成中...")
        
        # 推奨銘柄の選択（信頼度上位5銘柄）
        recommendations = []
        
        # predictionsがDataFrameの場合とdictの場合に対応
        if hasattr(predictions, 'iterrows'):  # DataFrame
            for _, row in predictions.iterrows():
                code = str(row['stock_code'])
                if code not in current_prices:
                    continue
                    
                confidence = row['confidence']
                price = current_prices[code]
                
                if confidence >= self.config['system']['confidence_threshold']:
                    # 会社名の修正（「銘柄XXXXX」の場合は正しい名前に置換）
                    company_name = row.get('company_name', f'株式会社{code}')
                    
                    # 既知の会社名マッピング（より多くの銘柄を追加）
                    known_names = {
                        '82670': 'ギグワークス',
                        '97660': 'モビルス',
                        '99830': 'オハラ',
                        '78320': 'ビットワングループ',
                        '63670': 'ANYCOLOR'
                    }
                    
                    # codeで直接置換
                    if code in known_names:
                        company_name = known_names[code]
                    
                    recommendations.append({
                        'code': code,
                        'company_name': company_name,
                        'current_price': price,
                        'confidence': confidence,
                        'predicted_direction': 'UP'
                    })
        else:  # dict
            for code, pred_data in predictions.items():
                if code not in current_prices:
                    continue
                    
                confidence = pred_data.get('confidence', 0)
                price = current_prices[code]
                
                if confidence >= self.config['system']['confidence_threshold']:
                    recommendations.append({
                        'code': code,
                        'company_name': pred_data.get('company_name', f'株式会社{code}'),
                        'current_price': price,
                        'confidence': confidence,
                        'predicted_direction': pred_data.get('direction', 'UP')
                    })
        
        # 信頼度順にソート
        recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
        top_recommendations = recommendations[:self.max_positions]
        
        # 会社名の最終修正
        known_names = {
            '82670': 'ギグワークス',
            '97660': 'モビルス',
            '99830': 'オハラ',
            '78320': 'ビットワングループ',
            '63670': 'ANYCOLOR'
        }
        
        for rec in top_recommendations:
            if rec['code'] in known_names:
                rec['company_name'] = known_names[rec['code']]
        
        # 投資額計算
        if len(top_recommendations) > 0:
            investment_per_stock = self.initial_capital * 0.95 / len(top_recommendations)
            
            for rec in top_recommendations:
                price = rec['current_price']
                max_shares = int(investment_per_stock / price)
                investment_amount = max_shares * price
                
                # 利確・損切り価格計算
                profit_price = round(price * (1 + self.optimal_params['profit_target']))
                stop_loss_price = round(price * (1 - self.optimal_params['stop_loss']))
                
                rec.update({
                    'recommended_shares': max_shares,
                    'investment_amount': investment_amount,
                    'profit_target_price': profit_price,
                    'stop_loss_price': stop_loss_price,
                    'expected_hold_days': self.optimal_params['hold_days']
                })
        
        buy_report = {
            'report_type': '購入推奨レポート',
            'generated_at': datetime.now().isoformat(),
            'market_date': datetime.now().strftime('%Y-%m-%d'),
            'total_recommendations': len(top_recommendations),
            'available_capital': self.initial_capital,
            'max_positions': self.max_positions,
            'parameters': self.optimal_params,
            'recommendations': top_recommendations,
            'summary': {
                'total_investment': sum(rec['investment_amount'] for rec in top_recommendations),
                'average_confidence': np.mean([rec['confidence'] for rec in top_recommendations]) if top_recommendations else 0,
                'cash_remaining': self.initial_capital - sum(rec['investment_amount'] for rec in top_recommendations)
            }
        }
        
        return buy_report
    
    def generate_portfolio_management_report(self, current_portfolio: Dict, current_prices: Dict) -> Dict:
        """保有銘柄管理レポート生成"""
        logger.info("📊 保有銘柄管理レポート生成中...")
        
        portfolio_analysis = []
        total_unrealized_pl = 0
        total_portfolio_value = 0
        
        # current_portfolioがリスト形式の場合に対応
        if isinstance(current_portfolio, list):
            positions = current_portfolio
        else:
            # 辞書形式の場合は値を取得
            positions = current_portfolio.values() if hasattr(current_portfolio, 'values') else []
        
        for position in positions:
            code = str(position['code'])
            if code not in current_prices:
                continue
                
            current_price = current_prices[code]
            buy_price = position['buy_price']
            shares = position['shares']
            buy_date = pd.to_datetime(position['buy_date'])
            
            # 損益計算
            current_value = shares * current_price
            cost_basis = shares * buy_price
            unrealized_pl = current_value - cost_basis
            unrealized_pl_pct = (current_price - buy_price) / buy_price
            
            # 保有日数
            days_held = (datetime.now() - buy_date).days
            
            # 売却判定
            profit_target_price = buy_price * (1 + self.optimal_params['profit_target'])
            stop_loss_price = buy_price * (1 - self.optimal_params['stop_loss'])
            
            sell_action = None
            sell_reason = None
            
            if days_held >= self.optimal_params['hold_days']:
                sell_action = "売却推奨"
                sell_reason = "期間満了"
            elif current_price <= stop_loss_price:
                sell_action = "即座売却"
                sell_reason = "損切り"
            elif current_price >= profit_target_price:
                sell_action = "売却推奨"
                sell_reason = "利確"
            else:
                sell_action = "保有継続"
                sell_reason = "条件未達"
            
            portfolio_analysis.append({
                'code': code,
                'company_name': position.get('company_name', f'株式会社{code}'),
                'shares': shares,
                'buy_price': buy_price,
                'buy_date': buy_date.strftime('%Y-%m-%d'),
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'unrealized_pl': unrealized_pl,
                'unrealized_pl_pct': unrealized_pl_pct,
                'days_held': days_held,
                'profit_target_price': profit_target_price,
                'stop_loss_price': stop_loss_price,
                'sell_action': sell_action,
                'sell_reason': sell_reason
            })
            
            total_unrealized_pl += unrealized_pl
            total_portfolio_value += current_value
        
        # ポートフォリオサマリー
        cash_balance = self.initial_capital  # 簡略化、実際は取引履歴から計算
        total_value = total_portfolio_value + cash_balance
        
        portfolio_report = {
            'report_type': '保有銘柄管理レポート',
            'generated_at': datetime.now().isoformat(),
            'portfolio_date': datetime.now().strftime('%Y-%m-%d'),
            'total_positions': len(portfolio_analysis),
            'portfolio_summary': {
                'total_portfolio_value': total_portfolio_value,
                'cash_balance': cash_balance,
                'total_value': total_value,
                'total_unrealized_pl': total_unrealized_pl,
                'total_unrealized_pl_pct': (total_unrealized_pl / (total_value - total_unrealized_pl)) if (total_value - total_unrealized_pl) > 0 else 0,
                'portfolio_weight': total_portfolio_value / total_value if total_value > 0 else 0
            },
            'positions': portfolio_analysis,
            'sell_recommendations': [pos for pos in portfolio_analysis if pos['sell_action'] in ['売却推奨', '即座売却']],
            'parameters': self.optimal_params
        }
        
        return portfolio_report
    
    def generate_performance_summary(self, trade_history: List[Dict]) -> Dict:
        """パフォーマンスサマリー生成"""
        logger.info("📊 パフォーマンスサマリー生成中...")
        
        if not trade_history:
            return {
                'report_type': 'パフォーマンスサマリー',
                'generated_at': datetime.now().isoformat(),
                'message': '取引履歴がありません'
            }
        
        trades_df = pd.DataFrame(trade_history)
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(sell_trades) == 0:
            return {
                'report_type': 'パフォーマンスサマリー',
                'generated_at': datetime.now().isoformat(),
                'message': '完了した取引がありません'
            }
        
        # パフォーマンス指標計算
        total_trades = len(sell_trades)
        winning_trades = len(sell_trades[sell_trades['profit_loss'] > 0])
        losing_trades = len(sell_trades[sell_trades['profit_loss'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit_loss = sell_trades['profit_loss'].sum()
        avg_profit_loss = sell_trades['profit_loss'].mean()
        avg_holding_days = sell_trades['days_held'].mean()
        
        # 月別パフォーマンス
        sell_trades['sell_date'] = pd.to_datetime(sell_trades['date'])
        monthly_performance = sell_trades.groupby(sell_trades['sell_date'].dt.to_period('M')).agg({
            'profit_loss': ['sum', 'count', 'mean']
        }).round(2)
        
        performance_summary = {
            'report_type': 'パフォーマンスサマリー',
            'generated_at': datetime.now().isoformat(),
            'period_start': sell_trades['date'].min(),
            'period_end': sell_trades['date'].max(),
            'overall_performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit_loss': total_profit_loss,
                'avg_profit_loss': avg_profit_loss,
                'avg_holding_days': avg_holding_days,
                'total_return_pct': (total_profit_loss / self.initial_capital) if self.initial_capital > 0 else 0
            },
            'recent_trades': sell_trades.tail(10).to_dict('records'),
            'parameters_used': self.optimal_params
        }
        
        return performance_summary
    
    def save_reports_to_files(self, buy_report: Dict, portfolio_report: Dict, performance_report: Dict):
        """レポートをMarkdown形式で保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Markdown形式で保存（メインレポート）
        markdown_file = self.daily_reports_dir / f"trading_report_{timestamp}.md"
        markdown_content = self.format_reports_as_markdown(buy_report, portfolio_report, performance_report)
        
        # 文字列置換で会社名を修正
        replacements = {
            '銘柄97660': 'モビルス',
            '銘柄99830': 'オハラ',
            '銘柄78320': 'ビットワングループ',
            '銘柄63670': 'ANYCOLOR',
            '銘柄91040': 'フォーバルテレコム',
            '銘柄65030': 'アバールデータ',
            '銘柄25010': 'セブン&アイ',
            '銘柄70120': '川本産業',
            '銘柄45680': 'KADOKAWA',
            '銘柄56310': 'ジーニー',
            '銘柄79740': 'キャンバス',
            '銘柄70130': 'テイコクセン'
        }
        
        for old_name, new_name in replacements.items():
            markdown_content = markdown_content.replace(old_name, new_name)
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"📁 レポート保存完了: {markdown_file}")
        
        return markdown_file
    
    def format_reports_as_markdown(self, buy_report: Dict, portfolio_report: Dict, performance_report: Dict) -> str:
        """レポートをMarkdown形式でフォーマット"""
        md_lines = []
        
        # 次の営業日を計算
        from datetime import datetime, timedelta
        now = datetime.now()
        hour = now.hour
        
        # 深夜実行の場合は当日の予測、それ以外は翌営業日の予測
        if hour >= 0 and hour < 6:
            target_day_text = f"本日（{now.strftime('%Y-%m-%d')}）の市場開場後"
        else:
            # 翌営業日の計算（簡易版：土日をスキップ）
            next_day = now + timedelta(days=1)
            while next_day.weekday() >= 5:  # 土日の場合
                next_day += timedelta(days=1)
            target_day_text = f"翌営業日（{next_day.strftime('%Y-%m-%d')}）の市場開場後"
        
        # ヘッダー
        md_lines.append("# 🤖 AI株式取引システム - 運用レポート")
        md_lines.append("")
        md_lines.append(f"**📅 生成日時:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**📊 基準日:** {now.strftime('%Y-%m-%d')}")
        md_lines.append(f"**🎯 予測対象:** {target_day_text}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 購入推奨レポート
        md_lines.append("## 📈 購入推奨レポート")
        md_lines.append("")
        
        if buy_report['total_recommendations'] > 0:
            # サマリー情報をカードスタイルで表示
            md_lines.append("### 📊 投資サマリー")
            md_lines.append("")
            md_lines.append("| 項目 | 金額 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 💰 利用可能資金 | ¥{buy_report['available_capital']:,} |")
            md_lines.append(f"| 🎯 推奨銘柄数 | {buy_report['total_recommendations']}銘柄 |")
            md_lines.append(f"| 💵 合計投資予定額 | ¥{buy_report['summary']['total_investment']:,} |")
            md_lines.append(f"| 💰 残り現金 | ¥{buy_report['summary']['cash_remaining']:,} |")
            md_lines.append(f"| 📊 平均信頼度 | {buy_report['summary']['average_confidence']:.1%} |")
            md_lines.append("")
            
            # 運用パラメータ
            params = buy_report['parameters']
            md_lines.append("### ⚙️ 運用パラメータ")
            md_lines.append("")
            md_lines.append("| パラメータ | 設定値 |")
            md_lines.append("|------------|---------|")
            md_lines.append(f"| 📅 保有期間 | {params['hold_days']}日 |")
            md_lines.append(f"| 📈 利確閾値 | {params['profit_target']:.1%} |")
            md_lines.append(f"| 📉 損切閾値 | {params['stop_loss']:.1%} |")
            md_lines.append(f"| 🎯 年率リターン | {params['annual_return']:.2%} |")
            md_lines.append("")
            
            # 推奨銘柄一覧
            md_lines.append("### 🎯 推奨銘柄一覧")
            md_lines.append("")
            md_lines.append("| 銘柄コード | 会社名 | 現在値 | 推奨株数 | 投資額 | 利確価格 | 損切価格 | AI信頼度 |")
            md_lines.append("|------------|--------|--------|----------|---------|----------|----------|----------|")
            
            for rec in buy_report['recommendations']:
                md_lines.append(
                    f"| **{rec['code']}** | {rec['company_name']} | "
                    f"¥{rec['current_price']:,} | {rec['recommended_shares']:,}株 | "
                    f"¥{rec['investment_amount']:,} | ¥{rec['profit_target_price']:,.0f} | "
                    f"¥{rec['stop_loss_price']:,.0f} | **{rec['confidence']:.1%}** |"
                )
            
            md_lines.append("")
            
            # 各銘柄の詳細カード
            md_lines.append("### 📋 銘柄詳細")
            md_lines.append("")
            
            for i, rec in enumerate(buy_report['recommendations'], 1):
                md_lines.append(f"#### {i}. {rec['company_name']} ({rec['code']})")
                md_lines.append("")
                md_lines.append("```")
                md_lines.append(f"💰 投資金額: ¥{rec['investment_amount']:,}")
                md_lines.append(f"📊 購入株数: {rec['recommended_shares']:,}株")
                md_lines.append(f"💹 現在価格: ¥{rec['current_price']:,}")
                md_lines.append(f"🎯 利確価格: ¥{rec['profit_target_price']:,.0f} (+{((rec['profit_target_price']/rec['current_price'])-1):.1%})")
                md_lines.append(f"⚠️  損切価格: ¥{rec['stop_loss_price']:,.0f} ({((rec['stop_loss_price']/rec['current_price'])-1):.1%})")
                md_lines.append(f"🤖 AI信頼度: {rec['confidence']:.1%}")
                md_lines.append("```")
                md_lines.append("")
        else:
            md_lines.append("❌ **推奨銘柄なし**")
            md_lines.append("")
            md_lines.append("> 現在の市場状況では、設定した信頼度閾値を満たす銘柄が見つかりませんでした。")
            md_lines.append("")
        
        md_lines.append("---")
        md_lines.append("")
        
        # 保有銘柄管理レポート
        md_lines.append("## 📊 保有銘柄管理レポート")
        md_lines.append("")
        
        if portfolio_report['total_positions'] > 0:
            summary = portfolio_report['portfolio_summary']
            
            # ポートフォリオサマリー
            md_lines.append("### 💼 ポートフォリオサマリー")
            md_lines.append("")
            md_lines.append("| 項目 | 金額 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 📈 保有銘柄数 | {portfolio_report['total_positions']}銘柄 |")
            md_lines.append(f"| 💰 ポートフォリオ評価額 | ¥{summary['total_portfolio_value']:,} |")
            md_lines.append(f"| 💵 現金残高 | ¥{summary['cash_balance']:,} |")
            md_lines.append(f"| 📊 総評価額 | ¥{summary['total_value']:,} |")
            
            # 評価損益の色分け
            pl_emoji = "📈" if summary['total_unrealized_pl'] >= 0 else "📉"
            pl_sign = "+" if summary['total_unrealized_pl'] >= 0 else ""
            md_lines.append(f"| {pl_emoji} **評価損益** | **¥{summary['total_unrealized_pl']:+,} ({summary['total_unrealized_pl_pct']:+.2%})** |")
            md_lines.append("")
            
            # 保有銘柄一覧
            md_lines.append("### 📋 保有銘柄一覧")
            md_lines.append("")
            md_lines.append("| 銘柄コード | 会社名 | 株数 | 買値 | 現在値 | 評価損益 | 保有日数 | 売却判定 |")
            md_lines.append("|------------|--------|------|------|--------|----------|----------|----------|")
            
            for pos in portfolio_report['positions']:
                # 評価損益の色分け
                pl_emoji = "📈" if pos['unrealized_pl'] >= 0 else "📉"
                
                # 売却判定の色分け
                action_emoji = {
                    "即座売却": "🚨",
                    "売却推奨": "⚠️",
                    "保有継続": "✅"
                }.get(pos['sell_action'], "❓")
                
                md_lines.append(
                    f"| **{pos['code']}** | {pos['company_name']} | "
                    f"{pos['shares']:,}株 | ¥{pos['buy_price']:,} | ¥{pos['current_price']:,} | "
                    f"{pl_emoji} ¥{pos['unrealized_pl']:+,} | {pos['days_held']}日 | "
                    f"{action_emoji} **{pos['sell_action']}** |"
                )
            
            md_lines.append("")
            
            # 売却推奨があれば詳細表示
            sell_recs = portfolio_report['sell_recommendations']
            if sell_recs:
                md_lines.append("### 🚨 売却推奨銘柄")
                md_lines.append("")
                
                for rec in sell_recs:
                    priority_emoji = {"損切り": "🚨", "利確": "💰", "期間満了": "⏰"}.get(rec['sell_reason'], "⚠️")
                    
                    md_lines.append(f"#### {priority_emoji} {rec['company_name']} ({rec['code']})")
                    md_lines.append("")
                    md_lines.append(f"**売却理由:** {rec['sell_reason']}")
                    md_lines.append("")
                    md_lines.append("```")
                    md_lines.append(f"📊 保有株数: {rec['shares']:,}株")
                    md_lines.append(f"💰 買値: ¥{rec['buy_price']:,}")
                    md_lines.append(f"💹 現在値: ¥{rec['current_price']:,}")
                    md_lines.append(f"📈 損益率: {rec['unrealized_pl_pct']:+.2%}")
                    md_lines.append(f"📅 保有日数: {rec['days_held']}日")
                    md_lines.append("```")
                    md_lines.append("")
        else:
            md_lines.append("📝 **現在保有銘柄なし**")
            md_lines.append("")
            md_lines.append("> 新しい投資機会を検討してください。")
            md_lines.append("")
        
        md_lines.append("---")
        md_lines.append("")
        
        # パフォーマンスサマリー
        md_lines.append("## 📊 パフォーマンスサマリー")
        md_lines.append("")
        
        if 'overall_performance' in performance_report:
            perf = performance_report['overall_performance']
            
            md_lines.append("### 📈 運用実績")
            md_lines.append("")
            md_lines.append(f"**📅 集計期間:** {performance_report['period_start']} ～ {performance_report['period_end']}")
            md_lines.append("")
            
            # パフォーマンス指標
            md_lines.append("| 指標 | 実績 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 🔄 総取引数 | {perf['total_trades']}回 |")
            md_lines.append(f"| ✅ 勝ちトレード | {perf['winning_trades']}回 |")
            md_lines.append(f"| ❌ 負けトレード | {perf['losing_trades']}回 |")
            md_lines.append(f"| 🎯 **勝率** | **{perf['win_rate']:.1%}** |")
            md_lines.append(f"| 💰 **累計損益** | **¥{perf['total_profit_loss']:+,}** |")
            md_lines.append(f"| 📊 平均損益 | ¥{perf['avg_profit_loss']:+,} |")
            md_lines.append(f"| 📅 平均保有日数 | {perf['avg_holding_days']:.1f}日 |")
            md_lines.append(f"| 📈 **リターン率** | **{perf['total_return_pct']:+.2%}** |")
            md_lines.append("")
            
        else:
            message = performance_report.get('message', 'データなし')
            md_lines.append(f"ℹ️ {message}")
            md_lines.append("")
        
        # フッター
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## 📱 次のアクション")
        md_lines.append("")
        md_lines.append("### 🛒 購入手順")
        md_lines.append("1. 楽天証券にログイン")
        md_lines.append("2. 上記推奨銘柄を指値で注文")
        md_lines.append("3. 利確・損切価格も同時に指値設定")
        md_lines.append("")
        md_lines.append("### 📤 売却手順")
        md_lines.append("1. 売却推奨銘柄を確認")
        md_lines.append("2. 楽天証券で該当銘柄を売却注文")
        md_lines.append("3. 成行または指値で売却実行")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("*🤖 AI株式取引システム - Powered by Claude*")
        md_lines.append("")
        md_lines.append(f"*生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(md_lines)
    
    def format_reports_as_text(self, buy_report: Dict, portfolio_report: Dict, performance_report: Dict) -> str:
        """レポートをテキスト形式でフォーマット"""
        text_lines = []
        
        # ヘッダー
        text_lines.append("=" * 100)
        text_lines.append(f"🤖 AI株式取引システム - 運用レポート")
        text_lines.append(f"📅 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_lines.append("=" * 100)
        
        # 購入推奨レポート
        text_lines.append("\n📈 【購入推奨レポート】")
        text_lines.append("-" * 60)
        
        if buy_report['total_recommendations'] > 0:
            text_lines.append(f"💰 利用可能資金: ¥{buy_report['available_capital']:,}")
            text_lines.append(f"🎯 推奨銘柄数: {buy_report['total_recommendations']}銘柄")
            text_lines.append(f"📊 運用パラメータ: {buy_report['parameters']['hold_days']}日保有, {buy_report['parameters']['profit_target']:.1%}利確, {buy_report['parameters']['stop_loss']:.1%}損切")
            text_lines.append("")
            
            text_lines.append("推奨銘柄一覧:")
            text_lines.append("コード | 会社名     | 現在値  | 推奨株数 | 投資額     | 利確価格 | 損切価格 | 信頼度")
            text_lines.append("-" * 90)
            
            for rec in buy_report['recommendations']:
                # 会社名の修正
                company_name = rec['company_name']
                known_names = {
                    '82670': 'ギグワークス',
                    '97660': 'モビルス',
                    '99830': 'オハラ',
                    '78320': 'ビットワングループ',
                    '63670': 'ANYCOLOR'
                }
                if rec['code'] in known_names:
                    company_name = known_names[rec['code']]
                
                text_lines.append(
                    f"{rec['code']:>6} | {company_name[:10]:<10} | "
                    f"¥{rec['current_price']:>6,.0f} | {rec['recommended_shares']:>6,}株 | "
                    f"¥{rec['investment_amount']:>8,.0f} | ¥{rec['profit_target_price']:>6,.0f} | "
                    f"¥{rec['stop_loss_price']:>6,.0f} | {rec['confidence']:>5.1%}"
                )
            
            text_lines.append("")
            text_lines.append(f"💵 合計投資額: ¥{buy_report['summary']['total_investment']:,}")
            text_lines.append(f"💰 残り現金: ¥{buy_report['summary']['cash_remaining']:,}")
        else:
            text_lines.append("❌ 推奨銘柄なし（条件を満たす銘柄が見つかりませんでした）")
        
        # 保有銘柄管理レポート
        text_lines.append(f"\n📊 【保有銘柄管理レポート】")
        text_lines.append("-" * 60)
        
        if portfolio_report['total_positions'] > 0:
            summary = portfolio_report['portfolio_summary']
            text_lines.append(f"📈 保有銘柄数: {portfolio_report['total_positions']}銘柄")
            text_lines.append(f"💰 ポートフォリオ評価額: ¥{summary['total_portfolio_value']:,}")
            text_lines.append(f"📊 評価損益: ¥{summary['total_unrealized_pl']:+,} ({summary['total_unrealized_pl_pct']:+.2%})")
            text_lines.append("")
            
            text_lines.append("保有銘柄一覧:")
            text_lines.append("コード | 会社名     | 株数   | 買値    | 現在値  | 評価損益  | 保有日数 | 売却判定")
            text_lines.append("-" * 90)
            
            for pos in portfolio_report['positions']:
                text_lines.append(
                    f"{pos['code']:>6} | {pos['company_name'][:10]:<10} | "
                    f"{pos['shares']:>5,}株 | ¥{pos['buy_price']:>6,.0f} | "
                    f"¥{pos['current_price']:>6,.0f} | ¥{pos['unrealized_pl']:>+8,.0f} | "
                    f"{pos['days_held']:>6}日 | {pos['sell_action']}"
                )
            
            # 売却推奨があれば表示
            sell_recs = portfolio_report['sell_recommendations']
            if sell_recs:
                text_lines.append(f"\n🚨 売却推奨銘柄: {len(sell_recs)}銘柄")
                for rec in sell_recs:
                    text_lines.append(f"  ▶ {rec['code']} - {rec['sell_reason']} ({rec['sell_action']})")
        else:
            text_lines.append("📝 現在保有銘柄なし")
        
        # パフォーマンスサマリー
        text_lines.append(f"\n📊 【パフォーマンスサマリー】")
        text_lines.append("-" * 60)
        
        if 'overall_performance' in performance_report:
            perf = performance_report['overall_performance']
            text_lines.append(f"📅 集計期間: {performance_report['period_start']} ～ {performance_report['period_end']}")
            text_lines.append(f"🔄 総取引数: {perf['total_trades']}回")
            text_lines.append(f"✅ 勝率: {perf['win_rate']:.1%} ({perf['winning_trades']}勝 {perf['losing_trades']}敗)")
            text_lines.append(f"💰 累計損益: ¥{perf['total_profit_loss']:+,}")
            text_lines.append(f"📊 平均損益: ¥{perf['avg_profit_loss']:+,.0f}")
            text_lines.append(f"📅 平均保有日数: {perf['avg_holding_days']:.1f}日")
            text_lines.append(f"📈 リターン率: {perf['total_return_pct']:+.2%}")
        else:
            text_lines.append(performance_report.get('message', 'データなし'))
        
        text_lines.append("\n" + "=" * 100)
        text_lines.append("🤖 AI株式取引システム by Claude")
        text_lines.append("=" * 100)
        
        return "\n".join(text_lines)


def create_sample_data():
    """サンプルデータ作成（テスト用）"""
    sample_predictions = {
        '7203': {'confidence': 0.75, 'direction': 'UP', 'company_name': 'トヨタ自動車'},
        '9984': {'confidence': 0.68, 'direction': 'UP', 'company_name': 'ソフトバンクG'},
        '6758': {'confidence': 0.62, 'direction': 'UP', 'company_name': 'ソニーG'},
        '8306': {'confidence': 0.58, 'direction': 'UP', 'company_name': '三菱UFJ'},
        '4063': {'confidence': 0.55, 'direction': 'UP', 'company_name': '信越化学'},
    }
    
    sample_prices = {
        '7203': 2800,
        '9984': 5200,
        '6758': 12500,
        '8306': 1250,
        '4063': 28000,
    }
    
    # 正しい形式のポートフォリオデータ（リスト形式）
    sample_portfolio = [
        {
            'code': '7203',
            'shares': 100,
            'buy_price': 2750,
            'buy_date': '2025-08-25',
            'company_name': 'トヨタ自動車'
        },
        {
            'code': '9984',
            'shares': 50,
            'buy_price': 5100,
            'buy_date': '2025-08-20',
            'company_name': 'ソフトバンクG'
        }
    ]
    
    sample_trades = [
        {
            'date': '2025-08-30',
            'action': 'SELL',
            'code': '6758',
            'profit_loss': 15000,
            'days_held': 8
        },
        {
            'date': '2025-08-28',
            'action': 'SELL', 
            'code': '8306',
            'profit_loss': -5000,
            'days_held': 5
        }
    ]
    
    return sample_predictions, sample_prices, sample_portfolio, sample_trades


def main():
    """テスト実行"""
    logger.info("🧪 レポート生成テスト実行中...")
    
    generator = ProductionReportGenerator()
    
    # サンプルデータ作成
    predictions, prices, portfolio, trades = create_sample_data()
    
    # レポート生成
    buy_report = generator.generate_buy_recommendations(predictions, prices)
    portfolio_report = generator.generate_portfolio_management_report(portfolio, prices)
    performance_report = generator.generate_performance_summary(trades)
    
    # ファイル保存（Markdownのみ）
    markdown_file = generator.save_reports_to_files(buy_report, portfolio_report, performance_report)
    
    print(f"\n✅ テスト完了!")
    print(f"📊 Markdownレポート: {markdown_file}")
    print(f"📊 レポート内容をご確認ください")

if __name__ == "__main__":
    main()