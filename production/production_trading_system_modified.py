#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
運用株式取引システム（本番用）
1つのコマンドで以下を実行:
1. 最新データ取得・更新
2. AI予測実行
3. 売買推奨レポート生成
4. 保有銘柄管理レポート生成
"""

import sys
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
import argparse

# 自作モジュール
from production_reports import ProductionReportGenerator
from stock_info_utils import get_multiple_company_names

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProductionTradingSystem:
    """運用取引システムメインクラス"""
    
    def __init__(self, config_path="production_config.yaml", execution_date=None):
        self.config_path = Path(config_path)
        self.execution_date = execution_date or datetime.now()
        if isinstance(self.execution_date, str):
            self.execution_date = datetime.strptime(self.execution_date, "%Y%m%d")
        
        self.load_config()
        self.setup_paths()
        self.report_generator = ProductionReportGenerator(config_path, execution_date=self.execution_date)
        
    def load_config(self):
        """設定ファイル読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # パラメータ展開
        self.optimal_params = self.config['optimal_params']
        self.initial_capital = self.config['system']['initial_capital']
        self.max_positions = self.config['system']['max_positions']
        self.confidence_threshold = self.config['system']['confidence_threshold']
        self.transaction_cost_rate = self.config['system']['transaction_cost_rate']
        self.optimal_features = self.config['features']['optimal_features']
        
        logger.info(f"✅ 設定ファイル読み込み完了: {self.config_path}")
        logger.info(f"🎯 運用パラメータ: {self.optimal_params['hold_days']}日保有, {self.optimal_params['profit_target']:.1%}利確, {self.optimal_params['stop_loss']:.1%}損切")
        
    def setup_paths(self):
        """パス設定"""
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / self.config['data']['processed_dir'].replace('data/', '')
        self.integrated_file = self.processed_dir / self.config['data']['integrated_file']
        
        # 運用データディレクトリ
        self.production_data_dir = Path("production_data")
        self.production_data_dir.mkdir(exist_ok=True)
        
        # ポートフォリオファイル
        self.portfolio_file = self.production_data_dir / "current_portfolio.json"
        self.trades_file = self.production_data_dir / "trade_history.json"
        
    def load_data_for_date(self):
        """指定日付に対応するデータ読み込み"""
        logger.info("📊 データ読み込み中...")
        
        if not self.integrated_file.exists():
            logger.error(f"❌ データファイルが見つかりません: {self.integrated_file}")
            logger.error("💡 先にデータ収集スクリプトを実行してください")
            return None
        
        df = pd.read_parquet(self.integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 基準日（実行日）を設定
        base_date = self.execution_date.strftime('%Y-%m-%d')
        base_date_pd = pd.to_datetime(base_date)
        
        # 基準日以前のデータのみ使用（学習用）
        training_df = clean_df[pd.to_datetime(clean_df['Date']) <= base_date_pd].copy()
        
        if len(training_df) == 0:
            logger.error(f"❌ {base_date}以前のデータが見つかりません")
            return None
        
        # 基準日に最も近い日付のデータを予測用として抽出
        actual_latest_date = training_df['Date'].max()
        latest_data = training_df[training_df['Date'] == actual_latest_date]
        
        logger.info(f"✅ データ読み込み完了: 基準日 {base_date}, 学習データ最新日 {actual_latest_date}, {len(latest_data)}銘柄")
        logger.info(f"📚 学習データ期間: {training_df['Date'].min()} ～ {actual_latest_date} ({len(training_df)}件)")
        
        return training_df, latest_data, actual_latest_date
    
    def train_prediction_model(self, df):
        """AI予測モデル学習"""
        logger.info("🤖 AI予測モデル学習中...")
        
        # 特徴量選択
        feature_cols = self.optimal_features
        
        # 学習データ準備
        train_data = df[df[feature_cols].notna().all(axis=1)].copy()
        
        if len(train_data) < 100:
            logger.warning("⚠️  学習データが不足しています")
            return None, None
        
        X = train_data[feature_cols]
        y = train_data['Binary_Direction']
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # モデル学習
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_scaled, y)
        
        logger.info(f"✅ モデル学習完了: {len(train_data)}件のデータで学習")
        
        return model, scaler
    
    def make_predictions(self, model, scaler, latest_data):
        """最新データでAI予測実行"""
        logger.info("🎯 AI予測実行中...")
        
        feature_cols = self.optimal_features
        
        # 予測対象データ準備
        pred_data = latest_data[latest_data[feature_cols].notna().all(axis=1)].copy()
        
        if len(pred_data) == 0:
            logger.warning("⚠️  予測対象データがありません")
            return pd.DataFrame()
        
        X_pred = pred_data[feature_cols]
        X_pred_scaled = scaler.transform(X_pred)
        
        # 予測実行
        predictions = model.predict(X_pred)
        probabilities = model.predict_proba(X_pred)
        
        # 予測結果を追加
        pred_data['prediction'] = predictions
        pred_data['confidence'] = np.max(probabilities, axis=1)
        pred_data['predicted_direction'] = np.where(predictions == 1, 'UP', 'DOWN')
        
        # 信頼度フィルタリング
        # 市場環境が悪い場合は闾値を調整
        vix_change = pred_data['vix_change'].iloc[0] if 'vix_change' in pred_data.columns else 0
        adjusted_threshold = self.confidence_threshold
        if vix_change > 0.1:  # VIXが10%以上上昇
            adjusted_threshold = max(0.40, self.confidence_threshold - 0.10)
        
        high_confidence = pred_data[
            (pred_data['confidence'] >= adjusted_threshold) & 
            (pred_data['predicted_direction'] == 'UP')
        ].copy()
        
        logger.info(f"✅ 予測完了: {len(pred_data)}銘柄中 {len(high_confidence)}銘柄が購入候補")
        
        return high_confidence
    
    def load_current_portfolio(self):
        """現在のポートフォリオ読み込み"""
        if not self.portfolio_file.exists():
            # 初期ポートフォリオ作成
            portfolio = {
                'cash_balance': self.initial_capital,
                'positions': [],
                'last_updated': datetime.now().isoformat()
            }
            self.save_portfolio(portfolio)
            return portfolio
        
        with open(self.portfolio_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_portfolio(self, portfolio):
        """ポートフォリオ保存"""
        portfolio['last_updated'] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)
    
    def load_trade_history(self):
        """取引履歴読み込み"""
        if not self.trades_file.exists():
            return []
        
        with open(self.trades_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_trade_history(self, trades):
        """取引履歴保存"""
        with open(self.trades_file, 'w', encoding='utf-8') as f:
            json.dump(trades, f, ensure_ascii=False, indent=2)
    
    def run_full_analysis(self):
        """フル分析実行"""
        logger.info("🚀 フル分析を開始します...")
        
        try:
            # 1. データ読み込み
            data_result = self.load_data_for_date()
            if data_result is None:
                return False
            
            training_df, latest_data, actual_latest_date = data_result
            
            # 2. AI予測モデル学習
            model, scaler = self.train_prediction_model(training_df)
            if model is None:
                return False
            
            # 3. 予測実行
            predictions = self.make_predictions(model, scaler, latest_data)
            
            # 4. ポートフォリオ読み込み
            portfolio = self.load_current_portfolio()
            trade_history = self.load_trade_history()
            
            # 5. レポート生成
            logger.info("📊 レポート生成中...")
            
            # レポート生成用のデータを準備
            buy_recommendations = self.prepare_buy_recommendations(predictions)
            portfolio_management = self.prepare_portfolio_management(portfolio, latest_data)
            performance_data = self.prepare_performance_data(trade_history)
            
            # レポート生成
            markdown_file, text_file, json_file = self.report_generator.save_reports_to_files(
                buy_recommendations, portfolio_management, performance_data
            )
            
            logger.info("✅ フル分析完了!")
            logger.info(f"📁 レポートファイル:")
            logger.info(f"  📝 Markdown: {markdown_file}")
            logger.info(f"  📄 テキスト: {text_file}")
            logger.info(f"  📊 JSON: {json_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 分析実行エラー: {e}")
            import traceback
            logger.error(f"詳細: {traceback.format_exc()}")
            return False
    
    def prepare_buy_recommendations(self, predictions):
        """購入推奨データ準備"""
        if len(predictions) == 0:
            return {
                'report_type': '購入推奨レポート',
                'generated_at': self.execution_date.isoformat(),
                'market_date': self.execution_date.strftime('%Y-%m-%d'),
                'total_recommendations': 0,
                'available_capital': self.initial_capital,
                'max_positions': self.max_positions,
                'parameters': self.optimal_params,
                'recommendations': [],
                'summary': {
                    'total_investment': 0,
                    'average_confidence': 0,
                    'cash_remaining': self.initial_capital
                }
            }
        
        # 信頼度でソート
        sorted_predictions = predictions.sort_values('confidence', ascending=False).head(self.max_positions)
        
        # 会社名を一括取得
        stock_codes = sorted_predictions['Code'].astype(str).tolist()
        company_names = get_multiple_company_names(stock_codes)
        
        recommendations = []
        total_investment = 0
        
        for _, row in sorted_predictions.iterrows():
            # 投資額計算（分散投資）
            investment_per_position = self.initial_capital / self.max_positions
            current_price = row.get('Close', 1000)  # デフォルト価格
            recommended_shares = int(investment_per_position / current_price)
            actual_investment = recommended_shares * current_price
            
            if actual_investment > 0:
                stock_code = str(row['Code'])
                recommendation = {
                    'code': stock_code,
                    'company_name': company_names.get(stock_code, f"銘柄{stock_code}"),
                    'current_price': int(current_price),
                    'confidence': float(row['confidence']),
                    'predicted_direction': row['predicted_direction'],
                    'recommended_shares': recommended_shares,
                    'investment_amount': int(actual_investment),
                    'profit_target_price': int(current_price * (1 + self.optimal_params['profit_target'])),
                    'stop_loss_price': int(current_price * (1 - self.optimal_params['stop_loss'])),
                    'expected_hold_days': self.optimal_params['hold_days']
                }
                recommendations.append(recommendation)
                total_investment += actual_investment
        
        return {
            'report_type': '購入推奨レポート',
            'generated_at': self.execution_date.isoformat(),
            'market_date': self.execution_date.strftime('%Y-%m-%d'),
            'total_recommendations': len(recommendations),
            'available_capital': self.initial_capital,
            'max_positions': self.max_positions,
            'parameters': self.optimal_params,
            'recommendations': recommendations,
            'summary': {
                'total_investment': int(total_investment),
                'average_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations) if recommendations else 0,
                'cash_remaining': self.initial_capital - int(total_investment)
            }
        }
    
    def prepare_portfolio_management(self, portfolio, latest_data):
        """ポートフォリオ管理データ準備"""
        positions = portfolio.get('positions', [])
        
        # 保有銘柄の会社名を一括取得
        if positions:
            position_codes = [str(pos.get('code', pos.get('Code', ''))) for pos in positions]
            company_names = get_multiple_company_names(position_codes)
        else:
            company_names = {}
        
        # 各ポジションの現在価値を更新
        updated_positions = []
        sell_recommendations = []
        
        for position in positions:
            # 最新価格取得（実際の実装では最新データから取得）
            current_price = 1000  # デフォルト価格（実装時は latest_data から取得）
            
            # 損益計算
            buy_price = position['buy_price']
            shares = position['shares']
            current_value = shares * current_price
            cost_basis = shares * buy_price
            unrealized_pl = current_value - cost_basis
            
            # 売却判定
            buy_date = datetime.fromisoformat(position['buy_date'])
            days_held = (self.execution_date - buy_date).days
            
            sell_action = "保有継続"
            sell_reason = "条件未達"
            
            # 売却条件チェック
            if days_held >= self.optimal_params['hold_days']:
                sell_action = "売却推奨"
                sell_reason = "期間満了"
            elif current_price >= buy_price * (1 + self.optimal_params['profit_target']):
                sell_action = "売却推奨"
                sell_reason = "利確"
            elif current_price <= buy_price * (1 - self.optimal_params['stop_loss']):
                sell_action = "売却推奨"
                sell_reason = "損切"
            
            # 会社名を追加
            stock_code = str(position.get('code', position.get('Code', '')))
            
            updated_position = {
                **position,
                'company_name': company_names.get(stock_code, f"銘柄{stock_code}"),
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'unrealized_pl': unrealized_pl,
                'unrealized_pl_pct': unrealized_pl / cost_basis if cost_basis > 0 else 0,
                'days_held': days_held,
                'sell_action': sell_action,
                'sell_reason': sell_reason
            }
            
            updated_positions.append(updated_position)
            
            if sell_action == "売却推奨":
                sell_recommendations.append(updated_position)
        
        # ポートフォリオサマリー
        total_portfolio_value = sum(pos['current_value'] for pos in updated_positions)
        total_unrealized_pl = sum(pos['unrealized_pl'] for pos in updated_positions)
        cash_balance = portfolio.get('cash_balance', self.initial_capital)
        total_value = total_portfolio_value + cash_balance
        
        return {
            'report_type': '保有銘柄管理レポート',
            'generated_at': self.execution_date.isoformat(),
            'portfolio_date': self.execution_date.strftime('%Y-%m-%d'),
            'total_positions': len(updated_positions),
            'portfolio_summary': {
                'total_portfolio_value': total_portfolio_value,
                'cash_balance': cash_balance,
                'total_value': total_value,
                'total_unrealized_pl': total_unrealized_pl,
                'total_unrealized_pl_pct': total_unrealized_pl / total_value if total_value > 0 else 0,
                'portfolio_weight': total_portfolio_value / total_value if total_value > 0 else 0
            },
            'positions': updated_positions,
            'sell_recommendations': sell_recommendations,
            'parameters': self.optimal_params
        }
    
    def prepare_performance_data(self, trade_history):
        """パフォーマンスデータ準備"""
        if len(trade_history) == 0:
            # パフォーマンス期間
            period_end = self.execution_date.strftime('%Y-%m-%d')
            period_start = (self.execution_date - timedelta(days=7)).strftime('%Y-%m-%d')
            
            return {
                'report_type': 'パフォーマンスサマリー',
                'generated_at': self.execution_date.isoformat(),
                'period_start': period_start,
                'period_end': period_end,
                'overall_performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit_loss': 0,
                    'avg_profit_loss': 0,
                    'avg_holding_days': 0,
                    'total_return_pct': 0
                },
                'recent_trades': [],
                'parameters_used': self.optimal_params
            }
        
        # 最近の取引のみ分析（直近1週間）
        recent_date = self.execution_date - timedelta(days=7)
        recent_trades = [
            trade for trade in trade_history 
            if datetime.fromisoformat(trade.get('date', '2020-01-01')) >= recent_date
        ]
        
        if len(recent_trades) == 0:
            recent_trades = trade_history[-5:]  # 最新5件
        
        # 統計計算
        total_trades = len(recent_trades)
        winning_trades = len([t for t in recent_trades if t.get('profit_loss', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit_loss = sum(t.get('profit_loss', 0) for t in recent_trades)
        avg_profit_loss = total_profit_loss / total_trades if total_trades > 0 else 0
        avg_holding_days = sum(t.get('days_held', 0) for t in recent_trades) / total_trades if total_trades > 0 else 0
        
        # パフォーマンス期間
        period_end = self.execution_date.strftime('%Y-%m-%d')
        period_start = (self.execution_date - timedelta(days=7)).strftime('%Y-%m-%d')
        
        return {
            'report_type': 'パフォーマンスサマリー',
            'generated_at': self.execution_date.isoformat(),
            'period_start': period_start,
            'period_end': period_end,
            'overall_performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit_loss': int(total_profit_loss),
                'avg_profit_loss': avg_profit_loss,
                'avg_holding_days': avg_holding_days,
                'total_return_pct': total_profit_loss / self.initial_capital if self.initial_capital > 0 else 0
            },
            'recent_trades': recent_trades,
            'parameters_used': self.optimal_params
        }
    
    def run_test_mode(self):
        """テストモード実行"""
        logger.info("🧪 テストモード実行中...")
        
        # テスト用データでレポート生成
        from production_reports import main as test_reports
        test_reports()
        
        return True

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='🤖 AI株式取引システム')
    parser.add_argument('command', nargs='?', default='run', 
                       help='実行コマンド (run, test)')
    parser.add_argument('--date', '-d', 
                       help='実行日付 (YYYYMMDD形式, 例: 20250901)')
    parser.add_argument('--help-detailed', action='store_true',
                       help='詳細ヘルプを表示')
    
    args = parser.parse_args()
    
    if args.help_detailed:
        print("🤖 AI株式取引システム 使用方法:")
        print("")
        print("基本コマンド:")
        print("  python production_trading_system.py                    # フル分析実行")
        print("  python production_trading_system.py test               # テストモード")
        print("  python production_trading_system.py --date 20250901    # 指定日付で実行")
        print("  python production_trading_system.py test --date 20250901 # 指定日付でテスト")
        print("")
        print("📊 実行内容:")
        print("  1. 最新データ読み込み")
        print("  2. AI予測モデル学習・予測")
        print("  3. 購入推奨レポート生成")
        print("  4. 保有銘柄管理レポート生成")
        print("  5. パフォーマンス分析")
        print("")
        print("日付引数について:")
        print("  --date オプションで指定した日付でレポートが生成されます")
        print("  形式: YYYYMMDD (例: 20250901)")
        print("  指定しない場合は現在日時で実行されます")
        return
    
    execution_date = args.date
    
    if args.command == "test":
        print("🧪 テストモード実行中...")
        if execution_date:
            print(f"📅 指定日付: {execution_date}")
        
        # テスト用データでレポート生成
        system = ProductionTradingSystem(execution_date=execution_date)
        success = system.run_test_mode()
        
        if success:
            print("\n✅ テスト完了!")
        else:
            print("\n❌ テストでエラーが発生しました")
        return
    
    # メインシステム実行
    if execution_date:
        print(f"🚀 AI株式取引システム - フル分析を開始します... (日付: {execution_date})")
    else:
        print("🚀 AI株式取引システム - フル分析を開始します...")
        
    system = ProductionTradingSystem(execution_date=execution_date)
    success = system.run_full_analysis()
    
    if success:
        print("\n✅ フル分析完了!")
        print("📊 生成されたレポートを確認して、楽天証券で売買を実行してください")
    else:
        print("\n❌ 分析でエラーが発生しました")
        print("🔍 ログを確認して問題を解決してください")


if __name__ == "__main__":
    main()