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
    
    def __init__(self, config_path="production_config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_paths()
        self.report_generator = ProductionReportGenerator(config_path)
        
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
        
    def setup_paths(self):
        """パス設定"""
        self.data_dir = Path(self.config['data']['processed_dir'])
        self.integrated_file = self.data_dir / self.config['data']['integrated_file']
        
        # ディレクトリ作成
        for dir_path in [self.data_dir, Path('production_data'), Path('production_reports')]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def update_data_from_api(self):
        """J-Quants APIから最新データを取得して更新"""
        logger.info("🔄 最新データ取得中...")
        
        try:
            # daily_update.pyを実行して最新データを取得
            import subprocess
            result = subprocess.run(
                ['python', 'scripts/daily_update.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ データ更新完了")
                return True
            else:
                logger.warning(f"⚠️ データ更新に失敗: {result.stderr}")
                logger.info("📊 既存データを使用します")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ データ更新エラー: {e}")
            logger.info("📊 既存データを使用します")
            return False

    def load_latest_data(self):
        """最新データ読み込み"""
        logger.info(f"📥 統合データ読み込み: {self.integrated_file}")
        
        if not self.integrated_file.exists():
            logger.error(f"❌ 統合データファイルが見つかりません: {self.integrated_file}")
            logger.info("💡 まず daily_update.py を実行してデータを更新してください")
            return None
        
        # データ読み込み
        df = pd.read_parquet(self.integrated_file)
        logger.info(f"✅ データ読み込み完了: {len(df):,}レコード")
        
        # Targetカラムを生成（Binary_Directionを使用）
        if 'Target' not in df.columns and 'Binary_Direction' in df.columns:
            df['Target'] = df['Binary_Direction']
        
        # 必要な列チェック（Code列をStock列として扱う）
        if 'Stock' not in df.columns and 'Code' in df.columns:
            df['Stock'] = df['Code']
        
        required_cols = ['Date', 'Stock', 'Close', 'Volume', 'Target'] + self.optimal_features
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ 必要な列が不足: {missing_cols}")
            return None
        
        # 欠損値処理
        clean_df = df[required_cols].dropna()
        logger.info(f"✅ 有効データ: {len(clean_df):,}レコード")
        
        if len(clean_df) == 0:
            logger.error("❌ 有効なデータがありません")
            return None
        
        # 日付でソート
        clean_df['Date'] = pd.to_datetime(clean_df['Date'])
        clean_df = clean_df.sort_values('Date')
        
        # 最新日付のデータ取得
        latest_date = clean_df['Date'].max()
        latest_data = clean_df[clean_df['Date'] == latest_date]
        historical_data = clean_df[clean_df['Date'] < latest_date]
        
        logger.info(f"📅 最新データ日付: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"📊 最新データ: {len(latest_data)}銘柄")
        logger.info(f"📊 履歴データ: {len(historical_data):,}レコード")
        
        return historical_data, latest_data, latest_date
    
    def train_prediction_model(self, historical_data):
        """AIモデル学習"""
        logger.info("🤖 AI予測モデル学習中...")
        
        # 特徴量とターゲット準備
        X = historical_data[self.optimal_features]
        y = historical_data['Target']
        
        # スケーリング
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # モデル学習
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        # 学習精度確認
        train_score = self.model.score(X_scaled, y)
        logger.info(f"✅ モデル学習完了 (精度: {train_score:.2%})")
    
    def generate_predictions(self, latest_data):
        """予測実行"""
        logger.info("🔮 AI予測実行中...")
        
        # 特徴量準備
        X = latest_data[self.optimal_features]
        X_scaled = self.scaler.transform(X)
        
        # 予測実行
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # 結果まとめ
        results = pd.DataFrame({
            'stock_code': latest_data['Stock'].values,
            'confidence': predictions,
            'buy_signal': predictions >= self.confidence_threshold
        })
        
        # 信頼度でソート
        results = results.sort_values('confidence', ascending=False)
        
        # 会社名取得
        stock_codes = results['stock_code'].tolist()
        company_names = get_multiple_company_names(stock_codes)
        results['company_name'] = results['stock_code'].map(company_names)
        
        buy_recommendations = results[results['buy_signal']].head(self.max_positions)
        
        logger.info(f"✅ 予測完了: {len(buy_recommendations)}銘柄が購入推奨基準を満たしました")
        
        return buy_recommendations
    
    def get_current_prices(self, latest_data):
        """最新価格取得"""
        price_dict = {}
        for _, row in latest_data.iterrows():
            price_dict[str(row['Stock'])] = row['Close']
        return price_dict
    
    def load_current_portfolio(self):
        """現在のポートフォリオ読み込み"""
        portfolio_file = Path('production_data') / 'current_portfolio.json'
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                data = json.load(f)
                # 新しい形式の場合はpositionsを返す
                if isinstance(data, dict) and 'positions' in data:
                    return data['positions']
                # 古い形式の場合はそのまま返す
                return data
        return []
    
    def load_trade_history(self):
        """取引履歴読み込み"""
        history_file = Path('production_data') / 'trade_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def simulate_automatic_sells(self, current_portfolio, current_prices):
        """自動売却シミュレーション"""
        sell_recommendations = []
        
        for position in current_portfolio:
            code = str(position['code'])
            buy_price = position['buy_price']
            buy_date = datetime.fromisoformat(position['buy_date'])
            
            # 現在価格取得
            if code not in current_prices:
                continue
            
            current_price = current_prices[code]
            
            # 保有日数と損益率計算
            days_held = (datetime.now() - buy_date).days
            profit_rate = (current_price - buy_price) / buy_price
            
            # 売却条件チェック
            sell_reason = None
            priority = 0
            
            if days_held >= self.optimal_params['hold_days']:
                sell_reason = "保有期間満了"
                priority = 1
            elif days_held >= self.optimal_params['hold_days'] - 1:
                sell_reason = "保有期間まもなく満了"
                priority = 2
            elif profit_rate <= -self.optimal_params['stop_loss']:
                sell_reason = "損切り"
                priority = 3
            elif profit_rate >= self.optimal_params['profit_target']:
                sell_reason = "利確"
                priority = 3
            
            if sell_reason:
                sell_value = position['shares'] * current_price
                transaction_cost = sell_value * self.transaction_cost_rate
                net_proceeds = sell_value - transaction_cost
                profit_loss = net_proceeds - (position['shares'] * position['buy_price'])
                
                sell_recommendations.append({
                    'code': code,
                    'company_name': position.get('company_name', f'株式会社{code}'),
                    'shares': position['shares'],
                    'buy_price': buy_price,
                    'current_price': current_price,
                    'sell_reason': sell_reason,
                    'priority': priority,
                    'days_held': days_held,
                    'profit_rate': profit_rate,
                    'estimated_proceeds': net_proceeds,
                    'estimated_profit_loss': profit_loss
                })
        
        # 優先度順にソート
        sell_recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"✅ 売却条件チェック完了: {len(sell_recommendations)}銘柄が売却対象")
        
        return sell_recommendations
    
    def run_full_analysis(self):
        """フル分析実行"""
        logger.info("🚀 運用システム開始")
        logger.info("="*80)
        
        # 1. 最新データを取得
        self.update_data_from_api()
        
        # 2. データ読み込み
        data_result = self.load_latest_data()
        if data_result is None:
            return False
        
        historical_data, latest_data, latest_date = data_result
        
        # 3. モデル学習
        self.train_prediction_model(historical_data)
        
        # 4. 予測実行
        predictions = self.generate_predictions(latest_data)
        
        # 5. 現在価格取得
        current_prices = self.get_current_prices(latest_data)
        
        # 6. ポートフォリオ・履歴読み込み
        current_portfolio = self.load_current_portfolio()
        trade_history = self.load_trade_history()
        
        # 7. 売却推奨チェック
        sell_recommendations = self.simulate_automatic_sells(current_portfolio, current_prices)
        
        # 8. レポート生成
        logger.info("📊 レポート生成中...")
        
        buy_report = self.report_generator.generate_buy_recommendations(predictions, current_prices)
        portfolio_report = self.report_generator.generate_portfolio_management_report(current_portfolio, current_prices)
        performance_report = self.report_generator.generate_performance_summary(trade_history)
        
        # 9. レポート保存
        markdown_file = self.report_generator.save_reports_to_files(
            buy_report, portfolio_report, performance_report
        )
        
        # 10. コンソール出力
        self.print_summary(buy_report, portfolio_report, sell_recommendations)
        
        logger.info("="*80)
        logger.info(f"✅ 運用分析完了!")
        logger.info(f"📁 詳細レポート: {markdown_file}")
        
        return True
    
    def print_summary(self, buy_report, portfolio_report, sell_recommendations):
        """サマリー情報をコンソール出力"""
        print("\n" + "="*80)
        print("🤖 AI株式取引システム - 運用サマリー")
        print("="*80)
        
        # 購入推奨
        print(f"\n📈 【購入推奨】")
        if buy_report['total_recommendations'] > 0:
            print(f"   推奨銘柄数: {buy_report['total_recommendations']}銘柄")
            print(f"   投資予定額: ¥{buy_report['summary']['total_investment']:,}")
            print(f"   平均信頼度: {buy_report['summary']['average_confidence']:.1%}")
            
            print(f"\n   推奨銘柄:")
            for i, rec in enumerate(buy_report['recommendations'][:3], 1):
                print(f"   {i}. {rec['code']} - ¥{rec['current_price']:,} ({rec['confidence']:.1%}信頼度)")
        else:
            print("   ❌ 推奨銘柄なし")
        
        # 保有銘柄
        print(f"\n📊 【保有銘柄】")
        if portfolio_report['total_positions'] > 0:
            summary = portfolio_report['portfolio_summary']
            print(f"   保有銘柄数: {portfolio_report['total_positions']}銘柄")
            print(f"   評価損益: ¥{summary['total_unrealized_pl']:+,} ({summary['total_unrealized_pl_pct']:+.1%})")
            
            sell_recs = portfolio_report['sell_recommendations']
            if sell_recs:
                print(f"   🚨 売却推奨: {len(sell_recs)}銘柄")
                for rec in sell_recs[:3]:
                    print(f"      ▶ {rec['code']} - {rec['sell_reason']}")
        else:
            print("   📝 保有銘柄なし")
        
        # パラメータ情報
        print(f"\n⚙️  【運用設定】")
        print(f"   保有期間: {self.optimal_params['hold_days']}日")
        print(f"   利確閾値: {self.optimal_params['profit_target']:.1%}")
        print(f"   損切閾値: {self.optimal_params['stop_loss']:.1%}")
        print(f"   最大保有: {self.max_positions}銘柄")
        
        print("\n" + "="*80)
        print("💡 詳細な指値価格等は上記レポートファイルをご確認ください")
        print("="*80)


def main():
    """メイン実行関数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--help" or command == "-h":
            print("🤖 AI株式取引システム 使用方法:")
            print("")
            print("  python production_trading_system.py        # フル分析実行")
            print("  python production_trading_system.py test   # テストモード")
            print("  python production_trading_system.py --help # ヘルプ表示")
            print("")
            print("📊 実行内容:")
            print("  1. 最新データ読み込み")
            print("  2. AI予測モデル学習・予測")
            print("  3. 購入推奨レポート生成")
            print("  4. 保有銘柄管理レポート生成")
            print("  5. パフォーマンス分析")
            return
        
        elif command == "test":
            print("🧪 テストモード実行中...")
            # テスト用データでレポート生成
            from production_reports import main as test_reports
            test_reports()
            return
    
    # メインシステム実行
    system = ProductionTradingSystem()
    success = system.run_full_analysis()
    
    if success:
        print("\n🎉 運用分析が正常に完了しました!")
        print("📱 楽天証券での注文設定をお忘れなく!")
    else:
        print("\n❌ 運用分析でエラーが発生しました")
        print("💡 ログを確認して問題を解決してください")


if __name__ == "__main__":
    main()