#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日次自動取引システム - ワンコマンド実行
毎日必要な処理を順番に自動実行します

実行方法: python daily_trading_automation.py
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent))
from utils.market_calendar import JapanMarketCalendar

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('daily_automation.log')
    ]
)
logger = logging.getLogger(__name__)

class DailyTradingAutomation:
    """日次取引自動化クラス"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.base_dir = Path(__file__).parent
        self.success_count = 0
        self.total_steps = 5
        
        # 分析対象日を決定
        self.target_date = JapanMarketCalendar.get_target_date_for_analysis(self.start_time)
        self.next_date = JapanMarketCalendar.get_next_market_day(self.target_date)
        
        logger.info("="*60)
        logger.info("🚀 日次AI株式取引システム 自動実行開始")
        logger.info(f"⏰ 実行開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 分析対象日: {self.target_date}")
        logger.info(f"🎯 推奨取引日: {self.next_date.strftime('%Y-%m-%d')}")
        logger.info("="*60)
    
    def run_command(self, command: str, description: str, timeout: int = 1800) -> bool:
        """コマンド実行"""
        logger.info(f"\n📊 STEP {self.success_count + 1}/{self.total_steps}: {description}")
        logger.info(f"🔧 実行コマンド: {command}")
        
        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=timeout
            )
            
            elapsed = time.time() - start
            logger.info(f"✅ {description} 完了 (実行時間: {elapsed:.1f}秒)")
            
            # 出力があれば最後の数行を表示
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    logger.info("📄 実行結果 (最後の5行):")
                    for line in lines[-5:]:
                        logger.info(f"   {line}")
                else:
                    logger.info("📄 実行結果:")
                    for line in lines:
                        logger.info(f"   {line}")
            
            self.success_count += 1
            return True
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.error(f"⏰ {description} タイムアウト (制限時間: {timeout}秒)")
            return False
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            logger.error(f"❌ {description} 実行エラー (実行時間: {elapsed:.1f}秒)")
            logger.error(f"   エラーコード: {e.returncode}")
            if e.stderr:
                logger.error(f"   エラー内容: {e.stderr}")
            return False
            
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"💥 {description} 予期しないエラー (実行時間: {elapsed:.1f}秒)")
            logger.error(f"   エラー: {str(e)}")
            return False
    
    def run_daily_automation(self):
        """日次自動化実行"""
        results = []
        
        # STEP 1: データヘルスチェック
        success = self.run_command(
            "python data_management/data_health_check.py --registry data/data_registry.json",
            "データヘルスチェック",
            timeout=180
        )
        results.append(("データヘルスチェック", success))

        if not success:
            self.show_summary(results)
            return

        # STEP 2: 日経225全銘柄データ取得
        success = self.run_command(
            "python data_management/nikkei225_complete_parallel_fetcher.py",
            "日経225全銘柄データ取得 (J-Quants API)",
            timeout=900  # 15分
        )
        results.append(("データ取得", success))

        # STEP 3: 外部市場データ統合
        success = self.run_command(
            "python data_management/enhanced_data_integration.py",
            "外部市場データ統合 (USD/JPY, VIX, S&P500等)",
            timeout=300  # 5分
        )
        results.append(("外部データ統合", success))
        
        # STEP 4: Enhanced V3 AI予測実行
        success = self.run_command(
            "python systems/enhanced_close_return_system_v1.py",
            "Close-to-Close V1 AI予測システム実行 (終値→終値)",
            timeout=1200  # 20分
        )
        results.append(("AI予測", success))

        # STEP 5: 日次推奨銘柄レポート生成
        success = self.run_command(
            "python reports/daily_stock_recommendation_close_v1.py",
            "終値ベース推奨銘柄レポート生成",
            timeout=300  # 5分
        )
        results.append(("推奨レポート", success))
        
        # 実行結果サマリー
        self.show_summary(results)
    
    def show_summary(self, results):
        """実行結果サマリー表示"""
        total_time = datetime.now() - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("📊 日次自動実行 完了サマリー")
        logger.info("="*60)
        
        logger.info(f"⏰ 開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏰ 終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  総実行時間: {str(total_time).split('.')[0]}")
        logger.info(f"📊 分析対象日: {self.target_date}")
        logger.info(f"🎯 推奨取引日: {self.next_date.strftime('%Y-%m-%d')}")
        
        logger.info(f"\n📋 実行結果: {self.success_count}/{self.total_steps} 成功")
        
        for i, (step_name, success) in enumerate(results, 1):
            status = "✅ 成功" if success else "❌ 失敗"
            logger.info(f"   STEP {i}: {step_name} - {status}")
        
        if self.success_count == self.total_steps:
            logger.info("\n🎉 全ステップが正常に完了しました！")
            logger.info(f"📈 {self.next_date.strftime('%Y-%m-%d')}の推奨銘柄レポートを確認してください")
            logger.info(f"📂 レポート: production_reports/{self.target_date.strftime('%Y-%m')}/{self.target_date}.md")
        else:
            failed_count = self.total_steps - self.success_count
            logger.warning(f"\n⚠️  {failed_count}個のステップで問題が発生しました")
            logger.warning("🔧 failed ステップを個別に実行して確認してください")
        
        logger.info("="*60)
    
    def check_environment(self):
        """環境確認"""
        logger.info("🔍 実行環境確認中...")
        
        # .envファイル確認
        env_file = self.base_dir / '.env'
        if not env_file.exists():
            logger.error("❌ .envファイルが見つかりません")
            logger.error("   J-Quants認証情報を.envファイルに設定してください")
            return False
        
        # 必要なディレクトリ確認
        required_dirs = [
            'data_management', 
            'systems', 
            'reports',
            'production_reports'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                logger.error(f"❌ 必要なディレクトリが見つかりません: {dir_name}")
                return False
        
        logger.info("✅ 実行環境確認完了")
        return True

def main():
    """メイン実行"""
    automation = DailyTradingAutomation()
    
    # 環境確認
    if not automation.check_environment():
        logger.error("❌ 環境確認に失敗しました。セットアップを確認してください。")
        sys.exit(1)
    
    # 日次自動実行
    try:
        automation.run_daily_automation()
    except KeyboardInterrupt:
        logger.info("\n🛑 ユーザーにより実行が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 予期しないエラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
