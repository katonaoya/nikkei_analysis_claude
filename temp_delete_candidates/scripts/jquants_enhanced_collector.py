#!/usr/bin/env python3
"""
J-Quantsスタンダードプランでの最大データ活用
10年分のデータを収集して精度向上を図る
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from datetime import datetime, timedelta
from dateutil import tz
import time
import os
import warnings
warnings.filterwarnings('ignore')

# J-Quants APIクライアント
try:
    import jquantsapi
except ImportError:
    print("J-Quants APIクライアントをインストールしてください: pip install jquants-api-client")
    exit(1)

class JQuantsEnhancedCollector:
    """J-Quantsからの拡張データ収集"""
    
    def __init__(self, mail_address: str = None, password: str = None):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed" 
        self.jquants_dir = self.raw_dir / "jquants_enhanced"
        
        # ディレクトリ作成
        for dir_path in [self.raw_dir, self.processed_dir, self.jquants_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 認証情報
        self.mail_address = mail_address or os.getenv('JQUANTS_MAIL')
        self.password = password or os.getenv('JQUANTS_PASSWORD')
        
        if not self.mail_address or not self.password:
            logger.error("J-Quants認証情報が必要です（環境変数 JQUANTS_MAIL, JQUANTS_PASSWORD またはパラメータ指定）")
            
        self.client = None
    
    def initialize_client(self):
        """クライアント初期化"""
        try:
            logger.info("J-Quants APIクライアント初期化中...")
            self.client = jquantsapi.Client(
                mail_address=self.mail_address, 
                password=self.password
            )
            logger.info("✅ J-Quants認証成功")
            return True
        except Exception as e:
            logger.error(f"❌ J-Quants認証失敗: {e}")
            return False
    
    def collect_basic_stock_data(self, years_back: int = 10):
        """基本株価データ収集（10年分）"""
        logger.info(f"📊 基本株価データ収集開始（{years_back}年分）")
        
        # 期間設定
        end_date = datetime.now(tz=tz.gettz("Asia/Tokyo"))
        start_date = end_date - timedelta(days=years_back * 365)
        
        try:
            # 日次株価取得
            logger.info(f"期間: {start_date.date()} ～ {end_date.date()}")
            df_prices = self.client.get_price_range(
                start_dt=start_date, 
                end_dt=end_date
            )
            
            if df_prices is not None and len(df_prices) > 0:
                # データ保存
                output_file = self.jquants_dir / f"daily_prices_{years_back}years.parquet"
                df_prices.to_parquet(output_file)
                logger.info(f"✅ 日次株価データ保存: {len(df_prices):,}件 -> {output_file}")
                return df_prices
            else:
                logger.warning("⚠️ 株価データが取得できませんでした")
                return None
                
        except Exception as e:
            logger.error(f"❌ 株価データ取得エラー: {e}")
            return None
    
    def collect_indices_data(self, years_back: int = 10):
        """指数データ収集"""
        logger.info("📈 指数データ収集開始")
        
        # 主要指数リスト
        indices = ["TOPIX", "NIKKEI", "MOTHERS", "JASDAQ"]
        all_indices_data = []
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
        
        for index_code in indices:
            try:
                logger.info(f"指数データ取得中: {index_code}")
                df_index = self.client.get_indices(
                    start_date=start_date,
                    end_date=end_date,
                    index_code=index_code
                )
                
                if df_index is not None and len(df_index) > 0:
                    df_index['IndexCode'] = index_code
                    all_indices_data.append(df_index)
                    logger.info(f"  ✅ {index_code}: {len(df_index)}件")
                    time.sleep(0.1)  # API制限対策
                    
            except Exception as e:
                logger.warning(f"  ⚠️ {index_code}取得エラー: {e}")
        
        if all_indices_data:
            df_all_indices = pd.concat(all_indices_data, ignore_index=True)
            output_file = self.jquants_dir / f"indices_{years_back}years.parquet"
            df_all_indices.to_parquet(output_file)
            logger.info(f"✅ 指数データ保存: {len(df_all_indices)}件 -> {output_file}")
            return df_all_indices
        else:
            logger.warning("⚠️ 指数データが取得できませんでした")
            return None
    
    def collect_margin_credit_data(self, years_back: int = 10):
        """信用取引・空売りデータ収集"""
        logger.info("💳 信用取引・空売りデータ収集開始")
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
        
        collected_data = {}
        
        # 1. 週次信用残高
        try:
            logger.info("週次信用残高取得中...")
            df_weekly_margin = self.client.get_weekly_margin_range(
                start_date=start_date,
                end_date=end_date
            )
            if df_weekly_margin is not None and len(df_weekly_margin) > 0:
                collected_data['weekly_margin'] = df_weekly_margin
                logger.info(f"  ✅ 週次信用残高: {len(df_weekly_margin)}件")
        except Exception as e:
            logger.warning(f"  ⚠️ 週次信用残高エラー: {e}")
        
        # 2. 空売り比率
        try:
            logger.info("空売り比率取得中...")
            df_short_selling = self.client.get_short_selling_range(
                start_date=start_date,
                end_date=end_date
            )
            if df_short_selling is not None and len(df_short_selling) > 0:
                collected_data['short_selling'] = df_short_selling
                logger.info(f"  ✅ 空売り比率: {len(df_short_selling)}件")
        except Exception as e:
            logger.warning(f"  ⚠️ 空売り比率エラー: {e}")
        
        # 3. 空売り残高（最近のデータのみ）
        try:
            logger.info("空売り残高取得中...")
            recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            df_short_positions = self.client.get_short_selling_positions_range(
                start_date=recent_date,
                end_date=end_date
            )
            if df_short_positions is not None and len(df_short_positions) > 0:
                collected_data['short_positions'] = df_short_positions
                logger.info(f"  ✅ 空売り残高: {len(df_short_positions)}件")
        except Exception as e:
            logger.warning(f"  ⚠️ 空売り残高エラー: {e}")
        
        # データ保存
        for data_name, df in collected_data.items():
            output_file = self.jquants_dir / f"{data_name}_{years_back}years.parquet"
            df.to_parquet(output_file)
            logger.info(f"✅ {data_name}保存: {output_file}")
        
        return collected_data
    
    def collect_options_data(self, years_back: int = 2):
        """オプションデータ収集（2年分、データ量を考慮）"""
        logger.info("📊 オプションデータ収集開始")
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
        
        try:
            df_options = self.client.get_index_option_range(
                start_date=start_date,
                end_date=end_date
            )
            
            if df_options is not None and len(df_options) > 0:
                output_file = self.jquants_dir / f"options_{years_back}years.parquet"
                df_options.to_parquet(output_file)
                logger.info(f"✅ オプションデータ保存: {len(df_options)}件 -> {output_file}")
                return df_options
            else:
                logger.warning("⚠️ オプションデータが取得できませんでした")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ オプションデータエラー: {e}")
            return None
    
    def collect_financial_data(self):
        """財務データ収集"""
        logger.info("💼 財務データ収集開始")
        
        try:
            # 決算発表情報
            logger.info("決算発表情報取得中...")
            df_announcements = self.client.get_fins_announcement()
            
            if df_announcements is not None and len(df_announcements) > 0:
                output_file = self.jquants_dir / "financial_announcements.parquet"
                df_announcements.to_parquet(output_file)
                logger.info(f"✅ 決算発表情報保存: {len(df_announcements)}件 -> {output_file}")
                return {'announcements': df_announcements}
            else:
                logger.warning("⚠️ 決算発表情報が取得できませんでした")
                return {}
                
        except Exception as e:
            logger.warning(f"⚠️ 財務データエラー: {e}")
            return {}
    
    def collect_all_data(self, years_back: int = 10):
        """全データの一括収集"""
        logger.info("🚀 J-Quantsデータ一括収集開始")
        
        if not self.initialize_client():
            return False
        
        collected = {}
        
        # 1. 基本株価データ（最重要）
        collected['prices'] = self.collect_basic_stock_data(years_back)
        time.sleep(1)
        
        # 2. 指数データ
        collected['indices'] = self.collect_indices_data(years_back)
        time.sleep(1)
        
        # 3. 信用・空売りデータ
        collected['margin_credit'] = self.collect_margin_credit_data(years_back)
        time.sleep(1)
        
        # 4. オプションデータ
        collected['options'] = self.collect_options_data(2)  # 2年分
        time.sleep(1)
        
        # 5. 財務データ
        collected['financial'] = self.collect_financial_data()
        
        # 収集結果サマリー
        logger.info("\n" + "="*60)
        logger.info("📋 J-Quantsデータ収集完了")
        logger.info("="*60)
        
        total_files = 0
        for category, data in collected.items():
            if data is not None:
                if isinstance(data, dict):
                    for sub_name, sub_data in data.items():
                        if sub_data is not None:
                            logger.info(f"  ✅ {category}/{sub_name}: {len(sub_data):,}件")
                            total_files += 1
                else:
                    logger.info(f"  ✅ {category}: {len(data):,}件")
                    total_files += 1
            else:
                logger.info(f"  ❌ {category}: データなし")
        
        logger.info(f"\n📊 合計ファイル数: {total_files}")
        logger.info(f"📁 保存場所: {self.jquants_dir}")
        
        return collected

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="J-Quants enhanced data collection")
    parser.add_argument("--years", type=int, default=10, help="Years of data to collect")
    parser.add_argument("--mail", help="J-Quants mail address")
    parser.add_argument("--password", help="J-Quants password")
    
    args = parser.parse_args()
    
    try:
        collector = JQuantsEnhancedCollector(
            mail_address=args.mail,
            password=args.password
        )
        
        print(f"📊 J-Quantsから{args.years}年分のデータ収集を開始します")
        print("="*60)
        
        collected_data = collector.collect_all_data(years_back=args.years)
        
        if collected_data:
            print(f"\n🎉 データ収集が正常に完了しました")
            print(f"📁 ファイル保存場所: {collector.jquants_dir}")
            return 0
        else:
            print(f"\n❌ データ収集に失敗しました")
            return 1
            
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())