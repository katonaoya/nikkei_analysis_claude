#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225全銘柄×10年間データ取得システム
J-Quants APIから日経225構成銘柄の完全な10年間データを並列取得
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from pathlib import Path
import logging
from typing import List, Optional, Dict
from dotenv import load_dotenv
import concurrent.futures
import threading
from queue import Queue

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class Nikkei225Full10YearFetcher:
    """日経225全銘柄×10年間データ取得システム"""
    
    def __init__(self, max_workers=5):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.max_workers = max_workers
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_interval = 0.5  # 500ms間隔
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info(f"日経225全銘柄×10年間データ取得システム初期化完了 (並列度: {max_workers})")
    
    def _get_id_token(self) -> str:
        """IDトークンを取得"""
        if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.id_token
        
        logger.info("JQuants認証トークンを取得中...")
        time.sleep(3)
        
        try:
            # リフレッシュトークンを取得
            auth_payload = {
                "mailaddress": self.mail_address,
                "password": self.password
            }
            
            resp = requests.post(
                f"{JQUANTS_BASE_URL}/token/auth_user",
                data=json.dumps(auth_payload),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限により2分待機...")
                time.sleep(120)
                return self._get_id_token()
                
            resp.raise_for_status()
            refresh_token = resp.json().get("refreshToken")
            
            if not refresh_token:
                raise RuntimeError("リフレッシュトークンの取得に失敗しました")
            
            time.sleep(1)
            resp = requests.post(
                f"{JQUANTS_BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
                timeout=30
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限により2分待機...")
                time.sleep(120)
                return self._get_id_token()
                
            resp.raise_for_status()
            self.id_token = resp.json().get("idToken")
            
            if not self.id_token:
                raise RuntimeError("IDトークンの取得に失敗しました")
            
            self.token_expires_at = datetime.now() + timedelta(hours=1)
            
            logger.info("認証トークン取得完了")
            return self.id_token
            
        except Exception as e:
            logger.error(f"認証エラー: {str(e)}")
            raise
    
    def _rate_limit_wait(self):
        """レート制限対応の待機処理"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def get_nikkei225_companies(self) -> pd.DataFrame:
        """日経225構成銘柄一覧を取得"""
        logger.info("📋 日経225構成銘柄一覧取得開始...")
        
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            
            logger.info("J-Quants上場銘柄一覧API呼び出し中...")
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/listed/info",
                headers=headers,
                timeout=60
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限、30秒待機...")
                time.sleep(30)
                return self.get_nikkei225_companies()
            
            resp.raise_for_status()
            data = resp.json()
            
            companies_df = pd.DataFrame(data['info'])
            logger.info(f"✅ 上場銘柄一覧取得完了: {len(companies_df)}社")
            
            # 日経225相当の選択（プライム市場の大型株を最大225社選択）
            nikkei225_companies = companies_df[
                (companies_df['MarketCode'] == '0111') &  # プライム市場
                (companies_df['ScaleCategory'].isin(['TOPIX Large70', 'TOPIX Mid400']))
            ].copy()
            
            # 有名企業を追加で確保
            major_companies = companies_df[
                companies_df['CompanyName'].str.contains(
                    'トヨタ|ソフトバンク|ソニー|日本電信電話|三菱UFJ|日立|ホンダ|任天堂|キヤノン|パナソニック|'
                    'NTT|KDDI|武田薬品|ファーストリテイリング|ファナック|信越化学|東京エレクトロン|'
                    'ダイキン工業|村田製作所|日本電産|キーエンス|エムスリー|リクルート|オリエンタルランド|'
                    'セコム|テルモ|シスメックス|日本M&Aセンター|モノタロウ|ペプチドリーム', 
                    na=False
                )
            ]
            
            # 統合して225社を選択
            selected_companies = pd.concat([nikkei225_companies, major_companies]).drop_duplicates()
            
            # 225社に制限（時価総額の大きい順などで選択）
            if len(selected_companies) > 225:
                selected_companies = selected_companies.head(225)
            
            logger.info(f"📈 日経225相当選択銘柄数: {len(selected_companies)}社")
            
            # サンプル表示
            logger.info("📊 選択銘柄サンプル:")
            for i, (_, company) in enumerate(selected_companies.head(10).iterrows()):
                logger.info(f"  {company['Code']}: {company['CompanyName']}")
            
            return selected_companies
            
        except Exception as e:
            logger.error(f"銘柄一覧取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def get_single_stock_data(self, company_info: tuple) -> Optional[pd.DataFrame]:
        """単一銘柄の10年間データを取得"""
        code, company_name = company_info
        
        try:
            self._rate_limit_wait()
            
            # 10年間の期間設定
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            params = {
                "code": code,
                "from": from_date,
                "to": to_date
            }
            
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/prices/daily_quotes",
                headers=headers,
                params=params,
                timeout=120
            )
            
            if resp.status_code == 429:
                logger.warning(f"銘柄 {code}: レート制限、60秒待機...")
                time.sleep(60)
                return self.get_single_stock_data((code, company_name))
            
            if resp.status_code != 200:
                logger.warning(f"❌ 銘柄 {code} ({company_name}): エラー {resp.status_code}")
                return None
            
            data = resp.json()
            daily_quotes = data.get("daily_quotes", [])
            
            if daily_quotes:
                stock_df = pd.DataFrame(daily_quotes)
                stock_df['CompanyName'] = company_name
                logger.info(f"✅ 銘柄 {code} ({company_name}): {len(daily_quotes)}件取得成功")
                return stock_df
            else:
                logger.warning(f"❌ 銘柄 {code} ({company_name}): データなし")
                return None
                
        except Exception as e:
            logger.error(f"❌ 銘柄 {code} ({company_name}): 取得エラー {str(e)}")
            return None
    
    def get_all_nikkei225_data_parallel(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """日経225全銘柄の10年間データを並列取得"""
        logger.info(f"🚀 日経225全銘柄並列取得開始: {len(companies_df)}銘柄 × 10年間")
        
        # 期間設定表示
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        logger.info(f"📅 取得期間: {from_date} ～ {to_date}")
        
        # 予想データ量計算
        expected_records = len(companies_df) * 10 * 245  # 銘柄数 × 年数 × 営業日数
        logger.info(f"📊 予想データ量: 約{expected_records:,}件")
        
        all_stock_data = []
        successful_companies = []
        failed_companies = []
        
        # 銘柄情報のタプルリストを作成
        company_list = [(row['Code'], row['CompanyName']) for _, row in companies_df.iterrows()]
        
        # 並列処理で銘柄データを取得
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_company = {
                executor.submit(self.get_single_stock_data, company_info): company_info 
                for company_info in company_list
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_company), 1):
                company_info = future_to_company[future]
                code, company_name = company_info
                
                try:
                    stock_df = future.result()
                    if stock_df is not None:
                        all_stock_data.append(stock_df)
                        successful_companies.append(f"{code}({company_name})")
                    else:
                        failed_companies.append(f"{code}({company_name})")
                    
                    # 進行状況表示
                    if i % 10 == 0:
                        progress = i / len(company_list) * 100
                        logger.info(f"📊 進行状況: {i}/{len(company_list)} ({progress:.1f}%) - 成功: {len(successful_companies)}, 失敗: {len(failed_companies)}")
                        
                except Exception as e:
                    logger.error(f"❌ {code}({company_name}): 並列処理エラー {str(e)}")
                    failed_companies.append(f"{code}({company_name})")
        
        if not all_stock_data:
            logger.error("❌ 全銘柄でデータ取得に失敗")
            return pd.DataFrame()
        
        # データ統合
        logger.info("🔄 全データ統合中...")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("="*60)
        logger.info("📊 日経225全銘柄×10年間データ取得結果")
        logger.info("="*60)
        logger.info(f"✅ 成功銘柄数: {len(successful_companies)}/{len(companies_df)}銘柄")
        logger.info(f"❌ 失敗銘柄数: {len(failed_companies)}銘柄")
        logger.info(f"📊 総レコード数: {len(combined_df):,}件")
        logger.info(f"📅 期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        logger.info(f"📈 平均レコード数/銘柄: {len(combined_df)/len(successful_companies):.0f}件") if successful_companies else None
        
        if successful_companies:
            logger.info("✅ 成功銘柄（最初の20社）:")
            for company in successful_companies[:20]:
                logger.info(f"  {company}")
        
        if failed_companies:
            logger.info("❌ 失敗銘柄:")
            for company in failed_companies:
                logger.info(f"  {company}")
        
        return combined_df
    
    def create_nikkei225_full_dataset(self) -> pd.DataFrame:
        """日経225全銘柄×10年間の完全データセットを作成"""
        logger.info("🚀 日経225全銘柄×10年間完全データセット作成開始")
        
        # 1. 日経225構成銘柄一覧取得
        companies_df = self.get_nikkei225_companies()
        
        if companies_df.empty:
            logger.error("❌ 銘柄一覧取得に失敗")
            return pd.DataFrame()
        
        # 2. 全銘柄の10年間データを並列取得
        full_df = self.get_all_nikkei225_data_parallel(companies_df)
        
        if full_df.empty:
            logger.error("❌ 全銘柄データ取得に失敗")
            return pd.DataFrame()
        
        # 3. データ保存
        output_dir = Path("data/nikkei225_full")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"nikkei225_full_{len(full_df)}records_{timestamp}.parquet"
        
        full_df.to_parquet(output_file, index=False)
        logger.info(f"💾 完全データセット保存: {output_file}")
        
        # 4. データ統計表示
        logger.info("📊 最終データセット統計:")
        logger.info(f"  総レコード数: {len(full_df):,}件")
        logger.info(f"  銘柄数: {full_df['Code'].nunique()}銘柄")
        logger.info(f"  期間: {full_df['Date'].min()} ～ {full_df['Date'].max()}")
        
        # 上位銘柄のレコード数
        top_stocks = full_df['Code'].value_counts().head(10)
        logger.info("\n📈 レコード数上位10銘柄:")
        for code, count in top_stocks.items():
            company_name = full_df[full_df['Code'] == code]['CompanyName'].iloc[0] if 'CompanyName' in full_df.columns else 'N/A'
            logger.info(f"  {code} ({company_name}): {count:,}件")
        
        logger.info("🎉 日経225全銘柄×10年間完全データセット作成完了")
        return full_df


def main():
    """メイン実行関数"""
    logger.info("🚀 日経225全銘柄×10年間データ取得システム開始")
    
    try:
        # 並列度5でフェッチャーを初期化
        fetcher = Nikkei225Full10YearFetcher(max_workers=5)
        full_df = fetcher.create_nikkei225_full_dataset()
        
        if not full_df.empty:
            logger.info("="*60)
            logger.info("🎉 日経225全銘柄×10年間データ取得完了")
            logger.info("="*60)
            logger.info(f"📊 最終データ量: {len(full_df):,}件")
            logger.info(f"📊 取得銘柄数: {full_df['Code'].nunique()}銘柄")
            logger.info(f"📅 データ期間: {full_df['Date'].min()} ～ {full_df['Date'].max()}")
            
            # ユーザー要求との比較
            target_records = 550000
            achievement_rate = len(full_df) / target_records * 100
            logger.info(f"🎯 目標達成率: {achievement_rate:.1f}% ({len(full_df):,}/{target_records:,}件)")
            
            logger.info("🎯 次のステップ: enhanced_precision_with_full_data.pyでの精度検証を実行予定")
                
        else:
            logger.error("❌ データ取得に失敗しました")
            
    except Exception as e:
        logger.error(f"❌ システムエラー: {str(e)}")
        raise


if __name__ == "__main__":
    main()