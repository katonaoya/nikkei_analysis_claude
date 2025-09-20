#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張J-Quantsデータ取得システム
実際のJ-Quants APIから日経225銘柄の10年データを取得
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
from typing import List, Optional
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class ExpandedJQuantsFetcher:
    """拡張J-Quantsデータ取得システム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("拡張J-Quantsデータ取得システム初期化完了")
    
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
    
    def get_listed_companies(self) -> pd.DataFrame:
        """上場銘柄一覧を取得してコードを確認"""
        logger.info("📋 上場銘柄一覧取得中...")
        
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/listed/info",
                headers=headers,
                timeout=60
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限、30秒待機...")
                time.sleep(30)
                return self.get_listed_companies()
            
            resp.raise_for_status()
            data = resp.json()
            
            companies_df = pd.DataFrame(data['info'])
            logger.info(f"✅ 上場銘柄一覧取得完了: {len(companies_df)}社")
            
            # 日経225に含まれそうな大手企業をフィルタ
            major_companies = companies_df[
                (companies_df['MarketCode'] == '111') |  # 東証プライム
                (companies_df['CompanyName'].str.contains('トヨタ|ソフトバンク|ソニー|日本電信電話|三菱UFJ', na=False))
            ]
            
            logger.info(f"📊 主要銘柄候補: {len(major_companies)}社")
            return major_companies
            
        except Exception as e:
            logger.error(f"銘柄一覧取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_quotes_batch(self, codes: List[str], from_date: str, to_date: str) -> pd.DataFrame:
        """複数銘柄の株価データを一括取得"""
        logger.info(f"📈 株価データ一括取得: {len(codes)}銘柄")
        logger.info(f"期間: {from_date} ～ {to_date}")
        
        all_data = []
        successful_codes = []
        
        for idx, code in enumerate(codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(codes)}) - {idx/len(codes)*100:.1f}%完了")
                
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
                    logger.warning(f"銘柄 {code}: レート制限、30秒待機...")
                    time.sleep(30)
                    continue
                
                if resp.status_code != 200:
                    logger.warning(f"銘柄 {code}: エラー {resp.status_code}")
                    continue
                
                data = resp.json()
                daily_quotes = data.get("daily_quotes", [])
                
                if daily_quotes:
                    stock_df = pd.DataFrame(daily_quotes)
                    all_data.append(stock_df)
                    successful_codes.append(code)
                    logger.info(f"  ✅ 銘柄 {code}: {len(daily_quotes)}件取得成功")
                else:
                    logger.warning(f"  ❌ 銘柄 {code}: データなし")
                
                # レート制限対策
                time.sleep(1.5)
                
                # 10銘柄ごとに長い待機
                if idx % 10 == 0:
                    logger.info(f"  ⏸️  10銘柄処理完了、5秒待機...")
                    time.sleep(5)\n                \n            except Exception as e:\n                logger.error(f\"銘柄 {code} 取得エラー: {str(e)}\")\n                continue\n        \n        if not all_data:\n            logger.error(\"❌ 全銘柄でデータ取得に失敗\")\n            return pd.DataFrame()\n        \n        # データ統合\n        logger.info(\"🔄 データ統合中...\")\n        combined_df = pd.concat(all_data, ignore_index=True)\n        \n        logger.info(f\"✅ 一括取得完了: {len(combined_df):,}件, {len(successful_codes)}銘柄\")\n        logger.info(f\"成功銘柄: {successful_codes}\")\n        \n        return combined_df\n    \n    def expand_existing_data(self) -> pd.DataFrame:\n        \"\"\"既存データを拡張（期間・銘柄数を増加）\"\"\"\n        logger.info(\"🚀 既存データ拡張開始\")\n        \n        # 1. 上場銘柄一覧から有効な銘柄コードを取得\n        companies_df = self.get_listed_companies()\n        \n        if companies_df.empty:\n            logger.error(\"❌ 銘柄一覧が取得できません\")\n            return pd.DataFrame()\n        \n        # 2. 主要銘柄を選択（最大100銘柄）\n        selected_codes = companies_df['Code'].unique()[:100]  # 上位100銘柄\n        logger.info(f\"📊 選択銘柄数: {len(selected_codes)}銘柄\")\n        \n        # 3. 期間を拡張（過去5年間）\n        to_date = datetime.now().strftime('%Y-%m-%d')\n        from_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')\n        \n        logger.info(f\"📅 拡張期間: {from_date} ～ {to_date} (5年間)\")\n        \n        # 4. データ取得\n        expanded_df = self.get_daily_quotes_batch(selected_codes, from_date, to_date)\n        \n        if expanded_df.empty:\n            logger.error(\"❌ 拡張データ取得に失敗\")\n            return pd.DataFrame()\n        \n        # 5. データ保存\n        output_dir = Path(\"data/expanded_jquants_data\")\n        output_dir.mkdir(parents=True, exist_ok=True)\n        \n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        output_file = output_dir / f\"expanded_jquants_5years_{len(selected_codes)}stocks_{timestamp}.parquet\"\n        \n        expanded_df.to_parquet(output_file, index=False)\n        logger.info(f\"💾 拡張データ保存: {output_file}\")\n        \n        logger.info(\"🎉 データ拡張完了\")\n        logger.info(f\"📊 最終データ: {len(expanded_df):,}件, {expanded_df['Code'].nunique()}銘柄\")\n        logger.info(f\"📅 期間: {expanded_df['Date'].min()} ～ {expanded_df['Date'].max()}\")\n        \n        return expanded_df\n    \n    def create_enhanced_dataset(self) -> pd.DataFrame:\n        \"\"\"既存データと拡張データを統合した強化データセットを作成\"\"\"\n        logger.info(\"🔧 強化データセット作成開始\")\n        \n        # 既存データ読み込み\n        existing_path = Path(\"data/processed/real_jquants_data.parquet\")\n        existing_df = pd.DataFrame()\n        \n        if existing_path.exists():\n            existing_df = pd.read_parquet(existing_path)\n            logger.info(f\"📁 既存データ: {len(existing_df):,}件, {existing_df['Code'].nunique()}銘柄\")\n        \n        # 拡張データ取得\n        expanded_df = self.expand_existing_data()\n        \n        if expanded_df.empty and existing_df.empty:\n            logger.error(\"❌ 利用可能なデータがありません\")\n            return pd.DataFrame()\n        \n        # データ統合\n        if not existing_df.empty and not expanded_df.empty:\n            # 両方のデータがある場合は統合\n            combined_df = pd.concat([existing_df, expanded_df], ignore_index=True)\n            combined_df = combined_df.drop_duplicates(subset=['Date', 'Code']).sort_values(['Code', 'Date'])\n            logger.info(f\"📊 統合データ: {len(combined_df):,}件, {combined_df['Code'].nunique()}銘柄\")\n        elif not expanded_df.empty:\n            combined_df = expanded_df\n            logger.info(\"📊 拡張データのみ使用\")\n        else:\n            combined_df = existing_df\n            logger.info(\"📊 既存データのみ使用\")\n        \n        # 強化データセット保存\n        output_dir = Path(\"data/enhanced_datasets\")\n        output_dir.mkdir(parents=True, exist_ok=True)\n        \n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        output_file = output_dir / f\"enhanced_jquants_dataset_{timestamp}.parquet\"\n        \n        combined_df.to_parquet(output_file, index=False)\n        logger.info(f\"💾 強化データセット保存: {output_file}\")\n        \n        logger.info(\"🎉 強化データセット作成完了\")\n        return combined_df


def main():\n    \"\"\"メイン実行関数\"\"\"\n    logger.info(\"🚀 拡張J-Quantsデータ取得システム開始\")\n    \n    try:\n        fetcher = ExpandedJQuantsFetcher()\n        enhanced_df = fetcher.create_enhanced_dataset()\n        \n        if not enhanced_df.empty:\n            logger.info(\"=\"*60)\n            logger.info(\"🎉 拡張データ取得完了\")\n            logger.info(\"=\"*60)\n            logger.info(f\"📊 総レコード数: {len(enhanced_df):,}件\")\n            logger.info(f\"📊 総銘柄数: {enhanced_df['Code'].nunique()}銘柄\")\n            logger.info(f\"📅 期間: {enhanced_df['Date'].min()} ～ {enhanced_df['Date'].max()}\")\n            \n            # 簡単な統計\n            print(\"\\n📈 銘柄別データ件数（上位10銘柄）:\")\n            top_stocks = enhanced_df['Code'].value_counts().head(10)\n            for code, count in top_stocks.items():\n                print(f\"  {code}: {count:,}件\")\n                \n        else:\n            logger.error(\"❌ データ取得に失敗しました\")\n            \n    except Exception as e:\n        logger.error(f\"❌ システムエラー: {str(e)}\")\n        raise\n\n\nif __name__ == \"__main__\":\n    main()