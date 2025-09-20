#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J-Quants API正確な銘柄取得システム
上場銘柄一覧APIから正確な銘柄コードを取得してデータ拡張を実行
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

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class JQuantsCorrectFetcher:
    """J-Quants API正確な形式での日経225データ取得システム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("J-Quants正確な銘柄取得システム初期化完了")
    
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
    
    def get_all_listed_companies(self) -> pd.DataFrame:
        """上場銘柄一覧を取得（J-Quants APIの正確な形式）"""
        logger.info("📋 上場銘柄一覧取得開始...")
        
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
                return self.get_all_listed_companies()
            
            resp.raise_for_status()
            data = resp.json()
            
            companies_df = pd.DataFrame(data['info'])
            logger.info(f"✅ 上場銘柄一覧取得完了: {len(companies_df)}社")
            
            # 銘柄コード形式を確認
            logger.info("📊 銘柄コード形式確認:")
            sample_codes = companies_df['Code'].head(10).tolist()
            logger.info(f"銘柄コード例: {sample_codes}")
            
            # プライム市場の大型株を選択（日経225相当）
            prime_large_companies = companies_df[
                (companies_df['MarketCode'] == '0111') &  # プライム市場
                (companies_df['ScaleCategory'].isin(['TOPIX Large70', 'TOPIX Mid400']))
            ].copy()
            
            logger.info(f"📊 プライム市場大型株: {len(prime_large_companies)}社")
            
            # 有名企業名での追加フィルタ
            major_companies = companies_df[
                companies_df['CompanyName'].str.contains(
                    'トヨタ|ソフトバンク|ソニー|日本電信電話|三菱UFJ|日立|ホンダ|任天堂|キヤノン|パナソニック', 
                    na=False
                )
            ]
            
            # プライム大型株と有名企業を統合
            selected_companies = pd.concat([prime_large_companies, major_companies]).drop_duplicates()
            logger.info(f"📈 選択された銘柄数: {len(selected_companies)}社")
            
            return selected_companies
            
        except Exception as e:
            logger.error(f"銘柄一覧取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_data_by_correct_codes(self, companies_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
        """正確な銘柄コードで株価データを取得"""
        logger.info(f"📈 株価データ取得開始: {len(companies_df)}銘柄, {years}年間")
        
        # 期間設定
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
        logger.info(f"📅 取得期間: {from_date} ～ {to_date}")
        
        all_stock_data = []
        successful_companies = []
        failed_companies = []
        
        # 最大50銘柄に制限（API制限と処理時間考慮）
        selected_companies = companies_df.head(50)
        
        for idx, (_, company) in enumerate(selected_companies.iterrows(), 1):
            code = company['Code']
            company_name = company['CompanyName']
            
            try:
                logger.info(f"銘柄 {code} ({company_name}) データ取得中... ({idx}/{len(selected_companies)}) - {idx/len(selected_companies)*100:.1f}%完了")
                
                headers = {"Authorization": f"Bearer {self._get_id_token()}"}
                params = {
                    "code": code,  # J-Quantsの正確な5桁コードを使用
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
                    failed_companies.append(f"{code}({company_name})")
                    continue
                
                data = resp.json()
                daily_quotes = data.get("daily_quotes", [])
                
                if daily_quotes:
                    stock_df = pd.DataFrame(daily_quotes)
                    # 企業名を追加
                    stock_df['CompanyName'] = company_name
                    all_stock_data.append(stock_df)
                    successful_companies.append(f"{code}({company_name})")
                    logger.info(f"  ✅ 銘柄 {code}: {len(daily_quotes)}件取得成功")
                else:
                    logger.warning(f"  ❌ 銘柄 {code}: データなし")
                    failed_companies.append(f"{code}({company_name})")
                
                # レート制限対策
                time.sleep(2)
                
                # 10銘柄ごとに長い待機
                if idx % 10 == 0:
                    logger.info(f"  ⏸️  10銘柄処理完了、10秒待機...")
                    time.sleep(10)
                
            except Exception as e:
                logger.error(f"銘柄 {code} 取得エラー: {str(e)}")
                failed_companies.append(f"{code}({company_name})")
                continue
        
        if not all_stock_data:
            logger.error("❌ 全銘柄でデータ取得に失敗")
            return pd.DataFrame()
        
        # データ統合
        logger.info("🔄 データ統合中...")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("="*60)
        logger.info("📊 データ取得結果")
        logger.info("="*60)
        logger.info(f"✅ 成功銘柄数: {len(successful_companies)}銘柄")
        logger.info(f"❌ 失敗銘柄数: {len(failed_companies)}銘柄")
        logger.info(f"📊 総レコード数: {len(combined_df):,}件")
        logger.info(f"📅 期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        if successful_companies:
            logger.info("✅ 成功銘柄:")
            for company in successful_companies[:10]:  # 最初の10社のみ表示
                logger.info(f"  {company}")
        
        if failed_companies:
            logger.info("❌ 失敗銘柄:")
            for company in failed_companies[:5]:  # 最初の5社のみ表示
                logger.info(f"  {company}")
        
        return combined_df
    
    def create_enhanced_dataset(self) -> pd.DataFrame:
        """拡張データセットを作成"""
        logger.info("🚀 J-Quants拡張データセット作成開始")
        
        # 1. 上場銘柄一覧取得
        companies_df = self.get_all_listed_companies()
        
        if companies_df.empty:
            logger.error("❌ 銘柄一覧取得に失敗")
            return pd.DataFrame()
        
        # 2. 株価データ取得（5年間）
        expanded_df = self.get_stock_data_by_correct_codes(companies_df, years=5)
        
        if expanded_df.empty:
            logger.error("❌ 株価データ取得に失敗")
            return pd.DataFrame()
        
        # 3. 既存データとの統合
        existing_path = Path("data/processed/real_jquants_data.parquet")
        if existing_path.exists():
            existing_df = pd.read_parquet(existing_path)
            logger.info(f"📁 既存データ: {len(existing_df):,}件, {existing_df['Code'].nunique()}銘柄")
            
            # データ統合（重複除去）
            combined_df = pd.concat([existing_df, expanded_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date', 'Code']).sort_values(['Code', 'Date'])
            logger.info(f"📊 統合後データ: {len(combined_df):,}件, {combined_df['Code'].nunique()}銘柄")
        else:
            combined_df = expanded_df
            logger.info("📊 拡張データのみ使用")
        
        # 4. データ保存
        output_dir = Path("data/enhanced_jquants")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"enhanced_jquants_{len(combined_df)}records_{timestamp}.parquet"
        
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"💾 拡張データセット保存: {output_file}")
        
        logger.info("🎉 J-Quants拡張データセット作成完了")
        return combined_df


def main():
    """メイン実行関数"""
    logger.info("🚀 J-Quants正確な銘柄取得システム開始")
    
    try:
        fetcher = JQuantsCorrectFetcher()
        enhanced_df = fetcher.create_enhanced_dataset()
        
        if not enhanced_df.empty:
            logger.info("="*60)
            logger.info("🎉 拡張データ取得完了")
            logger.info("="*60)
            logger.info(f"📊 総レコード数: {len(enhanced_df):,}件")
            logger.info(f"📊 総銘柄数: {enhanced_df['Code'].nunique()}銘柄")
            logger.info(f"📅 期間: {enhanced_df['Date'].min()} ～ {enhanced_df['Date'].max()}")
            
            # 銘柄別統計
            print("\n📈 銘柄別データ件数（上位10銘柄）:")
            top_stocks = enhanced_df['Code'].value_counts().head(10)
            for code, count in top_stocks.items():
                company_name = enhanced_df[enhanced_df['Code'] == code]['CompanyName'].iloc[0] if 'CompanyName' in enhanced_df.columns else 'N/A'
                print(f"  {code} ({company_name}): {count:,}件")
            
            logger.info("🎯 次のステップ: enhanced_precision_with_full_data.pyで拡張データの精度テストを実行してください")
                
        else:
            logger.error("❌ データ取得に失敗しました")
            
    except Exception as e:
        logger.error(f"❌ システムエラー: {str(e)}")
        raise


if __name__ == "__main__":
    main()