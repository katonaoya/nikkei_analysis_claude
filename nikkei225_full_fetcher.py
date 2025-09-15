#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225全銘柄データ取得システム
J-Quants APIを使用して日経225全225銘柄のデータを取得
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

class Nikkei225FullFetcher:
    """日経225全銘柄データ取得システム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        # 日経225銘柄コード読み込み
        self.nikkei225_codes = self._load_nikkei225_codes()
        logger.info(f"日経225銘柄数: {len(self.nikkei225_codes)}銘柄")
    
    def _load_nikkei225_codes(self) -> List[str]:
        """日経225銘柄コード読み込み"""
        try:
            df = pd.read_csv('data/nikkei225_codes.csv')
            # 4桁形式で統一（先頭0埋め）
            codes = df['code'].astype(str).str.zfill(4).tolist()
            logger.info(f"日経225銘柄コード読み込み完了: {len(codes)}銘柄")
            logger.info(f"サンプルコード: {codes[:5]}")
            return codes
        except Exception as e:
            logger.error(f"日経225銘柄コード読み込みエラー: {e}")
            return []
    
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
    
    def get_listed_companies(self) -> Dict[str, str]:
        """上場銘柄一覧を取得して正確な銘柄コードを特定"""
        token = self._get_id_token()
        
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{JQUANTS_BASE_URL}/listed/info"
        
        all_companies = {}
        pagination_key = None
        
        logger.info("上場銘柄一覧を取得中...")
        
        while True:
            params = {}
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                time.sleep(2)  # レート制限対策
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 429:
                    logger.warning("レート制限により2分待機...")
                    time.sleep(120)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if "info" in data:
                    for company in data["info"]:
                        code = company.get("Code", "")  # 5桁APIコード
                        name = company.get("CompanyName", "")
                        
                        # 5桁コードから4桁コードを逆算（最後の0を除去）
                        if code and len(str(code)) == 5 and str(code).endswith('0'):
                            code_4digit = str(code)[:-1]  # 最後の0を除去
                        else:
                            continue
                        
                        # 日経225銘柄に含まれる場合のみ記録
                        if code_4digit in self.nikkei225_codes:
                            all_companies[code_4digit] = {
                                "api_code": code,  # 5桁APIコード
                                "name": name or f"銘柄{code_4digit}",
                                "code_4digit": code_4digit
                            }
                            
                            if len(all_companies) <= 10:  # 最初の10件をログ出力
                                logger.info(f"マッピング: {code_4digit} -> {code} ({name})")
                
                # 次のページがあるかチェック
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                    
            except Exception as e:
                logger.error(f"上場銘柄一覧取得エラー: {e}")
                time.sleep(10)
                continue
        
        logger.info(f"日経225銘柄のマッピング完了: {len(all_companies)}銘柄")
        return all_companies
    
    def fetch_stock_data(self, api_code: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """個別銘柄の株価データを取得"""
        token = self._get_id_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"{JQUANTS_BASE_URL}/prices/daily_quotes"
        params = {
            "code": api_code,
            "from": from_date,
            "to": to_date
        }
        
        all_data = []
        pagination_key = None
        
        while True:
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                time.sleep(2)  # レート制限対策
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 429:
                    logger.warning("レート制限により2分待機...")
                    time.sleep(120)
                    continue
                
                if response.status_code == 404:
                    logger.warning(f"銘柄 {api_code} のデータが見つかりません")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                if "daily_quotes" in data:
                    all_data.extend(data["daily_quotes"])
                
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                    
            except Exception as e:
                logger.error(f"銘柄 {api_code} のデータ取得エラー: {e}")
                time.sleep(5)
                return None
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"銘柄 {api_code}: {len(df)}件取得")
            return df
        
        return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標計算"""
        if df.empty:
            return df
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Code', 'Date'])
        
        # グループごとに計算
        result_dfs = []
        for code, group in df.groupby('Code'):
            group = group.sort_values('Date').copy()
            
            # 移動平均
            group['MA_5'] = group['Close'].rolling(window=5).mean()
            group['MA_20'] = group['Close'].rolling(window=20).mean()
            
            # リターン
            group['Returns'] = group['Close'].pct_change()
            
            # ボラティリティ（20日間）
            group['Volatility'] = group['Returns'].rolling(window=20).std()
            
            # RSI計算
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['RSI'] = 100 - (100 / (1 + rs))
            
            result_dfs.append(group)
        
        result = pd.concat(result_dfs, ignore_index=True)
        logger.info(f"テクニカル指標計算完了: {len(result)}件")
        
        return result
    
    def fetch_all_nikkei225_data(self, years: int = 10) -> pd.DataFrame:
        """日経225全銘柄のデータを取得"""
        # 期間設定
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=years * 365)
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"データ取得期間: {from_date} 〜 {to_date}")
        
        # 上場銘柄一覧から正確なコードを取得
        companies_mapping = self.get_listed_companies()
        
        if not companies_mapping:
            logger.error("銘柄マッピング取得に失敗")
            return pd.DataFrame()
        
        all_stock_data = []
        success_count = 0
        total_count = len(companies_mapping)
        
        logger.info(f"日経225全銘柄データ取得開始: {total_count}銘柄")
        
        for i, (code_4digit, company_info) in enumerate(companies_mapping.items(), 1):
            api_code = company_info["api_code"]
            company_name = company_info["name"]
            
            logger.info(f"進捗 {i}/{total_count}: {code_4digit} ({company_name})")
            
            # 株価データ取得
            stock_df = self.fetch_stock_data(api_code, from_date, to_date)
            
            if stock_df is not None and not stock_df.empty:
                # 4桁コードを追加
                stock_df['Code'] = code_4digit
                stock_df['CompanyName'] = company_name
                all_stock_data.append(stock_df)
                success_count += 1
                logger.info(f"✅ {code_4digit}: {len(stock_df)}件取得")
            else:
                logger.warning(f"❌ {code_4digit}: データ取得失敗")
            
            # 10銘柄ごとに長めの待機
            if i % 10 == 0:
                logger.info(f"10銘柄処理完了、15秒待機...")
                time.sleep(15)
        
        # 全データを統合
        if all_stock_data:
            logger.info("データ統合処理開始...")
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            
            # テクニカル指標計算
            logger.info("テクニカル指標計算開始...")
            final_df = self.calculate_technical_indicators(combined_df)
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/nikkei225_full_data_{timestamp}.parquet"
            final_df.to_parquet(output_file, index=False)
            
            logger.info(f"🎉 日経225全銘柄データ取得完了!")
            logger.info(f"成功: {success_count}/{total_count}銘柄")
            logger.info(f"総レコード数: {len(final_df):,}件")
            logger.info(f"保存先: {output_file}")
            
            return final_df
        else:
            logger.error("データ取得に失敗しました")
            return pd.DataFrame()

def main():
    """メイン実行"""
    try:
        fetcher = Nikkei225FullFetcher()
        
        # 日経225全銘柄データ取得（10年間）
        df = fetcher.fetch_all_nikkei225_data(years=10)
        
        if not df.empty:
            print(f"\n✅ 日経225全銘柄データ取得完了")
            print(f"📊 データ概要:")
            print(f"  - 総レコード数: {len(df):,}")
            print(f"  - 銘柄数: {df['Code'].nunique()}")
            print(f"  - 期間: {df['Date'].min()} 〜 {df['Date'].max()}")
        else:
            print("\n❌ データ取得に失敗しました")
            
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main()