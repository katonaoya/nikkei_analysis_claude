#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225全225銘柄完全並列データ取得システム
拡張マッピング結果を使用して日経225全225銘柄の10年分株価データを並列で高速取得
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
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class Nikkei225CompleteParallelFetcher:
    """日経225全225銘柄完全並列データ取得システム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.token_lock = threading.Lock()
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        # マッピング結果読み込み
        self.company_mapping = self._load_company_mapping()
        logger.info(f"日経225完全マッピング銘柄数: {len(self.company_mapping)}銘柄")
        
        # 並列実行用の共有データ
        self.results_queue = Queue()
        self.progress_count = 0
        self.progress_lock = threading.Lock()
    
    def _load_company_mapping(self) -> Dict[str, Dict[str, str]]:
        """拡張マッピング結果を読み込み"""
        try:
            # 最新のマッピングファイルを検索
            mapping_files = list(Path("docment/ユーザー情報").glob("nikkei225_matched_companies_*.csv"))
            if not mapping_files:
                raise FileNotFoundError("マッピング結果ファイルが見つかりません")
            
            # 最新のファイルを使用
            latest_file = max(mapping_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"マッピングファイル使用: {latest_file}")
            
            df = pd.read_csv(latest_file)
            
            # 辞書形式に変換
            mapping = {}
            for _, row in df.iterrows():
                mapping[row['target_code']] = {
                    "api_code": row['api_code'],
                    "target_name": row['target_name'],
                    "api_name": row['api_name'],
                    "match_method": row['match_method']
                }
            
            logger.info(f"日経225完全マッピング読み込み完了: {len(mapping)}銘柄")
            return mapping
            
        except Exception as e:
            logger.error(f"マッピング結果読み込みエラー: {e}")
            return {}
    
    def _get_id_token(self) -> str:
        """IDトークンを取得（スレッドセーフ）"""
        with self.token_lock:
            if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
                return self.id_token
            
            logger.info("JQuants認証トークンを取得中...")
            time.sleep(1)  # 認証リクエストの間隔調整
            
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
                    logger.warning("認証レート制限により2分待機...")
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
                    logger.warning("認証レート制限により2分待機...")
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
    
    def fetch_stock_data_worker(self, company_info: Tuple[str, Dict[str, str]], 
                               from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """個別銘柄の株価データを取得（ワーカー関数）"""
        target_code, info = company_info
        api_code = info["api_code"]
        company_name = info["target_name"]
        match_method = info.get("match_method", "不明")
        
        try:
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
                    time.sleep(0.3)  # 並列実行用のより短い間隔
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    
                    if response.status_code == 429:
                        logger.warning(f"銘柄 {target_code}: レート制限により30秒待機...")
                        time.sleep(30)
                        continue
                    
                    if response.status_code == 404:
                        logger.warning(f"銘柄 {target_code}: データが見つかりません (API: {api_code})")
                        return None
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if "daily_quotes" in data:
                        all_data.extend(data["daily_quotes"])
                    
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                        
                except Exception as e:
                    logger.error(f"銘柄 {target_code} のデータ取得エラー: {e}")
                    time.sleep(2)
                    return None
            
            if all_data:
                df = pd.DataFrame(all_data)
                # ターゲットコードと企業名を追加
                df['Code'] = target_code
                df['CompanyName'] = company_name
                df['MatchMethod'] = match_method
                df['ApiCode'] = api_code
                
                # 進捗更新
                with self.progress_lock:
                    self.progress_count += 1
                    logger.info(f"✅ {target_code} ({company_name}): {len(df)}件取得 [{self.progress_count}/225] - {match_method}")
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"銘柄 {target_code} の処理エラー: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標計算（警告を抑制）"""
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
            
            # リターン（警告抑制）
            group['Returns'] = group['Close'].pct_change(fill_method=None)
            
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
    
    def fetch_all_nikkei225_complete_data(self, years: int = 10, max_workers: int = 10) -> pd.DataFrame:
        """日経225全225銘柄の完全データを並列で取得"""
        if not self.company_mapping:
            logger.error("マッピングデータがありません")
            return pd.DataFrame()
        
        # 期間設定
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=years * 365)
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"データ取得期間: {from_date} 〜 {to_date}")
        logger.info(f"並列ワーカー数: {max_workers}")
        logger.info(f"🚀 日経225全225銘柄完全データ並列取得開始!")
        
        # 並列実行
        all_stock_data = []
        company_items = list(self.company_mapping.items())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全銘柄のタスクを投入
            futures = {
                executor.submit(self.fetch_stock_data_worker, company_info, from_date, to_date): company_info[0]
                for company_info in company_items
            }
            
            # 結果を収集
            for future in as_completed(futures):
                code = futures[future]
                try:
                    result = future.result(timeout=300)  # 5分タイムアウト
                    if result is not None and not result.empty:
                        all_stock_data.append(result)
                except Exception as e:
                    logger.error(f"銘柄 {code} の並列処理エラー: {e}")
        
        # 全データを統合
        if all_stock_data:
            logger.info("データ統合処理開始...")
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            
            # テクニカル指標計算
            logger.info("テクニカル指標計算開始...")
            final_df = self.calculate_technical_indicators(combined_df)
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/nikkei225_complete_225stocks_{timestamp}.parquet"
            
            # ディレクトリ作成
            os.makedirs("data/processed", exist_ok=True)
            final_df.to_parquet(output_file, index=False)
            
            # マッチング統計
            match_stats = final_df.groupby('MatchMethod').size().to_dict()
            
            logger.info(f"🎉 日経225全225銘柄完全データ取得完了!")
            logger.info(f"成功: {len(all_stock_data)}/225銘柄")
            logger.info(f"総レコード数: {len(final_df):,}件")
            logger.info(f"マッチング手法別統計:")
            for method, count in match_stats.items():
                logger.info(f"  - {method}: {count:,}件")
            logger.info(f"保存先: {output_file}")
            
            return final_df
        else:
            logger.error("データ取得に失敗しました")
            return pd.DataFrame()

def main():
    """メイン実行"""
    try:
        fetcher = Nikkei225CompleteParallelFetcher()
        
        # 日経225全225銘柄完全データ並列取得（10年間、10並列）
        df = fetcher.fetch_all_nikkei225_complete_data(years=10, max_workers=10)
        
        if not df.empty:
            print(f"\n🎉 日経225全225銘柄完全データ取得完了!")
            print(f"📊 データ概要:")
            print(f"  - 総レコード数: {len(df):,}")
            print(f"  - 銘柄数: {df['Code'].nunique()}")
            print(f"  - 期間: {df['Date'].min()} 〜 {df['Date'].max()}")
            print(f"  - 平均レコード数/銘柄: {len(df)/df['Code'].nunique():.0f}")
            
            # マッチング手法別統計表示
            match_stats = df.groupby('MatchMethod').agg({
                'Code': 'nunique',
                'Date': 'count'
            }).rename(columns={'Code': '銘柄数', 'Date': 'レコード数'})
            print(f"\n📈 マッチング手法別統計:")
            print(match_stats)
            
        else:
            print("\n❌ データ取得に失敗しました")
            
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main()