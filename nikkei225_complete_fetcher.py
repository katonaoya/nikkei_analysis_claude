#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225全225銘柄データ完全取得システム
正確なユーザー提供リストを使用してJ-Quants APIから日経225全225銘柄のデータを並列で高速取得
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

class Nikkei225CompleteFetcher:
    """日経225全225銘柄データ完全取得システム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        
        # スレッドセーフなトークン管理
        self.token_lock = threading.Lock()
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        # 日経225銘柄コード読み込み（正確なリスト使用）
        self.nikkei225_codes = self._load_complete_nikkei225_codes()
        logger.info(f"日経225銘柄数: {len(self.nikkei225_codes)}銘柄")
        
        # 並列実行用の共有データ
        self.results_queue = Queue()
        self.progress_count = 0
        self.progress_lock = threading.Lock()
    
    def _load_complete_nikkei225_codes(self) -> List[str]:
        """正確な日経225銘柄コード読み込み（ユーザー提供リスト使用、4桁形式に変換）"""
        try:
            # ユーザー提供の正確なリストを読み込み
            df = pd.read_csv('/Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop/docment/ユーザー情報/nikkei225_4digit_list.csv')
            
            # 5桁コードを4桁コードに変換（最後の0を除去）
            codes = []
            for code_str in df['code'].astype(str):
                if len(code_str) == 5 and code_str.endswith('0'):
                    # 5桁コードの場合、最後の0を除去して4桁に変換
                    code_4digit = code_str[:-1]
                else:
                    # その他の場合は4桁でゼロパディング
                    code_4digit = code_str.zfill(4)
                codes.append(code_4digit)
            
            logger.info(f"日経225銘柄コード読み込み完了: {len(codes)}銘柄")
            logger.info(f"サンプルコード: {codes[:5]}")
            return codes
        except Exception as e:
            logger.error(f"日経225銘柄コード読み込みエラー: {e}")
            return []
    
    def _get_id_token(self) -> str:
        """IDトークンを取得（スレッドセーフ、リフレッシュトークン方式）"""
        with self.token_lock:
            if self.id_token:
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
                
                logger.info("認証トークン取得完了")
                return self.id_token
                
            except Exception as e:
                logger.error(f"認証エラー: {str(e)}")
                raise
    
    def _get_listed_companies(self) -> Dict[str, Dict[str, str]]:
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
                time.sleep(1)  # レート制限対策
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
                
                # 次のページがあるかチェック
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                    
            except Exception as e:
                logger.error(f"上場銘柄一覧取得エラー: {e}")
                time.sleep(5)
                continue
        
        logger.info(f"日経225銘柄のマッピング完了: {len(all_companies)}銘柄")
        return all_companies
    
    def _fetch_stock_data_worker(self, company_info: tuple, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """個別銘柄の株価データを取得（ワーカー関数）"""
        code_4digit, info = company_info
        api_code = info["api_code"]
        company_name = info["name"]
        
        try:
            token = self._get_id_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            url = f"{JQUANTS_BASE_URL}/prices/daily_quotes"
            params = {
                "code": api_code,
                "from": start_date,
                "to": end_date
            }
            
            all_data = []
            pagination_key = None
            
            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key
                
                try:
                    time.sleep(0.5)  # 並列実行用のより短い間隔
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    
                    if response.status_code == 429:
                        logger.warning(f"銘柄 {code_4digit}: レート制限により30秒待機...")
                        time.sleep(30)
                        continue
                    
                    if response.status_code == 404:
                        logger.warning(f"銘柄 {code_4digit}: データが見つかりません")
                        return None
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if "daily_quotes" in data:
                        all_data.extend(data["daily_quotes"])
                    
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                        
                except Exception as e:
                    logger.error(f"銘柄 {code_4digit} のデータ取得エラー: {e}")
                    time.sleep(2)
                    return None
            
            if all_data:
                df = pd.DataFrame(all_data)
                # データ前処理
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                
                # 必要な列のみ選択
                columns = ["Date", "Code", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "AdjustmentClose"]
                available_columns = [col for col in columns if col in df.columns]
                df = df[available_columns]
                
                # 数値型に変換
                numeric_columns = ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "AdjustmentClose"]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 4桁コードと企業名を追加
                df['Code'] = code_4digit
                df['CompanyName'] = company_name
                
                # 進捗更新
                with self.progress_lock:
                    self.progress_count += 1
                    logger.info(f"✅ {code_4digit} ({company_name}): {len(df)}件取得 [{self.progress_count}/{len(self.nikkei225_codes)}]")
                
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"銘柄 {code_4digit} の処理エラー: {e}")
            return None

    def _fetch_stock_data(self, code: str, company_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """指定銘柄の株価データを取得"""
        url = f"{JQUANTS_BASE_URL}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        params = {
            "code": code,
            "from": start_date,
            "to": end_date
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if "daily_quotes" not in data:
                return None
                
            df = pd.DataFrame(data["daily_quotes"])
            if df.empty:
                return None
            
            # データ前処理
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            
            # 必要な列のみ選択
            columns = ["Date", "Code", "Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "AdjustmentClose"]
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns]
            
            # 数値型に変換
            numeric_columns = ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "AdjustmentClose"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['CompanyName'] = company_name
            
            # 進捗更新
            with self.progress_lock:
                self.progress_count += 1
                logger.info(f"✅ {code} ({company_name}): {len(df)}件取得 [{self.progress_count}/{len(self.nikkei225_codes)}]")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ {code} ({company_name}): データ取得エラー - {e}")
            return None
    
    def _worker_thread(self, codes_companies: List[Tuple[str, str]], start_date: str, end_date: str):
        """ワーカースレッド関数"""
        for code, company_name in codes_companies:
            try:
                df = self._fetch_stock_data(code, company_name, start_date, end_date)
                if df is not None and not df.empty:
                    self.results_queue.put(df)
                time.sleep(0.1)  # API制限対応
            except Exception as e:
                logger.error(f"ワーカーエラー {code}: {e}")
    
    def fetch_complete_nikkei225_data(self, years: int = 10) -> pd.DataFrame:
        """日経225全225銘柄のデータを完全取得"""
        # 期間設定
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * years + 30)  # 余裕をもって30日追加
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"データ取得期間: {start_date_str} 〜 {end_date_str}")
        
        # 上場銘柄一覧を取得
        companies_mapping = self._get_listed_companies()
        
        if not companies_mapping:
            logger.error("銘柄マッピング取得に失敗")
            return pd.DataFrame()
            
        logger.info(f"日経225銘柄のマッピング完了: {len(companies_mapping)}銘柄")
        
        # 並列処理設定  
        max_workers = 8
        logger.info(f"並列ワーカー数: {max_workers}")
        logger.info(f"日経225全銘柄データ並列取得開始: {len(companies_mapping)}銘柄")
        
        # 並列実行
        all_stock_data = []
        company_items = list(companies_mapping.items())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全銘柄のタスクを投入
            futures = {
                executor.submit(self._fetch_stock_data_worker, company_info, start_date_str, end_date_str): company_info[0]
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
        all_dataframes = all_stock_data
        
        if not all_dataframes:
            logger.error("データが取得できませんでした")
            return pd.DataFrame()
        
        # 全データを結合
        result = pd.concat(all_dataframes, ignore_index=True)
        result = result.sort_values(['Code', 'Date']).reset_index(drop=True)
        
        # テクニカル指標追加
        logger.info("テクニカル指標計算開始...")
        result = self._add_technical_indicators(result)
        
        # データ保存
        output_file = f"nikkei225_complete_10years_{end_date_str}.csv"
        result.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        return result
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加"""
        result_dfs = []
        
        for code in df['Code'].unique():
            group = df[df['Code'] == code].copy()
            group = group.sort_values('Date').reset_index(drop=True)
            
            # 基本指標
            group['Returns'] = group['Close'].pct_change()
            group['Volume_MA_5'] = group['Volume'].rolling(window=5, min_periods=1).mean()
            
            # 移動平均
            for window in [5, 25, 75]:
                group[f'MA_{window}'] = group['Close'].rolling(window=window, min_periods=1).mean()
            
            # ボラティリティ
            group['Volatility_20'] = group['Returns'].rolling(window=20, min_periods=1).std()
            
            # RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            group['RSI_14'] = calculate_rsi(group['Close'])
            
            # MACD
            exp1 = group['Close'].ewm(span=12, min_periods=1).mean()
            exp2 = group['Close'].ewm(span=26, min_periods=1).mean()
            group['MACD'] = exp1 - exp2
            group['MACD_Signal'] = group['MACD'].ewm(span=9, min_periods=1).mean()
            group['MACD_Histogram'] = group['MACD'] - group['MACD_Signal']
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)

def main():
    """メイン処理"""
    try:
        fetcher = Nikkei225CompleteFetcher()
        
        # 日経225全225銘柄データ取得（10年間）
        df = fetcher.fetch_complete_nikkei225_data(years=10)
        
        if not df.empty:
            print(f"\n✅ 日経225全225銘柄データ完全取得完了")
            print(f"📊 データ概要:")
            print(f"  - 総レコード数: {len(df):,}")
            print(f"  - 銘柄数: {df['Code'].nunique()}")
            print(f"  - 期間: {df['Date'].min()} 〜 {df['Date'].max()}")
            print(f"  - 平均レコード数/銘柄: {len(df) // df['Code'].nunique():,}")
            
            # 取得銘柄の詳細表示
            company_counts = df.groupby(['Code', 'CompanyName']).size().reset_index(name='Count')
            print(f"\n📋 取得銘柄詳細:")
            for _, row in company_counts.iterrows():
                print(f"  {row['Code']}: {row['CompanyName']} ({row['Count']:,}件)")
                
        else:
            print("❌ データ取得に失敗しました")
            
    except Exception as e:
        logger.error(f"メイン処理エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()