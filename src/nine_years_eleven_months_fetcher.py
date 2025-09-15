"""
9年11ヶ月分のJ-Quantsデータ取得スクリプト
APIの10年制限を回避するため、9年11ヶ月間でデータを取得
期間: 2015年2月1日 〜 2025年1月31日
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import logging

import pandas as pd
import requests
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API設定
JQUANTS_BASE_URL = "https://api.jquants.com/v1"


class NineYearsElevenMonthsFetcher:
    """9年11ヶ月分のJ-Quantsデータを取得するクライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("9年11ヶ月分J-Quantsデータ取得クライアント初期化完了")
    
    def _get_id_token(self) -> str:
        """IDトークンを取得"""
        # トークンが有効な場合はそのまま返す
        if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.id_token
        
        logger.info("JQuants認証トークンを取得中...")
        time.sleep(3)  # レート制限対策
        
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
            
            # トークンの有効期限を設定
            self.token_expires_at = datetime.now() + timedelta(hours=1)
            
            logger.info("認証トークン取得完了")
            return self.id_token
            
        except Exception as e:
            logger.error(f"認証エラー: {str(e)}")
            raise
    
    def get_confirmed_working_stocks(self) -> List[str]:
        """
        前回確実に動作した銘柄コードリストを返す
        """
        # 前回成功した38銘柄（確実に動作する）
        working_stocks = [
            "72030", "99840", "67580", "94320", "83060", "80350", "63670", "79740",
            "99830", "40630", "65010", "72670", "69020", "80010", "29140", "45190",
            "45430", "69540", "83090", "45020", "68610", "49010", "45680", "62730", 
            "69200", "78320", "84110", "88020", "45230", "61780", "60980", "40050", 
            "45070", "69710", "68570", "69050", "80310", "90200"
        ]
        
        logger.info(f"確実動作銘柄リスト: {len(working_stocks)}銘柄")
        return working_stocks
    
    def get_daily_quotes_nine_years_eleven_months(
        self, 
        code: str,
        from_date: str = "2015-02-01",  # 9年11ヶ月前
        to_date: str = "2025-01-31"     # ちょうど9年11ヶ月後
    ) -> pd.DataFrame:
        """
        9年11ヶ月分の日次株価データ取得
        """
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            results: List[Dict] = []
            pagination_key: Optional[str] = None
            
            logger.info(f"銘柄 {code}: 9年11ヶ月分データ取得開始")
            logger.info(f"  期間: {from_date} ～ {to_date}")
            
            while True:
                params = {
                    "code": code,
                    "from": from_date,
                    "to": to_date
                }
                if pagination_key:
                    params["pagination_key"] = pagination_key
                
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
                
                if resp.status_code == 400:
                    logger.warning(f"銘柄 {code}: 無効なコード（400エラー）")
                    return pd.DataFrame()
                
                resp.raise_for_status()
                data = resp.json()
                
                items = data.get("daily_quotes", [])
                if items:
                    results.extend(items)
                    logger.info(f"  取得件数: {len(items)} (累計: {len(results)})")
                
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                
                # ページネーション間の待機
                time.sleep(0.3)
            
            if results:
                df = pd.DataFrame(results)
                logger.info(f"銘柄 {code}: 9年11ヶ月分データ取得完了 ({len(df):,}件)")
                return df
            else:
                logger.warning(f"銘柄 {code}: データなし")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"銘柄 {code} 取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def fetch_nine_years_eleven_months_data(self) -> pd.DataFrame:
        """
        9年11ヶ月分の大規模データを取得
        """
        logger.info("=== 9年11ヶ月分データ取得開始 ===")
        logger.info("期間: 2015年2月1日 ～ 2025年1月31日 (9年11ヶ月)")
        
        # Step 1: 確実に動作する銘柄リスト取得
        stock_codes = self.get_confirmed_working_stocks()
        
        # Step 2: データ取得
        all_stock_data = []
        failed_stocks = []
        
        # 中間保存用ディレクトリ
        intermediate_dir = Path("data/nine_years_eleven_months_data")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(stock_codes)}) - {idx/len(stock_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_nine_years_eleven_months(code)
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data):,}件取得成功")
                    
                    # 中間保存（10銘柄ごと）
                    if idx % 10 == 0:
                        intermediate_df = pd.concat(all_stock_data, ignore_index=True)
                        intermediate_file = intermediate_dir / f"intermediate_9y11m_{idx}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                        intermediate_df.to_pickle(intermediate_file)
                        logger.info(f"  💾 中間保存: {intermediate_file} ({len(intermediate_df):,}件)")
                
                else:
                    failed_stocks.append(code)
                    logger.warning(f"  ❌ 銘柄 {code}: データ取得失敗")
                
                # 銘柄間の待機（重要）
                wait_time = 4.0  # 4秒待機
                if idx % 5 == 0:
                    wait_time = 8.0  # 5銘柄ごとに長い待機
                    logger.info(f"  ⏸️  API制限対応で{wait_time:.1f}秒待機...")
                
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"  ❌ 銘柄 {code} でエラー: {str(e)}")
                failed_stocks.append(code)
                continue
        
        # Step 3: データ統合
        if not all_stock_data:
            raise RuntimeError("全ての銘柄でデータ取得に失敗しました")
        
        logger.info("=== データ統合中 ===")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("=== 9年11ヶ月分データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        if failed_stocks:
            logger.info(f"失敗銘柄リスト: {failed_stocks}")
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # Step 4: データ処理
        processed_df = self._process_data(combined_df)
        
        # Step 5: 保存
        output_dir = Path("data/nine_years_eleven_months_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nine_years_eleven_months_{len(all_stock_data)}stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        processed_df.to_pickle(output_file)
        
        logger.info(f"🎉 9年11ヶ月分データ保存完了: {output_file}")
        
        return processed_df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ処理"""
        logger.info("9年11ヶ月分データ処理開始...")
        
        # 列名標準化
        column_mapping = {
            'Date': 'date',
            'Code': 'symbol', 
            'Close': 'close_price',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Volume': 'volume',
            'AdjustmentFactor': 'adjustment_factor',
            'AdjustmentOpen': 'adj_open',
            'AdjustmentHigh': 'adj_high', 
            'AdjustmentLow': 'adj_low',
            'AdjustmentClose': 'adj_close',
            'AdjustmentVolume': 'adj_volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # データ型変換
        df['date'] = pd.to_datetime(df['date'])
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 重複除去・ソート
        df = df.drop_duplicates(subset=['date', 'symbol'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # リターン計算
        logger.info("リターン・ターゲット計算中...")
        df['daily_return'] = df.groupby('symbol')['close_price'].pct_change(fill_method=None)
        df['next_day_return'] = df.groupby('symbol')['close_price'].pct_change(fill_method=None).shift(-1)
        
        # ターゲット作成
        df['target'] = (df['next_day_return'] >= 0.01).astype(int)
        
        # 不完全なデータを除去
        df = df.dropna(subset=['close_price', 'next_day_return', 'target'])
        
        logger.info(f"9年11ヶ月分データ処理完了: {len(df):,}レコード")
        logger.info(f"対象銘柄数: {df['symbol'].nunique()}銘柄")
        logger.info(f"ターゲット分布: {df['target'].mean():.1%} (上昇)")
        logger.info(f"期間: {df['date'].min().date()} ～ {df['date'].max().date()}")
        
        # データ期間の詳細確認
        period_start = df['date'].min()
        period_end = df['date'].max()
        total_days = (period_end - period_start).days
        total_years = total_days / 365.25
        
        logger.info(f"実際のデータ期間: {total_days}日 ({total_years:.2f}年)")
        logger.info(f"期待期間: 9年11ヶ月 (約3,621日)")
        
        return df


def main():
    """メイン実行関数"""
    try:
        fetcher = NineYearsElevenMonthsFetcher()
        
        # 9年11ヶ月分のデータ取得
        nine_years_eleven_months_data = fetcher.fetch_nine_years_eleven_months_data()
        
        print("\n=== 9年11ヶ月分J-Quantsデータ取得結果 ===")
        print(f"総レコード数: {len(nine_years_eleven_months_data):,}件")
        print(f"銘柄数: {nine_years_eleven_months_data['symbol'].nunique()}銘柄") 
        print(f"期間: {nine_years_eleven_months_data['date'].min().date()} ～ {nine_years_eleven_months_data['date'].max().date()}")
        print(f"ターゲット分布: {nine_years_eleven_months_data['target'].mean():.1%}")
        
        # 期間確認
        period_start = nine_years_eleven_months_data['date'].min()
        period_end = nine_years_eleven_months_data['date'].max()
        total_days = (period_end - period_start).days
        total_years = total_days / 365.25
        print(f"実際の期間: {total_days}日 ({total_years:.2f}年)")
        
        print("✅ 9年11ヶ月分の100%実データで取得完了")
        
        return nine_years_eleven_months_data
        
    except Exception as e:
        logger.error(f"9年11ヶ月分データ取得に失敗: {str(e)}")
        print("❌ 9年11ヶ月分データ取得失敗")
        raise


if __name__ == "__main__":
    main()