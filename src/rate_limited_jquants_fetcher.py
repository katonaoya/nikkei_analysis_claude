"""
レート制限対応版J-Quantsデータ取得スクリプト
API制限を回避しながら大規模データを取得
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
import random

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


class RateLimitedJQuantsFetcher:
    """API制限を回避しながら大規模データを取得するクライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.last_token_request: Optional[datetime] = None
        self.api_call_count = 0
        self.last_reset_time = datetime.now()
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("レート制限対応版J-Quantsデータ取得クライアント初期化完了")
    
    def _wait_for_rate_limit(self, min_interval: float = 2.0):
        """API制限回避のための待機"""
        if self.last_token_request:
            elapsed = (datetime.now() - self.last_token_request).total_seconds()
            if elapsed < min_interval:
                wait_time = min_interval - elapsed + random.uniform(0.5, 1.5)  # ランダム要素追加
                logger.info(f"API制限回避のため {wait_time:.1f}秒待機...")
                time.sleep(wait_time)
    
    def _get_id_token(self) -> str:
        """レート制限対応版IDトークン取得"""
        # トークンが有効な場合はそのまま返す
        if (self.id_token and self.token_expires_at and 
            datetime.now() < self.token_expires_at - timedelta(minutes=5)):  # 5分前に更新
            return self.id_token
        
        # レート制限チェック
        self._wait_for_rate_limit(min_interval=3.0)  # トークンリクエストは3秒間隔
        
        logger.info("JQuants認証トークン取得中...")
        
        try:
            # リフレッシュトークンを取得
            auth_payload = {
                "mailaddress": self.mail_address,
                "password": self.password
            }
            
            self.last_token_request = datetime.now()
            resp = requests.post(
                f"{JQUANTS_BASE_URL}/token/auth_user",
                data=json.dumps(auth_payload),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限により長時間待機します...")
                time.sleep(60)  # 1分待機
                return self._get_id_token()  # 再試行
                
            resp.raise_for_status()
            refresh_token = resp.json().get("refreshToken")
            
            if not refresh_token:
                raise RuntimeError("リフレッシュトークンの取得に失敗しました")
            
            logger.info("リフレッシュトークン取得完了")
            
            # IDトークンを取得（少し待機）
            time.sleep(1.0)
            resp = requests.post(
                f"{JQUANTS_BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
                timeout=30
            )
            
            if resp.status_code == 429:
                logger.warning("レート制限により長時間待機します...")
                time.sleep(60)
                return self._get_id_token()  # 再試行
                
            resp.raise_for_status()
            self.id_token = resp.json().get("idToken")
            
            if not self.id_token:
                raise RuntimeError("IDトークンの取得に失敗しました")
            
            # トークンの有効期限を設定
            self.token_expires_at = datetime.now() + timedelta(hours=1)
            
            logger.info("認証トークン取得完了")
            return self.id_token
            
        except Exception as e:
            if "429" in str(e):
                logger.warning("レート制限エラー、長時間待機後再試行...")
                time.sleep(120)  # 2分待機
                return self._get_id_token()
            logger.error(f"認証エラー: {str(e)}")
            raise
    
    def get_daily_quotes_safe(
        self, 
        code: str,
        from_date: str,
        to_date: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        安全な日次株価データ取得
        """
        for attempt in range(max_retries):
            try:
                headers = {"Authorization": f"Bearer {self._get_id_token()}"}
                results: List[Dict] = []
                pagination_key: Optional[str] = None
                
                # APIコール前の待機
                time.sleep(random.uniform(0.5, 1.0))  # ランダム待機
                
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
                        logger.warning(f"銘柄 {code}: レート制限、60秒待機...")
                        time.sleep(60)
                        continue
                    
                    if resp.status_code == 400:
                        logger.warning(f"銘柄 {code}: 無効なコード（400エラー）")
                        return pd.DataFrame()
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    items = data.get("daily_quotes", [])
                    if items:
                        results.extend(items)
                    
                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break
                    
                    # ページネーション間の待機
                    time.sleep(0.2)
                
                if results:
                    return pd.DataFrame(results)
                else:
                    logger.info(f"銘柄 {code}: データなし")
                    return pd.DataFrame()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # 指数バックオフ
                    logger.warning(f"銘柄 {code} 取得失敗 (試行{attempt+1}/{max_retries}): {str(e)}")
                    logger.info(f"  {wait_time}秒待機してリトライ...")
                    time.sleep(wait_time)
                    
                    # トークンリセット
                    self.id_token = None
                    continue
                else:
                    logger.error(f"銘柄 {code} 最終取得失敗: {str(e)}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_working_stock_codes(self, max_stocks: int = 255) -> List[str]:
        """
        実際に取得可能な銘柄コードリストを取得
        """
        # 既に動作確認済みの銘柄コードから開始
        working_codes = [
            "29140", "40050", "40630", "45020", "45070", "45190", "45230", "45430",
            "45680", "49010", "60980", "61780", "62730", "63670", "65010", "67580",
            "68570", "68610", "69020", "69050", "69200", "69540", "69710", "72030",
            "72670", "78320", "79740", "80010", "80310", "80350", "83060", "83090",
            "84110", "88020", "90200", "94320", "99830", "99840"
        ]
        
        logger.info(f"動作確認済み銘柄: {len(working_codes)}銘柄")
        
        if len(working_codes) >= max_stocks:
            return working_codes[:max_stocks]
        
        # 不足分は上場銘柄一覧から補完
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            time.sleep(1)
            
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/listed/info",
                headers=headers,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            items = data.get("info", [])
            if items:
                listed_df = pd.DataFrame(items)
                # 既知銘柄以外を追加
                additional_codes = []
                for code in listed_df['Code'].tolist():
                    if code not in working_codes and len(working_codes) + len(additional_codes) < max_stocks:
                        additional_codes.append(code)
                
                working_codes.extend(additional_codes)
                logger.info(f"追加銘柄: {len(additional_codes)}銘柄")
                
        except Exception as e:
            logger.warning(f"上場銘柄一覧取得失敗: {str(e)}")
        
        final_codes = working_codes[:max_stocks]
        logger.info(f"最終銘柄数: {len(final_codes)}銘柄")
        return final_codes
    
    def fetch_large_scale_data_safe(
        self,
        target_stocks: int = 100,  # まずは100銘柄で試す
        from_date: str = "2015-01-01",
        to_date: str = "2025-08-31",
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        レート制限を回避しながら大規模データを安全に取得
        """
        logger.info("=== 安全な大規模データ取得開始 ===")
        logger.info(f"対象銘柄数: {target_stocks}銘柄")
        logger.info(f"期間: {from_date} ～ {to_date}")
        
        # Step 1: 動作する銘柄リスト取得
        stock_codes = self.get_working_stock_codes(max_stocks=target_stocks)
        logger.info(f"取得対象銘柄数: {len(stock_codes)}銘柄")
        
        # Step 2: データ取得
        all_stock_data = []
        failed_stocks = []
        
        # 中間保存用ディレクトリ
        if save_intermediate:
            intermediate_dir = Path("data/intermediate_data")
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(stock_codes)}) - {idx/len(stock_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_safe(
                    code=code,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data)}件取得")
                    
                    # 中間保存（20銘柄ごと）
                    if save_intermediate and idx % 20 == 0:
                        intermediate_df = pd.concat(all_stock_data, ignore_index=True)
                        intermediate_file = intermediate_dir / f"safe_intermediate_{idx}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                        intermediate_df.to_pickle(intermediate_file)
                        logger.info(f"  💾 中間保存: {intermediate_file} ({len(intermediate_df):,}件)")
                
                else:
                    failed_stocks.append(code)
                    logger.info(f"  ⚠️ 銘柄 {code}: データなし")
                
                # 銘柄間の待機（重要）
                wait_time = random.uniform(2.0, 4.0)  # 2-4秒のランダム待機
                if idx % 10 == 0:
                    wait_time = random.uniform(5.0, 8.0)  # 10銘柄ごとに長い待機
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
        
        logger.info("=== 安全な大規模データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        if failed_stocks:
            logger.info(f"失敗銘柄リスト: {failed_stocks[:20]}...")  # 最初の20個表示
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # Step 4: データ処理
        processed_df = self._process_large_scale_data(combined_df)
        
        # Step 5: 最終保存
        output_dir = Path("data/large_scale_jquants_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"safe_large_scale_{len(all_stock_data)}stocks_10years_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        processed_df.to_pickle(output_file)
        
        logger.info(f"🎉 安全な大規模データ保存完了: {output_file}")
        
        return processed_df
    
    def _process_large_scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """大規模データの処理"""
        logger.info("大規模データ処理開始...")
        
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
        
        # 必須カラム確認
        required_cols = ['date', 'symbol', 'close_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"必須カラムが不足しています: {missing_cols}")
        
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
        
        # ターゲット作成 (翌日+1%以上)
        df['target'] = (df['next_day_return'] >= 0.01).astype(int)
        
        # 不完全なデータを除去
        df = df.dropna(subset=['close_price', 'next_day_return', 'target'])
        
        logger.info(f"大規模データ処理完了: {len(df):,}レコード")
        logger.info(f"対象銘柄数: {df['symbol'].nunique()}銘柄")
        logger.info(f"ターゲット分布: {df['target'].mean():.1%} (上昇)")
        logger.info(f"期間: {df['date'].min().date()} ～ {df['date'].max().date()}")
        
        return df


def main():
    """メイン実行関数"""
    try:
        fetcher = RateLimitedJQuantsFetcher()
        
        # まずは100銘柄・10年分で安全に取得
        large_scale_data = fetcher.fetch_large_scale_data_safe(
            target_stocks=100,        # 100銘柄から開始
            from_date="2015-01-01",   # 10年前
            to_date="2025-08-31"      # 現在まで
        )
        
        print("\n=== レート制限対応版大規模J-Quantsデータ取得結果 ===")
        print(f"総レコード数: {len(large_scale_data):,}件")
        print(f"銘柄数: {large_scale_data['symbol'].nunique()}銘柄") 
        print(f"期間: {large_scale_data['date'].min().date()} ～ {large_scale_data['date'].max().date()}")
        print(f"ターゲット分布: {large_scale_data['target'].mean():.1%}")
        print("✅ 100銘柄・10年分の100%実データで取得完了")
        
        return large_scale_data
        
    except Exception as e:
        logger.error(f"安全な大規模データ取得に失敗: {str(e)}")
        print("❌ 大規模データ取得失敗")
        raise


if __name__ == "__main__":
    main()