"""
データ取得モジュール
J-Quants APIからの株価データ取得を行う
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

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

# API設定
JQUANTS_BASE_URL = "https://api.jquants.com/v1"


class JQuantsClient:
    """J-Quants APIクライアント"""
    
    def __init__(self, mail_address: Optional[str] = None, password: Optional[str] = None):
        """
        初期化
        
        Args:
            mail_address: JQuantsメールアドレス（環境変数から取得可能）
            password: JQuantsパスワード（環境変数から取得可能）
        """
        self.mail_address = mail_address or os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = password or os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません")
    
    def _get_id_token(self) -> str:
        """
        IDトークンを取得
        
        Returns:
            IDトークン文字列
        """
        # トークンが有効な場合はそのまま返す
        if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.id_token
        
        logger.info("JQuants認証トークンを取得中...")
        
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
        resp.raise_for_status()
        refresh_token = resp.json().get("refreshToken")
        
        if not refresh_token:
            raise RuntimeError("リフレッシュトークンの取得に失敗しました")
        
        # IDトークンを取得
        resp = requests.post(
            f"{JQUANTS_BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
            timeout=30
        )
        resp.raise_for_status()
        self.id_token = resp.json().get("idToken")
        
        if not self.id_token:
            raise RuntimeError("IDトークンの取得に失敗しました")
        
        # トークンの有効期限を設定（通常1時間）
        self.token_expires_at = datetime.now() + timedelta(hours=1)
        
        logger.info("認証トークン取得完了")
        return self.id_token
    
    def get_daily_quotes(
        self, 
        code: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        日次株価データ取得
        
        Args:
            code: 銘柄コード（4桁または5桁）
            from_date: 開始日（YYYY-MM-DD形式）
            to_date: 終了日（YYYY-MM-DD形式）
            date: 特定日（YYYY-MM-DD形式）
            
        Returns:
            株価データのDataFrame
        """
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        results: List[Dict] = []
        pagination_key: Optional[str] = None
        
        while True:
            params: Dict[str, str] = {}
            
            if code:
                params["code"] = code
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date
            if date:
                params["date"] = date
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                resp = requests.get(
                    f"{JQUANTS_BASE_URL}/prices/daily_quotes",
                    headers=headers,
                    params=params,
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                
                items = data.get("daily_quotes", [])
                if items:
                    results.extend(items)
                
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                    
                time.sleep(0.2)  # API負荷軽減
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API呼び出しエラー: {e}")
                raise
        
        if not results:
            logger.warning(f"データが取得できませんでした: code={code}, from={from_date}, to={to_date}")
            return pd.DataFrame()
        
        # DataFrameに変換して整形
        df = pd.DataFrame(results)
        return self._normalize_quotes(df)
    
    def get_listed_info(self, code: Optional[str] = None) -> pd.DataFrame:
        """
        上場銘柄情報取得
        
        Args:
            code: 銘柄コード（指定しない場合は全銘柄）
            
        Returns:
            銘柄情報のDataFrame
        """
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        params = {}
        if code:
            params["code"] = code
        
        try:
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/listed/info",
                headers=headers,
                params=params,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            items = data.get("info", [])
            if not items:
                logger.warning("銘柄情報が取得できませんでした")
                return pd.DataFrame()
            
            return pd.DataFrame(items)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"銘柄情報取得エラー: {e}")
            raise
    
    def get_trading_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        取引カレンダー取得
        
        Args:
            from_date: 開始日（YYYY-MM-DD形式）
            to_date: 終了日（YYYY-MM-DD形式）
            
        Returns:
            取引カレンダーのDataFrame
        """
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        params = {}
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        try:
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/markets/trading_calendar",
                headers=headers,
                params=params,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            records = data.get("trading_calendar") or data.get("data") or []
            if not records:
                logger.warning("取引カレンダーが取得できませんでした")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # 日付カラムを特定して変換
            date_col = next((c for c in ["Date", "date", "biz_date", "BusinessDay"] 
                            if c in df.columns), None)
            if date_col:
                df["Date"] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: "Date"})
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"取引カレンダー取得エラー: {e}")
            raise
    
    @staticmethod
    def _normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データの正規化
        
        Args:
            df: 生の株価データ
            
        Returns:
            正規化されたDataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        # カラム名の統一
        column_mapping = {
            "Date": "date",
            "Code": "code",
            "Open": "open",
            "High": "high", 
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "TurnoverValue": "turnover_value",
            "AdjustmentFactor": "adjustment_factor",
            "AdjustmentOpen": "adj_open",
            "AdjustmentHigh": "adj_high",
            "AdjustmentLow": "adj_low",
            "AdjustmentClose": "adj_close",
            "AdjustmentVolume": "adj_volume"
        }
        
        df = df.rename(columns=column_mapping)
        
        # 必須カラムの確認
        required_cols = ["date", "code", "open", "high", "low", "close", "volume"]
        existing_cols = [col for col in required_cols if col in df.columns]
        
        if len(existing_cols) < len(required_cols):
            logger.warning(f"必須カラムが不足: {set(required_cols) - set(existing_cols)}")
        
        # データ型変換
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(4)
        
        # 数値カラムの変換
        numeric_cols = ["open", "high", "low", "close", "volume", 
                       "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # ソート
        if "date" in df.columns and "code" in df.columns:
            df = df.sort_values(["code", "date"])
        
        return df


class DataFetcher:
    """データ取得管理クラス"""
    
    def __init__(self, client: Optional[JQuantsClient] = None):
        """
        初期化
        
        Args:
            client: JQuantsクライアント（指定しない場合は新規作成）
        """
        self.client = client or JQuantsClient()
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
    
    def fetch_nikkei225_data(
        self,
        from_date: str,
        to_date: str,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """
        日経225銘柄のデータを取得
        
        Args:
            from_date: 開始日（YYYY-MM-DD形式）
            to_date: 終了日（YYYY-MM-DD形式）
            save_to_file: ファイルに保存するか
            
        Returns:
            全銘柄の株価データ
        """
        # 日経225銘柄リストを読み込み
        nikkei_codes = self._load_nikkei225_codes()
        
        if not nikkei_codes:
            logger.error("日経225銘柄リストが空です")
            return pd.DataFrame()
        
        logger.info(f"日経225銘柄データ取得開始: {len(nikkei_codes)}銘柄")
        logger.info(f"期間: {from_date} ～ {to_date}")
        
        all_data = []
        failed_codes = []
        
        for i, code in enumerate(nikkei_codes, 1):
            try:
                logger.info(f"[{i}/{len(nikkei_codes)}] 銘柄 {code} 取得中...")
                
                df = self.client.get_daily_quotes(
                    code=code,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  → {len(df)}行取得")
                else:
                    logger.warning(f"  → データなし")
                
                # API負荷軽減のため待機
                if i % 10 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"銘柄 {code} の取得失敗: {e}")
                failed_codes.append(code)
                continue
        
        if failed_codes:
            logger.warning(f"取得失敗銘柄: {failed_codes}")
        
        if not all_data:
            logger.error("データが1件も取得できませんでした")
            return pd.DataFrame()
        
        # 全データを結合
        result_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"取得完了: 全{len(result_df)}行")
        
        # ファイル保存
        if save_to_file:
            filename = f"nikkei225_quotes_{from_date}_{to_date}.parquet"
            filepath = self.data_dir / filename
            result_df.to_parquet(filepath, compression="snappy")
            logger.info(f"データ保存完了: {filepath}")
        
        return result_df
    
    def update_latest_data(self, lookback_days: int = 7) -> pd.DataFrame:
        """
        最新データの差分更新
        
        Args:
            lookback_days: 取得する過去日数
            
        Returns:
            更新されたデータ
        """
        to_date = date.today().strftime("%Y-%m-%d")
        from_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        logger.info(f"最新データ取得: {from_date} ～ {to_date}")
        
        return self.fetch_nikkei225_data(from_date, to_date)
    
    def _load_nikkei225_codes(self) -> List[str]:
        """
        日経225銘柄コードリストを読み込み
        
        Returns:
            銘柄コードのリスト
        """
        # 複数の可能なパスを試す
        possible_paths = [
            self.data_dir / "nikkei_225_stock_list.csv",
            Path("./nikkei_225_stock_list.csv"),
            Path("../reference/data/nikkei_225_stock_list.csv"),
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path, dtype={"code": str})
                    codes = df["code"].astype(str).str.strip().tolist()
                    logger.info(f"銘柄リスト読み込み完了: {len(codes)}銘柄 from {path}")
                    return codes
                except Exception as e:
                    logger.error(f"銘柄リスト読み込みエラー: {e}")
                    continue
        
        # デフォルトの日経225銘柄コード（主要銘柄のみ）
        default_codes = [
            "7203", "6758", "9984", "6861", "7267", "8306", "9432", "7974",
            "4502", "6501", "6902", "6954", "7751", "8035", "9433", "4063"
        ]
        
        logger.warning(f"銘柄リストファイルが見つからないため、デフォルトリストを使用: {len(default_codes)}銘柄")
        return default_codes


def main():
    """メイン実行関数"""
    # データ取得テスト
    fetcher = DataFetcher()
    
    # 過去30日のデータを取得
    to_date = date.today().strftime("%Y-%m-%d")
    from_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    df = fetcher.fetch_nikkei225_data(from_date, to_date)
    
    if not df.empty:
        print(f"データ取得成功: {len(df)}行")
        print(f"期間: {df['date'].min()} ～ {df['date'].max()}")
        print(f"銘柄数: {df['code'].nunique()}")
        print("\nサンプルデータ:")
        print(df.head())
    else:
        print("データ取得失敗")


if __name__ == "__main__":
    main()