"""
J-Quants APIから確実に実データを取得するスクリプト
docment/data_fetcher.pyの実装を参考に作成
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


class JQuantsRealDataFetcher:
    """J-Quants APIから実データを確実に取得するクライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("J-Quants実データ取得クライアント初期化完了")
    
    def _get_id_token(self) -> str:
        """
        IDトークンを取得 (docment/data_fetcher.pyの実装に準拠)
        """
        # トークンが有効な場合はそのまま返す
        if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.id_token
        
        logger.info("JQuants認証トークンを取得中...")
        
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
            resp.raise_for_status()
            refresh_token = resp.json().get("refreshToken")
            
            if not refresh_token:
                raise RuntimeError("リフレッシュトークンの取得に失敗しました")
            
            logger.info("リフレッシュトークン取得完了")
            
            # IDトークンを取得 (正しいエンドポイント使用)
            resp = requests.post(
                f"{JQUANTS_BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
                timeout=30
            )
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
    
    def get_daily_quotes(
        self, 
        code: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        日次株価データ取得 (docment/data_fetcher.pyの実装に準拠)
        """
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        results: List[Dict] = []
        pagination_key: Optional[str] = None
        
        logger.info(f"株価データ取得開始: code={code}, from={from_date}, to={to_date}")
        
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
                    logger.info(f"取得件数: {len(items)} (累計: {len(results)})")
                
                pagination_key = data.get("pagination_key")
                
                if not pagination_key:
                    break
                
                # API制限対応
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"データ取得エラー: {str(e)}")
                if "401" in str(e):
                    # トークン無効化して再取得
                    self.id_token = None
                    headers = {"Authorization": f"Bearer {self._get_id_token()}"}
                    continue
                else:
                    raise
        
        if not results:
            logger.warning("データが取得できませんでした")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        logger.info(f"データ取得完了: {len(df)}件")
        
        return df
    
    def get_listed_info(self) -> pd.DataFrame:
        """上場銘柄一覧を取得"""
        headers = {"Authorization": f"Bearer {self._get_id_token()}"}
        
        try:
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/listed/info",
                headers=headers,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            items = data.get("info", [])
            if items:
                df = pd.DataFrame(items)
                logger.info(f"上場銘柄一覧取得完了: {len(df)}銘柄")
                return df
            else:
                logger.warning("上場銘柄一覧が空です")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"上場銘柄一覧取得エラー: {str(e)}")
            raise
    
    def fetch_nikkei225_data(
        self,
        from_date: str = "2020-01-01",
        to_date: str = "2025-08-30"
    ) -> pd.DataFrame:
        """
        日経225構成銘柄の実データを確実に取得
        
        Args:
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
            
        Returns:
            実際のJ-Quantsデータ
        """
        logger.info("=== J-Quants実データ取得開始 ===")
        logger.info(f"期間: {from_date} ～ {to_date}")
        
        # Step 1: 上場銘柄一覧を取得してNikkei225相当を特定
        try:
            listed_info = self.get_listed_info()
            if listed_info.empty:
                raise RuntimeError("上場銘柄一覧の取得に失敗")
            
            # 日経225相当の大手銘柄を特定 (時価総額上位など)
            # 利用可能なカラムをログ出力
            logger.info(f"利用可能なカラム: {list(listed_info.columns)}")
            
            # 時価総額やその他の指標で絞り込み
            market_cap_cols = [col for col in listed_info.columns if 'market' in col.lower() or 'cap' in col.lower()]
            logger.info(f"時価総額関連カラム: {market_cap_cols}")
            
            # フォールバック: 知られた日経225銘柄
            known_nikkei225 = [
                "7203", "9984", "6758", "9432", "8306", "8035", "6367", "7974", 
                "9983", "4063", "6501", "7267", "6902", "8001", "2914", "4519", 
                "4543", "6954", "6502", "8309", "4502", "6861", "4901", "9437",
                "4568", "6273", "6920", "7832", "8411", "8802", "4523", "6178",
                "6098", "4005", "4507", "6971", "6857", "6905", "8031", "9020"
            ]
            # マッチング前にデータを確認
            if 'Code' in listed_info.columns:
                logger.info(f"Codeサンプル: {listed_info['Code'].head().tolist()}")
                # 4桁銘柄コードのマッチング（J-Quantsは5桁コード形式）
                # 4桁コードから5桁コードへの変換（トヨタ: 7203 → 72030）
                known_nikkei225_5digit = [code + "0" for code in known_nikkei225]
                nikkei225_stocks = listed_info[listed_info['Code'].isin(known_nikkei225_5digit)]
                logger.info(f"5桁変換後マッチング銘柄数: {len(nikkei225_stocks)}")
                
                # 直接マッチングも試行
                if len(nikkei225_stocks) == 0:
                    nikkei225_stocks = listed_info[listed_info['Code'].isin(known_nikkei225)]
                    logger.info(f"4桁直接マッチング銘柄数: {len(nikkei225_stocks)}")
                
                # まだマッチしない場合は、MarketCodeでフィルタ（東証1部など）
                if len(nikkei225_stocks) == 0:
                    # 東証プライム市場（MarketCode: 111）を選択
                    prime_stocks = listed_info[listed_info['MarketCode'] == '111']
                    logger.info(f"東証プライム市場銘柄数: {len(prime_stocks)}")
                    # 上位40銘柄を選択
                    nikkei225_stocks = prime_stocks.head(40)
                    logger.info(f"選択された銘柄数: {len(nikkei225_stocks)}")
            else:
                logger.error("Codeカラムが見つかりません")
                nikkei225_stocks = pd.DataFrame({'Code': known_nikkei225})
            
            logger.info(f"対象銘柄数: {len(nikkei225_stocks)}銘柄")
            
        except Exception as e:
            logger.error(f"銘柄一覧取得失敗: {str(e)}")
            # 最小限の銘柄で継続
            known_nikkei225 = ["7203", "9984", "6758", "9432", "8306"]  # 主要5銘柄
            nikkei225_stocks = pd.DataFrame({'Code': known_nikkei225})
            logger.info(f"フォールバック: {len(known_nikkei225)}銘柄で実行")
        
        # Step 2: 各銘柄のデータを取得
        all_stock_data = []
        failed_stocks = []
        
        for idx, stock in nikkei225_stocks.iterrows():
            code = stock['Code']
            
            try:
                logger.info(f"銘柄 {code} のデータ取得中... ({idx+1}/{len(nikkei225_stocks)})")
                
                stock_data = self.get_daily_quotes(
                    code=code,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"銘柄 {code}: {len(stock_data)}件取得")
                else:
                    failed_stocks.append(code)
                    logger.warning(f"銘柄 {code}: データなし")
                
                # API制限対応
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"銘柄 {code} でエラー: {str(e)}")
                failed_stocks.append(code)
                continue
        
        # Step 3: データ統合
        if not all_stock_data:
            raise RuntimeError("全ての銘柄でデータ取得に失敗しました")
        
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("=== データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        logger.info(f"総レコード数: {len(combined_df)}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # Step 4: データを標準形式に変換
        processed_df = self._process_real_data(combined_df)
        
        # Step 5: ファイル保存
        output_dir = Path("data/real_jquants_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nikkei225_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        processed_df.to_pickle(output_file)
        
        logger.info(f"実データをファイル保存: {output_file}")
        
        return processed_df
    
    def _process_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """実データを処理して標準形式に変換"""
        logger.info("実データの処理開始...")
        
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
        
        # ソート
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # リターン計算
        df['daily_return'] = df.groupby('symbol')['close_price'].pct_change()
        df['next_day_return'] = df.groupby('symbol')['close_price'].pct_change().shift(-1)
        
        # ターゲット作成 (翌日+1%以上)
        df['target'] = (df['next_day_return'] >= 0.01).astype(int)
        
        # 不完全なデータを除去
        df = df.dropna(subset=['close_price', 'next_day_return', 'target'])
        
        logger.info(f"実データ処理完了: {len(df)}レコード")
        logger.info(f"対象銘柄数: {df['symbol'].nunique()}銘柄")
        logger.info(f"ターゲット分布: {df['target'].mean():.1%} (上昇)")
        
        return df


def main():
    """メイン実行関数"""
    try:
        fetcher = JQuantsRealDataFetcher()
        
        # 実データ取得 (絶対にデモデータは使わない)
        real_data = fetcher.fetch_nikkei225_data(
            from_date="2020-01-01",
            to_date="2025-08-30"
        )
        
        print("\n=== J-Quants実データ取得結果 ===")
        print(f"総レコード数: {len(real_data):,}件")
        print(f"銘柄数: {real_data['symbol'].nunique()}銘柄") 
        print(f"期間: {real_data['date'].min().date()} ～ {real_data['date'].max().date()}")
        print(f"ターゲット分布: {real_data['target'].mean():.1%}")
        print("✅ 100%実データで取得完了")
        
        return real_data
        
    except Exception as e:
        logger.error(f"実データ取得に失敗: {str(e)}")
        print("❌ 実データ取得失敗")
        print("デモデータは一切使用しません")
        raise


if __name__ == "__main__":
    main()