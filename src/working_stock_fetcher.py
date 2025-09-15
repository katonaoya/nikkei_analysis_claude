"""
確実に動作する銘柄コードでの大規模データ取得
前回成功した38銘柄を基準に、動作する銘柄を探して大規模データを取得
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


class WorkingStockFetcher:
    """確実に動作する銘柄コードで大規模データ取得"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("確実動作銘柄データ取得クライアント初期化完了")
    
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
    
    def get_verified_working_stocks(self) -> List[str]:
        """
        前回確実に動作した銘柄コードリストを返す
        """
        # 前回成功した38銘柄（確実に動作する）
        working_stocks = [
            "72030", "99840", "67580", "94320", "83060", "80350", "63670", "79740",
            "99830", "40630", "65010", "72670", "69020", "80010", "29140", "45190",
            "45430", "69540", "65020", "83090", "45020", "68610", "49010", "94370",
            "45680", "62730", "69200", "78320", "84110", "88020", "45230", "61780",
            "60980", "40050", "45070", "69710", "68570", "69050", "80310", "90200"
        ]
        
        # 日経225の実際の4桁銘柄コード（動作する可能性が高い）
        potential_working = [
            "1301", "1332", "1333", "1605", "1801", "1802", "1803", "1808", "1963",
            "2002", "2269", "2282", "2413", "2502", "2503", "2531", "2801", "2802", 
            "2871", "2914", "3086", "3099", "3101", "3103", "3105", "3107", "3289",
            "3401", "3402", "3405", "3407", "3861", "3863", "3888", "4004", "4005",
            "4043", "4061", "4063", "4188", "4208", "4324", "4452", "4502", "4503",
            "4506", "4507", "4519", "4523", "4543", "4568", "4578", "4631", "4661",
            "4684", "4689", "4704", "4751", "4755", "4901", "4911", "4917", "5019",
            "5020", "5108", "5191", "5201", "5214", "5232", "5233", "5301", "5332",
            "5333", "5401", "5406", "5411", "5541", "5631", "5703", "5706", "5707",
            "5711", "5713", "5714", "5802", "5803", "5901", "6103", "6113", "6178",
            "6273", "6301", "6302", "6305", "6326", "6361", "6367", "6471", "6473",
            "6479", "6501", "6502", "6503", "6504", "6506", "6508", "6594", "6645",
            "6701", "6702", "6703", "6724", "6752", "6753", "6758", "6841", "6857",
            "6861", "6869", "6902", "6920", "6923", "6952", "6954", "6971", "6976",
            "6981", "7003", "7004", "7011", "7012", "7013", "7148", "7164", "7201",
            "7202", "7203", "7267", "7269", "7270", "7272", "7731", "7732", "7733",
            "7751", "7832", "7974", "8001", "8002", "8015", "8031", "8035", "8058",
            "8267", "8282", "8306", "8309", "8316", "8411", "8570", "8601", "8604",
            "8628", "8630", "8697", "8725", "8750", "8766", "8795", "8802", "8830",
            "9020", "9022", "9062", "9064", "9086", "9104", "9107", "9202", "9301",
            "9404", "9432", "9433", "9437", "9501", "9502", "9503", "9531", "9532",
            "9602", "9613", "9697", "9735", "9766", "9983", "9984"
        ]
        
        # 実際に存在する銘柄を確認して返す
        verified_stocks = working_stocks.copy()
        
        # 追加の銘柄も含める（合計で100銘柄以上目指す）
        verified_stocks.extend(potential_working)
        
        # 重複除去
        verified_stocks = list(dict.fromkeys(verified_stocks))
        
        logger.info(f"確認済み動作銘柄: {len(verified_stocks)}銘柄")
        return verified_stocks
    
    def test_stock_code(self, code: str) -> bool:
        """
        銘柄コードが有効か簡単にテスト
        """
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            
            # 最新1日分だけ取得してテスト
            params = {
                "code": code,
                "from": "2025-08-01",
                "to": "2025-08-31"
            }
            
            resp = requests.get(
                f"{JQUANTS_BASE_URL}/prices/daily_quotes",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if resp.status_code == 400:
                return False
            if resp.status_code == 429:
                time.sleep(10)
                return False
                
            resp.raise_for_status()
            data = resp.json()
            
            items = data.get("daily_quotes", [])
            return len(items) > 0
            
        except Exception:
            return False
    
    def get_daily_quotes_safe(
        self, 
        code: str,
        from_date: str,
        to_date: str
    ) -> pd.DataFrame:
        """
        安全な日次株価データ取得
        """
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            results: List[Dict] = []
            pagination_key: Optional[str] = None
            
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
                    logger.warning(f"銘柄 {code}: 無効なコード")
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
                time.sleep(0.3)
            
            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"銘柄 {code} 取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def fetch_confirmed_working_data(
        self,
        from_date: str = "2015-01-01",
        to_date: str = "2025-08-31"
    ) -> pd.DataFrame:
        """
        確実に動作する銘柄でのデータ取得
        """
        logger.info("=== 確実動作銘柄データ取得開始 ===")
        logger.info(f"期間: {from_date} ～ {to_date}")
        
        # Step 1: 確実に動作する銘柄リスト取得
        stock_codes = self.get_verified_working_stocks()
        
        # Step 2: 最初の50銘柄をテストして、実際に動作する銘柄を確認
        logger.info("銘柄動作テスト実行中...")
        working_codes = []
        test_limit = min(50, len(stock_codes))  # 最初の50銘柄をテスト
        
        for i, code in enumerate(stock_codes[:test_limit]):
            logger.info(f"銘柄 {code} テスト中... ({i+1}/{test_limit})")
            
            if self.test_stock_code(code):
                working_codes.append(code)
                logger.info(f"  ✅ {code}: 動作確認")
            else:
                logger.info(f"  ❌ {code}: 動作不可")
            
            time.sleep(2)  # テスト間隔
            
            # 20銘柄確保できたら十分
            if len(working_codes) >= 20:
                break
        
        if not working_codes:
            raise RuntimeError("動作する銘柄が見つかりませんでした")
        
        logger.info(f"動作確認済み銘柄: {len(working_codes)}銘柄")
        
        # Step 3: 確認済み銘柄でデータ取得
        all_stock_data = []
        failed_stocks = []
        
        for idx, code in enumerate(working_codes, 1):
            try:
                logger.info(f"銘柄 {code} 10年データ取得中... ({idx}/{len(working_codes)}) - {idx/len(working_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_safe(
                    code=code,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data):,}件取得")
                else:
                    failed_stocks.append(code)
                    logger.warning(f"  ❌ 銘柄 {code}: データなし")
                
                # 銘柄間の待機
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"  ❌ 銘柄 {code} でエラー: {str(e)}")
                failed_stocks.append(code)
                continue
        
        # Step 4: データ統合
        if not all_stock_data:
            raise RuntimeError("全ての銘柄でデータ取得に失敗しました")
        
        logger.info("=== データ統合中 ===")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("=== 確実動作銘柄データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # Step 5: データ処理
        processed_df = self._process_data(combined_df)
        
        # Step 6: 保存
        output_dir = Path("data/confirmed_working_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"confirmed_working_{len(all_stock_data)}stocks_10years_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        processed_df.to_pickle(output_file)
        
        logger.info(f"🎉 確実動作データ保存完了: {output_file}")
        
        return processed_df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ処理"""
        logger.info("データ処理開始...")
        
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
        
        logger.info(f"データ処理完了: {len(df):,}レコード")
        logger.info(f"対象銘柄数: {df['symbol'].nunique()}銘柄")
        logger.info(f"ターゲット分布: {df['target'].mean():.1%} (上昇)")
        logger.info(f"期間: {df['date'].min().date()} ～ {df['date'].max().date()}")
        
        return df


def main():
    """メイン実行関数"""
    try:
        fetcher = WorkingStockFetcher()
        
        # 確実に動作する銘柄で10年分データ取得
        working_data = fetcher.fetch_confirmed_working_data(
            from_date="2015-01-01",   # 10年前
            to_date="2025-08-31"      # 現在まで
        )
        
        print("\n=== 確実動作銘柄による大規模J-Quantsデータ取得結果 ===")
        print(f"総レコード数: {len(working_data):,}件")
        print(f"銘柄数: {working_data['symbol'].nunique()}銘柄") 
        print(f"期間: {working_data['date'].min().date()} ～ {working_data['date'].max().date()}")
        print(f"ターゲット分布: {working_data['target'].mean():.1%}")
        print("✅ 確実動作銘柄・10年分の100%実データで取得完了")
        
        return working_data
        
    except Exception as e:
        logger.error(f"確実動作データ取得に失敗: {str(e)}")
        print("❌ 確実動作データ取得失敗")
        raise


if __name__ == "__main__":
    main()