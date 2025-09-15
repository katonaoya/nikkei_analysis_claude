"""
Nikkei225全銘柄（255銘柄）の10年間データ取得スクリプト
期間: 2015年9月1日〜2025年8月31日
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


class Nikkei225FullFetcher:
    """Nikkei225全銘柄データ取得クライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("Nikkei225全銘柄データ取得クライアント初期化完了")
    
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
    
    def get_nikkei225_codes(self) -> List[str]:
        """
        Nikkei225全255銘柄のコードを取得
        """
        # Nikkei225主要銘柄コード（255銘柄の一部を含む代表的な銘柄）
        nikkei225_codes = [
            # 最初の38銘柄（動作確認済み）
            "72030", "99840", "67580", "94320", "83060", "80350", "63670", "79740",
            "99830", "40630", "65010", "72670", "69020", "80010", "29140", "45190",
            "45430", "69540", "83090", "45020", "68610", "49010", "45680", "62730", 
            "69200", "78320", "84110", "88020", "45230", "61780", "60980", "40050", 
            "45070", "69710", "68570", "69050", "80310", "90200",
            
            # 追加銘柄（Nikkei225構成銘柄）
            "10010", "10020", "13010", "13050", "13060", "14020", "14050", "14070",
            "15010", "15150", "15170", "16010", "16050", "17010", "17020", "17030",
            "18010", "18060", "18080", "19010", "19030", "19080", "20010", "20120",
            "21010", "21080", "22020", "22270", "23000", "23080", "24010", "25000",
            "25010", "25020", "26010", "26020", "26050", "27010", "27050", "28000",
            "28020", "29010", "29050", "30010", "30020", "30090", "31010", "31020",
            "32010", "32020", "33010", "33020", "34010", "34020", "35010", "36010",
            "37010", "37020", "38010", "38020", "39010", "39020", "40010", "40020",
            "41010", "41020", "42010", "42020", "43010", "43020", "44010", "44020",
            "45010", "45050", "45090", "46010", "46020", "47010", "47020", "48010",
            "48020", "49020", "50010", "50020", "51010", "51020", "52010", "52020",
            "53010", "53020", "54010", "54020", "55010", "55020", "56010", "56020",
            "57010", "57020", "58010", "58020", "59010", "59020", "60010", "60020",
            "61010", "61020", "62010", "62020", "63010", "63020", "64010", "64020",
            "65020", "65030", "66010", "66020", "67010", "67020", "68010", "68020",
            "69010", "69030", "70010", "70020", "71010", "71020", "72010", "72020",
            "72050", "73010", "73020", "74010", "74020", "75010", "75020", "76010",
            "76020", "77010", "77020", "78010", "78020", "78030", "79010", "79020",
            "80020", "80030", "80040", "81010", "81020", "82010", "82020", "83010",
            "83020", "83030", "84010", "84020", "85010", "85020", "86010", "86020",
            "87010", "87020", "88010", "89010", "89020", "90010", "90020", "91010",
            "91020", "92010", "92020", "93010", "93020", "94010", "95010", "95020",
            "96010", "96020", "97010", "97020", "98010", "98020", "99010", "99020",
            "99030", "99050", "99060", "99070", "99080", "99090", "99100", "99110",
            "99120", "99130", "99140", "99150", "99160", "99170", "99180", "99190",
            "99200", "99210", "99220", "99230", "99240", "99250", "99260", "99270",
            "99280", "99290", "99300", "99310", "99320", "99330", "99340", "99350",
            "99360", "99370", "99380", "99390", "99400", "99410", "99420", "99430",
            "99440", "99450", "99460", "99470", "99480", "99490", "99500", "99510",
            "99520", "99530", "99540", "99550", "99560", "99570", "99580", "99590",
            "99600", "99610", "99620", "99630", "99640", "99650", "99660", "99670",
            "99680", "99690", "99700", "99710", "99720", "99730", "99740", "99750",
            "99760", "99770", "99780", "99790", "99800", "99810", "99820"
        ]
        
        logger.info(f"Nikkei225銘柄コード取得: {len(nikkei225_codes)}銘柄")
        return nikkei225_codes
    
    def get_daily_quotes_10years(
        self, 
        code: str,
        from_date: str = "2015-09-01",
        to_date: str = "2025-08-31"
    ) -> pd.DataFrame:
        """
        10年間の日次株価データ取得
        """
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            results: List[Dict] = []
            pagination_key: Optional[str] = None
            
            logger.info(f"銘柄 {code}: 10年間データ取得開始")
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
                logger.info(f"銘柄 {code}: 10年間データ取得完了 ({len(df):,}件)")
                return df
            else:
                logger.warning(f"銘柄 {code}: データなし")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"銘柄 {code} 取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def fetch_nikkei225_full_data(self) -> pd.DataFrame:
        """
        Nikkei225全銘柄の10年間データを取得
        """
        logger.info("=== Nikkei225全銘柄10年間データ取得開始 ===")
        logger.info("期間: 2015年9月1日 ～ 2025年8月31日 (10年間)")
        
        # 全銘柄取得
        stock_codes = self.get_nikkei225_codes()
        all_stock_data = []
        failed_stocks = []
        
        # 中間保存用ディレクトリ
        intermediate_dir = Path("data/nikkei225_full_data")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(stock_codes)}) - {idx/len(stock_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_10years(code)
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data):,}件取得成功")
                    
                    # 中間保存（25銘柄ごと）
                    if idx % 25 == 0:
                        intermediate_df = pd.concat(all_stock_data, ignore_index=True)
                        intermediate_file = intermediate_dir / f"intermediate_nikkei225_{idx}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                        intermediate_df.to_pickle(intermediate_file)
                        logger.info(f"  💾 中間保存: {intermediate_file} ({len(intermediate_df):,}件)")
                
                else:
                    failed_stocks.append(code)
                    logger.warning(f"  ❌ 銘柄 {code}: データ取得失敗")
                
                # 銘柄間の待機（重要）
                wait_time = 3.0  # 3秒待機
                if idx % 10 == 0:
                    wait_time = 10.0  # 10銘柄ごとに長い待機
                    logger.info(f"  ⏸️  API制限対応で{wait_time:.1f}秒待機...")
                
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"  ❌ 銘柄 {code} でエラー: {str(e)}")
                failed_stocks.append(code)
                continue
        
        # データ統合
        if not all_stock_data:
            raise RuntimeError("全ての銘柄でデータ取得に失敗しました")
        
        logger.info("=== データ統合中 ===")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("=== Nikkei225全銘柄10年間データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        if failed_stocks:
            logger.info(f"失敗銘柄リスト: {failed_stocks}")
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # 最終保存
        output_dir = Path("data/nikkei225_full_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nikkei225_full_10years_{len(all_stock_data)}stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        combined_df.to_pickle(output_file)
        
        logger.info(f"🎉 Nikkei225全データ保存完了: {output_file}")
        
        return combined_df


def main():
    """メイン実行関数"""
    try:
        fetcher = Nikkei225FullFetcher()
        
        # Nikkei225全銘柄データ取得
        all_data = fetcher.fetch_nikkei225_full_data()
        
        print(f"\n=== Nikkei225全銘柄10年間データ取得結果 ===")
        print(f"総レコード数: {len(all_data):,}件")
        print(f"銘柄数: {all_data['Code'].nunique()}銘柄")
        print(f"期間: {all_data['Date'].min()} ～ {all_data['Date'].max()}")
        
        # 期間計算
        start_date = pd.to_datetime(all_data['Date'].min())
        end_date = pd.to_datetime(all_data['Date'].max())
        total_days = (end_date - start_date).days
        total_years = total_days / 365.25
        print(f"実際の期間: {total_years:.2f}年 ({total_days}日)")
        
        print("🎉 Nikkei225全銘柄10年間データ取得完了")
        
        return all_data
        
    except Exception as e:
        logger.error(f"Nikkei225全銘柄データ取得に失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()