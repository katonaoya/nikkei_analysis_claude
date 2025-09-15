"""
月単位での精密期間テスト
8.5年から1ヶ月ずつ期間を延長して最大取得可能期間を特定
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


class MonthlyPrecisionTester:
    """月単位での精密期間テスト用クライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("月単位精密テスト用J-Quantsクライアント初期化完了")
    
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
    
    def get_daily_quotes_test(
        self, 
        code: str,
        from_date: str,
        to_date: str
    ) -> pd.DataFrame:
        """
        指定期間の日次株価データ取得（テスト用）
        """
        try:
            headers = {"Authorization": f"Bearer {self._get_id_token()}"}
            results: List[Dict] = []
            pagination_key: Optional[str] = None
            
            logger.info(f"銘柄 {code}: データ取得開始")
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
                    logger.warning(f"銘柄 {code}: 無効なリクエスト（400エラー）")
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
                logger.info(f"銘柄 {code}: データ取得完了 ({len(df):,}件)")
                return df
            else:
                logger.warning(f"銘柄 {code}: データなし")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"銘柄 {code} 取得エラー: {str(e)}")
            return pd.DataFrame()
    
    def find_maximum_period(self, test_code: str = "72030", end_date: str = "2025-08-31") -> str:
        """
        1銘柄で最大取得可能期間を月単位で特定
        8.5年から1ヶ月ずつ延長
        """
        logger.info("=== 月単位精密期間テスト開始 ===")
        logger.info(f"テスト銘柄: {test_code}")
        logger.info(f"終了日固定: {end_date}")
        
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 8.5年から開始し、1ヶ月ずつ延長
        base_years = 8.5
        max_successful_period = None
        max_successful_start_date = None
        
        # 8.5年から最大12年まで月単位でテスト
        for additional_months in range(0, 43):  # 8.5年 + 3.5年 = 12年まで
            total_months = base_years * 12 + additional_months
            total_years = total_months / 12
            
            # 開始日を計算
            start_dt = end_dt - timedelta(days=int(total_years * 365.25))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            period_name = f"{total_years:.2f}年"
            logger.info(f"\n期間テスト: {start_date} ～ {end_date} ({period_name})")
            
            # データ取得試行
            df = self.get_daily_quotes_test(test_code, start_date, end_date)
            
            if not df.empty:
                logger.info(f"✅ 成功: {period_name} ({len(df):,}件)")
                max_successful_period = period_name
                max_successful_start_date = start_date
                time.sleep(2)  # 成功時は短い待機
            else:
                logger.warning(f"❌ 失敗: {period_name}")
                break  # 失敗したらそれ以上は試さない
            
            # 待機時間
            time.sleep(3)
        
        logger.info(f"\n=== 最大取得可能期間特定完了 ===")
        logger.info(f"銘柄 {test_code} の最大期間: {max_successful_period}")
        logger.info(f"期間: {max_successful_start_date} ～ {end_date}")
        
        return max_successful_start_date
    
    def get_all_working_stocks(self) -> List[str]:
        """
        確実に動作する全38銘柄のリストを返す
        """
        working_stocks = [
            "72030", "99840", "67580", "94320", "83060", "80350", "63670", "79740",
            "99830", "40630", "65010", "72670", "69020", "80010", "29140", "45190",
            "45430", "69540", "83090", "45020", "68610", "49010", "45680", "62730", 
            "69200", "78320", "84110", "88020", "45230", "61780", "60980", "40050", 
            "45070", "69710", "68570", "69050", "80310", "90200"
        ]
        
        logger.info(f"全38銘柄リスト取得完了")
        return working_stocks
    
    def fetch_maximum_period_data(self, start_date: str, end_date: str = "2025-08-31") -> pd.DataFrame:
        """
        最長期間で全銘柄のデータを取得
        """
        logger.info("=== 最長期間での全銘柄データ取得開始 ===")
        logger.info(f"期間: {start_date} ～ {end_date}")
        
        # 期間の年数を計算
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        total_years = total_days / 365.25
        
        logger.info(f"データ取得期間: {total_years:.2f}年 ({total_days}日)")
        
        # 全銘柄取得
        stock_codes = self.get_all_working_stocks()
        all_stock_data = []
        failed_stocks = []
        
        # 中間保存用ディレクトリ
        intermediate_dir = Path("data/maximum_period_data")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(stock_codes)}) - {idx/len(stock_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_test(code, start_date, end_date)
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data):,}件取得成功")
                    
                    # 中間保存（10銘柄ごと）
                    if idx % 10 == 0:
                        intermediate_df = pd.concat(all_stock_data, ignore_index=True)
                        intermediate_file = intermediate_dir / f"intermediate_max_{total_years:.2f}y_{idx}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
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
        
        # データ統合
        if not all_stock_data:
            raise RuntimeError("全ての銘柄でデータ取得に失敗しました")
        
        logger.info("=== データ統合中 ===")
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        logger.info("=== 最長期間データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        if failed_stocks:
            logger.info(f"失敗銘柄リスト: {failed_stocks}")
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # 最終保存
        output_dir = Path("data/maximum_period_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"maximum_period_{total_years:.2f}years_{len(all_stock_data)}stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        combined_df.to_pickle(output_file)
        
        logger.info(f"🎉 最長期間データ保存完了: {output_file}")
        
        return combined_df


def main():
    """メイン実行関数"""
    try:
        tester = MonthlyPrecisionTester()
        
        # Step 1: 1銘柄で最大期間を特定
        logger.info("Step 1: 1銘柄での最大期間特定...")
        max_start_date = tester.find_maximum_period()
        
        if not max_start_date:
            logger.error("最大期間の特定に失敗しました")
            return
        
        print(f"✅ 最大取得可能期間特定完了")
        print(f"期間: {max_start_date} ～ 2025-08-31")
        
        # 期間の年数を計算して表示
        start_dt = datetime.strptime(max_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime("2025-08-31", "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        total_years = total_days / 365.25
        print(f"合計期間: {total_years:.2f}年 ({total_days}日)")
        
        # Step 2: 全銘柄でデータ取得
        logger.info(f"\nStep 2: 最長期間{total_years:.2f}年で全38銘柄データ取得...")
        all_data = tester.fetch_maximum_period_data(max_start_date)
        
        print(f"\n=== 最長期間データ取得結果 ===")
        print(f"総レコード数: {len(all_data):,}件")
        print(f"銘柄数: {all_data['Code'].nunique()}銘柄")
        print(f"期間: {all_data['Date'].min()} ～ {all_data['Date'].max()}")
        print(f"実際の年数: {total_years:.2f}年")
        
        print("🎉 最長期間での全銘柄データ取得完了")
        
        return all_data
        
    except Exception as e:
        logger.error(f"月単位精密テストに失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()