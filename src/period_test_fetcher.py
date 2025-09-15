"""
期間延長テスト用のJ-Quantsデータ取得スクリプト
5銘柄で段階的に期間を延長して最大取得可能期間を特定
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


class PeriodTestFetcher:
    """期間延長テスト用のJ-Quantsデータ取得クライアント"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("期間テスト用J-Quantsデータ取得クライアント初期化完了")
    
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
    
    def get_test_stock_codes(self) -> List[str]:
        """
        テスト用の5銘柄を返す
        """
        # 確実に動作する銘柄から5つ選択
        test_stocks = [
            "72030",  # ニックス
            "99840",  # レック
            "67580",  # セイコーエプソン
            "94320",  # リクルートHD
            "83060"   # セコム
        ]
        
        logger.info(f"テスト対象銘柄: {test_stocks}")
        return test_stocks
    
    def get_daily_quotes_period_test(
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
    
    def test_period_extension(self, end_date: str = "2025-08-31") -> Dict[str, str]:
        """
        期間を段階的に延長してテスト
        終了日を固定し、開始日を3ヶ月ずつ遡る
        
        Args:
            end_date: 終了日（固定）
            
        Returns:
            Dict: 各銘柄の最大取得可能期間
        """
        logger.info("=== 期間延長テスト開始 ===")
        logger.info(f"終了日固定: {end_date}")
        
        stock_codes = self.get_test_stock_codes()
        results = {}
        
        # テスト期間のリスト（開始日を3ヶ月ずつ遡る）
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        test_periods = []
        
        # 5.5年前から開始（既知の動作期間）
        start_dt = end_dt - timedelta(days=int(5.5 * 365.25))
        test_periods.append(("5.5年", start_dt.strftime("%Y-%m-%d")))
        
        # 3ヶ月ずつ開始日を遡る（期間を延長）
        for months in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]:  # 最大8年まで
            start_dt = end_dt - timedelta(days=int((5.5 + months/12) * 365.25))
            period_name = f"{5.5 + months/12:.1f}年"
            test_periods.append((period_name, start_dt.strftime("%Y-%m-%d")))
        
        logger.info(f"テスト期間: {len(test_periods)}パターン")
        
        for code in stock_codes:
            logger.info(f"\n--- 銘柄 {code} のテスト開始 ---")
            max_successful_period = None
            
            for period_name, start_date in test_periods:
                logger.info(f"\n期間テスト: {start_date} ～ {end_date} ({period_name})")
                
                # データ取得試行
                df = self.get_daily_quotes_period_test(code, start_date, end_date)
                
                if not df.empty:
                    logger.info(f"✅ 成功: {period_name} ({len(df):,}件)")
                    max_successful_period = period_name
                    time.sleep(2)  # 成功時は短い待機
                else:
                    logger.warning(f"❌ 失敗: {period_name}")
                    break  # 失敗したらそれ以上は試さない
                
                # 待機時間
                time.sleep(3)
            
            results[code] = max_successful_period or "取得失敗"
            logger.info(f"銘柄 {code} の最大取得可能期間: {results[code]}")
        
        logger.info("\n=== 期間延長テスト結果 ===")
        for code, max_period in results.items():
            logger.info(f"銘柄 {code}: {max_period}")
        
        return results
    
    def test_55_years_baseline(self, end_date: str = "2025-08-31") -> bool:
        """
        5.5年の基準テスト（終了日固定）
        """
        logger.info("=== 5.5年基準テスト開始 ===")
        logger.info(f"終了日固定: {end_date}")
        
        # 5.5年前の開始日を計算
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=int(5.5 * 365.25))
        from_date = start_dt.strftime("%Y-%m-%d")
        
        logger.info(f"期間: {from_date} ～ {end_date} (約5.5年)")
        
        stock_codes = self.get_test_stock_codes()
        all_success = True
        
        for code in stock_codes:
            logger.info(f"\n銘柄 {code}: 5.5年テスト実行")
            df = self.get_daily_quotes_period_test(code, from_date, end_date)
            
            if not df.empty:
                logger.info(f"✅ 銘柄 {code}: 成功 ({len(df):,}件)")
            else:
                logger.warning(f"❌ 銘柄 {code}: 失敗")
                all_success = False
            
            time.sleep(3)  # 銘柄間待機
        
        logger.info(f"\n5.5年基準テスト結果: {'全て成功' if all_success else '一部失敗'}")
        return all_success


def main():
    """メイン実行関数"""
    try:
        fetcher = PeriodTestFetcher()
        
        # Step 1: 5.5年の基準テスト
        logger.info("Step 1: 5.5年基準テストを実行...")
        baseline_success = fetcher.test_55_years_baseline()
        
        if not baseline_success:
            logger.error("5.5年基準テストが失敗しました。処理を中止します。")
            return
        
        print("✅ 5.5年基準テスト完了")
        
        # Step 2: 期間延長テスト
        logger.info("\nStep 2: 期間延長テストを実行...")
        results = fetcher.test_period_extension()
        
        print("\n=== 期間延長テスト最終結果 ===")
        for code, max_period in results.items():
            print(f"銘柄 {code}: {max_period}")
        
        # 最も短い期間を特定
        successful_periods = [period for period in results.values() if period != "取得失敗"]
        if successful_periods:
            # 期間を数値に変換して最小値を求める（簡単な実装）
            print(f"\n推奨最大取得期間: {min(successful_periods, key=lambda x: float(x.replace('年', '')))}") 
        else:
            print("\n全ての銘柄で取得に失敗しました")
        
    except Exception as e:
        logger.error(f"期間テストに失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()