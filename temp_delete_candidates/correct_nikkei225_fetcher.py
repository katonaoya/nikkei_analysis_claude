#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正しい日経225銘柄コードを使用したデータ取得システム
実在する日経225銘柄の正確な4桁コードを使用
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
from typing import List, Optional

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class CorrectNikkei225Fetcher:
    """正確な日経225銘柄コードを使用したデータ取得クライアント"""
    
    def __init__(self):
        """初期化"""
        # .envファイルを明示的に読み込み
        from dotenv import load_dotenv
        load_dotenv()
        
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        logger.info("正確な日経225データ取得クライアント初期化完了")
    
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
    
    def get_real_nikkei225_codes(self) -> List[str]:
        """
        実在する日経225銘柄の正確な4桁コードを取得
        """
        # 実在する日経225銘柄の正確な4桁コード（主要銘柄から開始）
        real_nikkei225_codes = [
            # 主要銘柄（動作確認済み）
            "7203",  # トヨタ自動車
            "9984",  # ソフトバンクグループ
            "6758",  # ソニーグループ
            "9432",  # 日本電信電話
            "8306",  # 三菱UFJフィナンシャル・グループ
            "8035",  # 東京エレクトロン
            "6367",  # ダイキン工業
            "7974",  # 任天堂
            "9983",  # ファーストリテイリング
            "4063",  # 信越化学工業
            "6501",  # 日立製作所
            "7267",  # ホンダ
            "6902",  # デンソー
            "8001",  # 伊藤忠商事
            "2914",  # 日本たばこ産業
            "4519",  # 中外製薬
            "4543",  # テルモ
            "6954",  # ファナック
            "8309",  # 三井住友トラスト・ホールディングス
            "4502",  # 武田薬品工業
            "8411",  # みずほフィナンシャルグループ
            "4568",  # 第一三共
            "4523",  # エーザイ
            "4661",  # オリエンタルランド
            "6273",  # SMC
            "6200",  # インソース
            "6920",  # レーザーテック
            "7832",  # バンダイナムコホールディングス
            "8316",  # 三井住友フィナンシャルグループ
            "8031",  # 三井物産
            "8002",  # 丸紅
            "9201",  # 日本航空
            "9202",  # 全日本空輸
            "7751",  # キヤノン
            "6981",  # 村田製作所
            "8028",  # ファミリーマート
            "4005",  # 住友化学
            "4507",  # 塩野義製薬
            "4578",  # 大塚ホールディングス
            "3382",  # セブン&アイ・ホールディングス
            "4478",  # フリー
            "6098",  # リクルートホールディングス
            "9434",  # ソフトバンク
            "4755",  # 楽天グループ
            "6971",  # 京セラ
            "6752",  # パナソニック ホールディングス
            "7013",  # IHI
            "8804",  # 東京建物
            "8766",  # 東京海上ホールディングス
            "2801",  # キッコーマン
        ]
        
        logger.info(f"実在日経225銘柄コード取得: {len(real_nikkei225_codes)}銘柄")
        return real_nikkei225_codes
    
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
            results: List[dict] = []
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
    
    def fetch_nikkei225_real_data(self) -> pd.DataFrame:
        """
        実在する日経225銘柄の10年間データを取得
        """
        logger.info("=== 実在日経225銘柄10年間データ取得開始 ===")
        logger.info("期間: 2015年9月1日 ～ 2025年8月31日 (10年間)")
        
        # 実在銘柄取得
        stock_codes = self.get_real_nikkei225_codes()
        all_stock_data = []
        failed_stocks = []
        
        # 中間保存用ディレクトリ
        intermediate_dir = Path("data/real_nikkei225_data")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"銘柄 {code} データ取得中... ({idx}/{len(stock_codes)}) - {idx/len(stock_codes)*100:.1f}%完了")
                
                stock_data = self.get_daily_quotes_10years(code)
                
                if not stock_data.empty:
                    all_stock_data.append(stock_data)
                    logger.info(f"  ✅ 銘柄 {code}: {len(stock_data):,}件取得成功")
                    
                    # 中間保存（10銘柄ごと）
                    if idx % 10 == 0:
                        intermediate_df = pd.concat(all_stock_data, ignore_index=True)
                        intermediate_file = intermediate_dir / f"intermediate_real_nikkei225_{idx}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                        intermediate_df.to_pickle(intermediate_file)
                        logger.info(f"  💾 中間保存: {intermediate_file} ({len(intermediate_df):,}件)")
                
                else:
                    failed_stocks.append(code)
                    logger.warning(f"  ❌ 銘柄 {code}: データ取得失敗")
                
                # 銘柄間の待機（重要）
                wait_time = 2.0  # 2秒待機
                if idx % 5 == 0:
                    wait_time = 5.0  # 5銘柄ごとに長い待機
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
        
        logger.info("=== 実在日経225銘柄10年間データ取得完了 ===")
        logger.info(f"成功銘柄: {len(all_stock_data)}銘柄")
        logger.info(f"失敗銘柄: {len(failed_stocks)}銘柄")
        if failed_stocks:
            logger.info(f"失敗銘柄リスト: {failed_stocks}")
        logger.info(f"総レコード数: {len(combined_df):,}件")
        logger.info(f"期間: {combined_df['Date'].min()} ～ {combined_df['Date'].max()}")
        
        # 最終保存
        output_dir = Path("data/real_nikkei225_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"real_nikkei225_10years_{len(all_stock_data)}stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        combined_df.to_pickle(output_file)
        
        # Parquet形式でも保存（互換性のため）
        parquet_file = output_dir / f"real_nikkei225_10years_{len(all_stock_data)}stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        combined_df.to_parquet(parquet_file)
        
        logger.info(f"🎉 実在日経225データ保存完了:")
        logger.info(f"  PKL: {output_file}")
        logger.info(f"  Parquet: {parquet_file}")
        
        return combined_df


def main():
    """メイン実行関数"""
    logger.info("🚀 実在日経225銘柄データ取得開始")
    
    fetcher = CorrectNikkei225Fetcher()
    df = fetcher.fetch_nikkei225_real_data()
    
    logger.info(f"🎯 最終結果: {len(df):,}件, {df['Code'].nunique()}銘柄")


if __name__ == "__main__":
    main()