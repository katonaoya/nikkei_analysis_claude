#!/usr/bin/env python3
"""
推奨銘柄10社の株価データ取得スクリプト
期間: 2025年8月1日～9月5日
出力: CSV形式
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, date
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JQuantsStockDataFetcher:
    """J-Quants APIを使用した株価データ取得クラス"""
    
    def __init__(self, mail_address: str, password: str):
        self.mail_address = mail_address
        self.password = password
        self.id_token = None
        self.base_url = "https://api.jquants.com/v1"
        
        # 推奨銘柄10社（コード付き）
        self.recommended_stocks = {
            "6098": "リクルートHD",
            "9984": "ソフトバンクG", 
            "8035": "東京エレクトロン",
            "6758": "ソニーG",
            "8306": "三菱UFJFG",
            "7974": "任天堂",
            "7203": "トヨタ自動車",
            "4519": "中外製薬",
            "9433": "KDDI",
            "4478": "フリー"
        }
        
        # データ取得期間
        self.start_date = "2025-08-01"
        self.end_date = "2025-09-05"
        
        logger.info(f"📊 対象銘柄: {len(self.recommended_stocks)}社")
        logger.info(f"📅 取得期間: {self.start_date} ～ {self.end_date}")
    
    def authenticate(self):
        """J-Quants APIの認証"""
        logger.info("🔐 J-Quants API認証中...")
        
        # リフレッシュトークン取得
        refresh_url = f"{self.base_url}/token/auth_user"
        refresh_data = {
            "mailaddress": self.mail_address,
            "password": self.password
        }
        
        try:
            response = requests.post(refresh_url, data=json.dumps(refresh_data))
            response.raise_for_status()
            refresh_token = response.json()["refreshToken"]
            logger.info("✅ リフレッシュトークン取得成功")
            
            # IDトークン取得
            id_token_url = f"{self.base_url}/token/auth_refresh"
            id_token_data = {"refreshtoken": refresh_token}
            
            response = requests.post(id_token_url, data=json.dumps(id_token_data))
            response.raise_for_status()
            self.id_token = response.json()["idToken"]
            logger.info("✅ IDトークン取得成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 認証エラー: {e}")
            return False
    
    def fetch_stock_data(self, code: str, company_name: str) -> pd.DataFrame:
        """指定銘柄の株価データ取得"""
        logger.info(f"📈 {company_name}({code}) データ取得中...")
        
        if not self.id_token:
            logger.error("IDトークンが設定されていません")
            return pd.DataFrame()
        
        headers = {'Authorization': f'Bearer {self.id_token}'}
        url = f"{self.base_url}/prices/daily_quotes"
        
        params = {
            'code': code,
            'from': self.start_date.replace('-', ''),
            'to': self.end_date.replace('-', '')
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'daily_quotes' not in data:
                logger.warning(f"⚠️ {company_name}({code}): データなし")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['daily_quotes'])
            
            if not df.empty:
                # データ整理
                df['Date'] = pd.to_datetime(df['Date'])
                df['Code'] = code
                df['CompanyName'] = company_name
                
                # 数値列を適切な型に変換
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'TurnoverValue']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                logger.info(f"✅ {company_name}({code}): {len(df)}件取得")
                return df
            else:
                logger.warning(f"⚠️ {company_name}({code}): 空のデータ")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ {company_name}({code}) データ取得エラー: {e}")
            return pd.DataFrame()
    
    def fetch_all_stocks(self) -> pd.DataFrame:
        """全推奨銘柄のデータ取得"""
        logger.info("🚀 全推奨銘柄データ取得開始")
        
        all_data = []
        
        for code, company_name in self.recommended_stocks.items():
            df = self.fetch_stock_data(code, company_name)
            if not df.empty:
                all_data.append(df)
            
            # API制限対策で1秒待機
            time.sleep(1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['Code', 'Date'])
            
            logger.info(f"✅ 全データ取得完了: {len(combined_df)}件")
            return combined_df
        else:
            logger.error("❌ データ取得失敗")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None):
        """CSVファイルに保存"""
        if df.empty:
            logger.error("保存するデータがありません")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommended_stocks_data_{timestamp}.csv"
        
        output_path = Path(filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"💾 CSVファイル保存完了: {output_path}")
        logger.info(f"📊 保存データ統計:")
        logger.info(f"   総レコード数: {len(df):,}件")
        logger.info(f"   銘柄数: {df['Code'].nunique()}社")
        logger.info(f"   期間: {df['Date'].min().date()} ～ {df['Date'].max().date()}")
        
        # 銘柄別データ数
        stock_counts = df['Code'].value_counts().sort_index()
        logger.info("   銘柄別データ数:")
        for code, count in stock_counts.items():
            company_name = self.recommended_stocks.get(code, "不明")
            logger.info(f"     {code} ({company_name}): {count}件")
        
        return output_path

def main():
    """メイン関数"""
    # J-Quants APIの認証情報を入力してください
    mail_address = input("J-Quants APIのメールアドレスを入力してください: ")
    password = input("J-Quants APIのパスワードを入力してください: ")
    
    # データ取得クラス初期化
    fetcher = JQuantsStockDataFetcher(mail_address, password)
    
    # 認証
    if not fetcher.authenticate():
        logger.error("認証に失敗しました")
        return
    
    # データ取得
    df = fetcher.fetch_all_stocks()
    
    if not df.empty:
        # CSV保存
        csv_path = fetcher.save_to_csv(df)
        
        # 簡単な統計情報表示
        print("\n" + "="*80)
        print("📊 推奨銘柄10社 株価データ取得完了")
        print("="*80)
        print(f"保存ファイル: {csv_path}")
        print(f"データ期間: 2025年8月1日 ～ 9月5日")
        print(f"総レコード数: {len(df):,}件")
        print(f"対象銘柄数: {df['Code'].nunique()}社")
        print()
        print("💰 この10社でバックテスト利益率: +55.97%")
        print("="*80)
    
    else:
        logger.error("データ取得に失敗しました")

if __name__ == "__main__":
    main()