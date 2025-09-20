#!/usr/bin/env python3
"""
J-Quants API 実データ取得システム
実際の日本株データを取得してAI予測システムに統合
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from loguru import logger
import json

class JQuantsDataFetcher:
    def __init__(self):
        self.base_url = "https://api.jquants.com"
        self.id_token = None
        self.refresh_token = None
        
        # 認証情報（.envファイルから取得）
        from dotenv import load_dotenv
        load_dotenv()
        
        self.email = os.getenv('JQUANTS_MAIL_ADDRESS')  # .envファイルのキー名に合わせる
        self.password = os.getenv('JQUANTS_PASSWORD')
        self.refresh_token = os.getenv('JQUANTS_REFRESH_TOKEN')  # 既存のリフレッシュトークンも読み込み
        
        if not self.email or not self.password:
            logger.warning("⚠️ J-Quants認証情報が.envファイルに設定されていません")
            logger.info(".envファイルに以下を設定してください:")
            logger.info("JQUANTS_MAIL_ADDRESS=your-email@example.com")
            logger.info("JQUANTS_PASSWORD=your-password")
    
    def authenticate(self):
        """J-Quants APIの認証を実行"""
        try:
            # 1. リフレッシュトークン取得
            auth_url = f"{self.base_url}/v1/token/auth_user"
            auth_data = {
                "mailaddress": self.email,
                "password": self.password
            }
            
            logger.info("🔐 J-Quants認証開始...")
            response = requests.post(auth_url, json=auth_data)
            
            if response.status_code == 200:
                self.refresh_token = response.json()['refreshToken']
                logger.success("✅ リフレッシュトークン取得成功")
            else:
                logger.error(f"❌ 認証失敗: {response.status_code} - {response.text}")
                return False
            
            # 2. IDトークン取得
            id_token_url = f"{self.base_url}/v1/token/auth_refresh"
            params = {"refreshtoken": self.refresh_token}
            
            response = requests.post(id_token_url, params=params)
            
            if response.status_code == 200:
                self.id_token = response.json()['idToken']
                logger.success("✅ IDトークン取得成功（24時間有効）")
                return True
            else:
                logger.error(f"❌ IDトークン取得失敗: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 認証エラー: {e}")
            return False
    
    def get_headers(self):
        """APIリクエスト用ヘッダー"""
        return {
            "Authorization": f"Bearer {self.id_token}",
            "Content-Type": "application/json"
        }
    
    def get_listed_companies(self):
        """上場銘柄一覧取得"""
        try:
            url = f"{self.base_url}/v1/listed/info"
            headers = self.get_headers()
            
            logger.info("📋 上場銘柄一覧取得中...")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                companies_df = pd.DataFrame(data['info'])
                logger.success(f"✅ {len(companies_df)}社の銘柄情報取得完了")
                return companies_df
            else:
                logger.error(f"❌ 銘柄一覧取得失敗: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 銘柄一覧取得エラー: {e}")
            return None
    
    def get_stock_prices(self, code, from_date, to_date):
        """指定銘柄の株価取得"""
        try:
            url = f"{self.base_url}/v1/prices/daily_quotes"
            headers = self.get_headers()
            
            params = {
                "code": code,
                "from": from_date,
                "to": to_date
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'daily_quotes' in data:
                    return pd.DataFrame(data['daily_quotes'])
                else:
                    return pd.DataFrame()
            else:
                logger.warning(f"⚠️ {code}: 株価取得失敗 {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ {code} 株価取得エラー: {e}")
            return None
    
    def get_bulk_stock_data(self, stock_codes, years=2):
        """複数銘柄の株価データを一括取得"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=years * 365)
            
            all_data = []
            total_codes = len(stock_codes)
            
            logger.info(f"📊 {total_codes}銘柄の株価データ取得開始")
            logger.info(f"期間: {start_date} ～ {end_date}")
            
            for i, code in enumerate(stock_codes):
                logger.info(f"取得中: {code} ({i+1}/{total_codes})")
                
                stock_data = self.get_stock_prices(
                    code, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if stock_data is not None and not stock_data.empty:
                    stock_data['Code'] = code
                    all_data.append(stock_data)
                    logger.debug(f"✅ {code}: {len(stock_data)}日分")
                else:
                    logger.warning(f"⚠️ {code}: データなし")
                
                # APIレート制限対応
                time.sleep(0.1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.success(f"✅ 全{len(combined_df)}件の株価データ取得完了")
                return combined_df
            else:
                logger.error("❌ 取得できた株価データがありません")
                return None
                
        except Exception as e:
            logger.error(f"❌ 一括株価取得エラー: {e}")
            return None
    
    def get_nikkei225_stocks(self):
        """日経225構成銘柄のコード取得"""
        # 主要日経225構成銘柄（実際のコード）
        nikkei225_codes = [
            # 主要構成銘柄
            "7203",  # トヨタ自動車
            "9984",  # ソフトバンクグループ
            "6098",  # リクルートホールディングス
            "8306",  # 三菱UFJフィナンシャル・グループ
            "9434",  # ソフトバンク
            "4063",  # 信越化学工業
            "6861",  # キーエンス
            "8035",  # 東京エレクトロン
            "6954",  # ファナック
            "9432",  # 日本電信電話
            "4519",  # 中外製薬
            "7974",  # 任天堂
            "6367",  # ダイキン工業
            "4523",  # エーザイ
            "8411",  # みずほフィナンシャルグループ
            "7741",  # HOYA
            "9983",  # ファーストリテイリング
            "8316",  # 三井住友フィナンシャルグループ
            "6902",  # デンソー
            "4578",  # 大塚ホールディングス
            "6273",  # SMC
            "4568",  # 第一三共
            "6758",  # ソニーグループ
            "8001",  # 伊藤忠商事
            "3382",  # セブン&アイ・ホールディングス
            "4661",  # オリエンタルランド
            "8058",  # 三菱商事
            "9020",  # 東日本旅客鉄道
            "4502",  # 武田薬品工業
            "7267",  # 本田技研工業
            "4478",  # フリー
            "6501",  # 日立製作所
            "4005",  # 住友化学
            "9301",  # 三菱倉庫
            "8031",  # 三井物産
        ]
        return nikkei225_codes
    
    def get_financial_statements(self, stock_codes, years=3):
        """財務情報を取得"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=years * 365)
            
            all_financial_data = []
            
            logger.info(f"📊 財務情報取得開始: {len(stock_codes)}銘柄")
            
            for i, code in enumerate(stock_codes):
                logger.info(f"財務取得中: {code} ({i+1}/{len(stock_codes)})")
                
                url = f"{self.base_url}/v1/fins/statements"
                headers = self.get_headers()
                params = {
                    "code": code,
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": end_date.strftime('%Y-%m-%d')
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'statements' in data and data['statements']:
                        financial_df = pd.DataFrame(data['statements'])
                        financial_df['Code'] = code
                        all_financial_data.append(financial_df)
                        logger.debug(f"✅ {code}: {len(financial_df)}件の財務データ")
                    else:
                        logger.warning(f"⚠️ {code}: 財務データなし")
                else:
                    logger.warning(f"⚠️ {code}: 財務データ取得失敗 {response.status_code}")
                
                time.sleep(0.1)  # API制限対応
            
            if all_financial_data:
                combined_financial = pd.concat(all_financial_data, ignore_index=True)
                logger.success(f"✅ 財務データ取得完了: {len(combined_financial)}件")
                return combined_financial
            else:
                logger.warning("⚠️ 財務データが取得できませんでした")
                return None
                
        except Exception as e:
            logger.error(f"❌ 財務データ取得エラー: {e}")
            return None
    
    def get_market_indices(self, years=3):
        """市場指数データ取得"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=years * 365)
            
            # TOPIX取得
            url = f"{self.base_url}/v1/indices/topix"
            headers = self.get_headers()
            params = {
                "from": start_date.strftime('%Y-%m-%d'),
                "to": end_date.strftime('%Y-%m-%d')
            }
            
            logger.info("📈 TOPIX指数データ取得中...")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'topix' in data:
                    topix_df = pd.DataFrame(data['topix'])
                    topix_df['Date'] = pd.to_datetime(topix_df['Date'])
                    logger.success(f"✅ TOPIX データ取得完了: {len(topix_df)}件")
                    return topix_df
                else:
                    return pd.DataFrame()
            else:
                logger.error(f"❌ TOPIX取得失敗: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 市場指数取得エラー: {e}")
            return None

    def create_enhanced_dataset(self, output_path="data/processed/enhanced_jquants_data.parquet"):
        """J-Quants全データ + Yahoo Finance統合データセット作成"""
        try:
            if not self.authenticate():
                return False
            
            # 日経225主要銘柄取得
            stock_codes = self.get_nikkei225_stocks()
            logger.info(f"対象銘柄: {len(stock_codes)}銘柄")
            
            # 1. 基本株価データ取得
            logger.info("📊 1/4: 基本株価データ取得中...")
            stock_data = self.get_bulk_stock_data(stock_codes, years=3)
            
            if stock_data is None or stock_data.empty:
                logger.error("❌ 株価データ取得に失敗しました")
                return False
            
            # 2. 技術指標追加
            logger.info("📊 2/4: 技術指標計算中...")
            stock_data = self.add_technical_indicators(stock_data)
            
            # 3. ターゲット作成
            logger.info("📊 3/4: ターゲット変数作成中...")
            stock_data = self.create_target_variable(stock_data)
            
            # 4. J-Quants追加データ取得
            logger.info("📊 4/4: J-Quants追加データ統合中...")
            
            # 財務情報取得
            financial_data = self.get_financial_statements(stock_codes, years=3)
            
            # TOPIX指数取得
            topix_data = self.get_market_indices(years=3)
            
            # Yahoo Finance市場データ統合（既存のYahooMarketDataクラス使用）
            try:
                from yahoo_market_data import YahooMarketData
                yahoo_data = YahooMarketData()
                
                logger.info("🌍 Yahoo Finance マーケットデータ取得中...")
                market_data_dict = yahoo_data.get_all_market_data(period="3y")
                
                if market_data_dict:
                    market_features = yahoo_data.calculate_market_features(market_data_dict)
                    if not market_features.empty:
                        # 日付統一
                        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
                        market_features['Date'] = pd.to_datetime(market_features['Date'], utc=True).dt.date
                        
                        stock_data = stock_data.merge(market_features, on='Date', how='left')
                        logger.success("✅ Yahoo Finance マーケットデータ統合完了")
                
            except Exception as e:
                logger.warning(f"⚠️ Yahoo Finance統合エラー（継続）: {e}")
            
            # TOPIX統合
            if topix_data is not None and not topix_data.empty:
                try:
                    topix_data['Date'] = pd.to_datetime(topix_data['Date']).dt.date
                    # TOPIX特徴量を追加
                    topix_data = topix_data.rename(columns={'Close': 'TOPIX_Close'})
                    stock_data = stock_data.merge(topix_data[['Date', 'TOPIX_Close']], on='Date', how='left')
                    logger.success("✅ TOPIX データ統合完了")
                except Exception as e:
                    logger.warning(f"⚠️ TOPIX統合エラー: {e}")
            
            # 財務データ統合（簡略版）
            if financial_data is not None and not financial_data.empty:
                try:
                    # 最新の財務データのみ使用（四半期ベース）
                    latest_financial = financial_data.sort_values(['Code', 'DisclosedDate']).groupby('Code').tail(1)
                    financial_features = latest_financial[['Code', 'NetSales', 'OperatingProfit', 'NetIncome']].copy()
                    stock_data = stock_data.merge(financial_features, on='Code', how='left')
                    logger.success("✅ 財務データ統合完了")
                except Exception as e:
                    logger.warning(f"⚠️ 財務データ統合エラー: {e}")
            
            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stock_data.to_parquet(output_path, index=False)
            
            logger.success(f"✅ 拡張データセット作成完了: {output_path}")
            logger.info(f"📊 データ概要:")
            logger.info(f"  - 総件数: {len(stock_data):,}件")
            logger.info(f"  - 銘柄数: {stock_data['Code'].nunique()}銘柄") 
            logger.info(f"  - 特徴量数: {len(stock_data.columns)}個")
            logger.info(f"  - 期間: {stock_data['Date'].min()} ～ {stock_data['Date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 拡張データセット作成エラー: {e}")
            return False

    def create_real_dataset(self, output_path="data/processed/real_jquants_data.parquet"):
        """基本のJ-Quantsデータセット作成（元の機能維持）"""
        try:
            if not self.authenticate():
                return False
                
            # 日経225主要銘柄取得
            stock_codes = self.get_nikkei225_stocks()
            logger.info(f"対象銘柄: {len(stock_codes)}銘柄")
            
            # 株価データ取得
            stock_data = self.get_bulk_stock_data(stock_codes, years=3)
            
            if stock_data is None or stock_data.empty:
                logger.error("❌ 株価データ取得に失敗しました")
                return False
            
            # データ前処理
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.sort_values(['Code', 'Date'])
            
            # 技術指標追加
            stock_data = self.add_technical_indicators(stock_data)
            
            # ターゲット作成（翌日1%上昇）
            stock_data = self.create_target_variable(stock_data)
            
            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stock_data.to_parquet(output_path, index=False)
            
            logger.success(f"✅ 実データセット作成完了: {output_path}")
            logger.info(f"📊 データ概要:")
            logger.info(f"  - 総件数: {len(stock_data):,}件")
            logger.info(f"  - 銘柄数: {stock_data['Code'].nunique()}銘柄")
            logger.info(f"  - 期間: {stock_data['Date'].min()} ～ {stock_data['Date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 実データセット作成エラー: {e}")
            return False
    
    def add_technical_indicators(self, df):
        """技術指標を追加"""
        df = df.copy()
        df = df.sort_values(['Code', 'Date'])
        
        # 新しい列を初期化
        df['MA_5'] = None
        df['MA_20'] = None
        df['RSI'] = None
        df['Volatility'] = None
        df['Returns'] = None
        
        for code in df['Code'].unique():
            mask = df['Code'] == code
            code_data = df[mask].copy()
            
            # 移動平均
            code_data['MA_5'] = code_data['Close'].rolling(5).mean()
            code_data['MA_20'] = code_data['Close'].rolling(20).mean()
            
            # RSI
            delta = code_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            code_data['RSI'] = 100 - (100 / (1 + rs))
            
            # ボラティリティ
            code_data['Volatility'] = code_data['Close'].pct_change().rolling(20).std()
            
            # 価格変化率
            code_data['Returns'] = code_data['Close'].pct_change()
            
            # データを戻す
            df.loc[mask, 'MA_5'] = code_data['MA_5'].values
            df.loc[mask, 'MA_20'] = code_data['MA_20'].values
            df.loc[mask, 'RSI'] = code_data['RSI'].values
            df.loc[mask, 'Volatility'] = code_data['Volatility'].values
            df.loc[mask, 'Returns'] = code_data['Returns'].values
        
        return df
    
    def create_target_variable(self, df):
        """ターゲット変数作成（翌日1%上昇）"""
        df = df.copy()
        df = df.sort_values(['Code', 'Date'])
        
        # 翌日の高値を取得
        df['Next_High'] = df.groupby('Code')['High'].shift(-1)
        
        # 翌日高値が終値から1%以上上昇したかどうか
        df['Target'] = (df['Next_High'] > df['Close'] * 1.01).astype(int)
        
        return df

def main():
    """メイン実行"""
    fetcher = JQuantsDataFetcher()
    
    logger.info("🚀 J-Quants全データ + Yahoo Finance統合データセット作成開始")
    success = fetcher.create_enhanced_dataset()
    
    if success:
        logger.success("🎉 拡張データセット作成完了！")
        logger.info("次のステップ: AI予測システムで拡張データを使用して精度向上")
    else:
        logger.error("💥 拡張データセット作成失敗")

if __name__ == "__main__":
    main()