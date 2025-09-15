#!/usr/bin/env python3
"""
J-Quants ファンダメンタルデータ取得
スタンダードプラン対応
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import time
from jquants_auth import JQuantsAuth

class JQuantsFundamental:
    """J-Quants ファンダメンタルデータ取得クラス"""
    
    def __init__(self, auth: JQuantsAuth):
        self.auth = auth
        self.base_url = "https://api.jquants.com"
        
    def get_listed_companies(self, date: str = None) -> pd.DataFrame:
        """上場銘柄一覧取得"""
        url = f"{self.base_url}/v1/listed/info"
        headers = self.auth.get_headers()
        
        params = {}
        if date:
            params['date'] = date
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'info' in data and data['info']:
                df = pd.DataFrame(data['info'])
                logger.info(f"✅ 上場銘柄データ取得: {len(df)}件")
                return df
            else:
                logger.warning("上場銘柄データが空です")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ 上場銘柄取得失敗: {e}")
            return pd.DataFrame()
    
    def get_financial_statements(self, code: str, from_date: str, to_date: str) -> pd.DataFrame:
        """財務情報取得（単一銘柄）"""
        url = f"{self.base_url}/v1/fins/statements"
        headers = self.auth.get_headers()
        
        params = {
            'code': code,
            'from': from_date,
            'to': to_date
        }
        
        all_data = []
        pagination_key = None
        
        try:
            while True:
                if pagination_key:
                    params['pagination_key'] = pagination_key
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'statements' in data and data['statements']:
                    all_data.extend(data['statements'])
                
                # ページネーション確認
                pagination_key = data.get('pagination_key')
                if not pagination_key:
                    break
                
                # API制限対策
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                logger.debug(f"銘柄{code}: 財務データ{len(df)}件取得")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ 銘柄{code}の財務データ取得失敗: {e}")
            return pd.DataFrame()
    
    def get_bulk_financial_data(self, stock_codes: List[str], from_date: str, to_date: str) -> pd.DataFrame:
        """複数銘柄の財務データ一括取得"""
        logger.info(f"🔄 財務データ一括取得開始: {len(stock_codes)}銘柄")
        
        all_financial_data = []
        processed = 0
        
        for code in stock_codes:
            try:
                df = self.get_financial_statements(code, from_date, to_date)
                if not df.empty:
                    all_financial_data.append(df)
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"  進捗: {processed}/{len(stock_codes)} ({processed/len(stock_codes)*100:.1f}%)")
                
                # API制限対策
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"銘柄{code}をスキップ: {e}")
                continue
        
        if all_financial_data:
            combined_df = pd.concat(all_financial_data, ignore_index=True)
            logger.success(f"✅ 財務データ一括取得完了: {len(combined_df)}件")
            return combined_df
        else:
            logger.warning("財務データが取得できませんでした")
            return pd.DataFrame()
    
    def process_fundamental_features(self, financial_df: pd.DataFrame) -> pd.DataFrame:
        """財務データから特徴量を生成"""
        if financial_df.empty:
            return pd.DataFrame()
        
        logger.info("🔧 ファンダメンタル特徴量生成中...")
        
        # 必要な列の存在確認
        required_cols = ['Local Code', 'Disclosed Date', 'TypeOfDocument', 'TypeOfCurrentPeriod']
        missing_cols = [col for col in required_cols if col not in financial_df.columns]
        if missing_cols:
            logger.warning(f"必要な列が不足: {missing_cols}")
            return pd.DataFrame()
        
        # データ前処理
        df = financial_df.copy()
        
        # 日付変換
        df['Date'] = pd.to_datetime(df['Disclosed Date'])
        df['Stock'] = df['Local Code'].astype(str)
        
        # 最新の四半期データのみ抽出
        df = df[df['TypeOfDocument'] == 'FY']  # 通期決算のみ
        df = df[df['TypeOfCurrentPeriod'] == 'Actual']  # 実績値のみ
        
        # 重複削除（最新データを優先）
        df = df.sort_values(['Stock', 'Date']).groupby('Stock').tail(1)
        
        # 特徴量生成
        features_df = pd.DataFrame()
        features_df['Stock'] = df['Stock']
        features_df['Date'] = df['Date']
        
        # 1. PER（株価収益率）
        if 'ForecastPER' in df.columns:
            features_df['PER'] = pd.to_numeric(df['ForecastPER'], errors='coerce')
        
        # 2. PBR（株価純資産倍率）
        if 'ForecastPBR' in df.columns:
            features_df['PBR'] = pd.to_numeric(df['ForecastPBR'], errors='coerce')
        
        # 3. ROE（自己資本利益率）
        if 'ROE' in df.columns:
            features_df['ROE'] = pd.to_numeric(df['ROE'], errors='coerce')
        
        # 4. ROA（総資産利益率）
        if 'ROA' in df.columns:
            features_df['ROA'] = pd.to_numeric(df['ROA'], errors='coerce')
        
        # 5. EPS（1株当たり純利益）
        if 'ForecastEPS' in df.columns:
            features_df['EPS'] = pd.to_numeric(df['ForecastEPS'], errors='coerce')
        
        # 6. 営業利益率
        if 'OperatingProfitMargin' in df.columns:
            features_df['Operating_Margin'] = pd.to_numeric(df['OperatingProfitMargin'], errors='coerce')
        
        # 7. 自己資本比率
        if 'EquityRatio' in df.columns:
            features_df['Equity_Ratio'] = pd.to_numeric(df['EquityRatio'], errors='coerce')
        
        # 8. 予想配当利回り
        if 'ForecastDividendYield' in df.columns:
            features_df['Dividend_Yield'] = pd.to_numeric(df['ForecastDividendYield'], errors='coerce')
        
        # 9. 時価総額（対数変換）
        if 'MarketCapitalization' in df.columns:
            market_cap = pd.to_numeric(df['MarketCapitalization'], errors='coerce')
            features_df['Log_Market_Cap'] = np.log1p(market_cap.fillna(0))
        
        # 10. 流動比率
        if 'CurrentRatio' in df.columns:
            features_df['Current_Ratio'] = pd.to_numeric(df['CurrentRatio'], errors='coerce')
        
        # 欠損値処理
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
        
        # 異常値処理（99.5%タイル値でキャップ）
        for col in numeric_cols:
            if col not in ['Stock', 'Date']:
                upper_bound = features_df[col].quantile(0.995)
                lower_bound = features_df[col].quantile(0.005)
                features_df[col] = features_df[col].clip(lower_bound, upper_bound)
        
        logger.success(f"✅ ファンダメンタル特徴量生成完了: {len(features_df)}銘柄, {len(features_df.columns)-2}特徴量")
        
        # 生成された特徴量の統計情報
        logger.info("📊 生成された特徴量:")
        feature_cols = [col for col in features_df.columns if col not in ['Stock', 'Date']]
        for col in feature_cols:
            if not features_df[col].empty:
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                logger.info(f"  {col}: 平均{mean_val:.2f}, 標準偏差{std_val:.2f}")
        
        return features_df
    
    def save_fundamental_data(self, df: pd.DataFrame, filename: str = "fundamental_data.parquet"):
        """ファンダメンタルデータを保存"""
        try:
            df.to_parquet(filename)
            logger.success(f"✅ ファンダメンタルデータ保存完了: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ データ保存失敗: {e}")
            return False

# 使用例
if __name__ == "__main__":
    # 認証
    auth = JQuantsAuth()
    
    if not auth.test_auth():
        logger.error("認証に失敗しました")
        exit(1)
    
    # ファンダメンタルデータ取得
    fundamental = JQuantsFundamental(auth)
    
    # 上場銘柄一覧取得
    listed_df = fundamental.get_listed_companies()
    
    if not listed_df.empty:
        # 主要銘柄（日経225など）のコードを抽出（例：1000-9999の4桁コード）
        major_codes = listed_df[
            (listed_df['Local Code'].str.len() == 4) & 
            (listed_df['Local Code'].str.isdigit())
        ]['Local Code'].head(100).tolist()  # テスト用に100銘柄
        
        logger.info(f"対象銘柄数: {len(major_codes)}")
        
        # 財務データ取得（過去2年）
        from_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        financial_df = fundamental.get_bulk_financial_data(major_codes, from_date, to_date)
        
        if not financial_df.empty:
            # 特徴量生成
            features_df = fundamental.process_fundamental_features(financial_df)
            
            # 保存
            if not features_df.empty:
                fundamental.save_fundamental_data(features_df)
            else:
                logger.warning("特徴量が生成されませんでした")
        else:
            logger.warning("財務データが取得できませんでした")
    else:
        logger.error("上場銘柄一覧が取得できませんでした")