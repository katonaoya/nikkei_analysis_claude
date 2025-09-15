#!/usr/bin/env python3
"""
Yahoo Finance マーケットデータ取得
日経平均、TOPIX、ドル円、VIX等の市場指標取得
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class YahooMarketData:
    """Yahoo Finance マーケットデータ取得クラス"""
    
    def __init__(self):
        # マーケット指標のシンボル定義
        self.market_symbols = {
            'nikkei225': '^N225',      # 日経平均株価
            'topix': '^TOPX',          # TOPIX
            'usdjpy': 'USDJPY=X',      # ドル円
            'vix': '^VIX',             # VIX恐怖指数
            'us_10y': '^TNX',          # 米10年債利回り
            'jpy_index': 'JPY=X',      # 円インデックス
            'dow': '^DJI',             # ダウ平均
            'sp500': '^GSPC',          # S&P500
            'nasdaq': '^IXIC'          # NASDAQ
        }
    
    def get_market_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """単一マーケット指標データ取得"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if not df.empty:
                # インデックスをリセットしてDateカラムに
                df = df.reset_index()
                df['Symbol'] = symbol
                logger.debug(f"✅ {symbol}: {len(df)}日分のデータ取得")
                return df
            else:
                logger.warning(f"⚠️ {symbol}: データが取得できませんでした")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ {symbol}データ取得失敗: {e}")
            return pd.DataFrame()
    
    def get_all_market_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """全マーケット指標データ一括取得"""
        logger.info(f"🔄 マーケットデータ一括取得開始 (期間: {period})")
        
        market_data = {}
        
        for name, symbol in self.market_symbols.items():
            logger.info(f"  取得中: {name} ({symbol})")
            df = self.get_market_data(symbol, period)
            
            if not df.empty:
                market_data[name] = df
                logger.info(f"    ✅ {name}: {len(df)}日分")
            else:
                logger.warning(f"    ❌ {name}: 取得失敗")
        
        logger.success(f"✅ マーケットデータ取得完了: {len(market_data)}/{len(self.market_symbols)}指標")
        return market_data
    
    def calculate_market_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """マーケットデータから特徴量生成"""
        logger.info("🔧 マーケット特徴量生成中...")
        
        if not market_data:
            logger.warning("マーケットデータが空です")
            return pd.DataFrame()
        
        # 全データの日付範囲を統一
        date_ranges = []
        for name, df in market_data.items():
            if not df.empty:
                date_ranges.append(df['Date'])
        
        if not date_ranges:
            return pd.DataFrame()
        
        # 共通日付範囲
        all_dates = pd.concat(date_ranges).drop_duplicates().sort_values()
        
        # 特徴量DataFrame初期化
        features_df = pd.DataFrame({'Date': all_dates})
        
        # 各指標の特徴量計算
        for name, df in market_data.items():
            if df.empty:
                continue
                
            logger.debug(f"  処理中: {name}")
            
            # 日付でマージ（日付型を統一）
            df = df[['Date', 'Close', 'Volume']].copy()
            df.columns = ['Date', f'{name}_close', f'{name}_volume']
            
            # 日付型を統一
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            features_df = features_df.merge(df, on='Date', how='left')
            
            # 前方補完で欠損値埋め
            features_df[f'{name}_close'] = features_df[f'{name}_close'].fillna(method='ffill')
            features_df[f'{name}_volume'] = features_df[f'{name}_volume'].fillna(method='ffill')
            
            # 技術指標計算
            close_col = f'{name}_close'
            
            if close_col in features_df.columns:
                # 1. リターン
                features_df[f'{name}_return_1d'] = features_df[close_col].pct_change(1)
                features_df[f'{name}_return_5d'] = features_df[close_col].pct_change(5)
                features_df[f'{name}_return_20d'] = features_df[close_col].pct_change(20)
                
                # 2. 移動平均
                features_df[f'{name}_ma5'] = features_df[close_col].rolling(5).mean()
                features_df[f'{name}_ma20'] = features_df[close_col].rolling(20).mean()
                features_df[f'{name}_ma60'] = features_df[close_col].rolling(60).mean()
                
                # 3. 移動平均乖離率
                features_df[f'{name}_ma5_ratio'] = (features_df[close_col] - features_df[f'{name}_ma5']) / features_df[f'{name}_ma5']
                features_df[f'{name}_ma20_ratio'] = (features_df[close_col] - features_df[f'{name}_ma20']) / features_df[f'{name}_ma20']
                
                # 4. ボラティリティ
                features_df[f'{name}_volatility_5d'] = features_df[f'{name}_return_1d'].rolling(5).std()
                features_df[f'{name}_volatility_20d'] = features_df[f'{name}_return_1d'].rolling(20).std()
                
                # 5. RSI風指標
                delta = features_df[close_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, 1)
                features_df[f'{name}_rsi'] = 100 - (100 / (1 + rs))
                
                # 6. トレンド強度（移動平均の傾き）
                features_df[f'{name}_ma5_slope'] = features_df[f'{name}_ma5'].pct_change(2)
                features_df[f'{name}_ma20_slope'] = features_df[f'{name}_ma20'].pct_change(5)
        
        # 相関系特徴量（主要ペア）
        if 'nikkei225_close' in features_df.columns and 'usdjpy_close' in features_df.columns:
            # 日経平均とドル円の相関
            features_df['nikkei_usdjpy_correlation'] = features_df['nikkei225_return_1d'].rolling(20).corr(features_df['usdjpy_return_1d'])
        
        if 'nikkei225_close' in features_df.columns and 'topix_close' in features_df.columns:
            # 日経平均とTOPIXの乖離
            features_df['nikkei_topix_spread'] = (features_df['nikkei225_return_1d'] - features_df['topix_return_1d'])
        
        if 'vix_close' in features_df.columns:
            # VIXレジーム（低リスク/高リスク）
            features_df['vix_regime_low'] = (features_df['vix_close'] < 20).astype(int)
            features_df['vix_regime_high'] = (features_df['vix_close'] > 30).astype(int)
        
        # 欠損値処理
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # 異常値処理
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Date':
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                features_df[col] = features_df[col].clip(q01, q99)
        
        logger.success(f"✅ マーケット特徴量生成完了: {len(features_df)}日, {len(features_df.columns)-1}特徴量")
        
        # 生成された主要特徴量の確認
        key_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in ['return', 'ratio', 'volatility', 'rsi'])]
        logger.info(f"📊 生成された主要特徴量数: {len(key_features)}")
        
        return features_df
    
    def get_sector_etf_data(self, period: str = "1y") -> pd.DataFrame:
        """セクター別ETFデータ取得（日本）"""
        sector_etfs = {
            'tse_reit': '1343.T',      # NEXT FUNDS 東証REIT指数連動型上場投信
            'tech': '1625.T',          # NEXT FUNDS 日経225連動型上場投信
            'financial': '1615.T'      # NEXT FUNDS TOPIX銀行業連動型上場投信
        }
        
        logger.info("🏢 セクターETFデータ取得中...")
        
        sector_data = {}
        for name, symbol in sector_etfs.items():
            df = self.get_market_data(symbol, period)
            if not df.empty:
                sector_data[name] = df
        
        if sector_data:
            logger.success(f"✅ セクターETFデータ取得完了: {len(sector_data)}セクター")
        
        return sector_data
    
    def save_market_data(self, features_df: pd.DataFrame, filename: str = "market_features.parquet"):
        """マーケット特徴量データ保存"""
        try:
            features_df.to_parquet(filename)
            logger.success(f"✅ マーケット特徴量保存完了: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ データ保存失敗: {e}")
            return False

# 使用例
if __name__ == "__main__":
    # マーケットデータ取得
    market_data = YahooMarketData()
    
    # 全マーケット指標取得
    data_dict = market_data.get_all_market_data(period="2y")
    
    if data_dict:
        # 特徴量生成
        features_df = market_data.calculate_market_features(data_dict)
        
        if not features_df.empty:
            # データ保存
            market_data.save_market_data(features_df)
            
            # 基本統計
            logger.info("📊 マーケット特徴量統計:")
            logger.info(f"データ期間: {features_df['Date'].min()} ～ {features_df['Date'].max()}")
            logger.info(f"データ点数: {len(features_df)}")
            
            # 主要指標の確認
            key_cols = ['nikkei225_return_1d', 'usdjpy_return_1d', 'vix_close', 'topix_return_1d']
            available_cols = [col for col in key_cols if col in features_df.columns]
            
            for col in available_cols:
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                logger.info(f"  {col}: 平均{mean_val:.4f}, 標準偏差{std_val:.4f}")
        else:
            logger.warning("特徴量が生成されませんでした")
    else:
        logger.error("マーケットデータが取得できませんでした")