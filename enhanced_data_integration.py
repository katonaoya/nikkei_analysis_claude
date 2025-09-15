#!/usr/bin/env python3
"""
拡張データ統合システム
既存テクニカル指標 + ファンダメンタル + マーケットデータ
60%精度達成を目指す
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from jquants_auth import JQuantsAuth
from jquants_fundamental import JQuantsFundamental
from yahoo_market_data import YahooMarketData

class EnhancedDataIntegration:
    """拡張データ統合クラス"""
    
    def __init__(self):
        self.base_data_file = "data/processed/integrated_with_external.parquet"
        self.output_file = "data/processed/enhanced_integrated_data.parquet"
        
    def load_base_data(self) -> pd.DataFrame:
        """既存のベースデータ読み込み"""
        try:
            if Path(self.base_data_file).exists():
                df = pd.read_parquet(self.base_data_file)
                logger.success(f"✅ ベースデータ読み込み: {len(df)}件")
                
                # カラム統一
                if 'date' in df.columns and 'Date' not in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                if 'code' in df.columns and 'Stock' not in df.columns:
                    df['Stock'] = df['code'].astype(str)
                
                return df
            else:
                logger.error(f"ベースデータファイルが見つかりません: {self.base_data_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ ベースデータ読み込み失敗: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self, base_df: pd.DataFrame, limit: int = 200) -> list:
        """対象銘柄リストを取得（主要銘柄に限定）"""
        if base_df.empty:
            return []
        
        # 銘柄別のデータ量を確認
        stock_counts = base_df['Stock'].value_counts()
        
        # データが十分にある銘柄を選択（最低100日以上）
        valid_stocks = stock_counts[stock_counts >= 100].head(limit).index.tolist()
        
        logger.info(f"対象銘柄選択: {len(valid_stocks)}銘柄（データ十分な銘柄から選択）")
        return valid_stocks
    
    def integrate_fundamental_data(self, base_df: pd.DataFrame, stock_list: list) -> pd.DataFrame:
        """ファンダメンタルデータを統合"""
        logger.info("🔄 ファンダメンタルデータ統合開始...")
        
        # J-Quants認証
        auth = JQuantsAuth()
        
        if not auth.test_auth():
            logger.warning("J-Quants認証失敗、ファンダメンタルデータなしで継続")
            return base_df
        
        # ファンダメンタルデータ取得
        fundamental = JQuantsFundamental(auth)
        
        # 期間設定（過去2年）
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        try:
            # 財務データ取得（制限付き）
            limited_stocks = stock_list[:50]  # API制限を考慮して50銘柄に制限
            logger.info(f"ファンダメンタルデータ取得対象: {len(limited_stocks)}銘柄")\n            \n            financial_df = fundamental.get_bulk_financial_data(limited_stocks, from_date, to_date)\n            \n            if financial_df.empty:\n                logger.warning("財務データが取得できませんでした")\n                return base_df\n            \n            # 特徴量生成\n            features_df = fundamental.process_fundamental_features(financial_df)\n            \n            if features_df.empty:\n                logger.warning("ファンダメンタル特徴量が生成できませんでした")\n                return base_df\n            \n            # ベースデータとマージ\n            logger.info("ファンダメンタルデータをベースデータと統合中...")\n            \n            # 日付範囲を調整（ファンダメンタルデータは四半期単位なので前方補完）\n            enhanced_df = base_df.copy()\n            \n            for _, fund_row in features_df.iterrows():\n                stock = fund_row['Stock']\n                fund_date = fund_row['Date']\n                \n                # 該当銘柄のデータを取得\n                stock_mask = (enhanced_df['Stock'] == stock) & (enhanced_df['Date'] >= fund_date)\n                \n                if stock_mask.sum() > 0:\n                    # ファンダメンタル特徴量を追加\n                    for col in features_df.columns:\n                        if col not in ['Stock', 'Date']:\n                            enhanced_df.loc[stock_mask, col] = fund_row[col]\n            \n            # ファンダメンタル特徴量の前方補完\n            fund_cols = [col for col in features_df.columns if col not in ['Stock', 'Date']]\n            \n            for stock in enhanced_df['Stock'].unique():\n                stock_mask = enhanced_df['Stock'] == stock\n                enhanced_df.loc[stock_mask, fund_cols] = enhanced_df.loc[stock_mask, fund_cols].fillna(method='ffill')\n            \n            # 欠損値を中央値で補完\n            for col in fund_cols:\n                if col in enhanced_df.columns:\n                    enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())\n            \n            logger.success(f"✅ ファンダメンタルデータ統合完了: {len(fund_cols)}特徴量追加")\n            return enhanced_df\n            \n        except Exception as e:\n            logger.error(f"❌ ファンダメンタルデータ統合失敗: {e}")\n            return base_df\n    \n    def integrate_market_data(self, base_df: pd.DataFrame) -> pd.DataFrame:\n        """マーケットデータを統合"""\n        logger.info("🔄 マーケットデータ統合開始...")\n        \n        try:\n            # Yahoo Financeからマーケットデータ取得\n            market_data = YahooMarketData()\n            data_dict = market_data.get_all_market_data(period="2y")\n            \n            if not data_dict:\n                logger.warning("マーケットデータが取得できませんでした")\n                return base_df\n            \n            # マーケット特徴量生成\n            market_features = market_data.calculate_market_features(data_dict)\n            \n            if market_features.empty:\n                logger.warning("マーケット特徴量が生成できませんでした")\n                return base_df\n            \n            # ベースデータとマージ\n            logger.info("マーケットデータをベースデータと統合中...")\n            \n            # 日付でマージ\n            enhanced_df = base_df.merge(market_features, on='Date', how='left')\n            \n            # 前方補完で欠損値を埋める\n            market_cols = [col for col in market_features.columns if col != 'Date']\n            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='ffill')\n            \n            # 残りの欠損値を後方補完\n            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='bfill')\n            \n            # それでも残る欠損値を0で補完\n            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(0)\n            \n            logger.success(f"✅ マーケットデータ統合完了: {len(market_cols)}特徴量追加")\n            return enhanced_df\n            \n        except Exception as e:\n            logger.error(f"❌ マーケットデータ統合失敗: {e}")\n            return base_df\n    \n    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:\n        """相互作用特徴量を生成"""\n        logger.info("🔧 相互作用特徴量生成中...")\n        \n        enhanced_df = df.copy()\n        \n        try:\n            # 1. ファンダメンタル × テクニカル\n            if 'PER' in df.columns and 'RSI' in df.columns:\n                enhanced_df['PER_RSI_interaction'] = df['PER'] * (100 - df['RSI']) / 100  # 割安 × 売られ過ぎ\n            \n            if 'ROE' in df.columns and 'Price_vs_MA20' in df.columns:\n                enhanced_df['ROE_Momentum_interaction'] = df['ROE'] * df['Price_vs_MA20']  # 収益性 × モメンタム\n            \n            # 2. マーケット × 個別株\n            if 'nikkei225_return_1d' in df.columns and 'Volume_Ratio' in df.columns:\n                enhanced_df['Market_Volume_interaction'] = df['nikkei225_return_1d'] * df['Volume_Ratio']  # 市場動向 × 出来高\n            \n            if 'vix_close' in df.columns and 'Volatility' in df.columns:\n                enhanced_df['VIX_Stock_Vol_ratio'] = df['vix_close'] / (df['Volatility'] * 100 + 1)  # 市場恐怖 / 個別ボラ\n            \n            # 3. セクター相対強度（仮想）\n            if 'PBR' in df.columns and 'topix_return_1d' in df.columns:\n                enhanced_df['Value_Market_sync'] = (1 / (df['PBR'] + 0.1)) * df['topix_return_1d']  # バリュー × 市場\n            \n            # 4. トレンドフォロー強度\n            if 'Momentum_5' in df.columns and 'nikkei225_ma20_slope' in df.columns:\n                enhanced_df['Trend_Alignment'] = np.sign(df['Momentum_5']) * np.sign(df['nikkei225_ma20_slope'])  # トレンド一致\n            \n            # 5. リスク調整リターン予測\n            if 'ROE' in df.columns and 'vix_close' in df.columns:\n                enhanced_df['Risk_Adjusted_Quality'] = df['ROE'] * np.exp(-df['vix_close'] / 100)  # 品質 × リスク調整\n            \n            interaction_cols = len([col for col in enhanced_df.columns if 'interaction' in col.lower() or 'sync' in col.lower() or 'alignment' in col.lower() or 'adjusted' in col.lower()])\n            \n            logger.success(f"✅ 相互作用特徴量生成完了: {interaction_cols}特徴量追加")\n            \n        except Exception as e:\n            logger.warning(f"相互作用特徴量生成でエラー: {e}")\n        \n        return enhanced_df\n    \n    def finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:\n        """最終的な特徴量調整"""\n        logger.info("🎯 最終特徴量調整中...")\n        \n        # 必要な列の確認\n        required_base_cols = ['Date', 'Stock', 'close', 'high', 'low', 'open', 'volume']\n        missing_cols = [col for col in required_base_cols if col not in df.columns]\n        \n        if missing_cols:\n            logger.error(f"必要な基本列が不足: {missing_cols}")\n            return df\n        \n        # ターゲット変数生成\n        df = df.sort_values(['Stock', 'Date'])\n        \n        # 翌日の高値が当日終値より1%以上高い場合を予測ターゲット\n        df['next_high'] = df.groupby('Stock')['high'].shift(-1)\n        df['Target'] = (df['next_high'] > df['close'] * 1.01).astype(int)\n        \n        # 異常値処理\n        numeric_cols = df.select_dtypes(include=[np.number]).columns\n        \n        for col in numeric_cols:\n            if col not in ['Target', 'Date']:\n                # 99%と1%で異常値をクリップ\n                q99 = df[col].quantile(0.99)\n                q01 = df[col].quantile(0.01)\n                df[col] = df[col].clip(q01, q99)\n        \n        # 無限大値をNaNに変換\n        df = df.replace([np.inf, -np.inf], np.nan)\n        \n        # 最終的な欠損値処理\n        df = df.fillna(method='ffill').fillna(0)\n        \n        # 最終データ品質チェック\n        total_features = len([col for col in df.columns if col not in ['Date', 'Stock', 'Target', 'next_high']]) \n        target_count = df['Target'].sum()\n        target_rate = target_count / len(df) * 100\n        \n        logger.success(f"✅ 最終特徴量調整完了:")\n        logger.info(f"  総特徴量数: {total_features}")\n        logger.info(f"  データ件数: {len(df):,}")\n        logger.info(f"  ターゲット陽性率: {target_rate:.2f}%")\n        \n        return df\n    \n    def run_integration(self) -> pd.DataFrame:\n        """統合処理実行"""\n        logger.info("🚀 拡張データ統合プロセス開始")\n        \n        # 1. ベースデータ読み込み\n        base_df = self.load_base_data()\n        if base_df.empty:\n            logger.error("ベースデータの読み込みに失敗")\n            return pd.DataFrame()\n        \n        # 2. 対象銘柄選択\n        stock_list = self.get_stock_list(base_df)\n        if not stock_list:\n            logger.error("対象銘柄が選択できませんでした")\n            return pd.DataFrame()\n        \n        # ベースデータを対象銘柄に限定\n        base_df = base_df[base_df['Stock'].isin(stock_list)].copy()\n        \n        # 3. ファンダメンタルデータ統合\n        enhanced_df = self.integrate_fundamental_data(base_df, stock_list)\n        \n        # 4. マーケットデータ統合\n        enhanced_df = self.integrate_market_data(enhanced_df)\n        \n        # 5. 相互作用特徴量生成\n        enhanced_df = self.create_interaction_features(enhanced_df)\n        \n        # 6. 最終調整\n        final_df = self.finalize_features(enhanced_df)\n        \n        if not final_df.empty:\n            # 保存\n            try:\n                final_df.to_parquet(self.output_file)\n                logger.success(f"✅ 拡張統合データ保存完了: {self.output_file}")\n                \n                # 統計サマリー\n                feature_cols = [col for col in final_df.columns if col not in ['Date', 'Stock', 'Target']]\n                logger.info("📊 最終データ統計:")\n                logger.info(f"  期間: {final_df['Date'].min()} ～ {final_df['Date'].max()}")\n                logger.info(f"  銘柄数: {final_df['Stock'].nunique()}")\n                logger.info(f"  特徴量数: {len(feature_cols)}")\n                logger.info(f"  データ品質: {(1 - final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns))) * 100:.1f}%")\n                \n            except Exception as e:\n                logger.error(f"❌ データ保存失敗: {e}")\n        \n        return final_df

# 実行部分
if __name__ == "__main__":
    integrator = EnhancedDataIntegration()
    enhanced_data = integrator.run_integration()
    
    if not enhanced_data.empty:
        logger.success("🎉 拡張データ統合完了！60%精度向上の準備が整いました")
    else:
        logger.error("⚠️ 統合処理に失敗しました")