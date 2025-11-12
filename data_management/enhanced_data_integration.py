#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張データ統合システム
既存テクニカル指標 + ファンダメンタル + マーケットデータ
60%精度達成を目指す
"""

import argparse
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

from credit_balance_fetcher import load_latest_feature_file
from news_headline_fetcher import load_latest_news_feature
from highfreq_market_feature_builder import load_latest_highfreq_feature

class EnhancedDataIntegration:
    """拡張データ統合クラス"""
    
    def __init__(self, include_highfreq: bool = True):
        self.base_data_file = "data/processed/integrated_with_external.parquet"
        self.output_file = "data/processed/enhanced_integrated_data.parquet"
        self.margin_feature_dir = Path("data/external/margin_balances")
        self.news_feature_dir = Path("data/external/news_headlines")
        self.highfreq_feature_dir = Path("data/external/highfreq_market")
        self.include_highfreq = include_highfreq
        
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
                if 'Code' not in df.columns:
                    if 'code' in df.columns:
                        df['Code'] = df['code'].astype(str)
                    elif 'Stock' in df.columns:
                        df['Code'] = df['Stock'].astype(str)
                if 'Stock' not in df.columns and 'Code' in df.columns:
                    df['Stock'] = df['Code'].astype(str)

                # 既存モデル互換用の列を補完
                if 'TurnoverValue' not in df.columns and 'turnover_value' in df.columns:
                    df['TurnoverValue'] = df['turnover_value']
                if 'AdjustmentFactor' not in df.columns and 'adjustment_factor' in df.columns:
                    df['AdjustmentFactor'] = df['adjustment_factor']
                for src, dst in [
                    ('adj_open', 'AdjustmentOpen'),
                    ('adj_high', 'AdjustmentHigh'),
                    ('adj_low', 'AdjustmentLow'),
                    ('adj_close', 'AdjustmentClose'),
                    ('adj_volume', 'AdjustmentVolume'),
                ]:
                    if dst not in df.columns and src in df.columns:
                        df[dst] = df[src]
                for src, dst in [
                    ('close', 'Close_raw'),
                    ('high', 'High_raw'),
                    ('low', 'Low_raw'),
                    ('volume', 'Volume_raw'),
                ]:
                    if dst not in df.columns and src in df.columns:
                        df[dst] = df[src]
                if 'ApiCode' not in df.columns and 'Code' in df.columns:
                    df['ApiCode'] = df['Code']

                if not self.include_highfreq:
                    hf_cols = [col for col in df.columns if col.startswith('hf_')]
                    if hf_cols:
                        df = df.drop(columns=hf_cols)
                        logger.info(f"ベースデータから高頻度指標を除外: {len(hf_cols)}列")

                if 'Returns' not in df.columns and 'close' in df.columns:
                    df['Returns'] = df.groupby('Stock')['close'].pct_change()
                if 'Close_gap_prev' not in df.columns and 'Close' in df.columns:
                    df['Close_gap_prev'] = df.groupby('Stock')['Close'].pct_change()
                if 'Returns_3d' not in df.columns and 'close' in df.columns:
                    df['Returns_3d'] = df.groupby('Stock')['close'].pct_change(periods=3)
                if 'Volatility_5' not in df.columns and 'Returns' in df.columns:
                    df['Volatility_5'] = df.groupby('Stock')['Returns'].transform(lambda s: s.rolling(5, min_periods=1).std())
                if 'High_Low_Ratio' not in df.columns and {'high', 'low'} <= set(df.columns):
                    df['High_Low_Ratio'] = (df['high'] - df['low']) / df['low'].replace(0, pd.NA)
                if 'TargetReturn' not in df.columns and 'close' in df.columns:
                    df['TargetReturn'] = df.groupby('Stock')['close'].shift(-1) / df['close'] - 1
                if 'MACD' not in df.columns and 'close' in df.columns:
                    close_series = df.sort_values(['Stock', 'Date']).groupby('Stock')['close']
                    ema12 = close_series.transform(lambda s: s.ewm(span=12, adjust=False).mean())
                    ema26 = close_series.transform(lambda s: s.ewm(span=26, adjust=False).mean())
                    df['MACD'] = ema12 - ema26

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
            logger.info(f"ファンダメンタルデータ取得対象: {len(limited_stocks)}銘柄")
            
            financial_df = fundamental.get_bulk_financial_data(limited_stocks, from_date, to_date)
            
            if financial_df.empty:
                logger.warning("財務データが取得できませんでした")
                return base_df
            
            # 特徴量生成
            features_df = fundamental.process_fundamental_features(financial_df)
            
            if features_df.empty:
                logger.warning("ファンダメンタル特徴量が生成できませんでした")
                return base_df
            
            # ベースデータとマージ
            logger.info("ファンダメンタルデータをベースデータと統合中...")
            
            # 日付範囲を調整（ファンダメンタルデータは四半期単位なので前方補完）
            enhanced_df = base_df.copy()
            
            for _, fund_row in features_df.iterrows():
                stock = fund_row['Stock']
                fund_date = fund_row['Date']
                
                # 該当銘柄のデータを取得
                stock_mask = (enhanced_df['Stock'] == stock) & (enhanced_df['Date'] >= fund_date)
                
                if stock_mask.sum() > 0:
                    # ファンダメンタル特徴量を追加
                    for col in features_df.columns:
                        if col not in ['Stock', 'Date']:
                            enhanced_df.loc[stock_mask, col] = fund_row[col]
            
            # ファンダメンタル特徴量の前方補完
            fund_cols = [col for col in features_df.columns if col not in ['Stock', 'Date']]
            
            for stock in enhanced_df['Stock'].unique():
                stock_mask = enhanced_df['Stock'] == stock
                enhanced_df.loc[stock_mask, fund_cols] = enhanced_df.loc[stock_mask, fund_cols].fillna(method='ffill')
            
            # 欠損値を中央値で補完
            for col in fund_cols:
                if col in enhanced_df.columns:
                    enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
            
            logger.success(f"✅ ファンダメンタルデータ統合完了: {len(fund_cols)}特徴量追加")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"❌ ファンダメンタルデータ統合失敗: {e}")
            return base_df
    
    def integrate_market_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """マーケットデータを統合"""
        logger.info("🔄 マーケットデータ統合開始...")

        try:
            # Yahoo Financeからマーケットデータ取得
            market_data = YahooMarketData()
            data_dict = market_data.get_all_market_data(period="2y")
            
            if not data_dict:
                logger.warning("マーケットデータが取得できませんでした")
                return base_df
            
            # マーケット特徴量生成
            market_features = market_data.calculate_market_features(data_dict)
            
            if market_features.empty:
                logger.warning("マーケット特徴量が生成できませんでした")
                return base_df
            
            # ベースデータとマージ
            logger.info("マーケットデータをベースデータと統合中...")
            
            # 日付でマージ
            enhanced_df = base_df.merge(market_features, on='Date', how='left')
            
            # 前方補完で欠損値を埋める
            market_cols = [col for col in market_features.columns if col != 'Date']
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='ffill')
            
            # 残りの欠損値を後方補完
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='bfill')
            
            # それでも残る欠損値を0で補完
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(0)
            
            logger.success(f"✅ マーケットデータ統合完了: {len(market_cols)}特徴量追加")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"❌ マーケットデータ統合失敗: {e}")
            return base_df

    def integrate_margin_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """信用取引残データを統合"""

        margin_file = load_latest_feature_file(self.margin_feature_dir)
        if margin_file is None:
            logger.warning("信用残特徴量ファイルが見つかりません。統合をスキップします。")
            return base_df

        try:
            margin_df = pd.read_parquet(margin_file)
        except Exception as e:
            logger.error(f"信用残データ読み込み失敗: {e}")
            return base_df

        required_cols = {'Code', 'Date'}
        if not required_cols.issubset(margin_df.columns):
            logger.error(f"信用残データに必要な列が不足: {required_cols - set(margin_df.columns)}")
            return base_df

        margin_df['Code'] = margin_df['Code'].astype(str)
        margin_df['Date'] = pd.to_datetime(margin_df['Date'])

        if 'Code' not in base_df.columns:
            logger.error("ベースデータにCode列がありません。信用残を統合できません。")
            return base_df

        merged = base_df.merge(margin_df, on=['Code', 'Date'], how='left')

        margin_cols = [col for col in margin_df.columns if col not in {'Code', 'Date'}]
        if margin_cols:
            merged.sort_values(['Code', 'Date'], inplace=True)
            for code, idx in merged.groupby('Code').groups.items():
                merged.loc[idx, margin_cols] = merged.loc[idx, margin_cols].fillna(method='ffill')

        logger.success(f"✅ 信用残データ統合完了: {len(margin_cols)}特徴量")
        return merged

    def integrate_news_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """ニュース関連指標を統合"""

        news_file = load_latest_news_feature(self.news_feature_dir)
        if news_file is None:
            logger.warning("ニュース特徴量ファイルが見つかりません。統合をスキップします。")
            return base_df

        try:
            news_df = pd.read_parquet(news_file)
        except Exception as e:
            logger.error(f"ニュースデータ読み込み失敗: {e}")
            return base_df

        if 'Date' not in news_df.columns:
            logger.error("ニュース特徴量にDate列が存在しません")
            return base_df

        news_df['Date'] = pd.to_datetime(news_df['Date'])
        merged = base_df.merge(news_df, on='Date', how='left')
        logger.success(f"✅ ニュースデータ統合完了: {len([c for c in news_df.columns if c != 'Date'])}特徴量")
        return merged

    def integrate_highfreq_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """高頻度マーケット指標を統合"""

        highfreq_path = load_latest_highfreq_feature(self.highfreq_feature_dir)
        if highfreq_path is None:
            logger.warning("高頻度データが見つかりません。統合をスキップします。")
            return base_df

        try:
            hf_df = pd.read_parquet(highfreq_path)
        except Exception as e:
            logger.error(f"高頻度データ読み込み失敗: {e}")
            return base_df

        if 'Date' not in hf_df.columns:
            logger.error("高頻度特徴量にDate列が存在しません")
            return base_df

        hf_df['Date'] = pd.to_datetime(hf_df['Date'])
        merged = base_df.merge(hf_df, on='Date', how='left')
        logger.success(f"✅ 高頻度マーケットデータ統合完了: {len([c for c in hf_df.columns if c != 'Date'])}特徴量")
        return merged

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を生成"""
        logger.info("🔧 相互作用特徴量生成中...")
        
        enhanced_df = df.copy()
        
        try:
            # 1. ファンダメンタル × テクニカル
            if 'PER' in df.columns and 'RSI' in df.columns:
                enhanced_df['PER_RSI_interaction'] = df['PER'] * (100 - df['RSI']) / 100  # 割安 × 売られ過ぎ
            
            if 'ROE' in df.columns and 'Price_vs_MA20' in df.columns:
                enhanced_df['ROE_Momentum_interaction'] = df['ROE'] * df['Price_vs_MA20']  # 収益性 × モメンタム
            
            # 2. マーケット × 個別株
            if 'nikkei225_return_1d' in df.columns and 'Volume_Ratio' in df.columns:
                enhanced_df['Market_Volume_interaction'] = df['nikkei225_return_1d'] * df['Volume_Ratio']  # 市場動向 × 出来高
            
            if 'vix_close' in df.columns and 'Volatility' in df.columns:
                enhanced_df['VIX_Stock_Vol_ratio'] = df['vix_close'] / (df['Volatility'] * 100 + 1)  # 市場恐怖 / 個別ボラ
            
            # 3. セクター相対強度（仮想）
            if 'PBR' in df.columns and 'topix_return_1d' in df.columns:
                enhanced_df['Value_Market_sync'] = (1 / (df['PBR'] + 0.1)) * df['topix_return_1d']  # バリュー × 市場
            
            # 4. トレンドフォロー強度
            if 'Momentum_5' in df.columns and 'nikkei225_ma20_slope' in df.columns:
                enhanced_df['Trend_Alignment'] = np.sign(df['Momentum_5']) * np.sign(df['nikkei225_ma20_slope'])  # トレンド一致
            
            # 5. リスク調整リターン予測
            if 'ROE' in df.columns and 'vix_close' in df.columns:
                enhanced_df['Risk_Adjusted_Quality'] = df['ROE'] * np.exp(-df['vix_close'] / 100)  # 品質 × リスク調整
            
            interaction_cols = len([col for col in enhanced_df.columns if 'interaction' in col.lower() or 'sync' in col.lower() or 'alignment' in col.lower() or 'adjusted' in col.lower()])
            
            logger.success(f"✅ 相互作用特徴量生成完了: {interaction_cols}特徴量追加")
            
        except Exception as e:
            logger.warning(f"相互作用特徴量生成でエラー: {e}")
        
        return enhanced_df
    
    def finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """最終的な特徴量調整"""
        logger.info("🎯 最終特徴量調整中...")
        
        # 必要な列の確認
        required_base_cols = ['Date', 'Stock', 'close', 'high', 'low', 'open', 'volume']
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"必要な基本列が不足: {missing_cols}")
            return df
        
        # ターゲット変数生成
        df = df.sort_values(['Stock', 'Date'])
        
        # 翌日の高値が当日終値より1%以上高い場合を予測ターゲット
        df['next_high'] = df.groupby('Stock')['high'].shift(-1)
        df['Target'] = (df['next_high'] > df['close'] * 1.01).astype(int)
        
        # 異常値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['Target', 'Date']:
                # 99%と1%で異常値をクリップ
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(q01, q99)
        
        # 無限大値をNaNに変換
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 最終的な欠損値処理
        df = df.fillna(method='ffill').fillna(0)
        
        # 最終データ品質チェック
        total_features = len([col for col in df.columns if col not in ['Date', 'Stock', 'Target', 'next_high']]) 
        target_count = df['Target'].sum()
        target_rate = target_count / len(df) * 100
        
        logger.success(f"✅ 最終特徴量調整完了:")
        logger.info(f"  総特徴量数: {total_features}")
        logger.info(f"  データ件数: {len(df):,}")
        logger.info(f"  ターゲット陽性率: {target_rate:.2f}%")
        
        return df
    
    def run_integration(self) -> pd.DataFrame:
        """統合処理実行"""
        logger.info("🚀 拡張データ統合プロセス開始")
        
        # 1. ベースデータ読み込み
        base_df = self.load_base_data()
        if base_df.empty:
            logger.error("ベースデータの読み込みに失敗")
            return pd.DataFrame()
        
        # 2. 対象銘柄選択
        stock_list = self.get_stock_list(base_df)
        if not stock_list:
            logger.error("対象銘柄が選択できませんでした")
            return pd.DataFrame()
        
        # ベースデータを対象銘柄に限定
        base_df = base_df[base_df['Stock'].isin(stock_list)].copy()
        
        # 3. ファンダメンタルデータ統合
        enhanced_df = self.integrate_fundamental_data(base_df, stock_list)
        
        # 4. マーケットデータ統合
        enhanced_df = self.integrate_market_data(enhanced_df)
        
        # 5. 信用残データ統合
        enhanced_df = self.integrate_margin_features(enhanced_df)

        # 6. 高頻度マーケットデータ統合（任意）
        if self.include_highfreq:
            enhanced_df = self.integrate_highfreq_features(enhanced_df)
        else:
            hf_cols = [col for col in enhanced_df.columns if col.startswith('hf_')]
            if hf_cols:
                enhanced_df = enhanced_df.drop(columns=hf_cols)
                logger.info(f"高頻度指標を除外: {len(hf_cols)}列")

        # 7. ニュース指標統合
        enhanced_df = self.integrate_news_features(enhanced_df)

        # 8. 相互作用特徴量生成
        enhanced_df = self.create_interaction_features(enhanced_df)
        
        # 9. 最終調整
        final_df = self.finalize_features(enhanced_df)
        
        if not final_df.empty:
            # 保存
            try:
                final_df.to_parquet(self.output_file)
                logger.success(f"✅ 拡張統合データ保存完了: {self.output_file}")
                
                # 統計サマリー
                feature_cols = [col for col in final_df.columns if col not in ['Date', 'Stock', 'Target']]
                logger.info("📊 最終データ統計:")
                logger.info(f"  期間: {final_df['Date'].min()} ～ {final_df['Date'].max()}")
                logger.info(f"  銘柄数: {final_df['Stock'].nunique()}")
                logger.info(f"  特徴量数: {len(feature_cols)}")
                logger.info(f"  データ品質: {(1 - final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns))) * 100:.1f}%")
                
            except Exception as e:
                logger.error(f"❌ データ保存失敗: {e}")
        
        return final_df

# 実行部分
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Enhanced data integration pipeline")
    cli_parser.add_argument("--skip-highfreq", action="store_true", help="高頻度指標を統合せずに出力する")
    args = cli_parser.parse_args()

    integrator = EnhancedDataIntegration(include_highfreq=not args.skip_highfreq)
    enhanced_data = integrator.run_integration()
    
    if not enhanced_data.empty:
        logger.success("🎉 拡張データ統合完了！60%精度向上の準備が整いました")
    else:
        logger.error("⚠️ 統合処理に失敗しました")
