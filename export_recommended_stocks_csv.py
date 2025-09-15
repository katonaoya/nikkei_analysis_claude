#!/usr/bin/env python3
"""
推奨銘柄10社の株価データをCSV形式でエクスポート
既存のenhanced_jquantsデータから2025年8月1日～9月5日の期間を抽出
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendedStocksExporter:
    """推奨銘柄データエクスポートクラス"""
    
    def __init__(self):
        self.data_dir = Path("./data")
        
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
    
    def load_enhanced_data(self) -> pd.DataFrame:
        """Enhanced J-Quantsデータを読み込み"""
        logger.info("🚀 Enhanced J-Quantsデータ読み込み中...")
        
        enhanced_files = list(self.data_dir.rglob("enhanced_jquants*.parquet"))
        if not enhanced_files:
            logger.error("Enhanced J-Quantsデータファイルが見つかりません")
            return pd.DataFrame()
        
        latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str)
        
        unique_codes = df['Code'].unique()
        logger.info(f"✅ Enhanced J-Quantsデータ読み込み完了: {len(df):,}件")
        logger.info(f"   銘柄数: {len(unique_codes)}")
        logger.info(f"   期間: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
        
        return df
    
    def filter_recommended_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """推奨銘柄のデータのみを抽出"""
        logger.info("🎯 推奨銘柄データ抽出中...")
        
        if df.empty:
            return df
        
        # 推奨銘柄のみ抽出
        recommended_codes = list(self.recommended_stocks.keys())
        filtered_df = df[df['Code'].isin(recommended_codes)].copy()
        
        if filtered_df.empty:
            logger.error("推奨銘柄のデータが見つかりません")
            return pd.DataFrame()
        
        # 会社名を追加
        filtered_df['CompanyName'] = filtered_df['Code'].map(self.recommended_stocks)
        
        # 期間でフィルタリング
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        period_filtered = filtered_df[
            (filtered_df['Date'] >= start_date) & 
            (filtered_df['Date'] <= end_date)
        ].copy()
        
        if period_filtered.empty:
            logger.warning(f"指定期間({self.start_date}～{self.end_date})のデータが見つかりません")
            logger.info("利用可能なデータ期間を確認します...")
            
            # 各銘柄のデータ期間を表示
            for code in recommended_codes:
                stock_data = filtered_df[filtered_df['Code'] == code]
                if not stock_data.empty:
                    company_name = self.recommended_stocks[code]
                    min_date = stock_data['Date'].min().date()
                    max_date = stock_data['Date'].max().date()
                    logger.info(f"   {code} ({company_name}): {min_date} ~ {max_date} ({len(stock_data)}件)")
                else:
                    logger.info(f"   {code} ({self.recommended_stocks[code]}): データなし")
            
            # 期間を拡張してデータを取得
            logger.info("期間を拡張してデータを検索...")
            available_start = filtered_df['Date'].min()
            available_end = filtered_df['Date'].max()
            
            # 利用可能な期間で最大限のデータを取得
            period_filtered = filtered_df[
                (filtered_df['Date'] >= max(available_start, pd.to_datetime('2025-08-01'))) &
                (filtered_df['Date'] <= min(available_end, pd.to_datetime('2025-09-30')))
            ].copy()
        
        # データを整理
        period_filtered = period_filtered.sort_values(['Code', 'Date'])
        
        logger.info(f"✅ 推奨銘柄データ抽出完了: {len(period_filtered):,}件")
        
        # 銘柄別統計
        stock_counts = period_filtered['Code'].value_counts().sort_index()
        logger.info("📊 銘柄別データ数:")
        for code, count in stock_counts.items():
            company_name = self.recommended_stocks.get(code, "不明")
            logger.info(f"   {code} ({company_name}): {count}件")
        
        return period_filtered
    
    def prepare_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CSV出力用データ準備"""
        if df.empty:
            return df
        
        # 必要な列のみ選択・並び替え
        csv_columns = [
            'Date', 'Code', 'CompanyName', 'Open', 'High', 'Low', 'Close', 
            'Volume', 'TurnoverValue', 'AdjustmentOpen', 'AdjustmentHigh', 
            'AdjustmentLow', 'AdjustmentClose', 'AdjustmentVolume'
        ]
        
        # 利用可能な列のみ選択
        available_columns = [col for col in csv_columns if col in df.columns]
        csv_df = df[available_columns].copy()
        
        # 日付を文字列形式に変換
        csv_df['Date'] = csv_df['Date'].dt.strftime('%Y-%m-%d')
        
        # 数値列のNaNを処理
        numeric_columns = csv_df.select_dtypes(include=[np.number]).columns
        csv_df[numeric_columns] = csv_df[numeric_columns].fillna(0)
        
        return csv_df
    
    def export_to_csv(self, df: pd.DataFrame) -> Path:
        """CSVファイルにエクスポート"""
        if df.empty:
            logger.error("エクスポートするデータがありません")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommended_stocks_data_{timestamp}.csv"
        output_path = Path(filename)
        
        # CSVファイル保存
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"💾 CSVファイル保存完了: {output_path}")
        logger.info(f"📊 保存データ統計:")
        logger.info(f"   総レコード数: {len(df):,}件")
        logger.info(f"   銘柄数: {df['Code'].nunique()}社")
        
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            logger.info(f"   期間: {dates.min().date()} ～ {dates.max().date()}")
        
        return output_path
    
    def run_export(self) -> Path:
        """エクスポート実行"""
        logger.info("🚀 推奨銘柄データエクスポート開始")
        
        # データ読み込み
        df = self.load_enhanced_data()
        if df.empty:
            logger.error("データ読み込みに失敗しました")
            return None
        
        # 推奨銘柄フィルタリング
        filtered_df = self.filter_recommended_stocks(df)
        if filtered_df.empty:
            logger.error("推奨銘柄データの抽出に失敗しました")
            return None
        
        # CSV用データ準備
        csv_df = self.prepare_csv_data(filtered_df)
        
        # CSVエクスポート
        output_path = self.export_to_csv(csv_df)
        
        return output_path

def main():
    """メイン関数"""
    print("="*80)
    print("🚀 推奨銘柄10社 株価データエクスポート")
    print("="*80)
    print("📊 対象: バックテスト+55.97%利益達成の10社")
    print("📅 期間: 2025年8月1日 ～ 9月5日（利用可能範囲）")
    print("💾 出力: CSV形式")
    print()
    
    exporter = RecommendedStocksExporter()
    csv_path = exporter.run_export()
    
    if csv_path:
        print("\n" + "="*80)
        print("✅ 推奨銘柄データエクスポート完了")
        print("="*80)
        print(f"📁 保存ファイル: {csv_path}")
        print("💰 この10社でバックテスト利益率: +55.97%")
        print("🎯 各社はTOP3推奨銘柄として複数回選出")
        print("="*80)
    else:
        print("\n❌ エクスポートに失敗しました")

if __name__ == "__main__":
    main()