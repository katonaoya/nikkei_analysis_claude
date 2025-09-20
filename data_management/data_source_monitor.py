#!/usr/bin/env python3
"""
データソース監視システム
価格データの整合性を定期的に監視し、異常があれば即座に通知
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib
import json

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSourceMonitor:
    """データソース監視クラス"""
    
    def __init__(self):
        self.data_dir = Path("./data")
        self.monitor_dir = Path("./data_monitoring")
        self.monitor_dir.mkdir(exist_ok=True)
        
    def get_data_fingerprint(self, df):
        """データのフィンガープリントを生成"""
        # 主要統計値からハッシュを生成
        stats = {
            'record_count': len(df),
            'stock_count': df['Code'].nunique(),
            'price_mean': float(df['Close'].mean()),
            'price_std': float(df['Close'].std()),
            'date_range': f"{df['Date'].min()}_{df['Date'].max()}"
        }
        
        stats_str = json.dumps(stats, sort_keys=True)
        fingerprint = hashlib.md5(stats_str.encode()).hexdigest()
        return fingerprint, stats
    
    def check_data_consistency(self):
        """データソース間の整合性をチェック"""
        logger.info("🔍 データソース整合性チェック開始")
        
        # real_jquants_data.parquetをチェック
        real_path = self.data_dir / "processed" / "real_jquants_data.parquet"
        enhanced_path = self.data_dir / "processed" / "enhanced_jquants_data.parquet"
        
        issues = []
        
        if real_path.exists():
            real_df = pd.read_parquet(real_path)
            real_df['Date'] = pd.to_datetime(real_df['Date']).dt.date
            real_fingerprint, real_stats = self.get_data_fingerprint(real_df)
            
            logger.info(f"📊 real_jquants_data: {real_stats['record_count']:,}件, {real_stats['stock_count']}銘柄")
            logger.info(f"   価格範囲: {real_df['Close'].min():.0f} ~ {real_df['Close'].max():.0f}円")
            
            # 異常な価格データをチェック
            if real_df['Close'].min() < 10 or real_df['Close'].max() > 100000:
                issues.append({
                    'type': 'price_range_anomaly',
                    'source': 'real_jquants_data',
                    'details': f"価格範囲異常: {real_df['Close'].min():.0f} ~ {real_df['Close'].max():.0f}円"
                })
        
        if enhanced_path.exists():
            enhanced_df = pd.read_parquet(enhanced_path)
            enhanced_df['Date'] = pd.to_datetime(enhanced_df['Date']).dt.date
            enhanced_fingerprint, enhanced_stats = self.get_data_fingerprint(enhanced_df)
            
            logger.info(f"📊 enhanced_jquants_data: {enhanced_stats['record_count']:,}件, {enhanced_stats['stock_count']}銘柄")
            logger.info(f"   価格範囲: {enhanced_df['Close'].min():.0f} ~ {enhanced_df['Close'].max():.0f}円")
            
            # 両方のデータが存在する場合、価格の整合性をチェック
            if real_path.exists():
                # 共通銘柄・共通日付で価格比較
                common_codes = set(real_df['Code'].unique()) & set(enhanced_df['Code'].unique())
                logger.info(f"🔍 共通銘柄数: {len(common_codes)}")
                
                for code in list(common_codes)[:5]:  # サンプル5銘柄
                    real_sample = real_df[real_df['Code'] == code].tail(5)
                    enhanced_sample = enhanced_df[enhanced_df['Code'] == code].tail(5)
                    
                    for _, real_row in real_sample.iterrows():
                        enhanced_match = enhanced_sample[enhanced_sample['Date'] == real_row['Date']]
                        if not enhanced_match.empty:
                            real_price = real_row['Close']
                            enhanced_price = enhanced_match.iloc[0]['Close']
                            
                            if abs(real_price - enhanced_price) > 0.01:
                                issues.append({
                                    'type': 'price_mismatch',
                                    'stock_code': code,
                                    'date': real_row['Date'],
                                    'real_price': real_price,
                                    'enhanced_price': enhanced_price,
                                    'details': f"価格不一致: {real_price} vs {enhanced_price}"
                                })
        
        return issues
    
    def save_monitoring_report(self, issues):
        """監視レポートを保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.monitor_dir / f"data_monitoring_report_{timestamp}.json"
        
        monitoring_data = {
            'timestamp': timestamp,
            'issues_found': len(issues),
            'issues': issues,
            'status': 'clean' if len(issues) == 0 else 'issues_detected'
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📋 監視レポート保存: {report_path}")
        return report_path
    
    def run_monitoring(self):
        """監視実行"""
        logger.info("🚀 データソース監視開始")
        
        issues = self.check_data_consistency()
        report_path = self.save_monitoring_report(issues)
        
        if issues:
            logger.error(f"🚨 {len(issues)}件の問題を検出しました")
            for issue in issues:
                logger.error(f"   {issue['type']}: {issue['details']}")
        else:
            logger.info("✅ データソース整合性OK")
        
        return len(issues) == 0

def main():
    """メイン関数"""
    monitor = DataSourceMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main()