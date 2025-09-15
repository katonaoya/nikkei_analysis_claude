#!/usr/bin/env python3
"""
価格データ整合性バリデーター
レポート生成時に自動的に価格の正確性を検証し、異常があれば警告
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceIntegrityValidator:
    """価格データ整合性バリデーター"""
    
    def __init__(self):
        self.data_dir = Path("./data")
        self.tolerance = 0.01  # 1%以内の差異は許容
        
    def load_reference_data(self):
        """参照用の正確な株価データを読み込み"""
        try:
            # Enhanced J-Quantsデータを基準とする
            enhanced_files = list(self.data_dir.rglob("enhanced_jquants*.parquet"))
            if enhanced_files:
                latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest_file)
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df['Code'] = df['Code'].astype(str)
                logger.info(f"✅ 参照データ読み込み: {len(df):,}件")
                return df
            else:
                logger.error("❌ 参照データが見つかりません")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ 参照データ読み込みエラー: {e}")
            return pd.DataFrame()
    
    def validate_prediction_prices(self, prediction_result, reference_data):
        """予測結果の価格を検証"""
        if not prediction_result or reference_data.empty:
            return False, []
        
        target_date = prediction_result['date']
        top3 = prediction_result['top3_recommendations']
        validation_errors = []
        
        logger.info(f"🔍 {target_date}の価格整合性検証開始")
        
        for _, stock in top3.iterrows():
            stock_code = stock['Stock']
            report_price = stock['close']
            
            # 参照データから正確な価格を取得
            ref_data = reference_data[
                (reference_data['Code'] == stock_code) & 
                (reference_data['Date'] == target_date)
            ]
            
            if not ref_data.empty:
                actual_price = ref_data.iloc[-1]['Close']
                price_diff = abs((report_price - actual_price) / actual_price)
                
                if price_diff > self.tolerance:
                    error = {
                        'stock_code': stock_code,
                        'target_date': target_date,
                        'report_price': report_price,
                        'actual_price': actual_price,
                        'difference_pct': price_diff * 100,
                        'status': '❌ 重大な価格乖離'
                    }
                    validation_errors.append(error)
                    logger.error(f"❌ {stock_code}: レポート価格{report_price:.0f}円 vs 実際{actual_price:.0f}円 ({price_diff*100:.1f}%乖離)")
                else:
                    logger.info(f"✅ {stock_code}: 価格検証OK ({price_diff*100:.2f}%以内)")
            else:
                logger.warning(f"⚠️ {stock_code}: 参照データなし")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
    
    def create_validation_report(self, validation_errors, target_date):
        """検証エラーレポートを作成"""
        if not validation_errors:
            return None
        
        report = f"""
# 🚨 価格データ整合性エラー - {target_date}

## 検出された価格乖離

"""
        
        for error in validation_errors:
            report += f"""
### {error['stock_code']} - {error['status']}
- **レポート価格**: {error['report_price']:,.0f}円
- **実際価格**: {error['actual_price']:,.0f}円  
- **差異**: {error['difference_pct']:+.1f}%
- **許容範囲**: ±{self.tolerance*100:.1f}%

"""
        
        report += f"""
## 推奨対応
1. データソースの確認
2. 価格変換ロジックの見直し
3. データ更新タイミングの確認

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_validation_report(self, report_content, target_date):
        """検証エラーレポートを保存"""
        if not report_content:
            return False
        
        error_dir = Path("./validation_errors")
        error_dir.mkdir(exist_ok=True)
        
        filename = f"price_validation_error_{target_date}.md"
        filepath = error_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.error(f"🚨 検証エラーレポート保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ エラーレポート保存失敗: {e}")
            return False
    
    def validate_report_generation(self, generator_instance, prediction_result):
        """レポート生成時の価格検証フック"""
        reference_data = self.load_reference_data()
        is_valid, errors = self.validate_prediction_prices(prediction_result, reference_data)
        
        if not is_valid:
            # エラーレポート作成
            target_date = prediction_result['date']
            report_content = self.create_validation_report(errors, target_date)
            self.save_validation_report(report_content, target_date)
            
            # 重大エラーの場合は生成を停止
            critical_errors = [e for e in errors if e['difference_pct'] > 10.0]
            if critical_errors:
                logger.error(f"🚨 重大な価格乖離検出。レポート生成を停止します。")
                return False
        
        return True

# 使用例とテスト関数
def test_validator():
    """バリデーターのテスト"""
    validator = PriceIntegrityValidator()
    reference_data = validator.load_reference_data()
    
    if not reference_data.empty:
        logger.info("✅ 価格整合性バリデーター準備完了")
        logger.info(f"📊 参照データ: {len(reference_data):,}件, {reference_data['Code'].nunique()}銘柄")
    else:
        logger.error("❌ 参照データの読み込みに失敗")

if __name__ == "__main__":
    test_validator()