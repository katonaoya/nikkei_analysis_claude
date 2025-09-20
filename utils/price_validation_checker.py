#!/usr/bin/env python3
"""
production_reportsの価格データと実際の株価データの整合性チェッカー
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceValidationChecker:
    """価格データ整合性チェッククラス"""
    
    def __init__(self):
        self.reports_dir = Path("./production_reports")
        self.data_dir = Path("./data")
        
    def load_stock_data(self) -> pd.DataFrame:
        """株価データ読み込み"""
        logger.info("📊 株価データ読み込み中...")
        
        # Enhanced J-Quantsデータを使用
        enhanced_files = list(self.data_dir.rglob("enhanced_jquants*.parquet"))
        if enhanced_files:
            latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Code'] = df['Code'].astype(str)
            logger.info(f"✅ Enhanced J-Quantsデータ読み込み: {len(df):,}件")
            return df
        
        logger.error("株価データが見つかりません")
        return pd.DataFrame()
    
    def extract_report_prices(self, report_file: Path) -> list:
        """レポートから価格情報を抽出"""
        try:
            content = report_file.read_text(encoding='utf-8')
            
            # 日付抽出
            date_match = re.search(r'(\d{4})年(\d{2})月(\d{2})日', content)
            if not date_match:
                date_match = re.search(r'(\d{8})', report_file.name)
                if date_match:
                    date_str = date_match.group(1)
                    report_date = datetime.strptime(date_str, "%Y%m%d")
                else:
                    return []
            else:
                year, month, day = date_match.groups()
                report_date = datetime(int(year), int(month), int(day))
            
            prices = []
            
            # TOP3推奨銘柄の価格抽出
            top3_pattern = r'### \d+\.\s*【.*?】\s*(.+?)\s*\((\d{4})\).*?\n.*?現在価格.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)円'
            top3_matches = re.findall(top3_pattern, content, re.MULTILINE | re.DOTALL)
            
            for match in top3_matches:
                company_name, code, price_str = match
                price = float(price_str.replace(',', ''))
                prices.append({
                    'source': 'TOP3',
                    'company_name': company_name.strip(),
                    'code': code,
                    'report_price': price,
                    'report_date': report_date
                })
            
            # テーブル形式の価格抽出
            table_pattern = r'\|\s*\d+\s*\|\s*(\d{4})\s*\|\s*(.+?)\s*\|\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\|'
            table_matches = re.findall(table_pattern, content)
            
            for match in table_matches:
                code, company_name, price_str = match
                price = float(price_str.replace(',', ''))
                prices.append({
                    'source': 'テーブル',
                    'company_name': company_name.strip(),
                    'code': code,
                    'report_price': price,
                    'report_date': report_date
                })
            
            logger.info(f"📋 {report_file.name}: {len(prices)}銘柄の価格抽出")
            return prices
            
        except Exception as e:
            logger.error(f"レポート解析エラー {report_file.name}: {e}")
            return []
    
    def get_actual_price(self, stock_data: pd.DataFrame, code: str, target_date: datetime) -> dict:
        """実際の株価データから該当日の価格取得"""
        if stock_data.empty:
            return {'status': 'no_data', 'price': None, 'date': None}
        
        # 指定銘柄のデータ抽出
        code_data = stock_data[stock_data['Code'] == code].copy()
        if code_data.empty:
            return {'status': 'no_stock', 'price': None, 'date': None}
        
        # 目標日付の前日（レポート作成基準日）を探す
        target_dates = [
            target_date - timedelta(days=1),  # 前日
            target_date,                      # 当日
            target_date - timedelta(days=2),  # 2日前
            target_date - timedelta(days=3)   # 3日前
        ]
        
        for check_date in target_dates:
            day_data = code_data[code_data['Date'].dt.date == check_date.date()]
            if not day_data.empty:
                actual_data = day_data.iloc[-1]  # 最後のレコード使用
                return {
                    'status': 'found',
                    'price': actual_data['Close'],
                    'date': actual_data['Date'],
                    'open': actual_data.get('Open', None),
                    'high': actual_data.get('High', None),
                    'low': actual_data.get('Low', None)
                }
        
        # 近似日付を検索
        nearest = code_data.iloc[(code_data['Date'] - target_date).abs().argsort()[:1]]
        if not nearest.empty:
            nearest_row = nearest.iloc[0]
            return {
                'status': 'nearest',
                'price': nearest_row['Close'],
                'date': nearest_row['Date'],
                'days_diff': abs((nearest_row['Date'] - target_date).days)
            }
        
        return {'status': 'not_found', 'price': None, 'date': None}
    
    def validate_single_report(self, report_file: Path, stock_data: pd.DataFrame) -> dict:
        """単一レポートの価格検証"""
        logger.info(f"🔍 {report_file.name} 検証中...")
        
        report_prices = self.extract_report_prices(report_file)
        if not report_prices:
            return {'status': 'no_prices', 'results': []}
        
        results = []
        
        for price_info in report_prices:
            actual_info = self.get_actual_price(
                stock_data, 
                price_info['code'], 
                price_info['report_date']
            )
            
            result = {
                'company': price_info['company_name'],
                'code': price_info['code'],
                'source': price_info['source'],
                'report_date': price_info['report_date'],
                'report_price': price_info['report_price'],
                'actual_status': actual_info['status'],
                'actual_price': actual_info.get('price'),
                'actual_date': actual_info.get('date'),
                'price_diff': None,
                'percentage_diff': None,
                'validation_status': 'unknown'
            }
            
            if actual_info['status'] == 'found' and actual_info['price']:
                result['price_diff'] = result['report_price'] - actual_info['price']
                result['percentage_diff'] = (result['price_diff'] / actual_info['price']) * 100
                
                # 検証ステータス判定
                abs_diff = abs(result['percentage_diff'])
                if abs_diff <= 1.0:
                    result['validation_status'] = '✅ 一致'
                elif abs_diff <= 5.0:
                    result['validation_status'] = '🟡 軽微差異'
                elif abs_diff <= 10.0:
                    result['validation_status'] = '🟠 注意差異'
                else:
                    result['validation_status'] = '❌ 重大差異'
            else:
                result['validation_status'] = f"⚠️ {actual_info['status']}"
            
            results.append(result)
        
        return {'status': 'validated', 'results': results}
    
    def run_validation(self) -> dict:
        """全レポートの価格検証実行"""
        logger.info("🚀 価格データ整合性検証開始")
        
        # 株価データ読み込み
        stock_data = self.load_stock_data()
        if stock_data.empty:
            return {'error': '株価データの読み込みに失敗'}
        
        # レポートファイル取得
        report_files = sorted(list(self.reports_dir.glob("*.md")))
        if not report_files:
            return {'error': 'レポートファイルが見つかりません'}
        
        logger.info(f"📁 検証対象レポート: {len(report_files)}件")
        
        validation_results = []
        summary_stats = {
            'total_comparisons': 0,
            'perfect_matches': 0,
            'minor_differences': 0,
            'attention_differences': 0,
            'major_differences': 0,
            'data_unavailable': 0
        }
        
        for report_file in report_files:
            validation = self.validate_single_report(report_file, stock_data)
            validation['report_file'] = report_file.name
            validation_results.append(validation)
            
            # 統計集計
            if validation.get('results'):
                for result in validation['results']:
                    summary_stats['total_comparisons'] += 1
                    
                    status = result['validation_status']
                    if '✅' in status:
                        summary_stats['perfect_matches'] += 1
                    elif '🟡' in status:
                        summary_stats['minor_differences'] += 1
                    elif '🟠' in status:
                        summary_stats['attention_differences'] += 1
                    elif '❌' in status:
                        summary_stats['major_differences'] += 1
                    else:
                        summary_stats['data_unavailable'] += 1
        
        return {
            'validation_results': validation_results,
            'summary_stats': summary_stats,
            'stock_data_info': {
                'total_records': len(stock_data),
                'date_range': f"{stock_data['Date'].min().date()} ~ {stock_data['Date'].max().date()}",
                'stock_count': stock_data['Code'].nunique()
            }
        }
    
    def print_validation_report(self, results: dict):
        """検証結果レポート出力"""
        if 'error' in results:
            print(f"❌ エラー: {results['error']}")
            return
        
        print("="*80)
        print("🔍 PRODUCTION REPORTS 価格データ整合性検証結果")
        print("="*80)
        
        # サマリー統計
        stats = results['summary_stats']
        total = stats['total_comparisons']
        
        print(f"\n📊 検証サマリー:")
        print(f"   総比較数: {total}件")
        print(f"   ✅ 完全一致: {stats['perfect_matches']}件 ({stats['perfect_matches']/total*100:.1f}%)")
        print(f"   🟡 軽微差異: {stats['minor_differences']}件 ({stats['minor_differences']/total*100:.1f}%)")
        print(f"   🟠 注意差異: {stats['attention_differences']}件 ({stats['attention_differences']/total*100:.1f}%)")
        print(f"   ❌ 重大差異: {stats['major_differences']}件 ({stats['major_differences']/total*100:.1f}%)")
        print(f"   ⚠️ データなし: {stats['data_unavailable']}件 ({stats['data_unavailable']/total*100:.1f}%)")
        
        # 株価データ情報
        stock_info = results['stock_data_info']
        print(f"\n📈 参照株価データ:")
        print(f"   レコード数: {stock_info['total_records']:,}件")
        print(f"   期間: {stock_info['date_range']}")
        print(f"   銘柄数: {stock_info['stock_count']}社")
        
        # 重大差異の詳細表示
        print(f"\n❌ 重大差異 (10%以上) の詳細:")
        major_issues = []
        
        for validation in results['validation_results']:
            if validation.get('results'):
                for result in validation['results']:
                    if '❌' in result['validation_status']:
                        major_issues.append({
                            'report': validation['report_file'],
                            'company': result['company'],
                            'code': result['code'],
                            'report_price': result['report_price'],
                            'actual_price': result['actual_price'],
                            'diff_pct': result['percentage_diff']
                        })
        
        if major_issues:
            for issue in major_issues[:10]:  # 最大10件表示
                print(f"   📋 {issue['report']} - {issue['company']} ({issue['code']})")
                print(f"     レポート価格: {issue['report_price']:,.0f}円")
                print(f"     実際価格: {issue['actual_price']:,.0f}円")
                print(f"     差異: {issue['diff_pct']:+.1f}%")
        else:
            print("   なし（良好）")

def main():
    """メイン関数"""
    checker = PriceValidationChecker()
    results = checker.run_validation()
    checker.print_validation_report(results)

if __name__ == "__main__":
    main()