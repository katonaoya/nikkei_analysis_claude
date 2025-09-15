#!/usr/bin/env python3
"""
価格データ不整合の徹底的な原因調査スクリプト
全データソースを分析し、根本原因を特定する
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePriceInvestigator:
    """価格データ不整合の包括的調査クラス"""
    
    def __init__(self):
        self.reports_dir = Path("./production_reports")
        self.data_dir = Path("./data")
        self.results_dir = Path("./investigation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 調査対象銘柄（問題の多かった銘柄）
        self.target_stocks = {
            "9984": "ソフトバンクG",
            "6758": "ソニーG", 
            "7974": "任天堂",
            "4478": "フリー",
            "8035": "東京エレクトロン",
            "6098": "リクルートHD",
            "7203": "トヨタ自動車",
            "4519": "中外製薬"
        }
        
    def load_all_data_sources(self) -> dict:
        """全データソースを読み込み"""
        logger.info("🔍 全データソース読み込み開始...")
        
        data_sources = {}
        
        # 1. Enhanced J-Quantsデータ
        enhanced_files = list(self.data_dir.rglob("enhanced_jquants*.parquet"))
        if enhanced_files:
            latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Code'] = df['Code'].astype(str)
            data_sources['enhanced_jquants'] = {
                'data': df,
                'file': latest_file.name,
                'records': len(df),
                'date_range': f"{df['Date'].min().date()} ~ {df['Date'].max().date()}",
                'stocks': df['Code'].nunique()
            }
            logger.info(f"✅ Enhanced J-Quants: {len(df):,}件 ({df['Date'].min().date()} ~ {df['Date'].max().date()})")
        
        # 2. Nikkei225 Fullデータ
        nikkei_files = list(self.data_dir.rglob("nikkei225_full*.parquet"))
        if nikkei_files:
            latest_file = max(nikkei_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Code'] = df['Code'].astype(str)
            data_sources['nikkei225_full'] = {
                'data': df,
                'file': latest_file.name,
                'records': len(df),
                'date_range': f"{df['Date'].min().date()} ~ {df['Date'].max().date()}",
                'stocks': df['Code'].nunique()
            }
            logger.info(f"✅ Nikkei225 Full: {len(df):,}件 ({df['Date'].min().date()} ~ {df['Date'].max().date()})")
        
        # 3. その他のparquetファイル
        other_parquets = list(self.data_dir.rglob("*.parquet"))
        for file in other_parquets:
            if 'enhanced_jquants' not in file.name and 'nikkei225_full' not in file.name:
                try:
                    df = pd.read_parquet(file)
                    if 'Date' in df.columns and 'Code' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Code'] = df['Code'].astype(str)
                        key = file.stem
                        data_sources[key] = {
                            'data': df,
                            'file': file.name,
                            'records': len(df),
                            'date_range': f"{df['Date'].min().date()} ~ {df['Date'].max().date()}",
                            'stocks': df['Code'].nunique()
                        }
                        logger.info(f"✅ {key}: {len(df):,}件")
                except Exception as e:
                    logger.warning(f"⚠️ {file.name} 読み込みエラー: {e}")
        
        return data_sources
    
    def analyze_single_stock_across_sources(self, code: str, target_date: str) -> dict:
        """単一銘柄を全データソースで分析"""
        logger.info(f"🔍 {self.target_stocks.get(code, code)} ({code}) - {target_date} 詳細分析")
        
        data_sources = self.load_all_data_sources()
        target_dt = pd.to_datetime(target_date)
        
        results = {
            'code': code,
            'company': self.target_stocks.get(code, '不明'),
            'target_date': target_date,
            'data_sources': []
        }
        
        for source_name, source_info in data_sources.items():
            df = source_info['data']
            
            # 該当銘柄データ抽出
            stock_data = df[df['Code'] == code]
            
            if stock_data.empty:
                results['data_sources'].append({
                    'source': source_name,
                    'status': 'no_stock_data',
                    'file': source_info['file']
                })
                continue
            
            # 目標日付近辺のデータ検索
            date_variants = [
                target_dt,
                target_dt - timedelta(days=1),
                target_dt + timedelta(days=1),
                target_dt - timedelta(days=2),
                target_dt + timedelta(days=2)
            ]
            
            found_data = []
            for check_date in date_variants:
                day_data = stock_data[stock_data['Date'].dt.date == check_date.date()]
                if not day_data.empty:
                    row = day_data.iloc[-1]
                    found_data.append({
                        'date': row['Date'],
                        'open': row.get('Open', 'N/A'),
                        'high': row.get('High', 'N/A'),
                        'low': row.get('Low', 'N/A'),
                        'close': row.get('Close', 'N/A'),
                        'volume': row.get('Volume', 'N/A'),
                        'days_from_target': (check_date.date() - target_dt.date()).days
                    })
            
            if found_data:
                results['data_sources'].append({
                    'source': source_name,
                    'status': 'found',
                    'file': source_info['file'],
                    'price_data': found_data,
                    'total_records': len(stock_data),
                    'date_range': f"{stock_data['Date'].min().date()} ~ {stock_data['Date'].max().date()}"
                })
            else:
                # 最近接データを取得
                if not stock_data.empty:
                    nearest = stock_data.iloc[(stock_data['Date'] - target_dt).abs().argsort()[:1]]
                    if not nearest.empty:
                        row = nearest.iloc[0]
                        results['data_sources'].append({
                            'source': source_name,
                            'status': 'nearest_only',
                            'file': source_info['file'],
                            'nearest_data': {
                                'date': row['Date'],
                                'close': row.get('Close', 'N/A'),
                                'days_diff': abs((row['Date'] - target_dt).days)
                            },
                            'total_records': len(stock_data)
                        })
        
        return results
    
    def extract_report_price_with_context(self, report_file: Path) -> dict:
        """レポートから価格とその文脈を詳細抽出"""
        try:
            content = report_file.read_text(encoding='utf-8')
            
            # レポート日付抽出
            date_match = re.search(r'(\d{8})', report_file.name)
            if date_match:
                report_date = datetime.strptime(date_match.group(1), "%Y%m%d").date()
            else:
                return {'error': 'date_not_found'}
            
            # 生成時刻抽出
            generation_time_match = re.search(r'生成時刻.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', content)
            generation_time = generation_time_match.group(1) if generation_time_match else None
            
            # 対象日抽出
            target_date_match = re.search(r'予測対象日.*?(\d{4})年(\d{2})月(\d{2})日', content)
            target_date = None
            if target_date_match:
                year, month, day = target_date_match.groups()
                target_date = datetime(int(year), int(month), int(day)).date()
            
            # 銘柄価格抽出（詳細版）
            stock_prices = []
            
            # TOP3推奨銘柄のパターン
            top3_pattern = r'### (\d+)\.\s*【.*?】\s*(.+?)\s*\((\d{4})\).*?\n(.*?)\n.*?現在価格.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)円.*?\n.*?上昇確率.*?(\d+\.\d+)%'
            
            for match in re.finditer(top3_pattern, content, re.MULTILINE | re.DOTALL):
                rank, company, code, context, price_str, probability = match.groups()
                
                # 技術指標抽出
                tech_section = content[match.end():match.end()+500]
                tech_indicators = re.findall(r'- (.+)', tech_section)
                
                stock_prices.append({
                    'rank': int(rank),
                    'company': company.strip(),
                    'code': code,
                    'price': float(price_str.replace(',', '')),
                    'probability': float(probability),
                    'context': context.strip(),
                    'technical_indicators': tech_indicators[:5]  # 最初の5個
                })
            
            return {
                'report_file': report_file.name,
                'report_date': report_date,
                'target_date': target_date,
                'generation_time': generation_time,
                'stock_prices': stock_prices,
                'full_content_length': len(content)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def deep_dive_price_discrepancies(self) -> dict:
        """価格差異の深掘り分析"""
        logger.info("🚀 価格差異の深掘り分析開始...")
        
        # サンプルレポートで詳細分析
        sample_reports = [
            "20250801_prediction_report.md",
            "20250802_prediction_report.md", 
            "20250804_prediction_report.md"
        ]
        
        investigation_results = []
        
        for report_name in sample_reports:
            report_file = self.reports_dir / report_name
            if not report_file.exists():
                continue
            
            logger.info(f"📋 {report_name} 詳細分析中...")
            
            # レポート情報抽出
            report_info = self.extract_report_price_with_context(report_file)
            if 'error' in report_info:
                continue
            
            # 各銘柄を詳細分析
            stock_analyses = []
            
            for stock_price in report_info['stock_prices']:
                code = stock_price['code']
                
                if code in self.target_stocks:
                    # 複数日付での分析
                    analysis_dates = []
                    
                    if report_info['target_date']:
                        analysis_dates.append(report_info['target_date'].strftime('%Y-%m-%d'))
                    if report_info['report_date']:
                        analysis_dates.append(report_info['report_date'].strftime('%Y-%m-%d'))
                    
                    # 前後の日付も追加
                    base_date = report_info['report_date'] if report_info['report_date'] else report_info['target_date']
                    if base_date:
                        for offset in [-2, -1, 0, 1, 2]:
                            check_date = base_date + timedelta(days=offset)
                            analysis_dates.append(check_date.strftime('%Y-%m-%d'))
                    
                    # 重複除去
                    analysis_dates = list(set(analysis_dates))
                    
                    date_analyses = []
                    for date_str in analysis_dates[:3]:  # 最大3日分
                        data_analysis = self.analyze_single_stock_across_sources(code, date_str)
                        date_analyses.append(data_analysis)
                    
                    stock_analyses.append({
                        'report_stock_info': stock_price,
                        'multi_date_analysis': date_analyses
                    })
            
            investigation_results.append({
                'report_info': report_info,
                'stock_analyses': stock_analyses
            })
        
        return {
            'investigation_date': datetime.now().isoformat(),
            'analyzed_reports': len(investigation_results),
            'results': investigation_results
        }
    
    def identify_root_cause(self, investigation_data: dict) -> dict:
        """根本原因の特定"""
        logger.info("🔬 根本原因特定分析中...")
        
        patterns = {
            'price_scale_issues': [],
            'date_mismatches': [],
            'data_source_inconsistencies': [],
            'potential_mock_data': [],
            'adjustment_factor_issues': []
        }
        
        for result in investigation_data['results']:
            report_info = result['report_info']
            
            for stock_analysis in result['stock_analyses']:
                report_price = stock_analysis['report_stock_info']['price']
                code = stock_analysis['report_stock_info']['code']
                company = stock_analysis['report_stock_info']['company']
                
                # 複数データソースでの価格比較
                for date_analysis in stock_analysis['multi_date_analysis']:
                    for source_data in date_analysis['data_sources']:
                        if source_data['status'] == 'found':
                            for price_data in source_data['price_data']:
                                actual_close = price_data.get('close')
                                if actual_close and actual_close != 'N/A':
                                    try:
                                        actual_close = float(actual_close)
                                        diff_pct = ((report_price - actual_close) / actual_close) * 100
                                        
                                        # パターン分析
                                        if abs(diff_pct) > 50:
                                            patterns['price_scale_issues'].append({
                                                'company': company,
                                                'code': code,
                                                'report_price': report_price,
                                                'actual_price': actual_close,
                                                'diff_pct': diff_pct,
                                                'source': source_data['source'],
                                                'date': price_data['date']
                                            })
                                        
                                        # 特定の倍数関係をチェック
                                        ratios = [0.1, 0.5, 2.0, 10.0, 100.0, 1000.0]
                                        for ratio in ratios:
                                            if abs(report_price / actual_close - ratio) < 0.05:
                                                patterns['adjustment_factor_issues'].append({
                                                    'company': company,
                                                    'code': code,
                                                    'ratio': ratio,
                                                    'report_price': report_price,
                                                    'actual_price': actual_close
                                                })
                                    except:
                                        pass
        
        # 根本原因の推定
        root_causes = []
        
        if len(patterns['price_scale_issues']) > 5:
            root_causes.append({
                'cause': 'データソース間のスケール不整合',
                'evidence_count': len(patterns['price_scale_issues']),
                'confidence': '高',
                'description': 'レポートと実データで価格のスケールが大幅に異なる'
            })
        
        if len(patterns['adjustment_factor_issues']) > 3:
            root_causes.append({
                'cause': '株価調整係数の問題',
                'evidence_count': len(patterns['adjustment_factor_issues']),
                'confidence': '中',
                'description': '株式分割等の調整が正しく処理されていない可能性'
            })
        
        return {
            'patterns': patterns,
            'root_causes': root_causes,
            'recommendations': self.generate_recommendations(patterns, root_causes)
        }
    
    def generate_recommendations(self, patterns: dict, root_causes: list) -> list:
        """修正推奨事項の生成"""
        recommendations = []
        
        if any('データソース間のスケール不整合' in cause['cause'] for cause in root_causes):
            recommendations.append({
                'priority': '最高',
                'action': 'データソース統一',
                'description': '全ての価格データを単一の信頼できるソースに統一する',
                'implementation': '1. 最も正確なデータソースを特定 2. 全システムで統一使用'
            })
        
        recommendations.append({
            'priority': '高',
            'action': '価格データ検証システム構築',
            'description': 'リアルタイムで市場価格との整合性をチェック',
            'implementation': '1. 外部価格API連携 2. 自動アラート機能'
        })
        
        recommendations.append({
            'priority': '中',
            'action': 'バックテスト再実行',
            'description': '正確な価格データでバックテストを再実行',
            'implementation': '1. 修正データでの完全再計算 2. 結果比較レポート作成'
        })
        
        return recommendations
    
    def run_comprehensive_investigation(self):
        """包括的調査の実行"""
        logger.info("🔍 包括的価格データ調査開始...")
        
        # 深掘り分析実行
        investigation_data = self.deep_dive_price_discrepancies()
        
        # 根本原因特定
        root_cause_analysis = self.identify_root_cause(investigation_data)
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細結果保存
        investigation_file = self.results_dir / f"price_investigation_{timestamp}.json"
        with open(investigation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'investigation_data': investigation_data,
                'root_cause_analysis': root_cause_analysis
            }, f, ensure_ascii=False, indent=2, default=str)
        
        # サマリーレポート出力
        self.print_investigation_summary(root_cause_analysis)
        
        logger.info(f"💾 詳細調査結果保存: {investigation_file}")
        
        return root_cause_analysis
    
    def print_investigation_summary(self, analysis: dict):
        """調査結果サマリー出力"""
        print("\n" + "="*80)
        print("🔬 価格データ不整合 - 根本原因調査結果")
        print("="*80)
        
        print(f"\n📊 発見されたパターン:")
        patterns = analysis['patterns']
        for pattern_name, issues in patterns.items():
            if issues:
                print(f"   {pattern_name}: {len(issues)}件")
        
        print(f"\n🎯 特定された根本原因:")
        for i, cause in enumerate(analysis['root_causes'], 1):
            print(f"   {i}. {cause['cause']}")
            print(f"      証拠数: {cause['evidence_count']}件")
            print(f"      信頼度: {cause['confidence']}")
            print(f"      説明: {cause['description']}")
        
        print(f"\n🛠️ 修正推奨事項:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. [{rec['priority']}] {rec['action']}")
            print(f"      {rec['description']}")
            print(f"      実装: {rec['implementation']}")

def main():
    """メイン関数"""
    investigator = ComprehensivePriceInvestigator()
    investigator.run_comprehensive_investigation()

if __name__ == "__main__":
    main()