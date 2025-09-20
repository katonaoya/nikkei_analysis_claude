#!/usr/bin/env python3
"""
レポート生成の価格変換をデバッグ
"""

import pandas as pd
from datetime import datetime
from production_report_generator import ProductionReportGenerator

def debug_price_generation():
    """価格生成過程をデバッグ"""
    print("=== レポート生成デバッグ開始 ===")
    
    generator = ProductionReportGenerator()
    
    # データ読み込み
    df = generator.load_data()
    df, feature_cols = generator.create_target_and_features(df)
    
    # 2025年8月27日で予測実行
    target_date = datetime(2025, 8, 27)
    prediction_result = generator.predict_for_date(df, target_date, feature_cols)
    
    if prediction_result:
        print(f"\n📊 {target_date.date()}の予測結果:")
        top3 = prediction_result['top3_recommendations']
        
        print("\nTOP3銘柄の価格詳細:")
        for i, (_, stock) in enumerate(top3.iterrows(), 1):
            company_name = generator.get_company_name(stock['Stock'])
            print(f"{i}. {company_name} ({stock['Stock']}):")
            print(f"   stock['close']: {stock['close']}")
            print(f"   型: {type(stock['close'])}")
            print(f"   レポート表示: {stock['close']:.2f}円")
            
            # レポート生成ロジックをトレース
            expected_return = stock['close'] * 1.01
            potential_profit = expected_return - stock['close']
            
            print(f"   目標価格計算: {stock['close']} * 1.01 = {expected_return:.2f}")
            print(f"   期待利益計算: {expected_return:.2f} - {stock['close']} = {potential_profit:.2f}")
            print()
        
        # レポート生成
        report_content = generator.generate_daily_report(prediction_result)
        
        # レポート内の価格を確認
        if report_content:
            lines = report_content.split('\n')
            for line in lines:
                if '現在価格' in line:
                    print(f"レポート行: {line.strip()}")
    
    else:
        print("予測結果なし")

if __name__ == "__main__":
    debug_price_generation()