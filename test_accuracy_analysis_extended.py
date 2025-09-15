#!/usr/bin/env python3
"""
Test extended accuracy analysis functionality including visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.evaluation.accuracy_analyzer import AccuracyAnalyzer

def create_comprehensive_test_data():
    """Create comprehensive test data for accuracy analysis"""
    
    # Create date range for 4 months
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-04-30')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter to business days only
    business_dates = [d for d in date_range if d.weekday() < 5]
    
    stocks = ['1301', '1332', '1333', '4755', '6758', '7203', '8035', '9432', '9984']  # More stocks
    sectors = ['Materials', 'Energy', 'Energy', 'IT', 'Electronics', 'Auto', 'Trading', 'Telecom', 'IT']
    
    # Create predictions data with varying patterns over time
    predictions = []
    actual_returns = []
    np.random.seed(42)
    
    for i, date in enumerate(business_dates):
        # Simulate time-varying model performance
        performance_decay = 1 - (i / len(business_dates)) * 0.3  # Performance degrades over time
        
        # Generate 0-3 predictions per day
        n_predictions = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        
        if n_predictions > 0:
            selected_stocks = np.random.choice(stocks, size=min(n_predictions, len(stocks)), replace=False)
            for stock in selected_stocks:
                # Generate probability with time-varying quality
                base_prob = np.random.beta(3, 2) * 0.4 + 0.6
                probability = base_prob * performance_decay
                
                # Determine if target is achieved (biased by probability)
                achievement_prob = 0.3 + 0.5 * (probability - 0.6) / 0.4  # Linear relationship
                target_achieved = np.random.random() < achievement_prob
                
                predictions.append({
                    'Date': date,
                    'Code': stock,
                    'prediction': 1,
                    'prediction_probability': probability,
                    'sector': sectors[stocks.index(stock)]
                })
                
                # Generate actual returns
                if target_achieved:
                    # Positive return with some noise
                    return_val = np.random.normal(0.02, 0.01)  # 2% mean, 1% std
                else:
                    # Mixed returns
                    return_val = np.random.normal(-0.005, 0.02)  # Slight negative bias
                
                actual_returns.append({
                    'Date': date,
                    'Code': stock,
                    'return_1d': return_val,
                    'high_return_1d': max(return_val + np.random.uniform(0, 0.01), return_val),
                    'target_achieved': target_achieved
                })
    
    predictions_df = pd.DataFrame(predictions)
    actual_returns_df = pd.DataFrame(actual_returns)
    
    # Create stock metadata
    stock_metadata = pd.DataFrame({
        'Code': stocks,
        'sector': sectors,
        'market_cap': np.random.choice(['Large', 'Mid', 'Small'], size=len(stocks)),
        'listing_years': np.random.randint(5, 30, len(stocks))
    })
    
    # Create market data (simulate volatility regimes)
    market_data = []
    for date in business_dates:
        # Simulate market volatility with time-varying patterns
        base_vol = 0.15 + 0.10 * np.sin(2 * np.pi * (date - start_date).days / 90)  # 3-month cycle
        daily_vol = abs(np.random.normal(base_vol, 0.05))
        
        market_data.append({
            'Date': date,
            'market_volatility': daily_vol,
            'vix_level': daily_vol * 100,  # Convert to VIX-like scale
            'market_return': np.random.normal(0.0005, daily_vol)
        })
    
    market_data_df = pd.DataFrame(market_data)
    
    return predictions_df, actual_returns_df, stock_metadata, market_data_df

def test_extended_accuracy_analysis():
    """Test extended accuracy analysis functionality"""
    print("Testing extended accuracy analysis functionality...")
    
    # Create comprehensive test data
    predictions_df, actual_returns_df, stock_metadata, market_data_df = create_comprehensive_test_data()
    print(f"Created test data:")
    print(f"  - {len(predictions_df)} predictions")
    print(f"  - {len(actual_returns_df)} actual return records") 
    print(f"  - {len(stock_metadata)} stocks with metadata")
    print(f"  - {len(market_data_df)} market data records")
    
    # Initialize analyzer
    analyzer = AccuracyAnalyzer()
    
    # Test comprehensive report generation
    print("\nTesting comprehensive report generation...")
    try:
        report_path = analyzer.generate_comprehensive_report(
            predictions=predictions_df,
            actual_returns=actual_returns_df,
            stock_metadata=stock_metadata,
            market_data=market_data_df
        )
        print(f"✓ Comprehensive report saved to: {report_path}")
        
        # Display part of the report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_lines = f.readlines()[:40]
            print("\nReport Preview:")
            print("=" * 60)
            for line in report_lines:
                print(line.rstrip())
            print("=" * 60)
    except Exception as e:
        print(f"✗ Report generation failed: {str(e)}")
    
    # Test comprehensive visualization
    print("\nTesting comprehensive visualization...")
    try:
        plot_path = analyzer.plot_accuracy_analysis(
            predictions=predictions_df,
            actual_returns=actual_returns_df,
            stock_metadata=stock_metadata,
            market_data=market_data_df
        )
        if plot_path:
            print(f"✓ Accuracy analysis plots saved to: {plot_path}")
        else:
            print("⚠ Visualization skipped (matplotlib not available)")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")
    
    # Test individual analysis components
    print("\nTesting individual analysis components...")
    
    # Temporal analysis
    try:
        temporal_result = analyzer.analyze_temporal_accuracy(predictions_df, actual_returns_df, window_size='M')
        if 'error' not in temporal_result:
            monthly_metrics = temporal_result.get('monthly_metrics', pd.DataFrame())
            print(f"✓ Temporal analysis: {len(monthly_metrics)} months analyzed")
        else:
            print(f"⚠ Temporal analysis: {temporal_result['error']}")
    except Exception as e:
        print(f"✗ Temporal analysis failed: {str(e)}")
    
    # Stock-level analysis
    try:
        stock_result = analyzer.analyze_stock_level_accuracy(predictions_df, actual_returns_df, stock_metadata)
        if 'error' not in stock_result:
            stock_perf = stock_result.get('stock_performance', pd.DataFrame())
            print(f"✓ Stock-level analysis: {len(stock_perf)} stocks analyzed")
        else:
            print(f"⚠ Stock-level analysis: {stock_result['error']}")
    except Exception as e:
        print(f"✗ Stock-level analysis failed: {str(e)}")
    
    # Market condition analysis
    try:
        market_result = analyzer.analyze_market_condition_accuracy(predictions_df, actual_returns_df, market_data_df)
        if 'error' not in market_result:
            vol_analysis = market_result.get('volatility_analysis', pd.DataFrame())
            print(f"✓ Market condition analysis: {len(vol_analysis)} volatility regimes analyzed")
        else:
            print(f"⚠ Market condition analysis: {market_result['error']}")
    except Exception as e:
        print(f"✗ Market condition analysis failed: {str(e)}")
    
    # Confidence calibration analysis  
    try:
        calibration_result = analyzer.analyze_confidence_calibration(predictions_df, actual_returns_df)
        if 'error' not in calibration_result:
            cal_error = calibration_result.get('overall_calibration_error', 'N/A')
            print(f"✓ Confidence calibration analysis: Overall error = {cal_error}")
        else:
            print(f"⚠ Confidence calibration analysis: {calibration_result['error']}")
    except Exception as e:
        print(f"✗ Confidence calibration analysis failed: {str(e)}")
    
    print("\nExtended accuracy analysis test completed!")
    return predictions_df, actual_returns_df

if __name__ == "__main__":
    test_extended_accuracy_analysis()