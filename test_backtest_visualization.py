#!/usr/bin/env python3
"""
Test backtest visualization and reporting functionality
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

from src.evaluation.backtester import Backtester

def create_test_data():
    """Create test prediction and price data for backtesting"""
    
    # Create date range for 2 months
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-02-29')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter to business days only (Monday-Friday)
    business_dates = [d for d in date_range if d.weekday() < 5]
    
    stocks = ['1301', '1332', '1333']  # Sample stock codes
    
    # Create prediction data (daily predictions)
    predictions = []
    np.random.seed(42)
    
    for date in business_dates:
        # Generate 0-2 predictions per day with different probabilities
        n_predictions = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        
        if n_predictions > 0:
            selected_stocks = np.random.choice(stocks, size=min(n_predictions, len(stocks)), replace=False)
            for stock in selected_stocks:
                # Generate probability with bias towards higher values (since we're testing positive predictions)
                probability = np.random.beta(3, 2) * 0.4 + 0.6  # Range 0.6-1.0
                predictions.append({
                    'Date': date,
                    'Code': stock,
                    'prediction': 1,
                    'prediction_probability': probability
                })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Create price data
    price_data = []
    np.random.seed(43)
    
    for stock in stocks:
        current_price = 1000 + np.random.uniform(-200, 200)  # Starting price around 1000
        
        for date in business_dates:
            # Generate realistic daily price movements
            daily_return = np.random.normal(0.001, 0.025)  # 0.1% mean, 2.5% vol
            
            open_price = current_price
            high_price = open_price * (1 + abs(daily_return) + np.random.uniform(0, 0.01))
            low_price = open_price * (1 - abs(daily_return) - np.random.uniform(0, 0.01))
            close_price = open_price * (1 + daily_return)
            volume = np.random.randint(100000, 1000000)
            
            price_data.append({
                'Date': date,
                'Code': stock,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
    
    price_data_df = pd.DataFrame(price_data)
    
    return predictions_df, price_data_df

def test_backtest_visualization():
    """Test backtest visualization and reporting functionality"""
    print("Testing backtest visualization and reporting...")
    
    # Create test data
    predictions_df, price_data_df = create_test_data()
    print(f"Created test data: {len(predictions_df)} predictions, {len(price_data_df)} price records")
    
    # Initialize backtester
    backtester = Backtester()
    
    # Run backtest
    results = backtester.run_walkforward_backtest(
        predictions=predictions_df,
        price_data=price_data_df
    )
    
    print(f"\nBacktest Results Summary:")
    print(f"Total Trades: {results['performance']['total_trades']}")
    print(f"Win Rate: {results['performance']['win_rate']:.1%}")
    print(f"Total Return: {results['performance']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.1%}")
    
    # Test visualization
    print("\nTesting backtest visualization...")
    try:
        plot_path = backtester.plot_backtest_results(results)
        if plot_path:
            print(f"✓ Visualization saved to: {plot_path}")
        else:
            print("⚠ Visualization skipped (matplotlib not available)")
    except Exception as e:
        print(f"✗ Visualization failed: {str(e)}")
    
    # Test report generation
    print("\nTesting backtest report generation...")
    try:
        report_path = backtester.export_backtest_report(results)
        print(f"✓ Report saved to: {report_path}")
        
        # Display first part of report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_lines = f.readlines()[:30]  # First 30 lines
            print("\nReport Preview:")
            print("=" * 50)
            for line in report_lines:
                print(line.rstrip())
            print("=" * 50)
        
    except Exception as e:
        print(f"✗ Report generation failed: {str(e)}")
    
    print("\nBacktest visualization and reporting test completed!")
    return results

if __name__ == "__main__":
    test_backtest_visualization()