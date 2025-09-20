#!/usr/bin/env python3
"""
Test script for enhanced feature engineering (400+ features)
Tests the comprehensive feature generation pipeline
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features.feature_engineer import FeatureEngineer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_features_test.log')
        ]
    )

def create_sample_data(n_stocks: int = 10, n_days: int = 100) -> pd.DataFrame:
    """
    Create sample stock data for testing
    
    Args:
        n_stocks: Number of stocks to generate
        n_days: Number of days to generate
        
    Returns:
        DataFrame with sample stock data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating sample data: {n_stocks} stocks, {n_days} days")
    
    # Sample stock codes (major Nikkei 225)
    stock_codes = [
        '7203', '6758', '9984', '6861', '8001', 
        '8058', '4519', '6098', '8306', '7974'
    ][:n_stocks]
    
    # Generate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    data = []
    
    for code in stock_codes:
        # Base price varies by stock
        base_prices = {
            '7203': 2500,   # Toyota
            '6758': 12000,  # Sony
            '9984': 7000,   # SoftBank
            '6861': 50000,  # Keyence
            '8001': 4500,   # Itochu
            '8058': 3800,   # Mitsubishi
            '4519': 3500,   # Takeda
            '6098': 8000,   # Recruit
            '8306': 1200,   # MUFG
            '7974': 60000,  # Nintendo
        }
        
        base_price = base_prices.get(code, 5000)
        
        # Generate realistic price movements
        np.random.seed(int(code))  # Consistent data for each stock
        
        returns = np.random.normal(0.0005, 0.02, len(date_range))  # Daily returns
        prices = [base_price]
        
        for i in range(1, len(date_range)):
            new_price = prices[i-1] * (1 + returns[i])
            prices.append(max(new_price, base_price * 0.3))  # Prevent extreme drops
        
        for i, date in enumerate(date_range):
            close_price = prices[i]
            high_price = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            
            # Ensure OHLC relationships
            high_price = max(high_price, close_price, open_price, low_price)
            low_price = min(low_price, close_price, open_price, high_price)
            
            volume = max(100000, int(np.random.lognormal(13, 0.8)))  # Log-normal volume
            
            data.append({
                'Date': date.date(),
                'Code': code,
                'Open': round(open_price, 0),
                'High': round(high_price, 0),
                'Low': round(low_price, 0),
                'Close': round(close_price, 0),
                'Volume': volume
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Sample data created: {len(df)} records")
    return df

def test_feature_generation():
    """Test the enhanced feature generation pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Enhanced Feature Generation Pipeline ===")
    
    try:
        # Create sample data
        sample_data = create_sample_data(n_stocks=5, n_days=60)
        logger.info(f"Initial data shape: {sample_data.shape}")
        logger.info(f"Initial columns: {list(sample_data.columns)}")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Test basic data loading (simulated)
        logger.info("\n--- Testing data standardization ---")
        standardized_data = feature_engineer._standardize_columns(sample_data)
        logger.info(f"Standardized data shape: {standardized_data.shape}")
        
        # Test comprehensive feature generation
        logger.info("\n--- Testing comprehensive feature generation ---")
        
        enhanced_features = feature_engineer.generate_features(
            standardized_data,
            include_technical=True,
            include_market=True,
            include_time_series=True,
            include_fundamental=True,
            include_labels=True
        )
        
        logger.info(f"Enhanced features shape: {enhanced_features.shape}")
        
        # Feature analysis
        original_cols = len(sample_data.columns)
        total_features = len(enhanced_features.columns)
        added_features = total_features - original_cols
        
        logger.info(f"\n=== Feature Generation Summary ===")
        logger.info(f"Original columns: {original_cols}")
        logger.info(f"Total columns after feature generation: {total_features}")
        logger.info(f"Features added: {added_features}")
        
        # Categorize features
        feature_categories = {
            'Original': [],
            'Technical': [],
            'Market': [],
            'Time_Series': [],
            'Fundamental': [],
            'Labels': [],
            'Other': []
        }
        
        original_columns = set(sample_data.columns)
        
        for col in enhanced_features.columns:
            if col in original_columns:
                feature_categories['Original'].append(col)
            elif any(tech_word in col.lower() for tech_word in [
                'sma', 'ema', 'rsi', 'macd', 'bb_', 'stoch', 'williams', 'roc', 'adx', 'atr', 'ichimoku'
            ]):
                feature_categories['Technical'].append(col)
            elif any(market_word in col.lower() for market_word in [
                'volatility', 'trend', 'breadth', 'regime', 'sector', 'beta', 'correlation', 'rank'
            ]):
                feature_categories['Market'].append(col)
            elif any(ts_word in col.lower() for ts_word in [
                'lag', 'return', 'seasonal', 'entropy', 'fractal', 'hurst', 'momentum', 'drawdown'
            ]):
                feature_categories['Time_Series'].append(col)
            elif any(fund_word in col.lower() for fund_word in [
                'market_cap', 'per', 'pbr', 'roe', 'growth', 'usdjpy', 'oil', 'gold', 'vix', 'earnings'
            ]):
                feature_categories['Fundamental'].append(col)
            elif any(label_word in col.lower() for label_word in [
                'target', 'label', 'next_day'
            ]):
                feature_categories['Labels'].append(col)
            else:
                feature_categories['Other'].append(col)
        
        # Print feature category summary
        logger.info(f"\n=== Feature Categories ===")
        for category, features in feature_categories.items():
            logger.info(f"{category}: {len(features)} features")
            if len(features) <= 10:
                logger.info(f"  Sample features: {features}")
            else:
                logger.info(f"  Sample features: {features[:10]}...")
        
        # Data quality checks
        logger.info(f"\n=== Data Quality Analysis ===")
        missing_pct = (enhanced_features.isnull().sum() / len(enhanced_features) * 100)
        high_missing = missing_pct[missing_pct > 50]
        
        logger.info(f"Columns with >50% missing data: {len(high_missing)}")
        if len(high_missing) > 0:
            logger.info(f"High missing columns: {high_missing.head(10).to_dict()}")
        
        # Infinite values check
        inf_cols = []
        for col in enhanced_features.select_dtypes(include=[np.number]).columns:
            if np.isinf(enhanced_features[col]).any():
                inf_cols.append(col)
        
        logger.info(f"Columns with infinite values: {len(inf_cols)}")
        if len(inf_cols) > 0:
            logger.info(f"Infinite value columns: {inf_cols[:10]}")
        
        # Feature summary statistics
        numeric_features = enhanced_features.select_dtypes(include=[np.number])
        logger.info(f"\n=== Summary Statistics ===")
        logger.info(f"Numeric features: {len(numeric_features.columns)}")
        logger.info(f"Average missing percentage: {missing_pct.mean():.2f}%")
        logger.info(f"Features with zero variance: {(numeric_features.var() == 0).sum()}")
        
        # Test feature creation summary
        feature_summary = feature_engineer.create_feature_summary(enhanced_features)
        logger.info(f"\n=== Feature Summary ===")
        logger.info(f"Total records: {feature_summary.get('total_records', 'N/A')}")
        logger.info(f"Unique stocks: {feature_summary.get('unique_stocks', 'N/A')}")
        logger.info(f"Date range: {feature_summary.get('date_range', {}).get('start', 'N/A')} to {feature_summary.get('date_range', {}).get('end', 'N/A')}")
        
        logger.info("\n=== Feature Generation Test Completed Successfully ===")
        
        return enhanced_features, feature_categories
        
    except Exception as e:
        logger.error(f"Feature generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def test_save_load_features():
    """Test feature saving and loading"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Testing Feature Save/Load ===")
    
    try:
        # Create sample features
        enhanced_features, _ = test_feature_generation()
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Test saving features
        logger.info("Testing feature saving...")
        output_path = feature_engineer.save_features(
            enhanced_features, 
            filename="test_enhanced_features.parquet"
        )
        logger.info(f"Features saved to: {output_path}")
        
        # Test loading features
        logger.info("Testing feature loading...")
        loaded_features = pd.read_parquet(output_path)
        logger.info(f"Loaded features shape: {loaded_features.shape}")
        
        # Verify data integrity
        assert len(enhanced_features) == len(loaded_features), "Row count mismatch"
        assert len(enhanced_features.columns) == len(loaded_features.columns), "Column count mismatch"
        
        logger.info("Feature save/load test completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature save/load test failed: {e}")
        raise

def main():
    """Main test execution"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Enhanced Feature Engineering Tests")
        
        # Test 1: Feature generation
        enhanced_features, feature_categories = test_feature_generation()
        
        # Test 2: Save/Load functionality
        test_save_load_features()
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("Enhanced feature engineering pipeline is working correctly.")
        logger.info("="*60)
        
        # Print final summary
        logger.info(f"\nFinal Summary:")
        logger.info(f"- Successfully generated {len(enhanced_features.columns)} total features")
        logger.info(f"- Technical features: {len(feature_categories.get('Technical', []))}")
        logger.info(f"- Market features: {len(feature_categories.get('Market', []))}")
        logger.info(f"- Time series features: {len(feature_categories.get('Time_Series', []))}")
        logger.info(f"- Fundamental features: {len(feature_categories.get('Fundamental', []))}")
        logger.info(f"- Ready for machine learning model training!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()