#!/usr/bin/env python
"""
Simplified feature generation script for stock analysis
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleFeatureGenerator:
    """Simplified feature generator for stock data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_stock_data(self, file_pattern: str = "*nikkei225_historical*") -> pd.DataFrame:
        """Load stock data from parquet files"""
        data_files = list(self.raw_dir.glob(f"{file_pattern}.parquet"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load and combine all files
        dfs = []
        for file_path in sorted(data_files):
            logger.info(f"Loading data from: {file_path.name}")
            df_file = pd.read_parquet(file_path)
            dfs.append(df_file)
        
        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Sort by date and code
        df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records for {df['Code'].nunique()} stocks")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""
        
        # Column mapping
        column_mapping = {
            'date': 'Date',
            'code': 'Code', 
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Convert date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        result = df.copy()
        
        # Group by stock code for calculations
        for code, group in result.groupby('Code'):
            idx = group.index
            prices = group['Close']
            
            # Simple Moving Averages
            result.loc[idx, 'MA_5'] = prices.rolling(window=5, min_periods=3).mean()
            result.loc[idx, 'MA_10'] = prices.rolling(window=10, min_periods=5).mean()
            result.loc[idx, 'MA_20'] = prices.rolling(window=20, min_periods=10).mean()
            
            # Price position relative to moving averages
            result.loc[idx, 'Price_vs_MA5'] = (prices / result.loc[idx, 'MA_5'] - 1) * 100
            result.loc[idx, 'Price_vs_MA20'] = (prices / result.loc[idx, 'MA_20'] - 1) * 100
            
            # RSI (simplified)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
            rs = gain / loss
            result.loc[idx, 'RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility
            returns = prices.pct_change()
            result.loc[idx, 'Volatility_20'] = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252) * 100
            
            # Volume indicators
            if 'Volume' in group.columns:
                volume = group['Volume']
                result.loc[idx, 'Volume_MA_10'] = volume.rolling(window=10, min_periods=5).mean()
                result.loc[idx, 'Volume_Ratio'] = volume / result.loc[idx, 'Volume_MA_10']
        
        logger.info("Technical indicators calculated")
        return result
    
    def calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-wide features"""
        result = df.copy()
        
        # Daily returns for each stock
        result['Returns'] = result.groupby('Code')['Close'].pct_change()
        
        # Market breadth (percentage of stocks above MA20)
        result['Above_MA20'] = (result['Close'] > result['MA_20']).astype(int)
        daily_breadth = result.groupby('Date')['Above_MA20'].agg(['sum', 'count']).reset_index()
        daily_breadth['Market_Breadth'] = daily_breadth['sum'] / daily_breadth['count'] * 100
        
        # Merge back
        result = result.merge(daily_breadth[['Date', 'Market_Breadth']], on='Date', how='left')
        
        # Relative performance vs market average
        daily_market_return = result.groupby('Date')['Returns'].mean().reset_index()
        daily_market_return.columns = ['Date', 'Market_Return']
        result = result.merge(daily_market_return, on='Date', how='left')
        result['Relative_Return'] = result['Returns'] - result['Market_Return']
        
        logger.info("Market features calculated")
        return result
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction labels"""
        result = df.copy()
        
        # Next day return
        result['Next_Day_Return'] = result.groupby('Code')['Returns'].shift(-1)
        
        # Classification labels
        result['Return_Direction'] = np.where(
            result['Next_Day_Return'] > 0.01, 'UP',
            np.where(result['Next_Day_Return'] < -0.01, 'DOWN', 'NEUTRAL')
        )
        
        # Binary classification (UP/DOWN only)
        result['Binary_Direction'] = np.where(result['Next_Day_Return'] > 0, 1, 0)
        
        logger.info("Labels created")
        return result
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features"""
        
        # Remove rows with missing target values
        clean_df = df[df['Next_Day_Return'].notna()].copy()
        
        # Forward fill missing values within each stock
        feature_cols = [col for col in clean_df.columns 
                       if col not in ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        clean_df[feature_cols] = clean_df.groupby('Code')[feature_cols].fillna(method='ffill')
        
        # Remove infinite values
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        
        # Final cleanup
        clean_df = clean_df.dropna(subset=['Next_Day_Return'])
        
        logger.info(f"Data cleaned: {len(clean_df)} records remaining")
        return clean_df
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features"""
        logger.info("Starting feature generation...")
        
        # Technical indicators
        result = self.calculate_technical_indicators(df)
        
        # Market features  
        result = self.calculate_market_features(result)
        
        # Labels
        result = self.create_labels(result)
        
        # Clean
        result = self.clean_features(result)
        
        logger.info("Feature generation completed")
        return result
    
    def save_features(self, df: pd.DataFrame, filename: str = None) -> Path:
        """Save features to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"features_simple_{timestamp}.parquet"
        
        output_path = self.processed_dir / filename
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Features saved to: {output_path}")
        return output_path
    
    def create_summary(self, df: pd.DataFrame) -> dict:
        """Create feature summary"""
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        summary = {
            'total_records': len(df),
            'unique_stocks': df['Code'].nunique(),
            'date_range': {
                'start': str(df['Date'].min().date()),
                'end': str(df['Date'].max().date())
            },
            'feature_count': len(feature_cols),
            'features': feature_cols,
            'missing_values': df[feature_cols].isnull().sum().sum()
        }
        
        return summary


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Simple feature generation for stock data")
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*nikkei225_historical*",
        help="Pattern to match input files"
    )
    parser.add_argument(
        "--output-filename", 
        type=str,
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = SimpleFeatureGenerator()
        
        # Load data
        print("📊 Loading stock data...")
        df = generator.load_stock_data(args.file_pattern)
        
        # Generate features
        print("🔧 Generating features...")
        features_df = generator.generate_features(df)
        
        # Save features
        print("💾 Saving features...")
        output_path = generator.save_features(features_df, args.output_filename)
        
        # Create summary
        summary = generator.create_summary(features_df)
        
        # Display results
        print("\n" + "="*50)
        print("📈 FEATURE GENERATION RESULTS")
        print("="*50)
        print(f"📄 Total records: {summary['total_records']:,}")
        print(f"📈 Unique stocks: {summary['unique_stocks']}")
        print(f"🔧 Features generated: {summary['feature_count']}")
        print(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"💾 Output file: {output_path.name}")
        
        if summary['missing_values'] > 0:
            print(f"⚠️  Missing values: {summary['missing_values']}")
        else:
            print("✅ No missing values")
        
        print(f"\n📋 Generated features:")
        for feature in summary['features']:
            print(f"  - {feature}")
        
        print(f"\n🎯 Sample data preview:")
        sample_cols = ['Date', 'Code', 'Close', 'MA_20', 'RSI', 'Next_Day_Return', 'Binary_Direction']
        available_cols = [col for col in sample_cols if col in features_df.columns]
        print(features_df[available_cols].head(10).to_string(index=False))
        
        print("\n✅ Feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())