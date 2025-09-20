#!/usr/bin/env python
"""
Data overview and analysis script
Provides insights into collected data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.collect_historical_data import HistoricalDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_data_file(file_path: Path) -> dict:
    """Analyze a single data file"""
    try:
        # Read data
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            return None
        
        if df.empty:
            return None
        
        # Standardize column names and remove timezone info
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df['Date'] = df['date']
        
        # Basic statistics
        analysis = {
            'file': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'total_records': len(df),
            'unique_codes': df['Code'].nunique() if 'Code' in df.columns else df['code'].nunique() if 'code' in df.columns else 0,
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max(),
                'days': (df['Date'].max() - df['Date'].min()).days + 1
            },
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict()
        }
        
        # Price statistics if available
        price_col = 'Close' if 'Close' in df.columns else 'close' if 'close' in df.columns else None
        if price_col:
            analysis['price_stats'] = {
                'min_price': df[price_col].min(),
                'max_price': df[price_col].max(),
                'median_price': df[price_col].median(),
                'price_range': df[price_col].max() - df[price_col].min()
            }
        
        # Volume statistics if available  
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume' if 'volume' in df.columns else None
        if volume_col:
            analysis['volume_stats'] = {
                'total_volume': df[volume_col].sum(),
                'avg_volume': df[volume_col].mean(),
                'max_volume': df[volume_col].max()
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze {file_path}: {e}")
        return None


def generate_data_report():
    """Generate comprehensive data report"""
    collector = HistoricalDataCollector()
    
    print("=" * 60)
    print("📊 STOCK DATA OVERVIEW REPORT")
    print("=" * 60)
    
    # Get all data files
    data_files = list(collector.raw_dir.glob("*.parquet")) + list(collector.raw_dir.glob("*.csv"))
    data_files = [f for f in data_files if not f.name.startswith('collection_summary')]
    
    if not data_files:
        print("❌ No data files found in", collector.raw_dir)
        return
    
    print(f"📁 Data directory: {collector.raw_dir}")
    print(f"📄 Found {len(data_files)} data files")
    print()
    
    # Analyze each file
    all_analyses = []
    total_records = 0
    total_size_mb = 0
    all_codes = set()
    earliest_date = None
    latest_date = None
    
    for file_path in sorted(data_files):
        analysis = analyze_data_file(file_path)
        if analysis:
            all_analyses.append(analysis)
            total_records += analysis['total_records']
            total_size_mb += analysis['file_size_mb']
            all_codes.update(range(analysis['unique_codes']))  # Approximation
            
            if earliest_date is None or analysis['date_range']['start'] < earliest_date:
                earliest_date = analysis['date_range']['start']
            if latest_date is None or analysis['date_range']['end'] > latest_date:
                latest_date = analysis['date_range']['end']
    
    # Summary statistics
    print("📈 SUMMARY STATISTICS")
    print("-" * 30)
    print(f"Total records: {total_records:,}")
    print(f"Total file size: {total_size_mb:.1f} MB")
    print(f"Estimated unique stocks: {len(all_codes)}")
    if earliest_date and latest_date:
        print(f"Date range: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        print(f"Coverage days: {(latest_date - earliest_date).days + 1}")
    print()
    
    # Individual file details
    print("📋 FILE DETAILS")
    print("-" * 50)
    for analysis in all_analyses:
        print(f"📄 {analysis['file']}")
        print(f"   Size: {analysis['file_size_mb']:.1f} MB")
        print(f"   Records: {analysis['total_records']:,}")
        print(f"   Stocks: {analysis['unique_codes']}")
        print(f"   Period: {analysis['date_range']['start'].strftime('%Y-%m-%d')} to {analysis['date_range']['end'].strftime('%Y-%m-%d')}")
        
        if 'price_stats' in analysis:
            ps = analysis['price_stats']
            print(f"   Price range: ¥{ps['min_price']:.0f} - ¥{ps['max_price']:.0f} (median: ¥{ps['median_price']:.0f})")
        
        print()
    
    # Data quality assessment
    print("🔍 DATA QUALITY ASSESSMENT")
    print("-" * 35)
    
    if all_analyses:
        # Check for missing data
        total_missing = 0
        for analysis in all_analyses:
            for col, missing_count in analysis['missing_data'].items():
                if missing_count > 0:
                    total_missing += missing_count
        
        if total_missing == 0:
            print("✅ No missing values detected")
        else:
            print(f"⚠️  Total missing values: {total_missing:,}")
        
        # Check date coverage
        if earliest_date and latest_date:
            expected_days = (latest_date - earliest_date).days + 1
            # Rough estimate: 5 business days per week
            expected_business_days = expected_days * 5 / 7
            avg_records_per_day = total_records / expected_business_days if expected_business_days > 0 else 0
            
            print(f"📅 Average records per business day: {avg_records_per_day:.0f}")
    
    print("\n" + "=" * 60)
    print("Report generated at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def show_sample_data(file_pattern: str = "*", n_samples: int = 5):
    """Show sample data from files"""
    collector = HistoricalDataCollector()
    
    # Find matching files
    if file_pattern == "*":
        data_files = list(collector.raw_dir.glob("*.parquet")) + list(collector.raw_dir.glob("*.csv"))
    else:
        data_files = list(collector.raw_dir.glob(file_pattern))
    
    data_files = [f for f in data_files if not f.name.startswith('collection_summary')]
    
    if not data_files:
        print("❌ No matching data files found")
        return
    
    # Show samples from the most recent file
    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📄 Sample data from: {latest_file.name}")
    print("-" * 50)
    
    try:
        if latest_file.suffix == '.parquet':
            df = pd.read_parquet(latest_file)
        else:
            df = pd.read_csv(latest_file)
        
        # Show basic info
        print(f"Shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        print()
        
        # Show sample data
        display_columns = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if not available_columns:
            # Fallback to lowercase
            display_columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            print("Sample records:")
            print(df[available_columns].head(n_samples).to_string(index=False))
        else:
            print("Sample records (all columns):")
            print(df.head(n_samples))
    
    except Exception as e:
        print(f"❌ Failed to read sample data: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Stock data overview and analysis")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive data report"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Show sample data"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*",
        help="File pattern for sample data"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of sample records to show"
    )
    
    args = parser.parse_args()
    
    if args.report:
        generate_data_report()
    elif args.sample:
        show_sample_data(args.file_pattern, args.n_samples)
    else:
        # Default: show both
        generate_data_report()
        print("\n" + "=" * 60)
        show_sample_data(n_samples=3)


if __name__ == "__main__":
    main()