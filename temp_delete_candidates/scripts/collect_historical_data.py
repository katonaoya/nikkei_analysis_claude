#!/usr/bin/env python
"""
Historical data collection script for AI Stock Analysis System
Collects 1-2 years of historical data for Nikkei 225 stocks
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import argparse
import logging
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import StockDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Historical data collection and management"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data collector"""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / data_dir
        
        # Create directories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data fetcher
        self.fetcher = StockDataFetcher()
        
        logger.info(f"Data collector initialized. Data directory: {self.data_dir}")
    
    def load_nikkei225_codes(self) -> List[str]:
        """Load Nikkei 225 constituent codes from CSV"""
        csv_path = self.data_dir / "nikkei225_codes.csv"
        
        if not csv_path.exists():
            logger.error(f"Nikkei 225 codes file not found: {csv_path}")
            # Fallback to predefined list
            return [
                "7203", "6758", "9984", "6861", "7267", "8306", "9432", "7974",
                "4502", "6501", "6902", "6954", "7751", "8035", "9433", "4063",
                "6098", "4519", "8309", "4568", "9434", "8001", "2914", "4661",
                "8766", "4543", "3382", "8801", "4452", "6981", "7182", "8411"
            ]
        
        try:
            df = pd.read_csv(csv_path)
            codes = df['code'].astype(str).str.zfill(4).tolist()
            logger.info(f"Loaded {len(codes)} Nikkei 225 codes from CSV")
            return codes
        except Exception as e:
            logger.error(f"Failed to load Nikkei 225 codes: {e}")
            return []
    
    def collect_historical_data(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
        batch_size: int = 10,
        delay_between_batches: float = 2.0,
        save_format: str = "parquet"
    ) -> Dict[str, any]:
        """
        Collect historical data for multiple stocks
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            codes: List of stock codes (None for Nikkei 225)
            batch_size: Number of stocks to fetch at once
            delay_between_batches: Delay between batches in seconds
            save_format: File format ("parquet" or "csv")
            
        Returns:
            Collection summary
        """
        if codes is None:
            codes = self.load_nikkei225_codes()
        
        if not codes:
            raise ValueError("No stock codes provided")
        
        logger.info(f"Starting historical data collection")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Codes: {len(codes)} stocks")
        logger.info(f"Batch size: {batch_size}")
        
        # Collection summary
        summary = {
            "start_time": datetime.now(),
            "total_codes": len(codes),
            "successful_codes": [],
            "failed_codes": [],
            "total_records": 0,
            "errors": []
        }
        
        # Process in batches
        all_data = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(codes) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {batch_codes}")
            
            batch_data = []
            
            for code in batch_codes:
                try:
                    logger.info(f"  Fetching data for {code}...")
                    
                    stock_data = self.fetcher.get_stock_prices(
                        code=code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not stock_data.empty:
                        batch_data.append(stock_data)
                        summary["successful_codes"].append(code)
                        logger.info(f"    → {len(stock_data)} records retrieved")
                    else:
                        summary["failed_codes"].append(code)
                        logger.warning(f"    → No data retrieved")
                    
                    # Small delay between individual requests
                    time.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"Failed to fetch data for {code}: {str(e)}"
                    logger.error(f"    → {error_msg}")
                    summary["failed_codes"].append(code)
                    summary["errors"].append(error_msg)
                    continue
            
            if batch_data:
                all_data.extend(batch_data)
            
            # Delay between batches
            if i + batch_size < len(codes):
                logger.info(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        # Combine all data
        if all_data:
            logger.info("Combining all collected data...")
            combined_df = pd.concat(all_data, ignore_index=True)
            summary["total_records"] = len(combined_df)
            
            # Save combined data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"nikkei225_historical_{start_date}_{end_date}_{timestamp}"
            
            if save_format == "parquet":
                filepath = self.raw_dir / f"{filename_base}.parquet"
                combined_df.to_parquet(filepath, compression='snappy', index=False)
            else:
                filepath = self.raw_dir / f"{filename_base}.csv"
                combined_df.to_csv(filepath, index=False)
            
            summary["output_file"] = filepath
            logger.info(f"Data saved to: {filepath}")
            
        else:
            logger.error("No data collected")
            summary["output_file"] = None
        
        # Set end time and duration
        summary["end_time"] = datetime.now() 
        summary["duration"] = summary["end_time"] - summary["start_time"]
        
        # Save summary after setting all fields
        if summary["total_records"] > 0:
            self._save_collection_summary(summary, start_date, end_date)
        
        # Log final summary
        logger.info("=== COLLECTION SUMMARY ===")
        logger.info(f"Duration: {summary['duration']}")
        logger.info(f"Total codes: {summary['total_codes']}")
        logger.info(f"Successful: {len(summary['successful_codes'])}")
        logger.info(f"Failed: {len(summary['failed_codes'])}")
        logger.info(f"Total records: {summary['total_records']}")
        logger.info(f"Success rate: {len(summary['successful_codes'])/summary['total_codes']:.1%}")
        
        if summary["failed_codes"]:
            logger.warning(f"Failed codes: {summary['failed_codes']}")
        
        return summary
    
    def _save_collection_summary(self, summary: Dict, start_date: str, end_date: str):
        """Save collection summary to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.raw_dir / f"collection_summary_{start_date}_{end_date}_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        summary_copy = summary.copy()
        summary_copy["start_time"] = summary["start_time"].isoformat()
        summary_copy["end_time"] = summary["end_time"].isoformat()
        summary_copy["duration"] = str(summary["duration"])
        summary_copy["output_file"] = str(summary["output_file"]) if summary["output_file"] else None
        
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def update_daily_data(self, target_date: Optional[str] = None) -> Dict[str, any]:
        """
        Update data for a specific date
        
        Args:
            target_date: Target date (YYYY-MM-DD), defaults to today
            
        Returns:
            Update summary
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Updating daily data for {target_date}")
        
        return self.collect_historical_data(
            start_date=target_date,
            end_date=target_date,
            batch_size=20,  # Larger batch for single day
            delay_between_batches=1.0
        )
    
    def get_data_coverage(self) -> pd.DataFrame:
        """Get overview of available data coverage"""
        raw_files = list(self.raw_dir.glob("*.parquet")) + list(self.raw_dir.glob("*.csv"))
        
        coverage_info = []
        for file_path in raw_files:
            try:
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                if not df.empty and 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    info = {
                        'file': file_path.name,
                        'records': len(df),
                        'unique_codes': df['Code'].nunique() if 'Code' in df.columns else 0,
                        'start_date': df['Date'].min(),
                        'end_date': df['Date'].max(),
                        'file_size_mb': file_path.stat().st_size / 1024 / 1024
                    }
                    coverage_info.append(info)
                    
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        if coverage_info:
            return pd.DataFrame(coverage_info)
        else:
            return pd.DataFrame()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Collect historical stock data")
    parser.add_argument(
        "--start-date", 
        type=str,
        default=(datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD), defaults to 2 years ago"
    )
    parser.add_argument(
        "--end-date",
        type=str, 
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--codes",
        type=str,
        help="Comma-separated list of stock codes (optional)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for data collection"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between batches in seconds"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format"
    )
    parser.add_argument(
        "--daily-update",
        action="store_true",
        help="Perform daily update for today's date"
    )
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Show data coverage overview"
    )
    
    args = parser.parse_args()
    
    try:
        collector = HistoricalDataCollector()
        
        if args.coverage:
            # Show data coverage
            coverage_df = collector.get_data_coverage()
            if not coverage_df.empty:
                print("\n=== DATA COVERAGE OVERVIEW ===")
                print(coverage_df.to_string(index=False))
            else:
                print("No data files found")
            return 0
        
        if args.daily_update:
            # Perform daily update
            summary = collector.update_daily_data()
        else:
            # Collect historical data
            codes = None
            if args.codes:
                codes = [c.strip() for c in args.codes.split(",")]
            
            summary = collector.collect_historical_data(
                start_date=args.start_date,
                end_date=args.end_date,
                codes=codes,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                save_format=args.format
            )
        
        if summary["total_records"] > 0:
            print(f"\n✅ Data collection completed successfully!")
            print(f"📊 Collected {summary['total_records']:,} records")
            print(f"📈 Success rate: {len(summary['successful_codes'])/summary['total_codes']:.1%}")
            if summary["output_file"]:
                print(f"💾 Data saved to: {summary['output_file']}")
        else:
            print("❌ Data collection failed - no data retrieved")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())