#!/usr/bin/env python
"""
Daily data update script
Automatically updates stock data for the latest business day
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.collect_historical_data import HistoricalDataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_business_day(target_date: date) -> bool:
    """Check if date is a business day (Monday-Friday, excluding holidays)"""
    # Simple check for weekday (0=Monday, 6=Sunday)
    return target_date.weekday() < 5


def get_latest_business_day() -> date:
    """Get the most recent business day"""
    today = date.today()
    
    if is_business_day(today):
        return today
    
    # Go back to find the latest business day
    days_back = 1
    while days_back <= 7:  # Don't go back more than a week
        check_date = today - timedelta(days=days_back)
        if is_business_day(check_date):
            return check_date
        days_back += 1
    
    # Fallback to today if we can't find a business day
    return today


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Daily stock data update")
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to update (YYYY-MM-DD), defaults to latest business day"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if data might already exist"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for data collection"
    )
    
    args = parser.parse_args()
    
    try:
        collector = HistoricalDataCollector()
        
        # Determine target date
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            target_date = get_latest_business_day()
        
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        logger.info(f"Starting daily update for {target_date_str}")
        
        # Check if it's a business day
        if not is_business_day(target_date):
            logger.warning(f"{target_date_str} is not a business day")
            if not args.force:
                logger.info("Use --force to update anyway")
                return 0
        
        # Perform daily update
        summary = collector.collect_historical_data(
            start_date=target_date_str,
            end_date=target_date_str,
            batch_size=args.batch_size,
            delay_between_batches=1.0,
            save_format="parquet"
        )
        
        if summary["total_records"] > 0:
            print(f"✅ Daily update completed successfully!")
            print(f"📅 Date: {target_date_str}")
            print(f"📊 Collected {summary['total_records']:,} records")
            print(f"📈 Success rate: {len(summary['successful_codes'])/summary['total_codes']:.1%}")
            print(f"💾 Data saved to: {summary['output_file']}")
            
            # Log any failures
            if summary["failed_codes"]:
                print(f"⚠️  Failed codes ({len(summary['failed_codes'])}): {summary['failed_codes'][:10]}...")
        else:
            print(f"❌ Daily update failed - no data retrieved")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())