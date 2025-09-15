#!/usr/bin/env python
"""
Data fetching script for AI Stock Analysis System

Fetches stock data from J-Quants and Yahoo Finance APIs
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import JQuantsClient, YahooFinanceClient
from config.settings import (
    DATA_DIR, 
    DATA_START_DATE, 
    DATA_END_DATE,
    JQUANTS_TOKEN,
    JQUANTS_REFRESH_TOKEN
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Main data fetching orchestrator"""
    
    def __init__(self):
        """Initialize data fetcher with API clients"""
        
        # Initialize clients
        self.jquants = JQuantsClient(refresh_token=JQUANTS_REFRESH_TOKEN)
        self.yahoo = YahooFinanceClient()
        
        # Setup data directories
        self.raw_dir = DATA_DIR / "raw"
        self.stock_dir = self.raw_dir / "stock_prices"
        self.index_dir = self.raw_dir / "indices"
        self.metadata_dir = self.raw_dir / "metadata"
        
        for dir_path in [self.stock_dir, self.index_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_nikkei225_list(self) -> List[str]:
        """
        Fetch list of Nikkei 225 constituent stocks
        
        Returns:
            List of stock codes
        """
        
        logger.info("Fetching Nikkei 225 constituent list")
        
        # For now, use a predefined list of major Japanese stocks
        # In production, this would fetch the actual constituent list
        nikkei225_codes = [
            "7203",  # Toyota
            "6758",  # Sony
            "9984",  # SoftBank
            "6861",  # Keyence
            "8306",  # Mitsubishi UFJ
            "4063",  # Shin-Etsu Chemical
            "9432",  # NTT
            "7267",  # Honda
            "6098",  # Recruit
            "7974",  # Nintendo
            "6501",  # Hitachi
            "8035",  # Tokyo Electron
            "6367",  # Daikin
            "4502",  # Takeda
            "9433",  # KDDI
            "6902",  # Denso
            "7741",  # HOYA
            "8058",  # Mitsubishi Corp
            "6273",  # SMC
            "8031",  # Mitsui
            # Add more codes as needed...
        ]
        
        # Save constituent list
        constituent_df = pd.DataFrame({
            'code': nikkei225_codes,
            'index': 'nikkei225',
            'date': datetime.now().strftime('%Y-%m-%d')
        })
        
        constituent_file = self.metadata_dir / "nikkei225_constituents.csv"
        constituent_df.to_csv(constituent_file, index=False)
        logger.info(f"Saved constituent list to {constituent_file}")
        
        return nikkei225_codes
    
    def fetch_stock_data(self, codes: List[str], 
                        start_date: str = None,
                        end_date: str = None,
                        source: str = "jquants") -> pd.DataFrame:
        """
        Fetch stock price data
        
        Args:
            codes: List of stock codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ("jquants" or "yahoo")
            
        Returns:
            Combined DataFrame with stock data
        """
        
        start_date = start_date or DATA_START_DATE
        end_date = end_date or DATA_END_DATE
        
        logger.info(f"Fetching stock data from {source}: {start_date} to {end_date}")
        
        if source == "jquants" and JQUANTS_TOKEN:
            # Fetch from J-Quants
            df = self.jquants.fetch_historical_data(codes, start_date, end_date)
        else:
            # Fetch from Yahoo Finance
            df = self.yahoo.batch_fetch_stocks(
                codes, 
                start_date=start_date,
                end_date=end_date
            )
        
        if not df.empty:
            # Save to parquet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.stock_dir / f"stock_prices_{timestamp}.parquet"
            df.to_parquet(output_file)
            logger.info(f"Saved stock data to {output_file}")
        
        return df
    
    def fetch_market_data(self, start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """
        Fetch market indices and forex data
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with market data
        """
        
        start_date = start_date or DATA_START_DATE
        end_date = end_date or DATA_END_DATE
        
        logger.info(f"Fetching market data: {start_date} to {end_date}")
        
        market_data = {}
        
        # Fetch USD/JPY
        usdjpy = self.yahoo.get_forex_data(
            "USDJPY=X",
            start_date=start_date,
            end_date=end_date
        )
        if not usdjpy.empty:
            market_data['usdjpy'] = usdjpy
        
        # Fetch Nikkei 225
        nikkei = self.yahoo.get_index_data(
            "^N225",
            start_date=start_date,
            end_date=end_date
        )
        if not nikkei.empty:
            market_data['nikkei'] = nikkei
        
        # Fetch TOPIX
        topix = self.yahoo.get_index_data(
            "^TOPX",
            start_date=start_date,
            end_date=end_date
        )
        if not topix.empty:
            market_data['topix'] = topix
        
        # Fetch VIX
        vix = self.yahoo.get_vix_data(
            start_date=start_date,
            end_date=end_date
        )
        if not vix.empty:
            market_data['vix'] = vix
        
        # Save each dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, df in market_data.items():
            output_file = self.index_dir / f"{name}_{timestamp}.parquet"
            df.to_parquet(output_file)
            logger.info(f"Saved {name} data to {output_file}")
        
        return market_data
    
    def fetch_daily_update(self, date: str = None) -> dict:
        """
        Fetch daily data update
        
        Args:
            date: Date to fetch (defaults to today)
            
        Returns:
            Dictionary with fetched data
        """
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching daily update for {date}")
        
        results = {}
        
        # Get constituent list
        codes = self.fetch_nikkei225_list()
        
        # Fetch stock data for the date
        stock_data = self.fetch_stock_data(
            codes,
            start_date=date,
            end_date=date
        )
        results['stocks'] = stock_data
        
        # Fetch market indicators
        market_indicators = self.yahoo.get_market_indicators(date)
        results['market'] = market_indicators
        
        # Save daily snapshot
        snapshot_file = self.raw_dir / f"daily_snapshot_{date.replace('-', '')}.json"
        pd.Series(market_indicators).to_json(snapshot_file)
        
        logger.info(f"Daily update completed for {date}")
        
        return results
    
    def fetch_initial_data(self) -> dict:
        """
        Fetch initial historical data for system setup
        
        Returns:
            Dictionary with all fetched data
        """
        
        logger.info("Starting initial data fetch")
        
        results = {}
        
        # Get constituent list
        codes = self.fetch_nikkei225_list()
        
        # Fetch historical stock data
        logger.info("Fetching historical stock data")
        stock_data = self.fetch_stock_data(
            codes[:20],  # Start with first 20 stocks for testing
            start_date=DATA_START_DATE,
            end_date=DATA_END_DATE
        )
        results['stocks'] = stock_data
        
        # Fetch market data
        logger.info("Fetching market data")
        market_data = self.fetch_market_data(
            start_date=DATA_START_DATE,
            end_date=DATA_END_DATE
        )
        results['market'] = market_data
        
        logger.info("Initial data fetch completed")
        
        return results


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Fetch stock market data")
    parser.add_argument(
        "--mode",
        choices=["initial", "daily", "custom"],
        default="daily",
        help="Fetch mode"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date for daily fetch (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for custom fetch"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for custom fetch"
    )
    parser.add_argument(
        "--codes",
        type=str,
        help="Comma-separated list of stock codes for custom fetch"
    )
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = DataFetcher()
    
    try:
        if args.mode == "initial":
            # Fetch all historical data
            results = fetcher.fetch_initial_data()
            print(f"Initial fetch completed. Fetched {len(results)} datasets")
            
        elif args.mode == "daily":
            # Fetch daily update
            results = fetcher.fetch_daily_update(args.date)
            print(f"Daily fetch completed for {args.date or 'today'}")
            
        elif args.mode == "custom":
            # Custom fetch
            if not args.codes:
                print("Error: --codes required for custom mode")
                sys.exit(1)
            
            codes = [c.strip() for c in args.codes.split(",")]
            results = fetcher.fetch_stock_data(
                codes,
                start_date=args.start_date,
                end_date=args.end_date
            )
            print(f"Custom fetch completed for {len(codes)} stocks")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())