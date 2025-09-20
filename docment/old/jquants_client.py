"""J-Quants API Client for fetching Japanese stock data"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JQuantsClient:
    """
    J-Quants API Client
    
    Handles authentication, data fetching, and caching for J-Quants API
    """
    
    BASE_URL = "https://api.jquants.com/v1"
    
    def __init__(self, mail_address: str = None, password: str = None, 
                 refresh_token: str = None, cache_dir: Path = None):
        """
        Initialize J-Quants client
        
        Args:
            mail_address: J-Quants account email
            password: J-Quants account password
            refresh_token: Refresh token for authentication
            cache_dir: Directory for caching data
        """
        self.mail_address = mail_address or os.getenv("JQUANTS_MAIL")
        self.password = password or os.getenv("JQUANTS_PASSWORD")
        self.refresh_token = refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        self.id_token = None
        self.token_expires = None
        
        if cache_dir is None:
            from config.settings import DATA_DIR
            cache_dir = DATA_DIR / "cache" / "jquants"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Authenticate on initialization
        if self.refresh_token:
            self._refresh_id_token()
        elif self.mail_address and self.password:
            self._authenticate()
        else:
            logger.warning("No J-Quants credentials provided. Some features may be limited.")
    
    def _authenticate(self):
        """Authenticate with J-Quants API using email and password"""
        
        url = f"{self.BASE_URL}/token/auth_user"
        data = {
            "mailaddress": self.mail_address,
            "password": self.password
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            self.refresh_token = result.get("refreshToken")
            self._refresh_id_token()
            
            logger.info("Successfully authenticated with J-Quants API")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to authenticate with J-Quants: {e}")
            raise
    
    def _refresh_id_token(self):
        """Refresh ID token using refresh token"""
        
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        url = f"{self.BASE_URL}/token/auth_refresh"
        params = {"refreshtoken": self.refresh_token}
        
        try:
            response = requests.post(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            self.id_token = result.get("idToken")
            # Token expires in 24 hours
            self.token_expires = datetime.now() + timedelta(hours=23)
            
            logger.info("Successfully refreshed J-Quants ID token")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh ID token: {e}")
            raise
    
    def _ensure_authenticated(self):
        """Ensure we have a valid ID token"""
        
        if not self.id_token or (self.token_expires and datetime.now() >= self.token_expires):
            self._refresh_id_token()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make authenticated request to J-Quants API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        self._ensure_authenticated()
        
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_listed_info(self, code: str = None) -> pd.DataFrame:
        """
        Get listed company information
        
        Args:
            code: Stock code (optional, returns all if None)
            
        Returns:
            DataFrame with company information
        """
        
        cache_file = self.cache_dir / "listed_info.parquet"
        
        # Check cache first
        if cache_file.exists() and not code:
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < timedelta(days=7):
                logger.info("Loading listed info from cache")
                return pd.read_parquet(cache_file)
        
        # Fetch from API
        params = {"code": code} if code else {}
        data = self._make_request("listed/info", params)
        
        df = pd.DataFrame(data.get("info", []))
        
        if not df.empty:
            # Convert date columns
            date_columns = ["Date", "EffectiveDate", "UpdateDate"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Cache the data
            if not code:
                df.to_parquet(cache_file)
                logger.info(f"Cached listed info to {cache_file}")
        
        return df
    
    def get_daily_quotes(self, date: str = None, code: str = None,
                        from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """
        Get daily stock quotes
        
        Args:
            date: Specific date (YYYY-MM-DD)
            code: Stock code
            from_date: Start date for range
            to_date: End date for range
            
        Returns:
            DataFrame with daily quotes
        """
        
        # Determine cache file name
        cache_parts = []
        if code:
            cache_parts.append(f"code_{code}")
        if date:
            cache_parts.append(f"date_{date}")
        elif from_date and to_date:
            cache_parts.append(f"range_{from_date}_{to_date}")
        
        cache_name = "_".join(cache_parts) if cache_parts else "all"
        cache_file = self.cache_dir / f"daily_quotes_{cache_name}.parquet"
        
        # Check cache
        if cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < timedelta(hours=24):
                logger.info(f"Loading daily quotes from cache: {cache_file}")
                return pd.read_parquet(cache_file)
        
        # Build parameters
        params = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        # Fetch from API
        data = self._make_request("prices/daily_quotes", params)
        
        df = pd.DataFrame(data.get("daily_quotes", []))
        
        if not df.empty:
            # Process columns
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Convert numeric columns
            numeric_columns = ["Open", "High", "Low", "Close", "Volume", 
                              "TurnoverValue", "AdjustmentClose", "AdjustmentVolume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date and code
            df = df.sort_values(["Date", "Code"])
            
            # Cache the data
            df.to_parquet(cache_file)
            logger.info(f"Cached daily quotes to {cache_file}")
        
        return df
    
    def get_financial_data(self, code: str = None) -> pd.DataFrame:
        """
        Get financial statement data
        
        Args:
            code: Stock code
            
        Returns:
            DataFrame with financial data
        """
        
        params = {"code": code} if code else {}
        data = self._make_request("fins/statements", params)
        
        df = pd.DataFrame(data.get("statements", []))
        
        if not df.empty:
            # Convert date columns
            date_columns = ["DisclosedDate", "CurrentPeriodEndDate", "CurrentFiscalYearEndDate"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        return df
    
    def get_indices(self, date: str = None, from_date: str = None, 
                    to_date: str = None) -> pd.DataFrame:
        """
        Get index data (TOPIX, Nikkei 225, etc.)
        
        Args:
            date: Specific date
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with index data
        """
        
        params = {}
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        data = self._make_request("indices", params)
        
        df = pd.DataFrame(data.get("indices", []))
        
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Convert numeric columns
            numeric_columns = ["Open", "High", "Low", "Close"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_nikkei225_list(self) -> List[str]:
        """
        Get list of Nikkei 225 constituent stocks
        
        Returns:
            List of stock codes
        """
        
        # Get listed info
        df = self.get_listed_info()
        
        # Filter for Nikkei 225 (this is a simplified version)
        # In reality, you'd need the actual constituent list
        # For now, we'll use market cap and sector as proxy
        
        nikkei_codes = df[
            (df["MarketCode"] == "0111") &  # Prime market
            (df["Scale"] == "1")  # Large cap
        ]["Code"].tolist()[:225]
        
        logger.info(f"Retrieved {len(nikkei_codes)} Nikkei 225 stocks")
        
        return nikkei_codes
    
    def fetch_historical_data(self, codes: List[str], start_date: str, 
                            end_date: str, batch_size: int = 10) -> pd.DataFrame:
        """
        Fetch historical data for multiple stocks
        
        Args:
            codes: List of stock codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            batch_size: Number of stocks to fetch at once
            
        Returns:
            Combined DataFrame with all stock data
        """
        
        all_data = []
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}/{(len(codes)-1)//batch_size + 1}")
            
            for code in batch:
                try:
                    df = self.get_daily_quotes(
                        code=code,
                        from_date=start_date,
                        to_date=end_date
                    )
                    
                    if not df.empty:
                        all_data.append(df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch data for {code}: {e}")
                    continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Fetched {len(combined_df)} records for {len(codes)} stocks")
            return combined_df
        else:
            logger.warning("No data fetched")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data"""
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
        
        logger.info("Cleared J-Quants cache")