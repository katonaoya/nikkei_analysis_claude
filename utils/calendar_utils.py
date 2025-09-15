"""
Business calendar utilities for Japanese market
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Union, List, Optional

from .logger import get_logger
from .error_handler import DataError


class JapaneseMarketCalendar:
    """Japanese stock market calendar utilities"""
    
    def __init__(self):
        self.logger = get_logger("calendar")
        
        # Japanese holidays (basic set - can be extended)
        self.jp_holidays = {
            # 2023 holidays (example - should be updated for current year)
            date(2023, 1, 1),   # New Year's Day
            date(2023, 1, 2),   # Bank Holiday
            date(2023, 1, 9),   # Coming of Age Day
            date(2023, 2, 11),  # National Foundation Day
            date(2023, 2, 23),  # Emperor's Birthday
            date(2023, 3, 21),  # Vernal Equinox Day
            date(2023, 4, 29),  # Showa Day
            date(2023, 5, 3),   # Constitution Memorial Day
            date(2023, 5, 4),   # Greenery Day
            date(2023, 5, 5),   # Children's Day
            date(2023, 7, 17),  # Marine Day
            date(2023, 8, 11),  # Mountain Day
            date(2023, 9, 18),  # Respect for the Aged Day
            date(2023, 9, 23),  # Autumnal Equinox Day
            date(2023, 10, 9),  # Sports Day
            date(2023, 11, 3),  # Culture Day
            date(2023, 11, 23), # Labor Thanksgiving Day
            date(2023, 12, 29), # Year-end Holiday
            date(2023, 12, 30), # Year-end Holiday
            date(2023, 12, 31), # New Year's Eve
            # 2024 holidays
            date(2024, 1, 1),   # New Year's Day
            date(2024, 1, 8),   # Coming of Age Day
            date(2024, 2, 11),  # National Foundation Day
            date(2024, 2, 12),  # National Foundation Day (observed)
            date(2024, 2, 23),  # Emperor's Birthday
            date(2024, 3, 20),  # Vernal Equinox Day
            date(2024, 4, 29),  # Showa Day
            date(2024, 5, 3),   # Constitution Memorial Day
            date(2024, 5, 4),   # Greenery Day
            date(2024, 5, 5),   # Children's Day
            date(2024, 5, 6),   # Children's Day (observed)
            date(2024, 7, 15),  # Marine Day
            date(2024, 8, 11),  # Mountain Day
            date(2024, 8, 12),  # Mountain Day (observed)
            date(2024, 9, 16),  # Respect for the Aged Day
            date(2024, 9, 22),  # Autumnal Equinox Day
            date(2024, 9, 23),  # Autumnal Equinox Day (observed)
            date(2024, 10, 14), # Sports Day
            date(2024, 11, 3),  # Culture Day
            date(2024, 11, 4),  # Culture Day (observed)
            date(2024, 11, 23), # Labor Thanksgiving Day
            date(2024, 12, 30), # Year-end Holiday
            date(2024, 12, 31), # New Year's Eve
        }
    
    def is_business_day(self, check_date: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """
        Check if given date is a Japanese business day
        
        Args:
            check_date: Date to check
            
        Returns:
            True if business day, False otherwise
        """
        if isinstance(check_date, str):
            check_date = pd.to_datetime(check_date)
        elif isinstance(check_date, (date, datetime)):
            check_date = pd.Timestamp(check_date)
        
        # Check if weekend
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if Japanese holiday
        if check_date.date() in self.jp_holidays:
            return False
        
        return True
    
    def get_business_days(
        self, 
        start_date: Union[str, date, datetime], 
        end_date: Union[str, date, datetime],
        include_end: bool = True
    ) -> pd.DatetimeIndex:
        """
        Get business days between start and end dates
        
        Args:
            start_date: Start date
            end_date: End date
            include_end: Whether to include end date
            
        Returns:
            Business days as DatetimeIndex
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Use pandas business day logic + Japanese holidays
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        business_days = []
        
        for day in all_days:
            if self.is_business_day(day):
                business_days.append(day)
        
        if not include_end and business_days:
            end_ts = pd.to_datetime(end_date)
            if business_days[-1].date() == end_ts.date():
                business_days = business_days[:-1]
        
        return pd.DatetimeIndex(business_days)
    
    def next_business_day(
        self, 
        from_date: Union[str, date, datetime, pd.Timestamp],
        n_days: int = 1
    ) -> pd.Timestamp:
        """
        Get the next n-th business day from given date
        
        Args:
            from_date: Starting date
            n_days: Number of business days to advance
            
        Returns:
            Next business day
        """
        if isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
        elif isinstance(from_date, (date, datetime)):
            from_date = pd.Timestamp(from_date)
        
        current_date = from_date
        business_days_found = 0
        
        # Look ahead up to 20 days to find n business days
        for _ in range(20):
            current_date += pd.Timedelta(days=1)
            
            if self.is_business_day(current_date):
                business_days_found += 1
                
                if business_days_found >= n_days:
                    return current_date
        
        raise DataError(f"Could not find {n_days} business days after {from_date}")
    
    def prev_business_day(
        self, 
        from_date: Union[str, date, datetime, pd.Timestamp],
        n_days: int = 1
    ) -> pd.Timestamp:
        """
        Get the previous n-th business day from given date
        
        Args:
            from_date: Starting date
            n_days: Number of business days to go back
            
        Returns:
            Previous business day
        """
        if isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
        elif isinstance(from_date, (date, datetime)):
            from_date = pd.Timestamp(from_date)
        
        current_date = from_date
        business_days_found = 0
        
        # Look back up to 20 days to find n business days
        for _ in range(20):
            current_date -= pd.Timedelta(days=1)
            
            if self.is_business_day(current_date):
                business_days_found += 1
                
                if business_days_found >= n_days:
                    return current_date
        
        raise DataError(f"Could not find {n_days} business days before {from_date}")
    
    def business_days_between(
        self,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime]
    ) -> int:
        """
        Count business days between two dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of business days
        """
        business_days = self.get_business_days(start_date, end_date, include_end=False)
        return len(business_days)
    
    def align_to_business_day(
        self,
        target_date: Union[str, date, datetime],
        direction: str = "forward"
    ) -> pd.Timestamp:
        """
        Align date to nearest business day
        
        Args:
            target_date: Date to align
            direction: "forward" to find next business day, "backward" for previous
            
        Returns:
            Aligned business day
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, (date, datetime)):
            target_date = pd.Timestamp(target_date)
        
        if self.is_business_day(target_date):
            return target_date
        
        if direction == "forward":
            return self.next_business_day(target_date, 1)
        else:
            return self.prev_business_day(target_date, 1)
    
    def get_trading_calendar(
        self,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime]
    ) -> pd.DataFrame:
        """
        Get full trading calendar with metadata
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with trading calendar information
        """
        business_days = self.get_business_days(start_date, end_date)
        
        calendar_df = pd.DataFrame(index=business_days)
        calendar_df['is_trading_day'] = True
        calendar_df['day_of_week'] = calendar_df.index.dayofweek
        calendar_df['month'] = calendar_df.index.month
        calendar_df['quarter'] = calendar_df.index.quarter
        calendar_df['year'] = calendar_df.index.year
        
        # Mark month/quarter/year ends
        calendar_df['is_month_end'] = calendar_df.index == calendar_df.groupby(calendar_df['month']).apply(lambda x: x.index.max())
        calendar_df['is_quarter_end'] = calendar_df.index == calendar_df.groupby(calendar_df['quarter']).apply(lambda x: x.index.max())
        calendar_df['is_year_end'] = calendar_df.index == calendar_df.groupby(calendar_df['year']).apply(lambda x: x.index.max())
        
        return calendar_df


# Global calendar instance
_calendar_instance = None


def get_market_calendar() -> JapaneseMarketCalendar:
    """Get global market calendar instance"""
    global _calendar_instance
    
    if _calendar_instance is None:
        _calendar_instance = JapaneseMarketCalendar()
    
    return _calendar_instance


def is_business_day(check_date: Union[str, date, datetime]) -> bool:
    """Check if date is a business day"""
    calendar = get_market_calendar()
    return calendar.is_business_day(check_date)


def next_business_day(from_date: Union[str, date, datetime], n_days: int = 1) -> pd.Timestamp:
    """Get next business day"""
    calendar = get_market_calendar()
    return calendar.next_business_day(from_date, n_days)


def prev_business_day(from_date: Union[str, date, datetime], n_days: int = 1) -> pd.Timestamp:
    """Get previous business day"""
    calendar = get_market_calendar()
    return calendar.prev_business_day(from_date, n_days)


def get_business_days(
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    include_end: bool = True
) -> pd.DatetimeIndex:
    """Get business days between dates"""
    calendar = get_market_calendar()
    return calendar.get_business_days(start_date, end_date, include_end)