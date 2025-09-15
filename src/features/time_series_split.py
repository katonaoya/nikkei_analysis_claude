"""
Time series cross-validation splitters for stock data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Iterator, Union
from datetime import datetime, date, timedelta

from utils.config import get_config
from utils.logger import get_logger
from utils.error_handler import DataError, ValidationError, with_error_context, validate_positive
from utils.calendar_utils import get_business_days, is_business_day


class TimeSeriesSplitter:
    """Time series aware data splitter for financial data"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize time series splitter
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("time_series_splitter")
        
        # Configuration
        self.n_splits = self.config.get('models.n_splits', 6)
        self.gap_days = self.config.get('models.gap_days', 5)
        self.train_end_date = self.config.get('data.train_end_date', '2022-12-30')
        self.val_end_date = self.config.get('data.val_end_date', '2023-12-30')
        self.test_end_date = self.config.get('data.test_end_date', '2024-12-31')
    
    def time_series_split_simple(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Simple chronological train/validation/test split
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            train_ratio: Training data ratio
            val_ratio: Validation data ratio  
            test_ratio: Test data ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        with with_error_context("simple time series split"):
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            if date_col not in df.columns:
                raise DataError(f"Date column '{date_col}' not found")
            
            # Validate ratios
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValidationError("Ratios must sum to 1.0")
            
            # Sort by date
            sorted_df = df.sort_values(date_col)
            
            # Calculate split points
            n_total = len(sorted_df)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split data
            train_df = sorted_df.iloc[:n_train].copy()
            val_df = sorted_df.iloc[n_train:n_train+n_val].copy()
            test_df = sorted_df.iloc[n_train+n_val:].copy()
            
            self.logger.info(
                "Simple time series split completed",
                total_records=n_total,
                train_records=len(train_df),
                val_records=len(val_df),
                test_records=len(test_df),
                train_date_range=f"{train_df[date_col].min()} to {train_df[date_col].max()}",
                val_date_range=f"{val_df[date_col].min()} to {val_df[date_col].max()}" if not val_df.empty else "empty",
                test_date_range=f"{test_df[date_col].min()} to {test_df[date_col].max()}" if not test_df.empty else "empty"
            )
            
            return train_df, val_df, test_df
    
    def time_series_split_by_date(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
        train_end: Optional[Union[str, date, datetime]] = None,
        val_end: Optional[Union[str, date, datetime]] = None,
        test_end: Optional[Union[str, date, datetime]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by specific dates
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            train_end: End date for training data
            val_end: End date for validation data
            test_end: End date for test data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        with with_error_context("date-based time series split"):
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Use config defaults if not provided
            if train_end is None:
                train_end = self.train_end_date
            if val_end is None:
                val_end = self.val_end_date
            if test_end is None:
                test_end = self.test_end_date
            
            # Convert to datetime
            train_end = pd.to_datetime(train_end)
            val_end = pd.to_datetime(val_end)
            test_end = pd.to_datetime(test_end)
            
            # Sort by date
            sorted_df = df.sort_values(date_col)
            sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
            
            # Split data
            train_df = sorted_df[sorted_df[date_col] <= train_end].copy()
            val_df = sorted_df[
                (sorted_df[date_col] > train_end) & 
                (sorted_df[date_col] <= val_end)
            ].copy()
            test_df = sorted_df[
                (sorted_df[date_col] > val_end) & 
                (sorted_df[date_col] <= test_end)
            ].copy()
            
            self.logger.info(
                "Date-based time series split completed",
                train_end=train_end.strftime('%Y-%m-%d'),
                val_end=val_end.strftime('%Y-%m-%d'),
                test_end=test_end.strftime('%Y-%m-%d'),
                train_records=len(train_df),
                val_records=len(val_df),
                test_records=len(test_df)
            )
            
            return train_df, val_df, test_df
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
        n_splits: Optional[int] = None,
        gap_days: Optional[int] = None,
        min_train_size: Optional[int] = None,
        expanding_window: bool = True
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
        """
        Walk-forward cross-validation split
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            n_splits: Number of splits
            gap_days: Gap between train and validation in business days
            min_train_size: Minimum training set size
            expanding_window: If True, expand training window; if False, use sliding window
            
        Yields:
            Tuples of (train_df, val_df, split_info)
        """
        with with_error_context("walk-forward split"):
            if df.empty:
                return
            
            if n_splits is None:
                n_splits = self.n_splits
            if gap_days is None:
                gap_days = self.gap_days
            
            validate_positive(n_splits, "n_splits")
            validate_positive(gap_days, "gap_days")
            
            # Sort by date
            sorted_df = df.sort_values(date_col).reset_index(drop=True)
            sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
            
            # Get unique dates
            unique_dates = sorted(sorted_df[date_col].unique())
            n_dates = len(unique_dates)
            
            if n_dates < n_splits + 2:
                raise DataError(f"Not enough dates ({n_dates}) for {n_splits} splits")
            
            # Calculate split points
            if min_train_size is None:
                min_train_size = max(1, n_dates // (n_splits + 2))
            
            for split_idx in range(n_splits):
                # Calculate date ranges for this split
                if expanding_window:
                    # Expanding window: always start from beginning
                    train_start_idx = 0
                    train_end_idx = min_train_size + split_idx * (n_dates - min_train_size) // n_splits
                else:
                    # Sliding window: fixed-size training window
                    window_size = (n_dates - min_train_size) // n_splits
                    train_start_idx = split_idx * window_size
                    train_end_idx = train_start_idx + min_train_size
                
                # Add gap
                val_start_idx = min(train_end_idx + gap_days, n_dates - 1)
                val_end_idx = min(val_start_idx + (n_dates - train_end_idx) // (n_splits - split_idx), n_dates - 1)
                
                if val_start_idx >= n_dates or val_end_idx >= n_dates:
                    continue
                
                # Get date ranges
                train_start_date = unique_dates[train_start_idx]
                train_end_date = unique_dates[train_end_idx]
                val_start_date = unique_dates[val_start_idx]
                val_end_date = unique_dates[val_end_idx]
                
                # Create splits
                train_df = sorted_df[
                    (sorted_df[date_col] >= train_start_date) & 
                    (sorted_df[date_col] <= train_end_date)
                ].copy()
                
                val_df = sorted_df[
                    (sorted_df[date_col] >= val_start_date) & 
                    (sorted_df[date_col] <= val_end_date)
                ].copy()
                
                # Skip if either set is empty
                if train_df.empty or val_df.empty:
                    continue
                
                # Split info
                split_info = {
                    'split_idx': split_idx,
                    'train_start_date': train_start_date,
                    'train_end_date': train_end_date,
                    'val_start_date': val_start_date,
                    'val_end_date': val_end_date,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'gap_days': (val_start_date - train_end_date).days
                }
                
                self.logger.debug(
                    f"Walk-forward split {split_idx + 1}/{n_splits}",
                    train_range=f"{train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}",
                    val_range=f"{val_start_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}",
                    train_size=len(train_df),
                    val_size=len(val_df)
                )
                
                yield train_df, val_df, split_info
    
    def purged_group_time_series_split(
        self,
        df: pd.DataFrame,
        group_col: str = 'Code',
        date_col: str = 'Date',
        n_splits: Optional[int] = None,
        gap_days: Optional[int] = None,
        purge_days: int = 1
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
        """
        Time series split with purging for grouped data (e.g., multiple stocks)
        
        Args:
            df: DataFrame with grouped time series data
            group_col: Column for grouping (e.g., 'Code' for stocks)
            date_col: Date column name
            n_splits: Number of splits
            gap_days: Gap between train and validation
            purge_days: Additional purging days to prevent data leakage
            
        Yields:
            Tuples of (train_df, val_df, split_info)
        """
        with with_error_context("purged group time series split"):
            if df.empty:
                return
            
            if n_splits is None:
                n_splits = self.n_splits
            if gap_days is None:
                gap_days = self.gap_days
            
            # Sort data
            sorted_df = df.sort_values([group_col, date_col]).reset_index(drop=True)
            sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
            
            # Get unique dates across all groups
            unique_dates = sorted(sorted_df[date_col].unique())
            n_dates = len(unique_dates)
            
            if n_dates < n_splits + 2:
                raise DataError(f"Not enough dates ({n_dates}) for {n_splits} splits")
            
            # Calculate minimum training size
            min_train_size = max(1, n_dates // (n_splits + 2))
            
            for split_idx in range(n_splits):
                # Calculate date boundaries
                train_end_idx = min_train_size + split_idx * (n_dates - min_train_size) // n_splits
                
                # Apply gap and purging
                total_gap = gap_days + purge_days
                val_start_idx = min(train_end_idx + total_gap, n_dates - 1)
                val_end_idx = min(val_start_idx + (n_dates - train_end_idx) // (n_splits - split_idx), n_dates - 1)
                
                if val_start_idx >= n_dates:
                    continue
                
                # Get date thresholds
                train_end_date = unique_dates[train_end_idx]
                val_start_date = unique_dates[val_start_idx]
                val_end_date = unique_dates[val_end_idx] if val_end_idx < n_dates else unique_dates[-1]
                
                # Create train set (all data up to train_end_date)
                train_df = sorted_df[sorted_df[date_col] <= train_end_date].copy()
                
                # Create validation set (data in validation period)
                val_df = sorted_df[
                    (sorted_df[date_col] >= val_start_date) & 
                    (sorted_df[date_col] <= val_end_date)
                ].copy()
                
                # Additional purging: remove any validation group-dates that might cause leakage
                if purge_days > 0:
                    purge_threshold = val_start_date - pd.Timedelta(days=purge_days)
                    
                    # For each group in validation set, remove recent training data
                    val_groups = val_df[group_col].unique()
                    train_df = train_df[
                        ~(
                            (train_df[group_col].isin(val_groups)) & 
                            (train_df[date_col] > purge_threshold)
                        )
                    ]
                
                # Skip if either set is empty
                if train_df.empty or val_df.empty:
                    continue
                
                # Split info
                split_info = {
                    'split_idx': split_idx,
                    'train_end_date': train_end_date,
                    'val_start_date': val_start_date,
                    'val_end_date': val_end_date,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'gap_days': (val_start_date - train_end_date).days,
                    'purge_days': purge_days,
                    'train_groups': train_df[group_col].nunique(),
                    'val_groups': val_df[group_col].nunique()
                }
                
                self.logger.debug(
                    f"Purged group split {split_idx + 1}/{n_splits}",
                    train_end=train_end_date.strftime('%Y-%m-%d'),
                    val_range=f"{val_start_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}",
                    train_size=len(train_df),
                    val_size=len(val_df),
                    train_groups=split_info['train_groups'],
                    val_groups=split_info['val_groups']
                )
                
                yield train_df, val_df, split_info
    
    def embargo_time_series_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
        embargo_pct: float = 0.1,
        n_splits: Optional[int] = None
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
        """
        Time series split with embargo period to prevent data leakage
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            embargo_pct: Percentage of data to embargo between train and test
            n_splits: Number of splits
            
        Yields:
            Tuples of (train_df, test_df, split_info)
        """
        with with_error_context("embargo time series split"):
            if df.empty:
                return
            
            if n_splits is None:
                n_splits = self.n_splits
            
            validate_positive(n_splits, "n_splits")
            if not 0 < embargo_pct < 1:
                raise ValidationError("embargo_pct must be between 0 and 1")
            
            # Sort data
            sorted_df = df.sort_values(date_col).reset_index(drop=True)
            sorted_df[date_col] = pd.to_datetime(sorted_df[date_col])
            
            n_total = len(sorted_df)
            embargo_size = int(n_total * embargo_pct)
            
            # Calculate base sizes
            usable_size = n_total - embargo_size
            fold_size = usable_size // n_splits
            
            for split_idx in range(n_splits):
                # Calculate indices
                test_start = split_idx * fold_size
                test_end = min((split_idx + 1) * fold_size, usable_size)
                
                # Add embargo after test set
                embargo_end = min(test_end + embargo_size, n_total)
                
                # Training data: everything except test and embargo
                train_indices = list(range(0, test_start)) + list(range(embargo_end, n_total))
                test_indices = list(range(test_start, test_end))
                
                if not train_indices or not test_indices:
                    continue
                
                train_df = sorted_df.iloc[train_indices].copy()
                test_df = sorted_df.iloc[test_indices].copy()
                
                # Split info
                split_info = {
                    'split_idx': split_idx,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'embargo_size': embargo_size,
                    'test_start_date': test_df[date_col].min(),
                    'test_end_date': test_df[date_col].max()
                }
                
                self.logger.debug(
                    f"Embargo split {split_idx + 1}/{n_splits}",
                    train_size=len(train_df),
                    test_size=len(test_df),
                    embargo_size=embargo_size
                )
                
                yield train_df, test_df, split_info
    
    def validate_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        date_col: str = 'Date',
        group_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate time series split for data leakage
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            date_col: Date column name
            group_col: Group column name (optional)
            
        Returns:
            Validation results
        """
        validation_results = {
            "status": "passed",
            "issues": [],
            "warnings": []
        }
        
        if train_df.empty or val_df.empty:
            validation_results["issues"].append("Empty train or validation set")
            validation_results["status"] = "failed"
            return validation_results
        
        # Check date ordering
        train_max_date = train_df[date_col].max()
        val_min_date = val_df[date_col].min()
        
        if train_max_date >= val_min_date:
            validation_results["issues"].append(
                f"Training data extends into validation period: "
                f"train_max={train_max_date}, val_min={val_min_date}"
            )
            validation_results["status"] = "failed"
        
        # Check for data leakage in groups
        if group_col and group_col in train_df.columns and group_col in val_df.columns:
            train_groups = set(train_df[group_col].unique())
            val_groups = set(val_df[group_col].unique())
            
            # Groups can overlap, but check for suspicious patterns
            overlap = train_groups & val_groups
            if overlap:
                validation_results["info"] = f"Overlapping groups: {len(overlap)}/{len(val_groups)}"
        
        # Check gap between train and validation
        gap_days = (val_min_date - train_max_date).days
        validation_results["gap_days"] = gap_days
        
        if gap_days < 0:
            validation_results["issues"].append("Negative gap between train and validation")
            validation_results["status"] = "failed"
        elif gap_days == 0:
            validation_results["warnings"].append("No gap between train and validation")
        
        # Check sizes
        train_size = len(train_df)
        val_size = len(val_df)
        
        if train_size < 100:
            validation_results["warnings"].append(f"Small training set: {train_size} samples")
        
        if val_size < 50:
            validation_results["warnings"].append(f"Small validation set: {val_size} samples")
        
        validation_results.update({
            "train_size": train_size,
            "val_size": val_size,
            "train_date_range": (train_df[date_col].min(), train_df[date_col].max()),
            "val_date_range": (val_df[date_col].min(), val_df[date_col].max())
        })
        
        return validation_results
    
    def create_cv_splits_summary(
        self,
        df: pd.DataFrame,
        split_method: str = 'walk_forward',
        **kwargs
    ) -> pd.DataFrame:
        """
        Create summary of cross-validation splits
        
        Args:
            df: Input DataFrame
            split_method: Split method to use
            **kwargs: Additional arguments for split method
            
        Returns:
            DataFrame with split summaries
        """
        summaries = []
        
        # Get splits based on method
        if split_method == 'walk_forward':
            splits = self.walk_forward_split(df, **kwargs)
        elif split_method == 'purged_group':
            splits = self.purged_group_time_series_split(df, **kwargs)
        elif split_method == 'embargo':
            splits = self.embargo_time_series_split(df, **kwargs)
        else:
            raise ValidationError(f"Unknown split method: {split_method}")
        
        # Collect summaries
        for train_df, val_df, split_info in splits:
            summary = {
                'split_idx': split_info['split_idx'],
                'train_size': split_info['train_size'],
                'val_size': split_info['val_size'],
                'total_size': split_info['train_size'] + split_info['val_size']
            }
            
            # Add method-specific info
            if 'train_end_date' in split_info:
                summary['train_end_date'] = split_info['train_end_date']
            if 'val_start_date' in split_info:
                summary['val_start_date'] = split_info['val_start_date']
            if 'gap_days' in split_info:
                summary['gap_days'] = split_info['gap_days']
            
            summaries.append(summary)
        
        if not summaries:
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(summaries)
        
        self.logger.info(
            "CV splits summary created",
            method=split_method,
            n_splits=len(summary_df),
            avg_train_size=summary_df['train_size'].mean(),
            avg_val_size=summary_df['val_size'].mean()
        )
        
        return summary_df


def create_time_series_splitter(config_override: Optional[Dict] = None) -> TimeSeriesSplitter:
    """
    Create time series splitter instance
    
    Args:
        config_override: Configuration overrides
        
    Returns:
        TimeSeriesSplitter instance
    """
    return TimeSeriesSplitter(config_override)