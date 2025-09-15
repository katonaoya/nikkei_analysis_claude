"""
Label generation for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
from datetime import datetime, date, timedelta

# Import utilities with fallback
try:
    from ..utils.config import get_config
    from ..utils.logging import get_logger
    from ..utils.error_handling import DataError, ValidationError, with_error_context
    from ..utils.validation import validate_positive
    from ..utils.calendar_utils import next_business_day, is_business_day
except ImportError:
    # Fallback implementations
    import logging
    
    def get_config():
        return type('Config', (), {'get': lambda self, key, default=None: default})()
    
    def get_logger(name):
        return logging.getLogger(name)
    
    def with_error_context(context_msg):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in {context_msg}: {e}")
                    raise
            return wrapper
        return decorator
    
    class DataError(Exception):
        pass
    
    class ValidationError(Exception):
        pass
    
    def validate_positive(value, name):
        if value <= 0:
            raise ValidationError(f"{name} must be positive")
    
    def next_business_day(date):
        """Simple next business day calculation"""
        next_day = date + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        return next_day
    
    def is_business_day(date):
        """Simple business day check"""
        return date.weekday() < 5


class LabelGenerator:
    """Generate target labels for machine learning"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize label generator
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("label_generator")
        
        # Configuration
        self.target_return = self.config.get('labels.target_return', 0.01)  # 1% return threshold
        self.lookahead_days = self.config.get('labels.lookahead_days', 1)  # Next business day
    
    def create_next_day_return_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        target_return: Optional[float] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Create labels based on next business day returns
        
        Args:
            df: DataFrame with stock price data
            price_col: Column name for closing prices
            high_col: Column name for high prices
            target_return: Target return threshold (e.g., 0.01 for 1%)
            group_by: Column to group by (e.g., 'Code' for stock-wise calculation)
            
        Returns:
            DataFrame with target labels added
        """
        with with_error_context("creating next day return labels"):
            if df.empty:
                return df
            
            if price_col not in df.columns:
                raise DataError(f"Price column '{price_col}' not found")
            
            if high_col not in df.columns:
                raise DataError(f"High price column '{high_col}' not found")
            
            if target_return is None:
                target_return = self.target_return
            
            validate_positive(target_return, "target return")
            
            result_df = df.copy()
            
            # Sort data by group and date
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Calculate next day high price
            if group_by and group_by in result_df.columns:
                result_df['Next_High'] = result_df.groupby(group_by)[high_col].shift(-1)
            else:
                result_df['Next_High'] = result_df[high_col].shift(-1)
            
            # Calculate return from today's close to next day's high
            result_df['Next_Day_Return'] = (
                result_df['Next_High'] / result_df[price_col] - 1
            )
            
            # Create binary target label
            result_df['Target'] = (result_df['Next_Day_Return'] >= target_return).astype(int)
            
            # Remove last row for each group (no next day data)
            if group_by and group_by in result_df.columns:
                # Mark last date for each group
                result_df['Is_Last'] = result_df.groupby(group_by)['Date'].rank(method='dense', ascending=False) == 1
                result_df = result_df[~result_df['Is_Last']].drop('Is_Last', axis=1)
            else:
                result_df = result_df[:-1]  # Remove last row
            
            # Clean up intermediate columns
            result_df = result_df.drop(['Next_High'], axis=1)
            
            # Calculate label statistics
            positive_rate = result_df['Target'].mean()
            total_labels = len(result_df)
            
            self.logger.info(
                "Created next day return labels",
                target_return=f"{target_return:.1%}",
                total_labels=total_labels,
                positive_labels=int(result_df['Target'].sum()),
                positive_rate=f"{positive_rate:.2%}"
            )
            
            return result_df
    
    def create_multi_horizon_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        horizons: list = [1, 3, 5, 10],
        target_return: Optional[float] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Create labels for multiple time horizons
        
        Args:
            df: DataFrame with stock price data
            price_col: Column name for closing prices
            high_col: Column name for high prices
            horizons: List of horizons (in business days)
            target_return: Target return threshold
            group_by: Column to group by
            
        Returns:
            DataFrame with multi-horizon labels
        """
        with with_error_context("creating multi-horizon labels"):
            if df.empty:
                return df
            
            if target_return is None:
                target_return = self.target_return
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            for horizon in horizons:
                validate_positive(horizon, f"horizon {horizon}")
                
                # Calculate maximum high price within horizon days
                if group_by and group_by in result_df.columns:
                    result_df[f'Max_High_{horizon}d'] = (
                        result_df.groupby(group_by)[high_col]
                        .rolling(window=horizon, min_periods=1)
                        .max()
                        .shift(-horizon + 1)
                        .reset_index(level=0, drop=True)
                    )
                else:
                    result_df[f'Max_High_{horizon}d'] = (
                        result_df[high_col]
                        .rolling(window=horizon, min_periods=1)
                        .max()
                        .shift(-horizon + 1)
                    )
                
                # Calculate return and create target
                result_df[f'Return_{horizon}d'] = (
                    result_df[f'Max_High_{horizon}d'] / result_df[price_col] - 1
                )
                
                result_df[f'Target_{horizon}d'] = (
                    result_df[f'Return_{horizon}d'] >= target_return
                ).astype(int)
                
                # Clean up intermediate column
                result_df = result_df.drop([f'Max_High_{horizon}d'], axis=1)
            
            # Remove rows without sufficient future data
            max_horizon = max(horizons)
            if group_by and group_by in result_df.columns:
                # Remove last max_horizon-1 rows for each group
                result_df['Group_Rank'] = result_df.groupby(group_by)['Date'].rank(method='dense', ascending=False)
                result_df = result_df[result_df['Group_Rank'] > max_horizon - 1].drop('Group_Rank', axis=1)
            else:
                result_df = result_df[:-(max_horizon-1)]
            
            # Log statistics
            for horizon in horizons:
                target_col = f'Target_{horizon}d'
                positive_rate = result_df[target_col].mean()
                self.logger.info(
                    f"Created {horizon}-day labels",
                    positive_rate=f"{positive_rate:.2%}",
                    total_labels=len(result_df)
                )
            
            return result_df
    
    def create_volatility_adjusted_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        volatility_window: int = 20,
        volatility_multiplier: float = 1.5,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Create labels adjusted for stock volatility
        
        Args:
            df: DataFrame with stock price data
            price_col: Column name for closing prices
            high_col: Column name for high prices
            volatility_window: Window for volatility calculation
            volatility_multiplier: Multiplier for volatility-based threshold
            group_by: Column to group by
            
        Returns:
            DataFrame with volatility-adjusted labels
        """
        with with_error_context("creating volatility-adjusted labels"):
            if df.empty:
                return df
            
            validate_positive(volatility_window, "volatility window")
            validate_positive(volatility_multiplier, "volatility multiplier")
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Calculate daily returns
            if group_by and group_by in result_df.columns:
                result_df['Daily_Return'] = result_df.groupby(group_by)[price_col].pct_change()
            else:
                result_df['Daily_Return'] = result_df[price_col].pct_change()
            
            # Calculate rolling volatility
            if group_by and group_by in result_df.columns:
                result_df['Volatility'] = (
                    result_df.groupby(group_by)['Daily_Return']
                    .rolling(window=volatility_window, min_periods=5)
                    .std()
                    .reset_index(level=0, drop=True)
                )
            else:
                result_df['Volatility'] = result_df['Daily_Return'].rolling(
                    window=volatility_window, min_periods=5
                ).std()
            
            # Calculate dynamic threshold based on volatility
            result_df['Dynamic_Threshold'] = result_df['Volatility'] * volatility_multiplier
            
            # Ensure minimum threshold
            min_threshold = self.target_return
            result_df['Dynamic_Threshold'] = np.maximum(result_df['Dynamic_Threshold'], min_threshold)
            
            # Calculate next day return
            if group_by and group_by in result_df.columns:
                result_df['Next_High'] = result_df.groupby(group_by)[high_col].shift(-1)
            else:
                result_df['Next_High'] = result_df[high_col].shift(-1)
            
            result_df['Next_Day_Return'] = (
                result_df['Next_High'] / result_df[price_col] - 1
            )
            
            # Create volatility-adjusted target
            result_df['Target_Vol_Adj'] = (
                result_df['Next_Day_Return'] >= result_df['Dynamic_Threshold']
            ).astype(int)
            
            # Remove rows without next day data
            if group_by and group_by in result_df.columns:
                result_df['Is_Last'] = result_df.groupby(group_by)['Date'].rank(method='dense', ascending=False) == 1
                result_df = result_df[~result_df['Is_Last']].drop('Is_Last', axis=1)
            else:
                result_df = result_df[:-1]
            
            # Clean up intermediate columns
            result_df = result_df.drop(['Daily_Return', 'Next_High'], axis=1)
            
            # Log statistics
            positive_rate = result_df['Target_Vol_Adj'].mean()
            avg_threshold = result_df['Dynamic_Threshold'].mean()
            
            self.logger.info(
                "Created volatility-adjusted labels",
                positive_rate=f"{positive_rate:.2%}",
                average_threshold=f"{avg_threshold:.2%}",
                volatility_multiplier=volatility_multiplier
            )
            
            return result_df
    
    def create_regime_based_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        market_index_col: Optional[str] = None,
        regime_window: int = 60,
        target_return: Optional[float] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Create labels based on market regime (bull/bear/neutral)
        
        Args:
            df: DataFrame with stock price data
            price_col: Column name for closing prices
            high_col: Column name for high prices
            market_index_col: Column name for market index (optional)
            regime_window: Window for regime detection
            target_return: Base target return
            group_by: Column to group by
            
        Returns:
            DataFrame with regime-based labels
        """
        with with_error_context("creating regime-based labels"):
            if df.empty:
                return df
            
            if target_return is None:
                target_return = self.target_return
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Detect market regime
            if market_index_col and market_index_col in result_df.columns:
                # Use market index for regime detection
                regime_col = market_index_col
            else:
                # Use individual stock price for regime detection
                regime_col = price_col
            
            # Calculate regime indicators
            if group_by and group_by in result_df.columns:
                result_df['MA_Long'] = result_df.groupby(group_by)[regime_col].rolling(
                    window=regime_window, min_periods=10
                ).mean().reset_index(level=0, drop=True)
                
                result_df['MA_Short'] = result_df.groupby(group_by)[regime_col].rolling(
                    window=regime_window//3, min_periods=5
                ).mean().reset_index(level=0, drop=True)
            else:
                result_df['MA_Long'] = result_df[regime_col].rolling(
                    window=regime_window, min_periods=10
                ).mean()
                
                result_df['MA_Short'] = result_df[regime_col].rolling(
                    window=regime_window//3, min_periods=5
                ).mean()
            
            # Define regimes
            result_df['Bull_Market'] = (result_df['MA_Short'] > result_df['MA_Long'] * 1.02).astype(int)
            result_df['Bear_Market'] = (result_df['MA_Short'] < result_df['MA_Long'] * 0.98).astype(int)
            result_df['Neutral_Market'] = (
                (~result_df['Bull_Market'].astype(bool)) & 
                (~result_df['Bear_Market'].astype(bool))
            ).astype(int)
            
            # Adjust target returns based on regime
            bull_multiplier = 1.2  # Higher threshold in bull markets
            bear_multiplier = 0.8  # Lower threshold in bear markets
            
            result_df['Regime_Threshold'] = (
                target_return * bull_multiplier * result_df['Bull_Market'] +
                target_return * bear_multiplier * result_df['Bear_Market'] +
                target_return * result_df['Neutral_Market']
            )
            
            # Calculate next day return and create labels
            if group_by and group_by in result_df.columns:
                result_df['Next_High'] = result_df.groupby(group_by)[high_col].shift(-1)
            else:
                result_df['Next_High'] = result_df[high_col].shift(-1)
            
            result_df['Next_Day_Return'] = (
                result_df['Next_High'] / result_df[price_col] - 1
            )
            
            result_df['Target_Regime'] = (
                result_df['Next_Day_Return'] >= result_df['Regime_Threshold']
            ).astype(int)
            
            # Remove rows without next day data
            if group_by and group_by in result_df.columns:
                result_df['Is_Last'] = result_df.groupby(group_by)['Date'].rank(method='dense', ascending=False) == 1
                result_df = result_df[~result_df['Is_Last']].drop('Is_Last', axis=1)
            else:
                result_df = result_df[:-1]
            
            # Clean up intermediate columns
            result_df = result_df.drop(['MA_Long', 'MA_Short', 'Next_High'], axis=1)
            
            # Log regime statistics
            bull_days = result_df['Bull_Market'].sum()
            bear_days = result_df['Bear_Market'].sum()
            neutral_days = result_df['Neutral_Market'].sum()
            positive_rate = result_df['Target_Regime'].mean()
            
            self.logger.info(
                "Created regime-based labels",
                bull_days=bull_days,
                bear_days=bear_days,
                neutral_days=neutral_days,
                positive_rate=f"{positive_rate:.2%}"
            )
            
            return result_df
    
    def validate_labels(
        self,
        df: pd.DataFrame,
        target_cols: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Validate generated labels
        
        Args:
            df: DataFrame with target labels
            target_cols: List of target column names (auto-detect if None)
            
        Returns:
            Validation results
        """
        with with_error_context("validating labels"):
            if df.empty:
                return {"status": "empty", "message": "DataFrame is empty"}
            
            if target_cols is None:
                # Auto-detect target columns
                target_cols = [col for col in df.columns if col.startswith(('Target', 'target'))]
            
            if not target_cols:
                return {"status": "error", "message": "No target columns found"}
            
            validation_results = {
                "status": "success",
                "total_samples": len(df),
                "target_columns": target_cols,
                "column_stats": {}
            }
            
            for col in target_cols:
                if col not in df.columns:
                    validation_results["status"] = "error"
                    validation_results.setdefault("errors", []).append(f"Column {col} not found")
                    continue
                
                # Calculate statistics
                col_stats = {
                    "positive_samples": int(df[col].sum()),
                    "negative_samples": int((df[col] == 0).sum()),
                    "positive_rate": float(df[col].mean()),
                    "missing_values": int(df[col].isna().sum())
                }
                
                # Check for issues
                if col_stats["positive_rate"] < 0.01:
                    col_stats["warning"] = "Very low positive rate (<1%)"
                elif col_stats["positive_rate"] > 0.5:
                    col_stats["warning"] = "High positive rate (>50%)"
                
                if col_stats["missing_values"] > 0:
                    col_stats["warning"] = f"{col_stats['missing_values']} missing values"
                
                validation_results["column_stats"][col] = col_stats
            
            self.logger.info(
                "Label validation completed",
                total_samples=validation_results["total_samples"],
                target_columns=len(target_cols),
                status=validation_results["status"]
            )
            
            return validation_results
    
    def create_feature_target_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'Target',
        feature_prefix_exclude: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into features and targets
        
        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            feature_prefix_exclude: List of column prefixes to exclude from features
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        with with_error_context("creating feature-target split"):
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            if target_col not in df.columns:
                raise DataError(f"Target column '{target_col}' not found")
            
            if feature_prefix_exclude is None:
                feature_prefix_exclude = ['Target', 'target', 'Next_', 'Return_']
            
            # Identify feature columns
            feature_cols = []
            exclude_cols = ['Date', target_col]
            
            for col in df.columns:
                if col in exclude_cols:
                    continue
                
                # Check if column should be excluded
                should_exclude = any(col.startswith(prefix) for prefix in feature_prefix_exclude)
                
                if not should_exclude:
                    feature_cols.append(col)
            
            # Create features and targets DataFrames
            features_df = df[['Date'] + feature_cols].copy()
            targets_df = df[['Date', target_col]].copy()
            
            self.logger.info(
                "Created feature-target split",
                feature_columns=len(feature_cols),
                total_samples=len(df),
                target_column=target_col
            )
            
            return features_df, targets_df


def create_label_generator(config_override: Optional[Dict] = None) -> LabelGenerator:
    """
    Create label generator instance
    
    Args:
        config_override: Configuration overrides
        
    Returns:
        LabelGenerator instance
    """
    return LabelGenerator(config_override)