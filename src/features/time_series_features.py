"""
Time Series and Lag Features Module
Provides comprehensive time series analysis features for enhanced stock prediction
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import warnings

# Import utilities with fallback
try:
    from ..utils.config import get_config
    from ..utils.logging import get_logger
    from ..utils.error_handling import with_error_context, DataError, ValidationError
    from ..utils.validation import validate_positive
except ImportError:
    # Fallback implementations
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


class TimeSeriesFeatures:
    """Generate comprehensive time series and lag features - Enhanced for 400+ features"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize time series features generator
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("time_series_features")
        
        # Enhanced configuration for comprehensive features
        self.lag_periods = [1, 2, 3, 5, 10]
        self.return_periods = [1, 3, 5, 10, 20, 40, 60]
        self.rolling_windows = [5, 10, 20, 40, 60, 120]
        self.seasonal_periods = [5, 22, 63, 126, 252]  # 1 week, 1 month, 1 quarter, 6 months, 1 year
        
    def calculate_lag_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        volume_col: str = 'Volume',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive lag features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with lag features
        """
        with with_error_context("calculating lag features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Price lags
            for lag in self.lag_periods:
                if group_by and group_by in result_df.columns:
                    result_df[f'{price_col}_lag_{lag}'] = result_df.groupby(group_by)[price_col].shift(lag)
                    if high_col in result_df.columns:
                        result_df[f'{high_col}_lag_{lag}'] = result_df.groupby(group_by)[high_col].shift(lag)
                    if low_col in result_df.columns:
                        result_df[f'{low_col}_lag_{lag}'] = result_df.groupby(group_by)[low_col].shift(lag)
                else:
                    result_df[f'{price_col}_lag_{lag}'] = result_df[price_col].shift(lag)
                    if high_col in result_df.columns:
                        result_df[f'{high_col}_lag_{lag}'] = result_df[high_col].shift(lag)
                    if low_col in result_df.columns:
                        result_df[f'{low_col}_lag_{lag}'] = result_df[low_col].shift(lag)
                
                # Price change from lag
                result_df[f'price_change_from_lag_{lag}'] = (
                    result_df[price_col] - result_df[f'{price_col}_lag_{lag}']
                ) / result_df[f'{price_col}_lag_{lag}']
                
                # Price momentum (acceleration)
                if lag > 1:
                    result_df[f'price_momentum_lag_{lag}'] = (
                        result_df[f'price_change_from_lag_{lag}'] - 
                        result_df[f'price_change_from_lag_1']
                    )
            
            # Volume lags
            if volume_col in result_df.columns:
                for lag in self.lag_periods:
                    if group_by and group_by in result_df.columns:
                        result_df[f'{volume_col}_lag_{lag}'] = result_df.groupby(group_by)[volume_col].shift(lag)
                    else:
                        result_df[f'{volume_col}_lag_{lag}'] = result_df[volume_col].shift(lag)
                    
                    # Volume change from lag
                    result_df[f'volume_change_from_lag_{lag}'] = (
                        result_df[volume_col] - result_df[f'{volume_col}_lag_{lag}']
                    ) / (result_df[f'{volume_col}_lag_{lag}'] + 1e-10)
            
            # Cross-lag features (price vs volume)
            if volume_col in result_df.columns:
                for lag in [1, 3, 5]:
                    result_df[f'price_volume_cross_lag_{lag}'] = (
                        result_df[f'price_change_from_lag_{lag}'] * 
                        result_df[f'volume_change_from_lag_{lag}']
                    )
            
            self.logger.info(f"Calculated lag features for {len(self.lag_periods)} periods")
            return result_df

    def calculate_return_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive return features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with return features
        """
        with with_error_context("calculating return features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Basic returns for different periods
            for period in self.return_periods:
                return_col = f'return_{period}d'
                
                if group_by and group_by in result_df.columns:
                    result_df[return_col] = result_df.groupby(group_by)[price_col].pct_change(periods=period)
                else:
                    result_df[return_col] = result_df[price_col].pct_change(periods=period)
                
                # Cumulative returns
                result_df[f'cum_return_{period}d'] = (1 + result_df[return_col]).cumprod() - 1
                
                # Return volatility
                result_df[f'return_vol_{period}d'] = abs(result_df[return_col])
                
                # Return direction consistency
                if period > 1:
                    result_df[f'return_consistency_{period}d'] = (
                        result_df[return_col] * result_df['return_1d'] > 0
                    ).astype(int)
                
                # Return rank (cross-sectional)
                if group_by and group_by in result_df.columns:
                    result_df[f'return_rank_{period}d'] = result_df.groupby('Date')[return_col].rank(
                        method='min', ascending=False, pct=True
                    )
            
            # Return statistics
            for window in [10, 20, 60]:
                # Rolling mean return
                return_mean_col = f'return_mean_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[return_mean_col] = result_df.groupby(group_by)['return_1d'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[return_mean_col] = result_df['return_1d'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean()
                
                # Rolling return volatility
                return_std_col = f'return_std_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[return_std_col] = result_df.groupby(group_by)['return_1d'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).std().reset_index(level=0, drop=True)
                else:
                    result_df[return_std_col] = result_df['return_1d'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).std()
                
                # Sharpe-like ratio
                result_df[f'sharpe_ratio_{window}d'] = result_df[return_mean_col] / (
                    result_df[return_std_col] + 1e-10
                )
                
                # Return skewness and kurtosis
                if group_by and group_by in result_df.columns:
                    result_df[f'return_skew_{window}d'] = result_df.groupby(group_by)['return_1d'].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).skew().reset_index(level=0, drop=True)
                    
                    result_df[f'return_kurtosis_{window}d'] = result_df.groupby(group_by)['return_1d'].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).kurt().reset_index(level=0, drop=True)
                else:
                    result_df[f'return_skew_{window}d'] = result_df['return_1d'].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).skew()
                    
                    result_df[f'return_kurtosis_{window}d'] = result_df['return_1d'].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).kurt()
            
            # Maximum Drawdown
            for window in [20, 60, 120]:
                if group_by and group_by in result_df.columns:
                    running_max = result_df.groupby(group_by)[price_col].rolling(
                        window=window, min_periods=1
                    ).max().reset_index(level=0, drop=True)
                else:
                    running_max = result_df[price_col].rolling(window=window, min_periods=1).max()
                
                result_df[f'drawdown_{window}d'] = (result_df[price_col] - running_max) / running_max
                result_df[f'max_drawdown_{window}d'] = result_df[f'drawdown_{window}d'].rolling(
                    window=window, min_periods=1
                ).min()
            
            self.logger.info(f"Calculated return features for {len(self.return_periods)} periods")
            return result_df

    def calculate_seasonal_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate seasonal and calendar features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with seasonal features
        """
        with with_error_context("calculating seasonal features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['Date']):
                result_df['Date'] = pd.to_datetime(result_df['Date'])
            
            # Basic calendar features
            result_df['year'] = result_df['Date'].dt.year
            result_df['month'] = result_df['Date'].dt.month
            result_df['quarter'] = result_df['Date'].dt.quarter
            result_df['day_of_week'] = result_df['Date'].dt.dayofweek
            result_df['day_of_month'] = result_df['Date'].dt.day
            result_df['week_of_year'] = result_df['Date'].dt.isocalendar().week
            
            # Seasonal indicators
            result_df['is_january'] = (result_df['month'] == 1).astype(int)
            result_df['is_december'] = (result_df['month'] == 12).astype(int)
            result_df['is_q1'] = (result_df['quarter'] == 1).astype(int)
            result_df['is_q4'] = (result_df['quarter'] == 4).astype(int)
            
            # Day-of-week effects
            result_df['is_monday'] = (result_df['day_of_week'] == 0).astype(int)
            result_df['is_tuesday'] = (result_df['day_of_week'] == 1).astype(int)
            result_df['is_wednesday'] = (result_df['day_of_week'] == 2).astype(int)
            result_df['is_thursday'] = (result_df['day_of_week'] == 3).astype(int)
            result_df['is_friday'] = (result_df['day_of_week'] == 4).astype(int)
            
            # Month-end effects
            result_df['days_in_month'] = result_df['Date'].dt.days_in_month
            result_df['is_month_end'] = (
                result_df['day_of_month'] >= result_df['days_in_month'] - 2
            ).astype(int)
            result_df['is_month_start'] = (result_df['day_of_month'] <= 3).astype(int)
            
            # Year-end effects
            result_df['is_year_end'] = (
                (result_df['month'] == 12) & (result_df['day_of_month'] >= 15)
            ).astype(int)
            result_df['is_year_start'] = (
                (result_df['month'] == 1) & (result_df['day_of_month'] <= 15)
            ).astype(int)
            
            # Seasonal performance analysis
            if group_by and group_by in result_df.columns:
                result_df['returns'] = result_df.groupby(group_by)[price_col].pct_change()
            else:
                result_df['returns'] = result_df[price_col].pct_change()
            
            # Historical seasonal performance
            for period_name, period_col in [('month', 'month'), ('quarter', 'quarter'), ('day_of_week', 'day_of_week')]:
                seasonal_perf = result_df.groupby([group_by if group_by and group_by in result_df.columns else None, period_col])['returns'].mean().reset_index()
                seasonal_perf = seasonal_perf.dropna()
                
                if not seasonal_perf.empty:
                    seasonal_perf.columns = [group_by if group_by else 'dummy', period_col, f'seasonal_perf_{period_name}']
                    if group_by and group_by in result_df.columns:
                        result_df = result_df.merge(seasonal_perf[[group_by, period_col, f'seasonal_perf_{period_name}']], 
                                                  on=[group_by, period_col], how='left')
                    else:
                        result_df = result_df.merge(seasonal_perf[[period_col, f'seasonal_perf_{period_name}']], 
                                                  on=[period_col], how='left')
            
            # Cyclical features using sine/cosine
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
            result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['day_of_month_sin'] = np.sin(2 * np.pi * result_df['day_of_month'] / 31)
            result_df['day_of_month_cos'] = np.cos(2 * np.pi * result_df['day_of_month'] / 31)
            
            self.logger.info("Calculated seasonal and calendar features")
            return result_df

    def calculate_trend_decomposition_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate trend decomposition features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with trend decomposition features
        """
        with with_error_context("calculating trend decomposition features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Hodrick-Prescott filter approximation using moving averages
            for trend_window in [20, 60, 120]:
                trend_col = f'trend_{trend_window}d'
                cycle_col = f'cycle_{trend_window}d'
                
                if group_by and group_by in result_df.columns:
                    # Trend component (long-term moving average)
                    result_df[trend_col] = result_df.groupby(group_by)[price_col].rolling(
                        window=trend_window, min_periods=max(1, trend_window//2), center=True
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[trend_col] = result_df[price_col].rolling(
                        window=trend_window, min_periods=max(1, trend_window//2), center=True
                    ).mean()
                
                # Cyclical component (detrended)
                result_df[cycle_col] = result_df[price_col] - result_df[trend_col]
                
                # Trend strength
                result_df[f'trend_strength_{trend_window}d'] = (
                    result_df[trend_col].pct_change(5).abs()
                )
                
                # Cycle amplitude
                cycle_amplitude_col = f'cycle_amplitude_{trend_window}d'
                if group_by and group_by in result_df.columns:
                    result_df[cycle_amplitude_col] = result_df.groupby(group_by)[cycle_col].rolling(
                        window=20, min_periods=5
                    ).std().reset_index(level=0, drop=True)
                else:
                    result_df[cycle_amplitude_col] = result_df[cycle_col].rolling(
                        window=20, min_periods=5
                    ).std()
                
                # Trend direction
                result_df[f'trend_up_{trend_window}d'] = (
                    result_df[trend_col] > result_df[trend_col].shift(5)
                ).astype(int)
                result_df[f'trend_down_{trend_window}d'] = (
                    result_df[trend_col] < result_df[trend_col].shift(5)
                ).astype(int)
            
            # Long-term trend classification
            for window in [60, 120, 252]:
                trend_slope_col = f'trend_slope_{window}d'
                
                # Calculate trend slope using linear regression approximation
                if group_by and group_by in result_df.columns:
                    def calc_slope(series):
                        if len(series.dropna()) < 5:
                            return np.nan
                        x = np.arange(len(series))
                        y = series.values
                        valid_idx = ~np.isnan(y)
                        if valid_idx.sum() < 5:
                            return np.nan
                        return np.polyfit(x[valid_idx], y[valid_idx], 1)[0]
                    
                    result_df[trend_slope_col] = result_df.groupby(group_by)[price_col].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).apply(calc_slope).reset_index(level=0, drop=True)
                else:
                    def calc_slope(series):
                        if len(series.dropna()) < 5:
                            return np.nan
                        x = np.arange(len(series))
                        y = series.values
                        valid_idx = ~np.isnan(y)
                        if valid_idx.sum() < 5:
                            return np.nan
                        return np.polyfit(x[valid_idx], y[valid_idx], 1)[0]
                    
                    result_df[trend_slope_col] = result_df[price_col].rolling(
                        window=window, min_periods=max(5, window//4)
                    ).apply(calc_slope)
                
                # Trend classification based on slope
                result_df[f'strong_uptrend_{window}d'] = (
                    result_df[trend_slope_col] > result_df[trend_slope_col].quantile(0.8)
                ).astype(int)
                result_df[f'strong_downtrend_{window}d'] = (
                    result_df[trend_slope_col] < result_df[trend_slope_col].quantile(0.2)
                ).astype(int)
                result_df[f'sideways_{window}d'] = (
                    (result_df[trend_slope_col] >= result_df[trend_slope_col].quantile(0.4)) &
                    (result_df[trend_slope_col] <= result_df[trend_slope_col].quantile(0.6))
                ).astype(int)
            
            self.logger.info("Calculated trend decomposition features")
            return result_df

    def calculate_statistical_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate statistical features (entropy, fractal dimension, etc.)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            group_by: Column to group by
            
        Returns:
            DataFrame with statistical features
        """
        with with_error_context("calculating statistical features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Calculate returns
            if group_by and group_by in result_df.columns:
                result_df['returns'] = result_df.groupby(group_by)[price_col].pct_change()
            else:
                result_df['returns'] = result_df[price_col].pct_change()
            
            # Information-theoretic features
            for window in [20, 60]:
                # Approximate entropy using quantiles
                def calc_entropy(series):
                    if len(series.dropna()) < 10:
                        return np.nan
                    # Discretize returns into bins
                    bins = 10
                    counts, _ = np.histogram(series.dropna(), bins=bins)
                    counts = counts + 1e-10  # Avoid log(0)
                    probs = counts / counts.sum()
                    entropy = -np.sum(probs * np.log(probs))
                    return entropy
                
                entropy_col = f'entropy_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[entropy_col] = result_df.groupby(group_by)['returns'].rolling(
                        window=window, min_periods=max(10, window//2)
                    ).apply(calc_entropy).reset_index(level=0, drop=True)
                else:
                    result_df[entropy_col] = result_df['returns'].rolling(
                        window=window, min_periods=max(10, window//2)
                    ).apply(calc_entropy)
            
            # Fractal dimension approximation (Higuchi method simplified)
            for window in [20, 60]:
                def calc_fractal_dim(series):
                    if len(series.dropna()) < 10:
                        return np.nan
                    
                    series = series.dropna()
                    N = len(series)
                    
                    # Simplified Higuchi algorithm
                    k_max = min(8, N//2)
                    Lk = []
                    
                    for k in range(1, k_max + 1):
                        Lk_sum = 0
                        for m in range(k):
                            Lk_m = 0
                            max_i = (N - m - 1) // k
                            if max_i <= 0:
                                continue
                                
                            for i in range(1, max_i + 1):
                                idx1 = m + i * k
                                idx2 = m + (i - 1) * k
                                if idx1 < len(series) and idx2 < len(series):
                                    Lk_m += abs(series.iloc[idx1] - series.iloc[idx2])
                            
                            if max_i > 0:
                                Lk_m = Lk_m * (N - 1) / (max_i * k * k)
                            Lk_sum += Lk_m
                        
                        if k > 0:
                            Lk.append(Lk_sum / k)
                    
                    if len(Lk) < 2:
                        return np.nan
                    
                    # Linear regression to find fractal dimension
                    x = np.log(range(1, len(Lk) + 1))
                    y = np.log(Lk)
                    
                    try:
                        slope = np.polyfit(x, y, 1)[0]
                        fractal_dim = -slope
                        return fractal_dim
                    except:
                        return np.nan
                
                fractal_col = f'fractal_dim_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[fractal_col] = result_df.groupby(group_by)[price_col].rolling(
                        window=window, min_periods=max(10, window//2)
                    ).apply(calc_fractal_dim).reset_index(level=0, drop=True)
                else:
                    result_df[fractal_col] = result_df[price_col].rolling(
                        window=window, min_periods=max(10, window//2)
                    ).apply(calc_fractal_dim)
            
            # Hurst Exponent approximation
            for window in [40, 120]:
                def calc_hurst(series):
                    if len(series.dropna()) < 20:
                        return np.nan
                    
                    series = series.dropna()
                    N = len(series)
                    
                    # Calculate cumulative deviations
                    mean_series = series.mean()
                    cum_dev = (series - mean_series).cumsum()
                    
                    # Calculate R/S statistic for different lags
                    lags = [2**i for i in range(2, min(8, int(np.log2(N))))]
                    RS = []
                    
                    for lag in lags:
                        if lag >= N:
                            continue
                            
                        # Divide series into non-overlapping segments
                        n_segments = N // lag
                        rs_values = []
                        
                        for i in range(n_segments):
                            start = i * lag
                            end = start + lag
                            segment = cum_dev[start:end]
                            
                            if len(segment) == lag:
                                R = segment.max() - segment.min()
                                S = series[start:end].std()
                                if S > 0:
                                    rs_values.append(R / S)
                        
                        if rs_values:
                            RS.append(np.mean(rs_values))
                    
                    if len(RS) < 2:
                        return np.nan
                    
                    # Linear regression
                    try:
                        x = np.log(lags[:len(RS)])
                        y = np.log(RS)
                        hurst = np.polyfit(x, y, 1)[0]
                        return hurst
                    except:
                        return np.nan
                
                hurst_col = f'hurst_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[hurst_col] = result_df.groupby(group_by)[price_col].rolling(
                        window=window, min_periods=max(20, window//2)
                    ).apply(calc_hurst).reset_index(level=0, drop=True)
                else:
                    result_df[hurst_col] = result_df[price_col].rolling(
                        window=window, min_periods=max(20, window//2)
                    ).apply(calc_hurst)
            
            self.logger.info("Calculated statistical features")
            return result_df

    def calculate_all_time_series_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        volume_col: str = 'Volume',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate all time series features (100+ features)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with all time series features
        """
        result_df = df.copy()
        
        self.logger.info("Starting calculation of all time series features...")
        
        # Lag features
        result_df = self.calculate_lag_features(
            result_df, price_col, high_col, low_col, volume_col, group_by
        )
        
        # Return features
        result_df = self.calculate_return_features(result_df, price_col, group_by)
        
        # Seasonal features
        result_df = self.calculate_seasonal_features(result_df, price_col, group_by)
        
        # Trend decomposition features
        result_df = self.calculate_trend_decomposition_features(result_df, price_col, group_by)
        
        # Statistical features
        result_df = self.calculate_statistical_features(
            result_df, price_col, high_col, low_col, group_by
        )
        
        original_columns = len(df.columns)
        feature_count = len(result_df.columns) - original_columns
        
        self.logger.info(
            f"Calculated all time series features (100+ features)",
            total_features=feature_count,
            original_columns=original_columns,
            final_columns=len(result_df.columns)
        )
        
        return result_df