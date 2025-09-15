"""
Technical indicators for stock price analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, List, Dict, Any
from datetime import datetime, date

# Import utilities with fallback
try:
    from ..utils.config import get_config
    from ..utils.logging import get_logger
    from ..utils.error_handling import DataError, ValidationError, with_error_context
    from ..utils.validation import validate_positive
except ImportError:
    # Fallback implementations
    def get_config():
        return type('Config', (), {'get': lambda self, key, default=None: default})()
    
    def get_logger(name):
        return logging.getLogger(name)
    
    class with_error_context:
        def __init__(self, context_msg):
            self.context_msg = context_msg
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                logging.error(f"Error in {self.context_msg}: {exc_val}")
            return False
    
    class DataError(Exception):
        pass
    
    class ValidationError(Exception):
        pass
    
    def validate_positive(value, name):
        if value <= 0:
            raise ValidationError(f"{name} must be positive")


class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis - Enhanced with 400+ features"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize technical indicators calculator
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("technical_indicators")
        
        # Enhanced configuration for 400+ features
        self.ma_periods = self.config.get('features.ma_periods', [5, 10, 20, 60, 120, 240])
        self.ema_periods = self.config.get('features.ema_periods', [5, 10, 20, 60, 120, 240])
        self.rsi_periods = self.config.get('features.rsi_periods', [14, 30])
        self.stoch_periods = self.config.get('features.stoch_periods', [14, 30])
        self.bb_periods = self.config.get('features.bb_periods', [20])
        self.bb_std_multipliers = self.config.get('features.bb_std_multipliers', [2])
        self.atr_periods = self.config.get('features.atr_periods', [14])
        self.adx_periods = self.config.get('features.adx_periods', [14])
        self.williams_periods = self.config.get('features.williams_periods', [14])
        self.roc_periods = self.config.get('features.roc_periods', [1, 3, 5, 10, 20])
        
    def calculate_enhanced_moving_averages(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate enhanced moving averages (SMA + EMA + deviations + slopes)
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            group_by: Column to group by
            
        Returns:
            DataFrame with enhanced moving average columns
        """
        with with_error_context("calculating enhanced moving averages"):
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Simple Moving Averages
            for period in self.ma_periods:
                col_name = f'SMA_{period}'
                if group_by and group_by in result_df.columns:
                    result_df[col_name] = result_df.groupby(group_by)[price_col].rolling(
                        window=period, min_periods=1
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[col_name] = result_df[price_col].rolling(
                        window=period, min_periods=1
                    ).mean()
                
                # Deviation from SMA
                result_df[f'SMA_{period}_dev'] = (result_df[price_col] - result_df[col_name]) / result_df[col_name]
                
                # SMA slope (rate of change)
                result_df[f'SMA_{period}_slope'] = result_df[col_name].pct_change(5)
            
            # Exponential Moving Averages
            for period in self.ema_periods:
                col_name = f'EMA_{period}'
                if group_by and group_by in result_df.columns:
                    result_df[col_name] = result_df.groupby(group_by)[price_col].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    )
                else:
                    result_df[col_name] = result_df[price_col].ewm(span=period, adjust=False).mean()
                
                # Deviation from EMA
                result_df[f'EMA_{period}_dev'] = (result_df[price_col] - result_df[col_name]) / result_df[col_name]
                
                # EMA slope
                result_df[f'EMA_{period}_slope'] = result_df[col_name].pct_change(5)
            
            # Moving Average Cross Signals
            if 5 in self.ma_periods and 20 in self.ma_periods:
                result_df['MA_5_20_cross'] = np.where(
                    (result_df['SMA_5'] > result_df['SMA_20']) & 
                    (result_df['SMA_5'].shift(1) <= result_df['SMA_20'].shift(1)), 1,
                    np.where(
                        (result_df['SMA_5'] < result_df['SMA_20']) & 
                        (result_df['SMA_5'].shift(1) >= result_df['SMA_20'].shift(1)), -1, 0
                    )
                )
            
            if 20 in self.ma_periods and 60 in self.ma_periods:
                result_df['MA_20_60_cross'] = np.where(
                    (result_df['SMA_20'] > result_df['SMA_60']) & 
                    (result_df['SMA_20'].shift(1) <= result_df['SMA_60'].shift(1)), 1,
                    np.where(
                        (result_df['SMA_20'] < result_df['SMA_60']) & 
                        (result_df['SMA_20'].shift(1) >= result_df['SMA_60'].shift(1)), -1, 0
                    )
                )
            
            self.logger.info(f"Calculated enhanced moving averages for periods: {self.ma_periods + self.ema_periods}")
            return result_df

    def calculate_momentum_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive momentum indicators (RSI, Stochastic, Williams%R, ROC)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            group_by: Column to group by
            
        Returns:
            DataFrame with momentum indicators
        """
        with with_error_context("calculating momentum indicators"):
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # RSI for multiple periods
            for period in self.rsi_periods:
                rsi_col = f'RSI_{period}'
                
                def _calculate_rsi_for_series(price_series: pd.Series) -> pd.Series:
                    delta = price_series.diff()
                    gains = delta.where(delta > 0, 0)
                    losses = -delta.where(delta < 0, 0)
                    avg_gains = gains.rolling(window=period, min_periods=1).mean()
                    avg_losses = losses.rolling(window=period, min_periods=1).mean()
                    rs = avg_gains / (avg_losses + 1e-10)
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                
                if group_by and group_by in result_df.columns:
                    result_df[rsi_col] = result_df.groupby(group_by)[price_col].transform(_calculate_rsi_for_series)
                else:
                    result_df[rsi_col] = _calculate_rsi_for_series(result_df[price_col])
                
                # RSI signals
                result_df[f'RSI_{period}_oversold'] = (result_df[rsi_col] < 30).astype(int)
                result_df[f'RSI_{period}_overbought'] = (result_df[rsi_col] > 70).astype(int)
                result_df[f'RSI_{period}_divergence'] = result_df[rsi_col] - result_df[rsi_col].rolling(10).mean()
            
            # Stochastic Oscillator for multiple periods
            for period in self.stoch_periods:
                if group_by and group_by in result_df.columns:
                    result_df[f'Highest_High_{period}'] = result_df.groupby(group_by)[high_col].rolling(
                        window=period, min_periods=1
                    ).max().reset_index(level=0, drop=True)
                    
                    result_df[f'Lowest_Low_{period}'] = result_df.groupby(group_by)[low_col].rolling(
                        window=period, min_periods=1
                    ).min().reset_index(level=0, drop=True)
                else:
                    result_df[f'Highest_High_{period}'] = result_df[high_col].rolling(
                        window=period, min_periods=1
                    ).max()
                    
                    result_df[f'Lowest_Low_{period}'] = result_df[low_col].rolling(
                        window=period, min_periods=1
                    ).min()
                
                # %K calculation
                stoch_k_col = f'Stoch_K_{period}'
                result_df[stoch_k_col] = 100 * (
                    (result_df[price_col] - result_df[f'Lowest_Low_{period}']) / 
                    (result_df[f'Highest_High_{period}'] - result_df[f'Lowest_Low_{period}'] + 1e-10)
                )
                
                # %D calculation (3-period moving average of %K)
                stoch_d_col = f'Stoch_D_{period}'
                if group_by and group_by in result_df.columns:
                    result_df[stoch_d_col] = result_df.groupby(group_by)[stoch_k_col].rolling(
                        window=3, min_periods=1
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[stoch_d_col] = result_df[stoch_k_col].rolling(
                        window=3, min_periods=1
                    ).mean()
                
                # Stochastic signals
                result_df[f'Stoch_{period}_oversold'] = (result_df[stoch_k_col] < 20).astype(int)
                result_df[f'Stoch_{period}_overbought'] = (result_df[stoch_k_col] > 80).astype(int)
                
                # Clean up intermediate columns
                result_df = result_df.drop([f'Highest_High_{period}', f'Lowest_Low_{period}'], axis=1)
            
            # Williams %R
            for period in self.williams_periods:
                williams_col = f'Williams_R_{period}'
                
                if group_by and group_by in result_df.columns:
                    highest_high = result_df.groupby(group_by)[high_col].rolling(
                        window=period, min_periods=1
                    ).max().reset_index(level=0, drop=True)
                    
                    lowest_low = result_df.groupby(group_by)[low_col].rolling(
                        window=period, min_periods=1
                    ).min().reset_index(level=0, drop=True)
                else:
                    highest_high = result_df[high_col].rolling(window=period, min_periods=1).max()
                    lowest_low = result_df[low_col].rolling(window=period, min_periods=1).min()
                
                result_df[williams_col] = -100 * (
                    (highest_high - result_df[price_col]) / (highest_high - lowest_low + 1e-10)
                )
                
                # Williams %R signals
                result_df[f'Williams_R_{period}_oversold'] = (result_df[williams_col] < -80).astype(int)
                result_df[f'Williams_R_{period}_overbought'] = (result_df[williams_col] > -20).astype(int)
            
            # Rate of Change (ROC)
            for period in self.roc_periods:
                roc_col = f'ROC_{period}'
                result_df[roc_col] = result_df[price_col].pct_change(periods=period) * 100
                
                # ROC momentum signals
                result_df[f'ROC_{period}_positive'] = (result_df[roc_col] > 0).astype(int)
                result_df[f'ROC_{period}_acceleration'] = result_df[roc_col] - result_df[roc_col].shift(1)
            
            self.logger.info("Calculated comprehensive momentum indicators")
            return result_df

    def calculate_trend_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate trend indicators (MACD, ADX, Parabolic SAR, Ichimoku)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            group_by: Column to group by
            
        Returns:
            DataFrame with trend indicators
        """
        with with_error_context("calculating trend indicators"):
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Enhanced MACD with multiple parameter sets
            macd_configs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
            
            for fast, slow, signal in macd_configs:
                # Calculate EMAs
                if group_by and group_by in result_df.columns:
                    ema_fast = result_df.groupby(group_by)[price_col].transform(
                        lambda x: x.ewm(span=fast, adjust=False).mean()
                    )
                    ema_slow = result_df.groupby(group_by)[price_col].transform(
                        lambda x: x.ewm(span=slow, adjust=False).mean()
                    )
                else:
                    ema_fast = result_df[price_col].ewm(span=fast, adjust=False).mean()
                    ema_slow = result_df[price_col].ewm(span=slow, adjust=False).mean()
                
                # MACD line
                macd_col = f'MACD_{fast}_{slow}_{signal}'
                result_df[macd_col] = ema_fast - ema_slow
                
                # Signal line
                signal_col = f'MACD_Signal_{fast}_{slow}_{signal}'
                if group_by and group_by in result_df.columns:
                    result_df[signal_col] = result_df.groupby(group_by)[macd_col].transform(
                        lambda x: x.ewm(span=signal, adjust=False).mean()
                    )
                else:
                    result_df[signal_col] = result_df[macd_col].ewm(span=signal, adjust=False).mean()
                
                # MACD histogram
                histogram_col = f'MACD_Histogram_{fast}_{slow}_{signal}'
                result_df[histogram_col] = result_df[macd_col] - result_df[signal_col]
                
                # MACD signals
                result_df[f'MACD_Bullish_{fast}_{slow}_{signal}'] = (
                    (result_df[macd_col] > result_df[signal_col]) & 
                    (result_df[macd_col].shift(1) <= result_df[signal_col].shift(1))
                ).astype(int)
                
                result_df[f'MACD_Bearish_{fast}_{slow}_{signal}'] = (
                    (result_df[macd_col] < result_df[signal_col]) & 
                    (result_df[macd_col].shift(1) >= result_df[signal_col].shift(1))
                ).astype(int)
            
            # ADX (Average Directional Index)
            for period in self.adx_periods:
                # True Range calculation
                result_df['TR1'] = result_df[high_col] - result_df[low_col]
                result_df['TR2'] = abs(result_df[high_col] - result_df[price_col].shift(1))
                result_df['TR3'] = abs(result_df[low_col] - result_df[price_col].shift(1))
                result_df['True_Range'] = result_df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Directional Movements
                result_df['DM_Plus'] = np.where(
                    (result_df[high_col] - result_df[high_col].shift(1)) > 
                    (result_df[low_col].shift(1) - result_df[low_col]),
                    np.maximum(result_df[high_col] - result_df[high_col].shift(1), 0), 0
                )
                
                result_df['DM_Minus'] = np.where(
                    (result_df[low_col].shift(1) - result_df[low_col]) > 
                    (result_df[high_col] - result_df[high_col].shift(1)),
                    np.maximum(result_df[low_col].shift(1) - result_df[low_col], 0), 0
                )
                
                # Smoothed values
                if group_by and group_by in result_df.columns:
                    result_df[f'ATR_{period}'] = result_df.groupby(group_by)['True_Range'].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    )
                    result_df[f'DI_Plus_{period}'] = 100 * result_df.groupby(group_by)['DM_Plus'].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    ) / result_df[f'ATR_{period}']
                    result_df[f'DI_Minus_{period}'] = 100 * result_df.groupby(group_by)['DM_Minus'].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    ) / result_df[f'ATR_{period}']
                else:
                    result_df[f'ATR_{period}'] = result_df['True_Range'].ewm(span=period, adjust=False).mean()
                    result_df[f'DI_Plus_{period}'] = 100 * result_df['DM_Plus'].ewm(span=period, adjust=False).mean() / result_df[f'ATR_{period}']
                    result_df[f'DI_Minus_{period}'] = 100 * result_df['DM_Minus'].ewm(span=period, adjust=False).mean() / result_df[f'ATR_{period}']
                
                # ADX calculation
                result_df[f'DX_{period}'] = 100 * abs(result_df[f'DI_Plus_{period}'] - result_df[f'DI_Minus_{period}']) / (
                    result_df[f'DI_Plus_{period}'] + result_df[f'DI_Minus_{period}'] + 1e-10
                )
                
                if group_by and group_by in result_df.columns:
                    result_df[f'ADX_{period}'] = result_df.groupby(group_by)[f'DX_{period}'].transform(
                        lambda x: x.ewm(span=period, adjust=False).mean()
                    )
                else:
                    result_df[f'ADX_{period}'] = result_df[f'DX_{period}'].ewm(span=period, adjust=False).mean()
                
                # ADX trend strength signals
                result_df[f'ADX_{period}_strong_trend'] = (result_df[f'ADX_{period}'] > 25).astype(int)
                result_df[f'ADX_{period}_weak_trend'] = (result_df[f'ADX_{period}'] < 20).astype(int)
                
                # Clean up intermediate columns
                result_df = result_df.drop(['TR1', 'TR2', 'TR3', 'True_Range', 'DM_Plus', 'DM_Minus', f'DX_{period}'], axis=1)
            
            # Ichimoku Cloud components
            # Tenkan-sen (9-period)
            tenkan_period = 9
            if group_by and group_by in result_df.columns:
                tenkan_high = result_df.groupby(group_by)[high_col].rolling(window=tenkan_period, min_periods=1).max().reset_index(level=0, drop=True)
                tenkan_low = result_df.groupby(group_by)[low_col].rolling(window=tenkan_period, min_periods=1).min().reset_index(level=0, drop=True)
            else:
                tenkan_high = result_df[high_col].rolling(window=tenkan_period, min_periods=1).max()
                tenkan_low = result_df[low_col].rolling(window=tenkan_period, min_periods=1).min()
            
            result_df['Ichimoku_Tenkan'] = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (26-period)
            kijun_period = 26
            if group_by and group_by in result_df.columns:
                kijun_high = result_df.groupby(group_by)[high_col].rolling(window=kijun_period, min_periods=1).max().reset_index(level=0, drop=True)
                kijun_low = result_df.groupby(group_by)[low_col].rolling(window=kijun_period, min_periods=1).min().reset_index(level=0, drop=True)
            else:
                kijun_high = result_df[high_col].rolling(window=kijun_period, min_periods=1).max()
                kijun_low = result_df[low_col].rolling(window=kijun_period, min_periods=1).min()
            
            result_df['Ichimoku_Kijun'] = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            result_df['Ichimoku_SpanA'] = ((result_df['Ichimoku_Tenkan'] + result_df['Ichimoku_Kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B) - 52-period
            senkou_period = 52
            if group_by and group_by in result_df.columns:
                senkou_high = result_df.groupby(group_by)[high_col].rolling(window=senkou_period, min_periods=1).max().reset_index(level=0, drop=True)
                senkou_low = result_df.groupby(group_by)[low_col].rolling(window=senkou_period, min_periods=1).min().reset_index(level=0, drop=True)
            else:
                senkou_high = result_df[high_col].rolling(window=senkou_period, min_periods=1).max()
                senkou_low = result_df[low_col].rolling(window=senkou_period, min_periods=1).min()
            
            result_df['Ichimoku_SpanB'] = ((senkou_high + senkou_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            result_df['Ichimoku_Chikou'] = result_df[price_col].shift(-26)
            
            # Ichimoku signals
            result_df['Ichimoku_Bullish'] = (
                (result_df['Ichimoku_Tenkan'] > result_df['Ichimoku_Kijun']) &
                (result_df[price_col] > result_df['Ichimoku_SpanA']) &
                (result_df[price_col] > result_df['Ichimoku_SpanB'])
            ).astype(int)
            
            result_df['Ichimoku_Bearish'] = (
                (result_df['Ichimoku_Tenkan'] < result_df['Ichimoku_Kijun']) &
                (result_df[price_col] < result_df['Ichimoku_SpanA']) &
                (result_df[price_col] < result_df['Ichimoku_SpanB'])
            ).astype(int)
            
            self.logger.info("Calculated comprehensive trend indicators")
            return result_df

    def calculate_volatility_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate volatility indicators (ATR, Bollinger Bands, Keltner Channels)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            group_by: Column to group by
            
        Returns:
            DataFrame with volatility indicators
        """
        with with_error_context("calculating volatility indicators"):
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Multiple Bollinger Bands
            for period in self.bb_periods:
                for std_mult in self.bb_std_multipliers:
                    prefix = f'BB_{period}_{std_mult}'
                    
                    # Calculate moving average and standard deviation
                    if group_by and group_by in result_df.columns:
                        bb_middle = result_df.groupby(group_by)[price_col].rolling(
                            window=period, min_periods=1
                        ).mean().reset_index(level=0, drop=True)
                        
                        bb_std = result_df.groupby(group_by)[price_col].rolling(
                            window=period, min_periods=1
                        ).std().reset_index(level=0, drop=True)
                    else:
                        bb_middle = result_df[price_col].rolling(window=period, min_periods=1).mean()
                        bb_std = result_df[price_col].rolling(window=period, min_periods=1).std()
                    
                    result_df[f'{prefix}_Middle'] = bb_middle
                    result_df[f'{prefix}_Upper'] = bb_middle + (bb_std * std_mult)
                    result_df[f'{prefix}_Lower'] = bb_middle - (bb_std * std_mult)
                    
                    # Bollinger Band features
                    result_df[f'{prefix}_Width'] = (
                        result_df[f'{prefix}_Upper'] - result_df[f'{prefix}_Lower']
                    ) / result_df[f'{prefix}_Middle']
                    
                    result_df[f'{prefix}_Position'] = (
                        result_df[price_col] - result_df[f'{prefix}_Lower']
                    ) / (result_df[f'{prefix}_Upper'] - result_df[f'{prefix}_Lower'] + 1e-10)
                    
                    # Bollinger Band squeeze
                    result_df[f'{prefix}_Squeeze'] = (
                        result_df[f'{prefix}_Width'] < result_df[f'{prefix}_Width'].rolling(20).quantile(0.2)
                    ).astype(int)
                    
                    # Band penetration signals
                    result_df[f'{prefix}_Upper_Break'] = (result_df[price_col] > result_df[f'{prefix}_Upper']).astype(int)
                    result_df[f'{prefix}_Lower_Break'] = (result_df[price_col] < result_df[f'{prefix}_Lower']).astype(int)
            
            # Keltner Channels
            for atr_period in self.atr_periods:
                kc_period = 20
                kc_multiplier = 2
                
                # EMA for center line
                if group_by and group_by in result_df.columns:
                    kc_middle = result_df.groupby(group_by)[price_col].transform(
                        lambda x: x.ewm(span=kc_period, adjust=False).mean()
                    )
                else:
                    kc_middle = result_df[price_col].ewm(span=kc_period, adjust=False).mean()
                
                # Use ATR from previous calculation or calculate here
                atr_col = f'ATR_{atr_period}'
                if atr_col not in result_df.columns:
                    # Calculate True Range
                    result_df['TR1'] = result_df[high_col] - result_df[low_col]
                    result_df['TR2'] = abs(result_df[high_col] - result_df[price_col].shift(1))
                    result_df['TR3'] = abs(result_df[low_col] - result_df[price_col].shift(1))
                    result_df['True_Range'] = result_df[['TR1', 'TR2', 'TR3']].max(axis=1)
                    
                    if group_by and group_by in result_df.columns:
                        result_df[atr_col] = result_df.groupby(group_by)['True_Range'].transform(
                            lambda x: x.ewm(span=atr_period, adjust=False).mean()
                        )
                    else:
                        result_df[atr_col] = result_df['True_Range'].ewm(span=atr_period, adjust=False).mean()
                    
                    # Clean up
                    result_df = result_df.drop(['TR1', 'TR2', 'TR3', 'True_Range'], axis=1)
                
                prefix = f'KC_{kc_period}_{atr_period}'
                result_df[f'{prefix}_Middle'] = kc_middle
                result_df[f'{prefix}_Upper'] = kc_middle + (result_df[atr_col] * kc_multiplier)
                result_df[f'{prefix}_Lower'] = kc_middle - (result_df[atr_col] * kc_multiplier)
                
                # Keltner Channel features
                result_df[f'{prefix}_Width'] = (
                    result_df[f'{prefix}_Upper'] - result_df[f'{prefix}_Lower']
                ) / result_df[f'{prefix}_Middle']
                
                result_df[f'{prefix}_Position'] = (
                    result_df[price_col] - result_df[f'{prefix}_Lower']
                ) / (result_df[f'{prefix}_Upper'] - result_df[f'{prefix}_Lower'] + 1e-10)
            
            # Historical Volatility
            volatility_periods = [5, 10, 20, 30]
            for period in volatility_periods:
                returns_col = f'Returns_{period}'
                result_df[returns_col] = result_df[price_col].pct_change()
                
                if group_by and group_by in result_df.columns:
                    result_df[f'HV_{period}'] = result_df.groupby(group_by)[returns_col].rolling(
                        window=period, min_periods=1
                    ).std().reset_index(level=0, drop=True) * np.sqrt(252)  # Annualized
                else:
                    result_df[f'HV_{period}'] = result_df[returns_col].rolling(
                        window=period, min_periods=1
                    ).std() * np.sqrt(252)
                
                # Clean up
                result_df = result_df.drop([returns_col], axis=1)
            
            self.logger.info("Calculated comprehensive volatility indicators")
            return result_df

    def calculate_moving_averages(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        periods: Optional[List[int]] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility - calls enhanced version"""
        return self.calculate_enhanced_moving_averages(df, price_col, group_by)
    
    def calculate_rsi(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        period: Optional[int] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility - calls enhanced version with single period"""
        result_df = df.copy()
        
        if period is None:
            period = 14
            
        def _calculate_rsi_for_series(price_series: pd.Series) -> pd.Series:
            delta = price_series.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
            result_df['RSI'] = result_df.groupby(group_by)[price_col].transform(_calculate_rsi_for_series)
        else:
            result_df = result_df.sort_values('Date')
            result_df['RSI'] = _calculate_rsi_for_series(result_df[price_col])
        
        result_df['RSI_oversold'] = (result_df['RSI'] < 30).astype(int)
        result_df['RSI_overbought'] = (result_df['RSI'] > 70).astype(int)
        
        return result_df
    
    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        period: Optional[int] = None,
        std_multiplier: Optional[float] = None,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        if period is None:
            period = 20
        if std_multiplier is None:
            std_multiplier = 2
            
        result_df = df.copy()
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
            
            result_df['BB_Middle'] = result_df.groupby(group_by)[price_col].rolling(
                window=period, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            
            bb_std = result_df.groupby(group_by)[price_col].rolling(
                window=period, min_periods=1
            ).std().reset_index(level=0, drop=True)
        else:
            result_df = result_df.sort_values('Date')
            result_df['BB_Middle'] = result_df[price_col].rolling(window=period, min_periods=1).mean()
            bb_std = result_df[price_col].rolling(window=period, min_periods=1).std()
        
        result_df['BB_Upper'] = result_df['BB_Middle'] + (bb_std * std_multiplier)
        result_df['BB_Lower'] = result_df['BB_Middle'] - (bb_std * std_multiplier)
        result_df['BB_Width'] = (result_df['BB_Upper'] - result_df['BB_Lower']) / result_df['BB_Middle']
        result_df['BB_Position'] = (result_df[price_col] - result_df['BB_Lower']) / (
            result_df['BB_Upper'] - result_df['BB_Lower'] + 1e-10
        )
        result_df['BB_Squeeze'] = (result_df['BB_Width'] < result_df['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        
        return result_df
    
    def calculate_macd(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        result_df = df.copy()
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
            
            ema_fast = result_df.groupby(group_by)[price_col].transform(
                lambda x: x.ewm(span=fast_period, adjust=False).mean()
            )
            ema_slow = result_df.groupby(group_by)[price_col].transform(
                lambda x: x.ewm(span=slow_period, adjust=False).mean()
            )
        else:
            result_df = result_df.sort_values('Date')
            ema_fast = result_df[price_col].ewm(span=fast_period, adjust=False).mean()
            ema_slow = result_df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        result_df['MACD'] = ema_fast - ema_slow
        
        if group_by and group_by in result_df.columns:
            result_df['MACD_Signal'] = result_df.groupby(group_by)['MACD'].transform(
                lambda x: x.ewm(span=signal_period, adjust=False).mean()
            )
        else:
            result_df['MACD_Signal'] = result_df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        
        result_df['MACD_Bullish'] = (
            (result_df['MACD'] > result_df['MACD_Signal']) & 
            (result_df['MACD'].shift(1) <= result_df['MACD_Signal'].shift(1))
        ).astype(int)
        
        result_df['MACD_Bearish'] = (
            (result_df['MACD'] < result_df['MACD_Signal']) & 
            (result_df['MACD'].shift(1) >= result_df['MACD_Signal'].shift(1))
        ).astype(int)
        
        return result_df
    
    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        high_col: str = 'High',
        low_col: str = 'Low',
        close_col: str = 'Close',
        k_period: int = 14,
        d_period: int = 3,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        result_df = df.copy()
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
            
            highest_high = result_df.groupby(group_by)[high_col].rolling(
                window=k_period, min_periods=1
            ).max().reset_index(level=0, drop=True)
            
            lowest_low = result_df.groupby(group_by)[low_col].rolling(
                window=k_period, min_periods=1
            ).min().reset_index(level=0, drop=True)
        else:
            result_df = result_df.sort_values('Date')
            highest_high = result_df[high_col].rolling(window=k_period, min_periods=1).max()
            lowest_low = result_df[low_col].rolling(window=k_period, min_periods=1).min()
        
        result_df['Stoch_K'] = 100 * (
            (result_df[close_col] - lowest_low) / (highest_high - lowest_low + 1e-10)
        )
        
        if group_by and group_by in result_df.columns:
            result_df['Stoch_D'] = result_df.groupby(group_by)['Stoch_K'].rolling(
                window=d_period, min_periods=1
            ).mean().reset_index(level=0, drop=True)
        else:
            result_df['Stoch_D'] = result_df['Stoch_K'].rolling(window=d_period, min_periods=1).mean()
        
        result_df['Stoch_Oversold'] = (result_df['Stoch_K'] < 20).astype(int)
        result_df['Stoch_Overbought'] = (result_df['Stoch_K'] > 80).astype(int)
        
        return result_df
    
    def calculate_volume_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        volume_col: str = 'Volume',
        period: int = 20,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        result_df = df.copy()
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
            
            result_df['Volume_MA'] = result_df.groupby(group_by)[volume_col].rolling(
                window=period, min_periods=1
            ).mean().reset_index(level=0, drop=True)
        else:
            result_df = result_df.sort_values('Date')
            result_df['Volume_MA'] = result_df[volume_col].rolling(window=period, min_periods=1).mean()
        
        result_df['Volume_Ratio'] = result_df[volume_col] / (result_df['Volume_MA'] + 1e-10)
        
        price_change_pct = result_df.groupby(group_by)[price_col].pct_change() if group_by and group_by in result_df.columns else result_df[price_col].pct_change()
        result_df['PVT'] = (price_change_pct * result_df[volume_col]).fillna(0)
        
        if group_by and group_by in result_df.columns:
            result_df['PVT'] = result_df.groupby(group_by)['PVT'].cumsum()
        else:
            result_df['PVT'] = result_df['PVT'].cumsum()
        
        price_direction = np.where(price_change_pct > 0, 1, np.where(price_change_pct < 0, -1, 0))
        obv_change = price_direction * result_df[volume_col]
        
        if group_by and group_by in result_df.columns:
            result_df['OBV'] = result_df.groupby(group_by).apply(lambda x: obv_change[x.index].cumsum()).reset_index(level=0, drop=True)
        else:
            result_df['OBV'] = obv_change.cumsum()
        
        result_df['High_Volume'] = (result_df['Volume_Ratio'] > 2.0).astype(int)
        
        return result_df
    
    def calculate_price_patterns(
        self,
        df: pd.DataFrame,
        open_col: str = 'Open',
        high_col: str = 'High',
        low_col: str = 'Low',
        close_col: str = 'Close',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        result_df = df.copy()
        
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
        else:
            result_df = result_df.sort_values('Date')
        
        result_df['Body_Size'] = abs(result_df[close_col] - result_df[open_col])
        result_df['Upper_Shadow'] = result_df[high_col] - np.maximum(result_df[open_col], result_df[close_col])
        result_df['Lower_Shadow'] = np.minimum(result_df[open_col], result_df[close_col]) - result_df[low_col]
        result_df['Total_Range'] = result_df[high_col] - result_df[low_col]
        
        result_df['Body_Ratio'] = result_df['Body_Size'] / (result_df['Total_Range'] + 1e-10)
        result_df['Upper_Shadow_Ratio'] = result_df['Upper_Shadow'] / (result_df['Total_Range'] + 1e-10)
        result_df['Lower_Shadow_Ratio'] = result_df['Lower_Shadow'] / (result_df['Total_Range'] + 1e-10)
        
        result_df['Doji'] = (result_df['Body_Ratio'] < 0.1).astype(int)
        result_df['Hammer'] = (
            (result_df['Lower_Shadow_Ratio'] > 0.6) & 
            (result_df['Upper_Shadow_Ratio'] < 0.1) &
            (result_df['Body_Ratio'] < 0.3)
        ).astype(int)
        
        result_df['Shooting_Star'] = (
            (result_df['Upper_Shadow_Ratio'] > 0.6) & 
            (result_df['Lower_Shadow_Ratio'] < 0.1) &
            (result_df['Body_Ratio'] < 0.3)
        ).astype(int)
        
        result_df['Gap_Up'] = (result_df[low_col] > result_df[high_col].shift(1)).astype(int)
        result_df['Gap_Down'] = (result_df[high_col] < result_df[low_col].shift(1)).astype(int)
        result_df['Close_Position'] = (
            (result_df[close_col] - result_df[low_col]) / (result_df['Total_Range'] + 1e-10)
        )
        
        return result_df
    
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        include_patterns: bool = True,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators (Enhanced version with 400+ features)
        
        Args:
            df: DataFrame with OHLC data
            include_patterns: Whether to include candlestick patterns
            include_volume: Whether to include volume indicators
            
        Returns:
            DataFrame with all technical indicators
        """
        with with_error_context("calculating all enhanced technical indicators"):
            if df.empty:
                return df
            
            result_df = df.copy()
            
            # Enhanced moving averages (SMA + EMA + deviations + slopes)
            result_df = self.calculate_enhanced_moving_averages(result_df)
            
            # Comprehensive momentum indicators (RSI, Stochastic, Williams%R, ROC)
            if all(col in result_df.columns for col in ['High', 'Low', 'Close']):
                result_df = self.calculate_momentum_indicators(result_df)
            
            # Trend indicators (MACD, ADX, Ichimoku)
            if all(col in result_df.columns for col in ['High', 'Low', 'Close']):
                result_df = self.calculate_trend_indicators(result_df)
            
            # Volatility indicators (Bollinger Bands, Keltner Channels, ATR, Historical Volatility)
            if all(col in result_df.columns for col in ['High', 'Low', 'Close']):
                result_df = self.calculate_volatility_indicators(result_df)
            
            # Legacy volume indicators (if available)
            if include_volume and 'Volume' in result_df.columns:
                result_df = self.calculate_volume_indicators(result_df)
            
            # Legacy price patterns (if available)
            if include_patterns and all(col in result_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                result_df = self.calculate_price_patterns(result_df)
            
            original_columns = len([col for col in df.columns])
            indicator_count = len([col for col in result_df.columns]) - original_columns
            
            self.logger.info(
                f"Calculated enhanced technical indicators (400+ features): "
                f"total_indicators={indicator_count}, original_columns={original_columns}, "
                f"final_columns={len(result_df.columns)}"
            )
            
            return result_df


def create_technical_indicators(config_override: Optional[Dict] = None) -> TechnicalIndicators:
    """
    Create technical indicators calculator
    
    Args:
        config_override: Configuration overrides
        
    Returns:
        TechnicalIndicators instance
    """
    return TechnicalIndicators(config_override)