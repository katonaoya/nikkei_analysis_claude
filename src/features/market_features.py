"""
Market environment features for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from datetime import datetime, date, timedelta
from pathlib import Path
import warnings
import logging


class MarketFeatures:
    """Generate comprehensive market environment features - Enhanced with 200+ features"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize market features generator
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir or Path("data/raw")
        self.logger = logging.getLogger(__name__)
        
        # Enhanced feature configuration
        self.volatility_windows = [5, 10, 20, 30, 60]
        self.ma_periods = [5, 10, 20, 60, 120]
        self.correlation_windows = [10, 20, 60]
        self.sector_mapping = self._get_default_sector_mapping()
        
    def _get_default_sector_mapping(self) -> Dict[str, str]:
        """Get default sector mapping for Nikkei 225 stocks"""
        return {
            # Technology
            '6758': 'Technology', '9984': 'Technology', '6861': 'Technology', 
            '6098': 'Technology', '4751': 'Technology', '4689': 'Technology',
            '6701': 'Technology', '6702': 'Technology', '6981': 'Technology',
            
            # Automotive
            '7203': 'Automotive', '7201': 'Automotive', '7202': 'Automotive',
            '7267': 'Automotive', '7269': 'Automotive', '7270': 'Automotive',
            
            # Financial
            '8306': 'Financial', '8316': 'Financial', '8411': 'Financial',
            '8766': 'Financial', '8802': 'Financial', '8801': 'Financial',
            
            # Trading
            '8001': 'Trading', '8002': 'Trading', '8015': 'Trading',
            '8020': 'Trading', '8031': 'Trading', '8053': 'Trading',
            
            # Manufacturing
            '6103': 'Manufacturing', '6113': 'Manufacturing', '6178': 'Manufacturing',
            '6269': 'Manufacturing', '6301': 'Manufacturing', '6302': 'Manufacturing',
            
            # Pharma
            '4519': 'Pharma', '4502': 'Pharma', '4503': 'Pharma',
            '4506': 'Pharma', '4507': 'Pharma', '4523': 'Pharma',
            
            # Consumer
            '2501': 'Consumer', '2502': 'Consumer', '2503': 'Consumer',
            '2531': 'Consumer', '2801': 'Consumer', '2802': 'Consumer',
            
            # Energy
            '5019': 'Energy', '5020': 'Energy', '5101': 'Energy',
            '5108': 'Energy', '5201': 'Energy', '5232': 'Energy',
            
            # Transportation
            '9001': 'Transportation', '9005': 'Transportation', '9007': 'Transportation',
            '9008': 'Transportation', '9009': 'Transportation', '9020': 'Transportation',
            
            # Utilities
            '9502': 'Utilities', '9503': 'Utilities', '9531': 'Utilities',
            '9532': 'Utilities', '9602': 'Utilities', '9613': 'Utilities',
        }

    def calculate_enhanced_volatility_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        volume_col: str = 'Volume',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive volatility features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column  
            low_col: Low price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with volatility features
        """
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
        
        # Historical Volatility for multiple windows
        for window in self.volatility_windows:
            vol_col = f'HV_{window}d'
            
            if group_by and group_by in result_df.columns:
                result_df[vol_col] = result_df.groupby(group_by)['returns'].rolling(
                    window=window, min_periods=max(1, window//2)
                ).std().reset_index(level=0, drop=True) * np.sqrt(252)
            else:
                result_df[vol_col] = result_df['returns'].rolling(
                    window=window, min_periods=max(1, window//2)
                ).std() * np.sqrt(252)
            
            # Volatility regime indicators
            result_df[f'{vol_col}_high'] = (
                result_df[vol_col] > result_df[vol_col].rolling(60).quantile(0.8)
            ).astype(int)
            
            result_df[f'{vol_col}_low'] = (
                result_df[vol_col] < result_df[vol_col].rolling(60).quantile(0.2)
            ).astype(int)
            
            # Volatility momentum
            result_df[f'{vol_col}_momentum'] = result_df[vol_col].pct_change(5)
        
        # True Range and ATR
        if all(col in result_df.columns for col in [high_col, low_col]):
            result_df['TR1'] = result_df[high_col] - result_df[low_col]
            result_df['TR2'] = abs(result_df[high_col] - result_df[price_col].shift(1))
            result_df['TR3'] = abs(result_df[low_col] - result_df[price_col].shift(1))
            result_df['True_Range'] = result_df[['TR1', 'TR2', 'TR3']].max(axis=1)
            
            for window in [14, 30]:
                atr_col = f'ATR_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[atr_col] = result_df.groupby(group_by)['True_Range'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[atr_col] = result_df['True_Range'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean()
                
                # ATR-based indicators
                result_df[f'{atr_col}_ratio'] = result_df[atr_col] / result_df[price_col]
                result_df[f'{atr_col}_percentile'] = result_df[atr_col].rolling(120).rank(pct=True)
            
            # Clean up intermediate columns
            result_df = result_df.drop(['TR1', 'TR2', 'TR3', 'True_Range'], axis=1)
        
        # Realized Volatility (intraday)
        if all(col in result_df.columns for col in [high_col, low_col]):
            # Garman-Klass estimator
            result_df['GK_vol'] = np.log(result_df[high_col] / result_df[low_col]) ** 2
            
            for window in [5, 20]:
                gk_vol_col = f'GK_vol_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[gk_vol_col] = result_df.groupby(group_by)['GK_vol'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean().reset_index(level=0, drop=True) * np.sqrt(252)
                else:
                    result_df[gk_vol_col] = result_df['GK_vol'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean() * np.sqrt(252)
        
        # Volume-Price Volatility
        if volume_col in result_df.columns:
            # Price-Volume correlation
            for window in [10, 20]:
                pv_corr_col = f'PV_corr_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[pv_corr_col] = result_df.groupby(group_by).rolling(window=window)[
                        ['returns', volume_col]
                    ].corr().iloc[0::2, -1].reset_index(level=[0,1], drop=True)
                else:
                    result_df[pv_corr_col] = result_df[['returns', volume_col]].rolling(window=window).corr().iloc[0::2, -1]
        
        # Volatility clustering
        result_df['vol_cluster'] = (
            (result_df['returns'].abs() > result_df['returns'].abs().rolling(20).quantile(0.8)) &
            (result_df['returns'].abs().shift(1) > result_df['returns'].abs().rolling(20).quantile(0.8))
        ).astype(int)
        
        self.logger.info(f"Calculated enhanced volatility features for {len(self.volatility_windows)} windows")
        return result_df

    def calculate_comprehensive_market_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        volume_col: str = 'Volume',
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive market regime features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with market regime features
        """
        result_df = df.copy()
        
        # Sort data
        if group_by and group_by in result_df.columns:
            result_df = result_df.sort_values([group_by, 'Date'])
        else:
            result_df = result_df.sort_values('Date')
        
        # Multiple MA periods for trend identification
        for short_ma in [10, 20]:
            for long_ma in [50, 120, 200]:
                if short_ma >= long_ma:
                    continue
                    
                trend_col = f'trend_{short_ma}_{long_ma}'
                
                if group_by and group_by in result_df.columns:
                    ma_short = result_df.groupby(group_by)[price_col].rolling(
                        window=short_ma, min_periods=max(1, short_ma//2)
                    ).mean().reset_index(level=0, drop=True)
                    
                    ma_long = result_df.groupby(group_by)[price_col].rolling(
                        window=long_ma, min_periods=max(1, long_ma//2)
                    ).mean().reset_index(level=0, drop=True)
                else:
                    ma_short = result_df[price_col].rolling(
                        window=short_ma, min_periods=max(1, short_ma//2)
                    ).mean()
                    
                    ma_long = result_df[price_col].rolling(
                        window=long_ma, min_periods=max(1, long_ma//2)
                    ).mean()
                
                # Trend strength
                result_df[trend_col] = (ma_short / ma_long - 1) * 100
                
                # Trend classification
                result_df[f'{trend_col}_bullish'] = (result_df[trend_col] > 2).astype(int)
                result_df[f'{trend_col}_bearish'] = (result_df[trend_col] < -2).astype(int)
                result_df[f'{trend_col}_sideways'] = (
                    (result_df[trend_col] >= -2) & (result_df[trend_col] <= 2)
                ).astype(int)
                
                # Trend momentum
                result_df[f'{trend_col}_momentum'] = result_df[trend_col].diff(5)
        
        # Market strength indicators
        if group_by and group_by in result_df.columns:
            # Daily number of advancing/declining stocks
            result_df['daily_return'] = result_df.groupby(group_by)[price_col].pct_change()
            
            daily_market_data = result_df.groupby('Date').agg({
                'daily_return': ['count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
                price_col: 'mean',
                volume_col: 'sum' if volume_col in result_df.columns else 'count'
            }).reset_index()
            
            daily_market_data.columns = ['Date', 'total_stocks', 'advancing_stocks', 'declining_stocks', 
                                       'market_price_avg', 'total_volume']
            
            # Advance-Decline indicators
            daily_market_data['AD_ratio'] = daily_market_data['advancing_stocks'] / (
                daily_market_data['declining_stocks'] + 1e-10
            )
            daily_market_data['AD_line'] = (
                daily_market_data['advancing_stocks'] - daily_market_data['declining_stocks']
            ).cumsum()
            
            # McClellan Oscillator approximation
            daily_market_data['AD_net'] = daily_market_data['advancing_stocks'] - daily_market_data['declining_stocks']
            daily_market_data['AD_ema_19'] = daily_market_data['AD_net'].ewm(span=19).mean()
            daily_market_data['AD_ema_39'] = daily_market_data['AD_net'].ewm(span=39).mean()
            daily_market_data['McClellan_Osc'] = daily_market_data['AD_ema_19'] - daily_market_data['AD_ema_39']
            
            # Market breadth
            daily_market_data['market_breadth'] = daily_market_data['advancing_stocks'] / daily_market_data['total_stocks']
            
            # Market momentum
            daily_market_data['market_momentum'] = daily_market_data['market_price_avg'].pct_change(5)
            
            # Merge back to main dataframe
            merge_cols = ['Date', 'AD_ratio', 'AD_line', 'McClellan_Osc', 'market_breadth', 'market_momentum']
            result_df = result_df.merge(daily_market_data[merge_cols], on='Date', how='left')
            
            # Market regime based on breadth
            result_df['strong_market'] = (result_df['market_breadth'] > 0.6).astype(int)
            result_df['weak_market'] = (result_df['market_breadth'] < 0.4).astype(int)
            result_df['neutral_market'] = (
                (result_df['market_breadth'] >= 0.4) & (result_df['market_breadth'] <= 0.6)
            ).astype(int)
        
        self.logger.info("Calculated comprehensive market regime features")
        return result_df

    def calculate_advanced_relative_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        volume_col: str = 'Volume',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate advanced relative strength and correlation features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with relative features
        """
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
        
        # Market average performance (Nikkei 225 proxy)
        market_data = result_df.groupby('Date').agg({
            'returns': 'mean',
            price_col: 'mean',
            volume_col: 'sum' if volume_col in result_df.columns else 'count'
        }).reset_index()
        market_data.columns = ['Date', 'market_return', 'market_price', 'market_volume']
        
        # Merge market data
        result_df = result_df.merge(market_data, on='Date', how='left')
        
        # Relative performance measures
        for period in [1, 5, 10, 20]:
            if group_by and group_by in result_df.columns:
                stock_perf = result_df.groupby(group_by)['returns'].rolling(
                    window=period, min_periods=1
                ).mean().reset_index(level=0, drop=True)
                
                market_perf = result_df['market_return'].rolling(
                    window=period, min_periods=1
                ).mean()
            else:
                stock_perf = result_df['returns'].rolling(window=period, min_periods=1).mean()
                market_perf = result_df['market_return'].rolling(window=period, min_periods=1).mean()
            
            result_df[f'rel_perf_{period}d'] = stock_perf - market_perf
            result_df[f'rel_strength_{period}d'] = stock_perf / (market_perf + 1e-10)
        
        # Beta calculation (rolling)
        for window in [20, 60, 120]:
            beta_col = f'beta_{window}d'
            
            if group_by and group_by in result_df.columns:
                def calc_beta(group_df):
                    aligned_data = pd.concat([
                        group_df['returns'].rename('stock'),
                        group_df['market_return'].rename('market')
                    ], axis=1).dropna()
                    
                    if len(aligned_data) < 5:  # Minimum observations for beta
                        return pd.Series(index=group_df.index, data=np.nan)
                    
                    rolling_beta = []
                    for i in range(len(aligned_data)):
                        if i < window - 1:
                            rolling_beta.append(np.nan)
                        else:
                            start_idx = max(0, i - window + 1)
                            window_data = aligned_data.iloc[start_idx:i+1]
                            
                            if window_data['market'].var() > 1e-10:  # Avoid division by zero
                                beta = window_data['stock'].cov(window_data['market']) / window_data['market'].var()
                            else:
                                beta = 1.0
                            rolling_beta.append(beta)
                    
                    return pd.Series(index=aligned_data.index, data=rolling_beta)
                
                result_df[beta_col] = result_df.groupby(group_by).apply(
                    calc_beta
                ).reset_index(level=0, drop=True)
            else:
                # Simple case without grouping
                aligned_data = result_df[['returns', 'market_return']].dropna()
                rolling_beta = aligned_data['returns'].rolling(window=window).cov(
                    aligned_data['market_return']
                ) / aligned_data['market_return'].rolling(window=window).var()
                result_df[beta_col] = rolling_beta
            
            # Beta-based features
            result_df[f'{beta_col}_high'] = (result_df[beta_col] > 1.2).astype(int)
            result_df[f'{beta_col}_low'] = (result_df[beta_col] < 0.8).astype(int)
        
        # Correlation with market
        for window in self.correlation_windows:
            corr_col = f'market_corr_{window}d'
            
            if group_by and group_by in result_df.columns:
                def calc_correlation(group_df):
                    return group_df['returns'].rolling(window=window).corr(group_df['market_return'])
                
                result_df[corr_col] = result_df.groupby(group_by).apply(
                    calc_correlation
                ).reset_index(level=0, drop=True)
            else:
                result_df[corr_col] = result_df['returns'].rolling(window=window).corr(
                    result_df['market_return']
                )
            
            # Correlation regime
            result_df[f'{corr_col}_high'] = (result_df[corr_col] > 0.7).astype(int)
            result_df[f'{corr_col}_low'] = (result_df[corr_col] < 0.3).astype(int)
        
        # Rank-based features
        for period in [1, 5, 20]:
            perf_col = f'perf_{period}d'
            rank_col = f'rank_{period}d'
            
            if group_by and group_by in result_df.columns:
                result_df[perf_col] = result_df.groupby(group_by)['returns'].rolling(
                    window=period, min_periods=1
                ).mean().reset_index(level=0, drop=True)
                
                result_df[rank_col] = result_df.groupby('Date')[perf_col].rank(
                    method='min', ascending=False, pct=True
                )
            else:
                result_df[perf_col] = result_df['returns'].rolling(window=period, min_periods=1).mean()
                result_df[rank_col] = result_df[perf_col].rank(method='min', ascending=False, pct=True)
            
            # Rank-based signals
            result_df[f'{rank_col}_top_decile'] = (result_df[rank_col] <= 0.1).astype(int)
            result_df[f'{rank_col}_bottom_decile'] = (result_df[rank_col] >= 0.9).astype(int)
        
        self.logger.info("Calculated advanced relative strength and correlation features")
        return result_df

    def calculate_sector_rotation_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate sector rotation and style features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with sector features
        """
        result_df = df.copy()
        
        # Add sector information
        result_df['sector'] = result_df[group_by].map(self.sector_mapping).fillna('Other')
        
        # Calculate returns
        if group_by and group_by in result_df.columns:
            result_df['returns'] = result_df.groupby(group_by)[price_col].pct_change()
        else:
            result_df['returns'] = result_df[price_col].pct_change()
        
        # Sector performance
        for period in [1, 5, 20]:
            sector_perf_col = f'sector_perf_{period}d'
            
            if group_by and group_by in result_df.columns:
                stock_perf = result_df.groupby(group_by)['returns'].rolling(
                    window=period, min_periods=1
                ).mean().reset_index(level=0, drop=True)
            else:
                stock_perf = result_df['returns'].rolling(window=period, min_periods=1).mean()
            
            # Calculate sector average performance
            sector_data = result_df.copy()
            sector_data[f'stock_perf_{period}d'] = stock_perf
            
            sector_avg = sector_data.groupby(['Date', 'sector'])[f'stock_perf_{period}d'].mean().reset_index()
            sector_avg.columns = ['Date', 'sector', sector_perf_col]
            
            # Merge back
            result_df = result_df.merge(sector_avg, on=['Date', 'sector'], how='left')
            
            # Relative to sector performance
            result_df[f'rel_to_sector_{period}d'] = stock_perf - result_df[sector_perf_col]
            
            # Sector ranking
            result_df[f'sector_rank_{period}d'] = result_df.groupby('Date')[sector_perf_col].rank(
                method='min', ascending=False, pct=True
            )
        
        # Sector momentum and rotation
        result_df['sector_momentum'] = result_df['sector_perf_20d'].pct_change(5)
        
        # Leading/Lagging sector identification
        result_df['leading_sector'] = (result_df['sector_rank_20d'] <= 0.3).astype(int)
        result_df['lagging_sector'] = (result_df['sector_rank_20d'] >= 0.7).astype(int)
        
        # Sector strength vs market
        market_perf = result_df.groupby('Date')['returns'].mean().reset_index()
        market_perf.columns = ['Date', 'market_return']
        result_df = result_df.merge(market_perf, on='Date', how='left')
        
        result_df['sector_vs_market'] = result_df['sector_perf_20d'] - result_df['market_return']
        
        self.logger.info("Calculated sector rotation and style features")
        return result_df

    def calculate_macro_sentiment_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        volume_col: str = 'Volume',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate macro sentiment and external factor features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with macro sentiment features
        """
        result_df = df.copy()
        
        # VIX-like indicator (market fear gauge)
        if group_by and group_by in result_df.columns:
            result_df['returns'] = result_df.groupby(group_by)[price_col].pct_change()
            
            daily_vol = result_df.groupby('Date')['returns'].std().reset_index()
            daily_vol.columns = ['Date', 'market_volatility']
            
            result_df = result_df.merge(daily_vol, on='Date', how='left')
            
            # Fear/Greed indicators
            result_df['fear_index'] = result_df['market_volatility'].rolling(20).rank(pct=True)
            result_df['high_fear'] = (result_df['fear_index'] > 0.8).astype(int)
            result_df['extreme_fear'] = (result_df['fear_index'] > 0.95).astype(int)
            result_df['low_fear'] = (result_df['fear_index'] < 0.2).astype(int)
        
        # Risk-on/Risk-off sentiment
        if volume_col in result_df.columns:
            # High volume days (flight to quality indicator)
            result_df['volume_surge'] = (
                result_df[volume_col] > result_df[volume_col].rolling(20).quantile(0.9)
            ).astype(int)
            
            # Risk-on/off based on volume and price action
            result_df['risk_on'] = (
                (result_df['returns'] > 0) & 
                (result_df['volume_surge'] == 1)
            ).astype(int)
            
            result_df['risk_off'] = (
                (result_df['returns'] < 0) & 
                (result_df['volume_surge'] == 1)
            ).astype(int)
        
        # Market stress indicators
        result_df['price_dispersion'] = result_df.groupby('Date')['returns'].transform('std')
        result_df['high_dispersion'] = (
            result_df['price_dispersion'] > result_df['price_dispersion'].rolling(60).quantile(0.8)
        ).astype(int)
        
        # Seasonal effects (simplified)
        result_df['Date_dt'] = pd.to_datetime(result_df['Date'])
        result_df['month'] = result_df['Date_dt'].dt.month
        result_df['quarter'] = result_df['Date_dt'].dt.quarter
        result_df['is_january'] = (result_df['month'] == 1).astype(int)
        result_df['is_december'] = (result_df['month'] == 12).astype(int)
        result_df['is_q1'] = (result_df['quarter'] == 1).astype(int)
        result_df['is_q4'] = (result_df['quarter'] == 4).astype(int)
        
        # Week effects
        result_df['day_of_week'] = result_df['Date_dt'].dt.dayofweek
        result_df['is_monday'] = (result_df['day_of_week'] == 0).astype(int)
        result_df['is_friday'] = (result_df['day_of_week'] == 4).astype(int)
        
        # Month-end effects
        result_df['month_end'] = (
            result_df['Date_dt'].dt.day > 
            result_df['Date_dt'].dt.days_in_month - 3
        ).astype(int)
        
        # Clean up temporary columns
        result_df = result_df.drop(['Date_dt'], axis=1)
        
        self.logger.info("Calculated macro sentiment and external factor features")
        return result_df

    def calculate_liquidity_microstructure_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        volume_col: str = 'Volume',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate liquidity and microstructure features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with liquidity features
        """
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
        
        if volume_col in result_df.columns:
            # Amihud illiquidity measure
            result_df['amihud_illiq'] = abs(result_df['returns']) / (result_df[volume_col] + 1e-10)
            
            for window in [5, 20]:
                amihud_col = f'amihud_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[amihud_col] = result_df.groupby(group_by)['amihud_illiq'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[amihud_col] = result_df['amihud_illiq'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean()
                
                # Liquidity regime
                result_df[f'{amihud_col}_high'] = (
                    result_df[amihud_col] > result_df[amihud_col].rolling(60).quantile(0.8)
                ).astype(int)
            
            # Volume patterns
            result_df['volume_ma_20'] = result_df.groupby(group_by)[volume_col].rolling(
                window=20, min_periods=10
            ).mean().reset_index(level=0, drop=True) if group_by and group_by in result_df.columns else result_df[volume_col].rolling(20).mean()
            
            result_df['volume_ratio'] = result_df[volume_col] / (result_df['volume_ma_20'] + 1e-10)
            result_df['high_volume'] = (result_df['volume_ratio'] > 2.0).astype(int)
            result_df['low_volume'] = (result_df['volume_ratio'] < 0.5).astype(int)
            
            # Price impact measures
            result_df['price_impact'] = result_df['returns'] / np.log(result_df['volume_ratio'] + 1e-10)
        
        # Spread proxies (using High-Low)
        if all(col in result_df.columns for col in [high_col, low_col]):
            result_df['hl_spread'] = (result_df[high_col] - result_df[low_col]) / result_df[price_col]
            
            # Average spread
            for window in [5, 20]:
                spread_col = f'avg_spread_{window}d'
                if group_by and group_by in result_df.columns:
                    result_df[spread_col] = result_df.groupby(group_by)['hl_spread'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean().reset_index(level=0, drop=True)
                else:
                    result_df[spread_col] = result_df['hl_spread'].rolling(
                        window=window, min_periods=max(1, window//2)
                    ).mean()
        
        # Anomaly detection
        result_df['return_anomaly'] = (
            abs(result_df['returns']) > result_df['returns'].rolling(60).std() * 3
        ).astype(int)
        
        if volume_col in result_df.columns:
            result_df['volume_anomaly'] = (
                result_df['volume_ratio'] > result_df['volume_ratio'].rolling(60).quantile(0.95)
            ).astype(int)
        
        # Trading intensity
        result_df['trading_intensity'] = (
            (result_df.get('high_volume', 0) == 1) | 
            (result_df.get('return_anomaly', 0) == 1)
        ).astype(int)
        
        self.logger.info("Calculated liquidity and microstructure features")
        return result_df

    # Legacy methods for backward compatibility
    def calculate_market_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        window: int = 20,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method - calls enhanced version"""
        return self.calculate_enhanced_volatility_features(df, price_col, 'High', 'Low', 'Volume', group_by)
    
    def calculate_market_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        short_ma: int = 20,
        long_ma: int = 60,
        group_by: Optional[str] = 'Code'
    ) -> pd.DataFrame:
        """Legacy method - calls enhanced version"""
        return self.calculate_comprehensive_market_regime(df, price_col, 'Volume', group_by)
    
    def calculate_relative_strength(
        self,
        df: pd.DataFrame,
        benchmark_code: str = '1301',
        price_col: str = 'Close',
        window: int = 20,
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """Legacy method - calls enhanced version"""
        return self.calculate_advanced_relative_features(df, price_col, 'Volume', group_by)
    
    def calculate_market_breadth(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        ma_period: int = 20,
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """Legacy method - calls enhanced version"""
        return self.calculate_comprehensive_market_regime(df, price_col, 'Volume', group_by)
    
    def calculate_volume_features(
        self,
        df: pd.DataFrame,
        volume_col: str = 'Volume',
        window: int = 20,
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """Legacy method - calls enhanced version"""
        return self.calculate_liquidity_microstructure_features(df, 'Close', 'High', 'Low', volume_col, group_by)
    
    def calculate_all_enhanced_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        high_col: str = 'High',
        low_col: str = 'Low',
        volume_col: str = 'Volume',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate all enhanced market features (200+ features)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            high_col: High price column
            low_col: Low price column
            volume_col: Volume column
            group_by: Column to group by
            
        Returns:
            DataFrame with all market features
        """
        result_df = df.copy()
        
        self.logger.info("Starting calculation of all enhanced market features...")
        
        # Enhanced volatility features
        result_df = self.calculate_enhanced_volatility_features(
            result_df, price_col, high_col, low_col, volume_col, group_by
        )
        
        # Comprehensive market regime features
        result_df = self.calculate_comprehensive_market_regime(
            result_df, price_col, volume_col, group_by
        )
        
        # Advanced relative features
        result_df = self.calculate_advanced_relative_features(
            result_df, price_col, volume_col, group_by
        )
        
        # Sector rotation features
        result_df = self.calculate_sector_rotation_features(
            result_df, price_col, group_by
        )
        
        # Macro sentiment features
        result_df = self.calculate_macro_sentiment_features(
            result_df, price_col, volume_col, group_by
        )
        
        # Liquidity and microstructure features
        result_df = self.calculate_liquidity_microstructure_features(
            result_df, price_col, high_col, low_col, volume_col, group_by
        )
        
        original_columns = len(df.columns)
        feature_count = len(result_df.columns) - original_columns
        
        self.logger.info(
            f"Calculated all enhanced market features (200+ features)",
            total_features=feature_count,
            original_columns=original_columns,
            final_columns=len(result_df.columns)
        )
        
        return result_df