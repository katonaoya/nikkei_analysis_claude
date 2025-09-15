"""
Fundamental and External Factors Features Module
Provides comprehensive fundamental analysis and external factor features for enhanced stock prediction
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


class FundamentalFeatures:
    """Generate comprehensive fundamental and external factor features - Enhanced for 400+ features"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize fundamental features generator
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("fundamental_features")
        
        # Market cap tiers (in billion JPY)
        self.market_cap_tiers = {
            'mega': 3000,    # > 3T yen
            'large': 1000,   # 1-3T yen  
            'mid': 300,      # 300B-1T yen
            'small': 100     # 100-300B yen
        }
        
        # Sector-specific characteristics
        self.sector_characteristics = self._get_sector_characteristics()
        
    def _get_sector_characteristics(self) -> Dict[str, Dict[str, float]]:
        """Get sector-specific financial characteristics"""
        return {
            'Technology': {
                'typical_per': 25.0,
                'typical_pbr': 3.5,
                'typical_roe': 15.0,
                'growth_premium': 1.2
            },
            'Financial': {
                'typical_per': 12.0,
                'typical_pbr': 0.8,
                'typical_roe': 8.0,
                'growth_premium': 0.9
            },
            'Automotive': {
                'typical_per': 8.0,
                'typical_pbr': 0.7,
                'typical_roe': 6.0,
                'growth_premium': 0.8
            },
            'Consumer': {
                'typical_per': 16.0,
                'typical_pbr': 1.5,
                'typical_roe': 10.0,
                'growth_premium': 1.0
            },
            'Manufacturing': {
                'typical_per': 14.0,
                'typical_pbr': 1.2,
                'typical_roe': 9.0,
                'growth_premium': 0.9
            },
            'Energy': {
                'typical_per': 10.0,
                'typical_pbr': 1.0,
                'typical_roe': 7.0,
                'growth_premium': 0.8
            },
            'Pharma': {
                'typical_per': 20.0,
                'typical_pbr': 2.5,
                'typical_roe': 12.0,
                'growth_premium': 1.1
            }
        }
    
    def calculate_market_cap_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate market capitalization and size-based features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with market cap features
        """
        with with_error_context("calculating market cap features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Approximate market cap calculation (simplified)
            # Note: In real implementation, would use actual shares outstanding
            stock_codes = result_df[group_by].unique()
            
            # Estimated market caps for major Nikkei 225 stocks (in billion JPY)
            market_caps = {
                '7203': 25000,    # Toyota Motor
                '6758': 12000,    # Sony Group
                '9984': 8000,     # SoftBank Group
                '6861': 7000,     # Keyence
                '8001': 6500,     # Itochu
                '8058': 6000,     # Mitsubishi Corp
                '4519': 5500,     # Takeda
                '6098': 5000,     # Recruit
                '8306': 4500,     # MUFG
                '7974': 4000,     # Nintendo
                '9432': 3800,     # NTT
                '4502': 3500,     # Takeda Pharma
                '6501': 3200,     # Hitachi
                '7201': 3000,     # Nissan
                '8316': 2800,     # Sumitomo Mitsui
                '6954': 2500,     # Fanuc
                '9613': 2200,     # NTT Data
                '4063': 2000,     # Shin-Etsu Chemical
            }
            
            # Add market cap information
            result_df['market_cap_est'] = result_df[group_by].map(market_caps).fillna(1000)  # Default 1T yen
            
            # Market cap relative to price (adjustment factor)
            latest_prices = result_df.groupby(group_by)[price_col].last()
            base_prices = {code: 1000 for code in stock_codes}  # Assume base price of 1000 yen
            
            for code in stock_codes:
                if code in latest_prices.index:
                    current_price = latest_prices[code]
                    if not pd.isna(current_price) and current_price > 0:
                        adjustment = current_price / base_prices.get(code, 1000)
                        result_df.loc[result_df[group_by] == code, 'market_cap_adjusted'] = (
                            market_caps.get(code, 1000) * adjustment
                        )
            
            # Fill missing adjusted market caps
            result_df['market_cap_adjusted'] = result_df['market_cap_adjusted'].fillna(result_df['market_cap_est'])
            
            # Market cap tiers
            result_df['is_mega_cap'] = (result_df['market_cap_adjusted'] > self.market_cap_tiers['mega']).astype(int)
            result_df['is_large_cap'] = (
                (result_df['market_cap_adjusted'] > self.market_cap_tiers['large']) & 
                (result_df['market_cap_adjusted'] <= self.market_cap_tiers['mega'])
            ).astype(int)
            result_df['is_mid_cap'] = (
                (result_df['market_cap_adjusted'] > self.market_cap_tiers['mid']) & 
                (result_df['market_cap_adjusted'] <= self.market_cap_tiers['large'])
            ).astype(int)
            result_df['is_small_cap'] = (result_df['market_cap_adjusted'] <= self.market_cap_tiers['small']).astype(int)
            
            # Market cap rank
            result_df['market_cap_rank'] = result_df.groupby('Date')['market_cap_adjusted'].rank(
                method='min', ascending=False, pct=True
            )
            
            # Size premium/discount
            median_market_cap = result_df['market_cap_adjusted'].median()
            result_df['size_premium'] = np.log(result_df['market_cap_adjusted'] / median_market_cap)
            
            self.logger.info("Calculated market cap and size features")
            return result_df

    def calculate_valuation_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate valuation-based features (P/E, P/B, etc.)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with valuation features
        """
        with with_error_context("calculating valuation features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Add sector information for valuation context
            sector_mapping = {
                '7203': 'Automotive', '6758': 'Technology', '9984': 'Technology',
                '6861': 'Technology', '8001': 'Trading', '8058': 'Trading',
                '4519': 'Pharma', '6098': 'Technology', '8306': 'Financial',
                '7974': 'Technology', '9432': 'Technology', '4502': 'Pharma',
                '6501': 'Manufacturing', '7201': 'Automotive', '8316': 'Financial',
                '6954': 'Manufacturing', '9613': 'Technology', '4063': 'Manufacturing'
            }
            
            result_df['sector'] = result_df[group_by].map(sector_mapping).fillna('Other')
            
            # Estimated financial metrics (simplified for demonstration)
            # In production, this would come from fundamental data APIs
            estimated_metrics = {
                '7203': {'per': 8.5, 'pbr': 0.9, 'roe': 10.5, 'div_yield': 2.8},
                '6758': {'per': 18.2, 'pbr': 2.1, 'roe': 11.5, 'div_yield': 1.5},
                '9984': {'per': 12.3, 'pbr': 1.4, 'roe': 11.4, 'div_yield': 5.2},
                '6861': {'per': 35.6, 'pbr': 5.2, 'roe': 14.6, 'div_yield': 0.8},
                '8001': {'per': 9.8, 'pbr': 1.1, 'roe': 11.2, 'div_yield': 3.5},
                '8058': {'per': 8.7, 'pbr': 0.8, 'roe': 9.2, 'div_yield': 4.1},
                '4519': {'per': 45.2, 'pbr': 1.3, 'roe': 2.9, 'div_yield': 4.8},
                '6098': {'per': 28.4, 'pbr': 3.8, 'roe': 13.4, 'div_yield': 1.2},
            }
            
            # Default values by sector
            default_metrics = {
                'Technology': {'per': 25.0, 'pbr': 3.5, 'roe': 15.0, 'div_yield': 1.0},
                'Financial': {'per': 12.0, 'pbr': 0.8, 'roe': 8.0, 'div_yield': 3.0},
                'Automotive': {'per': 8.0, 'pbr': 0.7, 'roe': 6.0, 'div_yield': 3.5},
                'Trading': {'per': 9.0, 'pbr': 1.0, 'roe': 10.0, 'div_yield': 4.0},
                'Pharma': {'per': 20.0, 'pbr': 2.5, 'roe': 12.0, 'div_yield': 2.5},
                'Manufacturing': {'per': 14.0, 'pbr': 1.2, 'roe': 9.0, 'div_yield': 2.0},
                'Other': {'per': 15.0, 'pbr': 1.5, 'roe': 10.0, 'div_yield': 2.0}
            }
            
            # Assign valuation metrics
            for metric in ['per', 'pbr', 'roe', 'div_yield']:
                result_df[f'est_{metric}'] = result_df.apply(
                    lambda row: estimated_metrics.get(row[group_by], {}).get(
                        metric, default_metrics[row['sector']][metric]
                    ), axis=1
                )
            
            # Valuation relative to sector
            sector_medians = result_df.groupby('sector')[['est_per', 'est_pbr', 'est_roe', 'est_div_yield']].median()
            
            for metric in ['per', 'pbr', 'roe', 'div_yield']:
                result_df[f'{metric}_vs_sector'] = result_df.apply(
                    lambda row: row[f'est_{metric}'] / sector_medians.loc[row['sector'], f'est_{metric}'], axis=1
                )
                
                # Valuation quintiles within sector
                result_df[f'{metric}_sector_quintile'] = result_df.groupby('sector')[f'est_{metric}'].transform(
                    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
                ).fillna(3)  # Default to middle quintile
            
            # Growth vs Value classification
            result_df['is_growth_stock'] = (
                (result_df['per_vs_sector'] > 1.2) & 
                (result_df['est_roe'] > result_df.groupby('sector')['est_roe'].transform('median'))
            ).astype(int)
            
            result_df['is_value_stock'] = (
                (result_df['per_vs_sector'] < 0.8) | 
                (result_df['pbr_vs_sector'] < 0.8)
            ).astype(int)
            
            result_df['is_dividend_stock'] = (
                result_df['est_div_yield'] > result_df.groupby('sector')['est_div_yield'].transform('median') * 1.5
            ).astype(int)
            
            # Earnings quality indicators (simplified)
            result_df['high_roe'] = (result_df['est_roe'] > 15).astype(int)
            result_df['low_roe'] = (result_df['est_roe'] < 5).astype(int)
            
            # Price-to-fundamentals ratios
            result_df['price_to_roe'] = result_df['est_per'] / (result_df['est_roe'] + 1e-10)
            result_df['yield_spread'] = result_df['est_div_yield'] - 2.0  # vs risk-free rate approximation
            
            self.logger.info("Calculated valuation features")
            return result_df

    def calculate_growth_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate growth and momentum features
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with growth features
        """
        with with_error_context("calculating growth features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Sort data
            if group_by and group_by in result_df.columns:
                result_df = result_df.sort_values([group_by, 'Date'])
            else:
                result_df = result_df.sort_values('Date')
            
            # Estimated growth rates (simplified)
            estimated_growth = {
                '7203': {'sales_growth': 3.5, 'earnings_growth': 5.2, 'guidance_revision': 0.1},
                '6758': {'sales_growth': 6.8, 'earnings_growth': 12.5, 'guidance_revision': 0.3},
                '9984': {'sales_growth': -2.1, 'earnings_growth': -15.6, 'guidance_revision': -0.2},
                '6861': {'sales_growth': 8.9, 'earnings_growth': 15.3, 'guidance_revision': 0.5},
                '8001': {'sales_growth': 12.3, 'earnings_growth': 18.7, 'guidance_revision': 0.4},
                '8058': {'sales_growth': 15.2, 'earnings_growth': 22.1, 'guidance_revision': 0.3},
                '4519': {'sales_growth': 2.8, 'earnings_growth': -8.9, 'guidance_revision': -0.1},
                '6098': {'sales_growth': 9.4, 'earnings_growth': 16.8, 'guidance_revision': 0.2},
            }
            
            # Default growth rates by sector
            sector_growth = {
                'Technology': {'sales_growth': 8.0, 'earnings_growth': 12.0, 'guidance_revision': 0.2},
                'Financial': {'sales_growth': 3.0, 'earnings_growth': 5.0, 'guidance_revision': 0.0},
                'Automotive': {'sales_growth': 2.0, 'earnings_growth': 3.0, 'guidance_revision': -0.1},
                'Trading': {'sales_growth': 10.0, 'earnings_growth': 15.0, 'guidance_revision': 0.3},
                'Pharma': {'sales_growth': 4.0, 'earnings_growth': 2.0, 'guidance_revision': 0.0},
                'Manufacturing': {'sales_growth': 5.0, 'earnings_growth': 7.0, 'guidance_revision': 0.1},
                'Other': {'sales_growth': 4.0, 'earnings_growth': 6.0, 'guidance_revision': 0.0}
            }
            
            # Add sector information if not present
            if 'sector' not in result_df.columns:
                sector_mapping = {
                    '7203': 'Automotive', '6758': 'Technology', '9984': 'Technology',
                    '6861': 'Technology', '8001': 'Trading', '8058': 'Trading',
                    '4519': 'Pharma', '6098': 'Technology', '8306': 'Financial',
                }
                result_df['sector'] = result_df[group_by].map(sector_mapping).fillna('Other')
            
            # Assign growth metrics
            for metric in ['sales_growth', 'earnings_growth', 'guidance_revision']:
                result_df[f'est_{metric}'] = result_df.apply(
                    lambda row: estimated_growth.get(row[group_by], {}).get(
                        metric, sector_growth[row['sector']][metric]
                    ), axis=1
                )
            
            # Growth quality indicators
            result_df['high_growth'] = (
                (result_df['est_sales_growth'] > 10) & 
                (result_df['est_earnings_growth'] > 15)
            ).astype(int)
            
            result_df['stable_growth'] = (
                (result_df['est_sales_growth'] > 3) & 
                (result_df['est_sales_growth'] < 10) &
                (result_df['est_earnings_growth'] > 5) & 
                (result_df['est_earnings_growth'] < 15)
            ).astype(int)
            
            result_df['declining_growth'] = (
                (result_df['est_sales_growth'] < 0) | 
                (result_df['est_earnings_growth'] < 0)
            ).astype(int)
            
            # Guidance revision momentum
            result_df['positive_revision'] = (result_df['est_guidance_revision'] > 0.2).astype(int)
            result_df['negative_revision'] = (result_df['est_guidance_revision'] < -0.1).astype(int)
            
            # Growth vs sector comparison
            sector_median_growth = result_df.groupby('sector')[['est_sales_growth', 'est_earnings_growth']].median()
            
            result_df['sales_growth_vs_sector'] = result_df.apply(
                lambda row: row['est_sales_growth'] / (
                    sector_median_growth.loc[row['sector'], 'est_sales_growth'] + 1e-10
                ), axis=1
            )
            
            result_df['earnings_growth_vs_sector'] = result_df.apply(
                lambda row: row['est_earnings_growth'] / (
                    sector_median_growth.loc[row['sector'], 'est_earnings_growth'] + 1e-10
                ), axis=1
            )
            
            # Growth momentum (price performance vs growth)
            if group_by and group_by in result_df.columns:
                result_df['price_return_20d'] = result_df.groupby(group_by)[price_col].pct_change(20)
            else:
                result_df['price_return_20d'] = result_df[price_col].pct_change(20)
            
            result_df['growth_momentum'] = result_df['price_return_20d'] * result_df['est_earnings_growth']
            
            # Analyst sentiment proxy
            result_df['analyst_optimism'] = (
                result_df['est_guidance_revision'] * 10 + 
                np.random.normal(0, 0.1, len(result_df))  # Add some noise for realism
            ).clip(-1, 1)
            
            self.logger.info("Calculated growth features")
            return result_df

    def calculate_external_factor_features(
        self,
        df: pd.DataFrame,
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate external factor features (macro, currency, commodities)
        
        Args:
            df: DataFrame with price data
            group_by: Column to group by
            
        Returns:
            DataFrame with external factor features
        """
        with with_error_context("calculating external factor features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['Date']):
                result_df['Date'] = pd.to_datetime(result_df['Date'])
            
            # Simulated external factors (in production, these would come from external APIs)
            date_range = result_df['Date'].unique()
            
            # Generate synthetic macro indicators
            np.random.seed(42)  # For reproducibility
            n_dates = len(date_range)
            
            # USD/JPY exchange rate simulation
            usdjpy_base = 145.0
            usdjpy_changes = np.cumsum(np.random.normal(0, 0.5, n_dates))
            usdjpy_values = usdjpy_base + usdjpy_changes
            
            # 10-year JGB yield simulation
            jgb_base = 0.7
            jgb_changes = np.cumsum(np.random.normal(0, 0.05, n_dates))
            jgb_values = np.maximum(0, jgb_base + jgb_changes)
            
            # Oil price simulation (Brent crude proxy)
            oil_base = 85.0
            oil_changes = np.cumsum(np.random.normal(0, 2.0, n_dates))
            oil_values = np.maximum(20, oil_base + oil_changes)
            
            # Gold price simulation
            gold_base = 2000.0
            gold_changes = np.cumsum(np.random.normal(0, 15.0, n_dates))
            gold_values = np.maximum(1500, gold_base + gold_changes)
            
            # VIX proxy (Nikkei VI simulation)
            vix_base = 20.0
            vix_values = np.maximum(10, vix_base + np.random.gamma(2, 2, n_dates))
            
            # Create external factors dataframe
            external_df = pd.DataFrame({
                'Date': date_range,
                'usdjpy': usdjpy_values,
                'jgb_10y': jgb_values,
                'oil_price': oil_values,
                'gold_price': gold_values,
                'nikkei_vi': vix_values
            })
            
            # Merge external factors
            result_df = result_df.merge(external_df, on='Date', how='left')
            
            # Currency features
            result_df['usdjpy_change_1d'] = result_df['usdjpy'].pct_change(1)
            result_df['usdjpy_change_5d'] = result_df['usdjpy'].pct_change(5)
            result_df['usdjpy_change_20d'] = result_df['usdjpy'].pct_change(20)
            
            # Currency regime
            result_df['usdjpy_ma_20'] = result_df['usdjpy'].rolling(20).mean()
            result_df['yen_weakening'] = (result_df['usdjpy'] > result_df['usdjpy_ma_20']).astype(int)
            result_df['yen_strengthening'] = (result_df['usdjpy'] < result_df['usdjpy_ma_20']).astype(int)
            
            # Interest rate features
            result_df['jgb_change_1d'] = result_df['jgb_10y'].diff(1)
            result_df['jgb_change_5d'] = result_df['jgb_10y'].diff(5)
            result_df['rising_rates'] = (result_df['jgb_change_5d'] > 0.05).astype(int)
            result_df['falling_rates'] = (result_df['jgb_change_5d'] < -0.05).astype(int)
            
            # Yield curve steepness proxy
            result_df['yield_curve_steepness'] = result_df['jgb_10y'] - 0.1  # vs short rate proxy
            
            # Commodity features
            result_df['oil_change_1d'] = result_df['oil_price'].pct_change(1)
            result_df['oil_change_5d'] = result_df['oil_price'].pct_change(5)
            result_df['oil_change_20d'] = result_df['oil_price'].pct_change(20)
            
            result_df['gold_change_1d'] = result_df['gold_price'].pct_change(1)
            result_df['gold_change_5d'] = result_df['gold_price'].pct_change(5)
            result_df['gold_change_20d'] = result_df['gold_price'].pct_change(20)
            
            # Risk sentiment indicators
            result_df['vix_change_1d'] = result_df['nikkei_vi'].pct_change(1)
            result_df['vix_change_5d'] = result_df['nikkei_vi'].pct_change(5)
            
            result_df['high_vix'] = (result_df['nikkei_vi'] > 25).astype(int)
            result_df['low_vix'] = (result_df['nikkei_vi'] < 15).astype(int)
            result_df['vix_spike'] = (result_df['vix_change_1d'] > 0.2).astype(int)
            
            # Risk-on/Risk-off indicators
            result_df['risk_on'] = (
                (result_df['nikkei_vi'] < 20) & 
                (result_df['oil_change_5d'] > 0) &
                (result_df['usdjpy_change_5d'] > 0)
            ).astype(int)
            
            result_df['risk_off'] = (
                (result_df['nikkei_vi'] > 25) & 
                (result_df['gold_change_5d'] > 0.02) &
                (result_df['usdjpy_change_5d'] < -0.01)
            ).astype(int)
            
            # Sector-specific external sensitivities
            if 'sector' not in result_df.columns:
                sector_mapping = {
                    '7203': 'Automotive', '6758': 'Technology', '9984': 'Technology',
                    '6861': 'Technology', '8001': 'Trading', '8058': 'Trading',
                }
                result_df['sector'] = result_df[group_by].map(sector_mapping).fillna('Other')
            
            # Currency sensitivity by sector
            currency_sensitivity = {
                'Automotive': 1.5,    # High export sensitivity
                'Technology': 1.2,    # Moderate export sensitivity  
                'Trading': 0.8,       # Less direct sensitivity
                'Financial': -0.5,    # Inverse sensitivity
                'Other': 1.0
            }
            
            result_df['currency_sensitivity'] = result_df['sector'].map(currency_sensitivity).fillna(1.0)
            result_df['currency_impact'] = result_df['usdjpy_change_5d'] * result_df['currency_sensitivity']
            
            # Oil sensitivity by sector
            oil_sensitivity = {
                'Automotive': -0.8,   # Negative impact from high oil
                'Energy': 1.5,        # Positive impact from high oil
                'Trading': 0.3,       # Some positive impact
                'Transportation': -1.2,  # Negative impact
                'Other': -0.2
            }
            
            result_df['oil_sensitivity'] = result_df['sector'].map(oil_sensitivity).fillna(-0.2)
            result_df['oil_impact'] = result_df['oil_change_5d'] * result_df['oil_sensitivity']
            
            self.logger.info("Calculated external factor features")
            return result_df

    def calculate_event_calendar_features(
        self,
        df: pd.DataFrame,
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate event and calendar-based features
        
        Args:
            df: DataFrame with price data
            group_by: Column to group by
            
        Returns:
            DataFrame with event features
        """
        with with_error_context("calculating event calendar features"):
            if df.empty:
                return df
                
            result_df = df.copy()
            
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['Date']):
                result_df['Date'] = pd.to_datetime(result_df['Date'])
            
            # Earnings season indicators (Japanese fiscal year: April-March)
            result_df['month'] = result_df['Date'].dt.month
            result_df['is_earnings_season'] = result_df['month'].isin([5, 8, 11, 2]).astype(int)  # Quarterly reports
            result_df['is_annual_earnings'] = result_df['month'].isin([5, 6]).astype(int)  # Annual reports
            
            # Earnings announcement proximity (simplified)
            # In practice, this would use actual earnings calendar
            earnings_months = [5, 8, 11, 2]
            result_df['days_to_earnings'] = 999  # Default large value
            
            for month in earnings_months:
                # Approximate days to next earnings for each quarter
                if month == 5:  # May earnings
                    mask = result_df['month'].isin([3, 4, 5])
                    result_df.loc[mask, 'days_to_earnings'] = np.minimum(
                        result_df.loc[mask, 'days_to_earnings'],
                        ((pd.to_datetime(result_df.loc[mask, 'Date'].dt.year.astype(str) + '-05-15') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
                elif month == 8:  # August earnings  
                    mask = result_df['month'].isin([6, 7, 8])
                    result_df.loc[mask, 'days_to_earnings'] = np.minimum(
                        result_df.loc[mask, 'days_to_earnings'],
                        ((pd.to_datetime(result_df.loc[mask, 'Date'].dt.year.astype(str) + '-08-15') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
                elif month == 11:  # November earnings
                    mask = result_df['month'].isin([9, 10, 11])
                    result_df.loc[mask, 'days_to_earnings'] = np.minimum(
                        result_df.loc[mask, 'days_to_earnings'],
                        ((pd.to_datetime(result_df.loc[mask, 'Date'].dt.year.astype(str) + '-11-15') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
                elif month == 2:  # February earnings
                    mask = result_df['month'].isin([12, 1, 2])
                    result_df.loc[mask, 'days_to_earnings'] = np.minimum(
                        result_df.loc[mask, 'days_to_earnings'],
                        ((pd.to_datetime((result_df.loc[mask, 'Date'].dt.year + 1).astype(str) + '-02-15') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
            
            # Earnings announcement windows
            result_df['earnings_window'] = (result_df['days_to_earnings'] <= 7).astype(int)
            result_df['pre_earnings'] = (
                (result_df['days_to_earnings'] > 7) & 
                (result_df['days_to_earnings'] <= 30)
            ).astype(int)
            
            # Dividend ex-date proximity (simplified)
            # Japanese companies typically pay dividends in March and September
            div_months = [3, 9]
            result_df['days_to_dividend'] = 999
            
            for month in div_months:
                if month == 3:
                    mask = result_df['month'].isin([1, 2, 3])
                    result_df.loc[mask, 'days_to_dividend'] = np.minimum(
                        result_df.loc[mask, 'days_to_dividend'],
                        ((pd.to_datetime(result_df.loc[mask, 'Date'].dt.year.astype(str) + '-03-31') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
                elif month == 9:
                    mask = result_df['month'].isin([7, 8, 9])
                    result_df.loc[mask, 'days_to_dividend'] = np.minimum(
                        result_df.loc[mask, 'days_to_dividend'],
                        ((pd.to_datetime(result_df.loc[mask, 'Date'].dt.year.astype(str) + '-09-30') - result_df.loc[mask, 'Date']).dt.days % 365)
                    )
            
            result_df['dividend_window'] = (result_df['days_to_dividend'] <= 5).astype(int)
            result_df['pre_dividend'] = (
                (result_df['days_to_dividend'] > 5) & 
                (result_df['days_to_dividend'] <= 20)
            ).astype(int)
            
            # Stock split/merger events (randomized for demonstration)
            np.random.seed(42)
            result_df['corporate_action_prob'] = np.random.random(len(result_df))
            result_df['potential_corporate_action'] = (result_df['corporate_action_prob'] < 0.02).astype(int)
            
            # Index rebalancing events (quarterly)
            result_df['index_rebalance'] = result_df['month'].isin([3, 6, 9, 12]).astype(int)
            
            # Holiday effects
            result_df['is_golden_week'] = (
                (result_df['month'] == 5) & 
                (result_df['Date'].dt.day <= 7)
            ).astype(int)
            
            result_df['is_year_end'] = (
                (result_df['month'] == 12) & 
                (result_df['Date'].dt.day >= 25)
            ).astype(int)
            
            # Analyst coverage events (simplified)
            result_df['high_analyst_activity'] = (
                result_df['earnings_window'] | 
                result_df['is_earnings_season']
            ).astype(int)
            
            self.logger.info("Calculated event and calendar features")
            return result_df

    def calculate_all_fundamental_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        group_by: str = 'Code'
    ) -> pd.DataFrame:
        """
        Calculate all fundamental and external factor features (100+ features)
        
        Args:
            df: DataFrame with price data
            price_col: Close price column
            group_by: Column to group by
            
        Returns:
            DataFrame with all fundamental features
        """
        result_df = df.copy()
        
        self.logger.info("Starting calculation of all fundamental features...")
        
        # Market cap and size features
        result_df = self.calculate_market_cap_features(result_df, price_col, group_by)
        
        # Valuation features
        result_df = self.calculate_valuation_features(result_df, price_col, group_by)
        
        # Growth features
        result_df = self.calculate_growth_features(result_df, price_col, group_by)
        
        # External factor features
        result_df = self.calculate_external_factor_features(result_df, group_by)
        
        # Event and calendar features
        result_df = self.calculate_event_calendar_features(result_df, group_by)
        
        original_columns = len(df.columns)
        feature_count = len(result_df.columns) - original_columns
        
        self.logger.info(
            f"Calculated all fundamental features (100+ features)",
            total_features=feature_count,
            original_columns=original_columns,
            final_columns=len(result_df.columns)
        )
        
        return result_df