"""
Feature engineering pipeline for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, date
import warnings

from utils.config import get_config
from utils.logger import get_logger
from utils.error_handler import DataError, ValidationError, with_error_context
from src.data.data_preprocessor import create_data_preprocessor
from src.data.external_data_fetcher import create_external_data_fetcher
from .technical_indicators import create_technical_indicators
from .label_generator import create_label_generator


class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize feature pipeline
        
        Args:
            config_override: Configuration overrides
        """
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.logger = get_logger("feature_pipeline")
        
        # Initialize components
        self.preprocessor = create_data_preprocessor(config_override)
        self.external_fetcher = create_external_data_fetcher(config_override)
        self.technical_indicators = create_technical_indicators(config_override)
        self.label_generator = create_label_generator(config_override)
        
        # Configuration
        self.target_return = self.config.get('labels.target_return', 0.01)
        self.ma_periods = self.config.get('features.ma_periods', [5, 10, 20, 60, 120])
        
        # Feature selection configuration
        self.min_correlation_threshold = 0.05  # Minimum correlation with target
        self.max_correlation_threshold = 0.95  # Maximum correlation between features
    
    def create_basic_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_volume: bool = True,
        include_patterns: bool = True
    ) -> pd.DataFrame:
        """
        Create basic stock features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            include_technical: Include technical indicators
            include_volume: Include volume indicators
            include_patterns: Include price patterns
            
        Returns:
            DataFrame with basic features
        """
        with with_error_context("creating basic features"):
            if df.empty:
                return df
            
            # Validate required columns
            required_cols = ['Code', 'Date', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataError(f"Missing required columns: {missing_cols}")
            
            self.logger.info("Creating basic features", records=len(df))
            
            result_df = df.copy()
            
            # Sort data properly
            result_df = result_df.sort_values(['Code', 'Date'])
            
            # Basic price features
            result_df = self._create_price_features(result_df)
            
            # Technical indicators
            if include_technical:
                result_df = self.technical_indicators.calculate_all_indicators(
                    result_df,
                    include_patterns=include_patterns,
                    include_volume=include_volume
                )
            
            # Market environment features (from external data)
            try:
                result_df = self._add_market_environment_features(result_df)
            except Exception as e:
                self.logger.warning(f"Failed to add market environment features: {e}")
            
            # Feature engineering
            result_df = self._create_interaction_features(result_df)
            result_df = self._create_lag_features(result_df)
            result_df = self._create_rolling_features(result_df)
            
            feature_count = len([col for col in result_df.columns if col not in df.columns])
            
            self.logger.info(
                "Basic features created",
                new_features=feature_count,
                total_columns=len(result_df.columns)
            )
            
            return result_df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price-based features"""
        result_df = df.copy()
        
        # Daily returns
        result_df['Daily_Return'] = result_df.groupby('Code')['Close'].pct_change()
        
        # Log returns
        result_df['Log_Return'] = np.log(result_df['Close'] / result_df.groupby('Code')['Close'].shift(1))
        
        # Price momentum features
        for period in [1, 3, 5, 10, 20]:
            result_df[f'Return_{period}d'] = (
                result_df.groupby('Code')['Close'].pct_change(periods=period)
            )
            
            result_df[f'Price_Momentum_{period}d'] = (
                result_df['Close'] / result_df.groupby('Code')['Close'].shift(period) - 1
            )
        
        # Volatility features
        for window in [5, 10, 20]:
            result_df[f'Volatility_{window}d'] = (
                result_df.groupby('Code')['Daily_Return']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True) * np.sqrt(252)
            )
        
        # Price position features
        for window in [20, 60, 120]:
            result_df[f'High_{window}d'] = (
                result_df.groupby('Code')['High']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
            
            result_df[f'Low_{window}d'] = (
                result_df.groupby('Code')['Low']
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )
            
            # Position within range
            result_df[f'Price_Position_{window}d'] = (
                (result_df['Close'] - result_df[f'Low_{window}d']) / 
                (result_df[f'High_{window}d'] - result_df[f'Low_{window}d'] + 1e-10)
            )
        
        return result_df
    
    def _add_market_environment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market environment features from external data"""
        result_df = df.copy()
        
        if df.empty:
            return result_df
        
        # Get date range
        start_date = df['Date'].min() - pd.Timedelta(days=30)
        end_date = df['Date'].max()
        
        # Create market environment features
        target_dates = pd.DatetimeIndex(df['Date'].unique()).sort_values()
        
        try:
            market_features = self.external_fetcher.create_market_environment_features(
                target_dates=target_dates,
                lookback_days=30
            )
            
            if not market_features.empty:
                # Merge with main DataFrame
                result_df = pd.merge(
                    result_df,
                    market_features,
                    left_on='Date',
                    right_index=True,
                    how='left'
                )
                
                self.logger.info(
                    "Market environment features added",
                    features_count=len(market_features.columns)
                )
            
        except Exception as e:
            self.logger.warning(f"Could not create market environment features: {e}")
        
        return result_df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators"""
        result_df = df.copy()
        
        # Price-Volume interactions
        if 'Volume_Ratio' in result_df.columns and 'Daily_Return' in result_df.columns:
            result_df['PriceVol_Interaction'] = (
                result_df['Daily_Return'] * result_df['Volume_Ratio']
            )
        
        # MA crossovers
        if 'MA_5' in result_df.columns and 'MA_20' in result_df.columns:
            result_df['MA_5_20_Ratio'] = result_df['MA_5'] / result_df['MA_20']
            result_df['MA_5_20_Cross_Bull'] = (
                (result_df['MA_5'] > result_df['MA_20']) &
                (result_df['MA_5'].shift(1) <= result_df['MA_20'].shift(1))
            ).astype(int)
        
        if 'MA_10' in result_df.columns and 'MA_60' in result_df.columns:
            result_df['MA_10_60_Ratio'] = result_df['MA_10'] / result_df['MA_60']
        
        # RSI and price momentum
        if 'RSI' in result_df.columns and 'Return_5d' in result_df.columns:
            result_df['RSI_Momentum_Interaction'] = (
                result_df['RSI'] / 50 - 1) * result_df['Return_5d']
        
        # Bollinger Bands and volatility
        if 'BB_Position' in result_df.columns and 'Volatility_10d' in result_df.columns:
            result_df['BB_Vol_Interaction'] = (
                result_df['BB_Position'] * result_df['Volatility_10d']
            )
        
        return result_df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for key indicators"""
        result_df = df.copy()
        
        # Key features to lag
        lag_features = [
            'Daily_Return', 'RSI', 'MACD', 'BB_Position',
            'Volume_Ratio', 'Volatility_5d'
        ]
        
        # Lag periods
        lag_periods = [1, 3, 5]
        
        for feature in lag_features:
            if feature in result_df.columns:
                for lag in lag_periods:
                    result_df[f'{feature}_lag_{lag}'] = (
                        result_df.groupby('Code')[feature].shift(lag)
                    )
        
        return result_df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features"""
        result_df = df.copy()
        
        # Rolling features for key indicators
        roll_features = ['Daily_Return', 'Volume_Ratio', 'RSI']
        windows = [5, 10, 20]
        
        for feature in roll_features:
            if feature not in result_df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                result_df[f'{feature}_mean_{window}d'] = (
                    result_df.groupby('Code')[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Rolling std
                result_df[f'{feature}_std_{window}d'] = (
                    result_df.groupby('Code')[feature]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                
                # Z-score (current vs rolling mean)
                result_df[f'{feature}_zscore_{window}d'] = (
                    (result_df[feature] - result_df[f'{feature}_mean_{window}d']) / 
                    (result_df[f'{feature}_std_{window}d'] + 1e-10)
                )
        
        return result_df
    
    def create_targets(
        self,
        df: pd.DataFrame,
        target_types: List[str] = ['basic', 'volatility_adjusted'],
        target_return: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Create target variables
        
        Args:
            df: DataFrame with features
            target_types: List of target types to create
            target_return: Target return threshold
            
        Returns:
            DataFrame with targets added
        """
        with with_error_context("creating targets"):
            if df.empty:
                return df
            
            if target_return is None:
                target_return = self.target_return
            
            result_df = df.copy()
            
            # Basic next-day return target
            if 'basic' in target_types:
                result_df = self.label_generator.create_next_day_return_labels(
                    result_df,
                    target_return=target_return
                )
            
            # Volatility-adjusted target
            if 'volatility_adjusted' in target_types:
                result_df = self.label_generator.create_volatility_adjusted_labels(
                    result_df,
                    volatility_multiplier=1.5
                )
            
            # Multi-horizon targets
            if 'multi_horizon' in target_types:
                result_df = self.label_generator.create_multi_horizon_labels(
                    result_df,
                    horizons=[1, 3, 5],
                    target_return=target_return
                )
            
            # Regime-based targets
            if 'regime' in target_types:
                result_df = self.label_generator.create_regime_based_labels(
                    result_df,
                    target_return=target_return
                )
            
            self.logger.info(
                "Targets created",
                target_types=target_types,
                target_return=f"{target_return:.1%}",
                total_records=len(result_df)
            )
            
            return result_df
    
    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Target',
        method: str = 'correlation',
        max_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most relevant features
        
        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            method: Feature selection method
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (filtered DataFrame, selected feature names)
        """
        with with_error_context("selecting features"):
            if df.empty:
                return df, []
            
            if target_col not in df.columns:
                raise DataError(f"Target column '{target_col}' not found")
            
            # Get feature columns (exclude metadata and targets)
            exclude_prefixes = ['Date', 'Code', 'Target', 'Next_', 'Return_']
            feature_cols = [
                col for col in df.columns 
                if not any(col.startswith(prefix) for prefix in exclude_prefixes)
            ]
            
            if not feature_cols:
                self.logger.warning("No feature columns found for selection")
                return df, []
            
            selected_features = feature_cols.copy()
            
            if method == 'correlation' and len(feature_cols) > 0:
                selected_features = self._correlation_based_selection(
                    df, feature_cols, target_col, max_features
                )
            elif method == 'variance':
                selected_features = self._variance_based_selection(
                    df, feature_cols, max_features
                )
            
            # Create filtered DataFrame
            keep_cols = ['Date', 'Code', target_col] + selected_features
            keep_cols = [col for col in keep_cols if col in df.columns]
            
            result_df = df[keep_cols].copy()
            
            self.logger.info(
                "Feature selection completed",
                method=method,
                original_features=len(feature_cols),
                selected_features=len(selected_features),
                selection_ratio=f"{len(selected_features)/len(feature_cols):.2%}"
            )
            
            return result_df, selected_features
    
    def _correlation_based_selection(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        max_features: Optional[int] = None
    ) -> List[str]:
        """Select features based on correlation with target"""
        # Calculate correlations with target
        correlations = {}
        
        for col in feature_cols:
            if col in df.columns:
                try:
                    # Handle missing values
                    feature_data = df[col].fillna(df[col].median())
                    target_data = df[target_col]
                    
                    # Calculate correlation
                    corr = feature_data.corr(target_data)
                    
                    if not np.isnan(corr) and abs(corr) >= self.min_correlation_threshold:
                        correlations[col] = abs(corr)
                        
                except Exception as e:
                    self.logger.debug(f"Could not calculate correlation for {col}: {e}")
        
        # Sort by correlation strength
        sorted_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
        
        # Apply max features limit
        if max_features and len(sorted_features) > max_features:
            sorted_features = sorted_features[:max_features]
        
        return sorted_features
    
    def _variance_based_selection(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_features: Optional[int] = None
    ) -> List[str]:
        """Select features based on variance (remove low variance features)"""
        variances = {}
        
        for col in feature_cols:
            if col in df.columns:
                try:
                    feature_data = df[col].fillna(df[col].median())
                    var = feature_data.var()
                    
                    if not np.isnan(var) and var > 1e-6:  # Minimum variance threshold
                        variances[col] = var
                        
                except Exception as e:
                    self.logger.debug(f"Could not calculate variance for {col}: {e}")
        
        # Sort by variance
        sorted_features = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)
        
        # Apply max features limit
        if max_features and len(sorted_features) > max_features:
            sorted_features = sorted_features[:max_features]
        
        return sorted_features
    
    def run_full_pipeline(
        self,
        stock_data: pd.DataFrame,
        target_types: List[str] = ['basic'],
        feature_selection: bool = True,
        max_features: Optional[int] = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Run complete feature engineering pipeline
        
        Args:
            stock_data: Raw stock data
            target_types: List of target types to create
            feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (features_df, targets_df, pipeline_info)
        """
        with with_error_context("running full feature pipeline"):
            if stock_data.empty:
                return pd.DataFrame(), pd.DataFrame(), {}
            
            pipeline_info = {
                "start_time": datetime.now(),
                "input_records": len(stock_data),
                "input_columns": len(stock_data.columns)
            }
            
            # Step 1: Clean and preprocess data
            self.logger.info("Step 1: Data preprocessing")
            cleaned_data = self.preprocessor.clean_stock_price_data(
                stock_data,
                remove_outliers=True,
                handle_missing=True
            )
            
            # Step 2: Create features
            self.logger.info("Step 2: Feature creation")
            featured_data = self.create_basic_features(
                cleaned_data,
                include_technical=True,
                include_volume=True,
                include_patterns=True
            )
            
            # Step 3: Create targets
            self.logger.info("Step 3: Target creation")
            final_data = self.create_targets(
                featured_data,
                target_types=target_types
            )
            
            # Step 4: Feature selection
            if feature_selection and len(target_types) > 0:
                self.logger.info("Step 4: Feature selection")
                target_col = 'Target' if 'basic' in target_types else f'Target_{target_types[0]}'
                if target_col in final_data.columns:
                    final_data, selected_features = self.select_features(
                        final_data,
                        target_col=target_col,
                        max_features=max_features
                    )
                    pipeline_info['selected_features'] = selected_features
            
            # Step 5: Split features and targets
            self.logger.info("Step 5: Feature-target split")
            target_col = 'Target' if 'basic' in target_types else f'Target_{target_types[0]}'
            if target_col in final_data.columns:
                features_df, targets_df = self.label_generator.create_feature_target_split(
                    final_data,
                    target_col=target_col
                )
            else:
                # No targets, return all as features
                features_df = final_data.copy()
                targets_df = pd.DataFrame()
            
            # Update pipeline info
            pipeline_info.update({
                "end_time": datetime.now(),
                "output_records": len(features_df),
                "output_features": len(features_df.columns) - 1,  # Exclude Date
                "target_columns": len(targets_df.columns) - 1 if not targets_df.empty else 0,
                "processing_time": (datetime.now() - pipeline_info["start_time"]).total_seconds()
            })
            
            self.logger.info(
                "Feature pipeline completed",
                input_records=pipeline_info["input_records"],
                output_records=pipeline_info["output_records"],
                output_features=pipeline_info["output_features"],
                processing_time_sec=pipeline_info["processing_time"]
            )
            
            return features_df, targets_df, pipeline_info
    
    def get_feature_importance_analysis(
        self,
        df: pd.DataFrame,
        target_col: str = 'Target'
    ) -> pd.DataFrame:
        """
        Analyze feature importance based on correlations
        
        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            
        Returns:
            DataFrame with feature importance analysis
        """
        if df.empty or target_col not in df.columns:
            return pd.DataFrame()
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['Date', 'Code', target_col]]
        
        importance_data = []
        
        for col in feature_cols:
            try:
                # Calculate correlation with target
                feature_data = df[col].fillna(df[col].median())
                target_data = df[target_col]
                
                correlation = feature_data.corr(target_data)
                
                # Calculate basic statistics
                stats = {
                    'feature': col,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation) if not np.isnan(correlation) else 0,
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'missing_rate': df[col].isna().mean(),
                    'unique_values': df[col].nunique()
                }
                
                importance_data.append(stats)
                
            except Exception as e:
                self.logger.debug(f"Could not analyze feature {col}: {e}")
        
        if not importance_data:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        self.logger.info(
            "Feature importance analysis completed",
            analyzed_features=len(importance_df)
        )
        
        return importance_df


def create_feature_pipeline(config_override: Optional[Dict] = None) -> FeaturePipeline:
    """
    Create feature pipeline instance
    
    Args:
        config_override: Configuration overrides
        
    Returns:
        FeaturePipeline instance
    """
    return FeaturePipeline(config_override)