"""
Unified feature engineering pipeline for stock data
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime, date
from pathlib import Path
import logging
import warnings

from .technical_indicators import TechnicalIndicators
from .market_features import MarketFeatures
from .label_generator import LabelGenerator


class FeatureEngineer:
    """Unified feature engineering pipeline"""
    
    def __init__(
        self, 
        data_dir: Optional[Path] = None,
        config_override: Optional[Dict] = None
    ):
        """
        Initialize feature engineer
        
        Args:
            data_dir: Path to data directory
            config_override: Configuration overrides
        """
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component modules
        self.technical = TechnicalIndicators(config_override)
        self.market = MarketFeatures(self.raw_dir)
        self.labels = LabelGenerator(config_override)
        
        # Initialize enhanced feature modules
        try:
            from .time_series_features import TimeSeriesFeatures
            from .fundamental_features import FundamentalFeatures
            
            self.time_series = TimeSeriesFeatures(config_override)
            self.fundamental = FundamentalFeatures(config_override)
            
            self.enhanced_modules_available = True
            
        except Exception as e:
            logging.warning(f"Enhanced feature modules not available: {e}")
            self.enhanced_modules_available = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_stock_data(
        self, 
        file_pattern: str = "*nikkei225_historical*",
        date_range: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load stock data from parquet files
        
        Args:
            file_pattern: Pattern to match data files
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with stock data
        """
        # Find data files
        data_files = list(self.raw_dir.glob(f"{file_pattern}.parquet"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern: {file_pattern}")
        
        # Load and combine data
        dataframes = []
        for file_path in data_files:
            try:
                df = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {len(df)} records from {file_path.name}")
                dataframes.append(df)
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No data could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Standardize column names
        combined_df = self._standardize_columns(combined_df)
        
        # Filter by date range if specified
        if date_range:
            start_date, end_date = date_range
            combined_df = combined_df[
                (combined_df['Date'] >= start_date) & 
                (combined_df['Date'] <= end_date)
            ]
        
        # Sort by date and code
        combined_df = combined_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        self.logger.info(f"Total loaded: {len(combined_df)} records for {combined_df['Code'].nunique()} stocks")
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""
        
        # Ensure essential columns exist
        column_mapping = {
            'date': 'Date',
            'code': 'Code',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def generate_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_market: bool = True,
        include_time_series: bool = True,
        include_fundamental: bool = True,
        include_labels: bool = True,
        technical_params: Optional[Dict] = None,
        market_params: Optional[Dict] = None,
        time_series_params: Optional[Dict] = None,
        fundamental_params: Optional[Dict] = None,
        label_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set (400+ features)
        
        Args:
            df: Input DataFrame with stock data
            include_technical: Whether to include technical indicators
            include_market: Whether to include market features
            include_time_series: Whether to include time series features
            include_fundamental: Whether to include fundamental features
            include_labels: Whether to include prediction labels
            technical_params: Parameters for technical indicators
            market_params: Parameters for market features
            time_series_params: Parameters for time series features
            fundamental_params: Parameters for fundamental features
            label_params: Parameters for label generation
            
        Returns:
            DataFrame with generated features
        """
        result_df = df.copy()
        
        self.logger.info("Starting comprehensive feature generation (400+ features)...")
        initial_columns = len(result_df.columns)
        
        # Enhanced Technical Indicators (150+ features)
        if include_technical:
            self.logger.info("Generating enhanced technical indicators...")
            
            # Use enhanced version with all technical indicators
            result_df = self.technical.calculate_all_indicators(
                result_df, 
                include_patterns=True, 
                include_volume=True
            )
            
            technical_features = len(result_df.columns) - initial_columns
            self.logger.info(f"Added {technical_features} technical features")
        
        # Comprehensive Market Features (150+ features)
        if include_market:
            self.logger.info("Generating comprehensive market features...")
            current_columns = len(result_df.columns)
            
            # Use enhanced market features
            result_df = self.market.calculate_all_enhanced_features(
                result_df,
                price_col='Close',
                high_col='High',
                low_col='Low',
                volume_col='Volume',
                group_by='Code'
            )
            
            market_features = len(result_df.columns) - current_columns
            self.logger.info(f"Added {market_features} market features")
        
        # Time Series Features (100+ features)
        if include_time_series:
            self.logger.info("Generating comprehensive time series features...")
            current_columns = len(result_df.columns)
            
            # Import and initialize time series features
            try:
                from .time_series_features import TimeSeriesFeatures
                time_series_engine = TimeSeriesFeatures(time_series_params)
                
                result_df = time_series_engine.calculate_all_time_series_features(
                    result_df,
                    price_col='Close',
                    high_col='High',
                    low_col='Low',
                    volume_col='Volume',
                    group_by='Code'
                )
                
                time_series_features = len(result_df.columns) - current_columns
                self.logger.info(f"Added {time_series_features} time series features")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate time series features: {e}")
        
        # Fundamental Features (100+ features)
        if include_fundamental:
            self.logger.info("Generating comprehensive fundamental features...")
            current_columns = len(result_df.columns)
            
            # Import and initialize fundamental features
            try:
                from .fundamental_features import FundamentalFeatures
                fundamental_engine = FundamentalFeatures(fundamental_params)
                
                result_df = fundamental_engine.calculate_all_fundamental_features(
                    result_df,
                    price_col='Close',
                    group_by='Code'
                )
                
                fundamental_features = len(result_df.columns) - current_columns
                self.logger.info(f"Added {fundamental_features} fundamental features")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate fundamental features: {e}")
        
        # Generate labels
        if include_labels:
            self.logger.info("Generating prediction labels...")
            current_columns = len(result_df.columns)
            
            # Next day return labels
            result_df = self.labels.create_next_day_return_labels(result_df)
            
            # Classification labels
            result_df = self.labels.create_classification_labels(result_df)
            
            label_features = len(result_df.columns) - current_columns
            self.logger.info(f"Added {label_features} label features")
        
        # Clean up data
        result_df = self._clean_features(result_df)
        
        total_features = len(result_df.columns) - initial_columns
        self.logger.info(
            f"Comprehensive feature generation completed. "
            f"Total features added: {total_features} "
            f"(Initial: {initial_columns}, Final: {len(result_df.columns)})"
        )
        
        return result_df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate generated features"""
        
        # Remove rows with too many missing values
        missing_threshold = 0.5  # Remove rows with >50% missing values
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
        clean_df = df[missing_ratio <= missing_threshold].copy()
        
        if len(clean_df) < len(df):
            removed_rows = len(df) - len(clean_df)
            self.logger.warning(f"Removed {removed_rows} rows due to excessive missing values")
        
        # Forward fill remaining missing values within each stock
        if 'Code' in clean_df.columns:
            clean_df = clean_df.groupby('Code').fillna(method='ffill')
        
        # Replace infinite values with NaN
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        
        return clean_df
    
    def save_features(
        self, 
        df: pd.DataFrame, 
        filename: Optional[str] = None,
        format: str = "parquet"
    ) -> Path:
        """
        Save generated features to file
        
        Args:
            df: DataFrame with features
            filename: Output filename (auto-generated if None)
            format: File format ('parquet' or 'csv')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"features_{timestamp}.{format}"
        
        output_path = self.processed_dir / filename
        
        if format.lower() == "parquet":
            # Convert date to datetime for parquet compatibility
            df_to_save = df.copy()
            if 'Date' in df_to_save.columns:
                df_to_save['Date'] = pd.to_datetime(df_to_save['Date'])
            df_to_save.to_parquet(output_path, index=False)
        elif format.lower() == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Features saved to: {output_path}")
        return output_path
    
    def create_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary of generated features
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'unique_stocks': df['Code'].nunique() if 'Code' in df.columns else 0,
            'date_range': {
                'start': str(df['Date'].min()) if 'Date' in df.columns else None,
                'end': str(df['Date'].max()) if 'Date' in df.columns else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'feature_columns': list(df.columns)
        }
        
        return summary
    
    def run_full_pipeline(
        self,
        file_pattern: str = "*nikkei225_historical*",
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Path, Dict[str, Any]]:
        """
        Run the complete feature engineering pipeline
        
        Args:
            file_pattern: Pattern to match input data files
            output_filename: Output filename for features
            **kwargs: Additional parameters for feature generation
            
        Returns:
            Tuple of (features_df, output_path, summary)
        """
        self.logger.info("Starting full feature engineering pipeline...")
        
        # Load data
        stock_data = self.load_stock_data(file_pattern)
        
        # Generate features
        features_df = self.generate_features(stock_data, **kwargs)
        
        # Save features
        output_path = self.save_features(features_df, output_filename)
        
        # Create summary
        summary = self.create_feature_summary(features_df)
        
        self.logger.info("Feature engineering pipeline completed successfully!")
        
        return features_df, output_path, summary