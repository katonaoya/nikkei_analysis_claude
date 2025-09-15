"""
Tests for data modules (mocked tests to avoid external API calls)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import json

from src.data.jquants_client import JQuantsClient, create_jquants_client
from src.data.stock_data_fetcher import StockDataFetcher, create_stock_data_fetcher
from src.data.external_data_fetcher import ExternalDataFetcher, create_external_data_fetcher
from src.data.data_preprocessor import DataPreprocessor, create_data_preprocessor
from src.data.data_validator import DataValidator, create_data_validator


class TestJQuantsClient:
    """Test J-Quants API client (mocked)"""
    
    @patch('src.data.jquants_client.requests.Session')
    def test_jquants_client_initialization(self, mock_session):
        """Test J-Quants client initialization"""
        # Mock configuration
        with patch('src.data.jquants_client.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'api.jquants.base_url': 'https://api.jquants.com',
                'api.jquants.mail_address': 'test@example.com',
                'api.jquants.password': 'test_password',
                'api.jquants.rate_limit': 100,
                'api.jquants.timeout': 30
            }.get(key, default)
            mock_config.return_value = mock_config_instance
            
            client = JQuantsClient()
            assert client is not None
            assert client.base_url == 'https://api.jquants.com'
            assert client.mail_address == 'test@example.com'
    
    @patch('src.data.jquants_client.requests.Session')
    def test_jquants_client_get_stock_prices_mock(self, mock_session):
        """Test getting stock prices with mocked response"""
        with patch('src.data.jquants_client.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'api.jquants.base_url': 'https://api.jquants.com',
                'api.jquants.mail_address': 'test@example.com',
                'api.jquants.password': 'test_password',
                'api.jquants.rate_limit': 100,
                'api.jquants.timeout': 30
            }.get(key, default)
            mock_config.return_value = mock_config_instance
            
            # Mock the session and response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'daily_quotes': [
                    {
                        'Code': '7203',
                        'Date': '2023-01-01',
                        'Open': 1000,
                        'High': 1100,
                        'Low': 950,
                        'Close': 1050,
                        'Volume': 1000000
                    }
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_response.text = '{"daily_quotes": []}'
            
            mock_session_instance = MagicMock()
            mock_session_instance.request.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            client = JQuantsClient()
            client.id_token = 'mock_token'  # Mock authentication
            
            result = client.get_stock_prices('7203', '2023-01-01', '2023-01-02')
            assert 'daily_quotes' in result
            assert len(result['daily_quotes']) == 1
    
    def test_create_jquants_client(self):
        """Test client factory function"""
        with patch('src.data.jquants_client.JQuantsClient') as mock_client:
            create_jquants_client({'test': 'config'})
            mock_client.assert_called_once_with({'test': 'config'})


class TestStockDataFetcher:
    """Test stock data fetcher (mocked)"""
    
    @patch('src.data.stock_data_fetcher.create_jquants_client')
    def test_stock_data_fetcher_initialization(self, mock_create_client):
        """Test stock data fetcher initialization"""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        with patch('src.data.stock_data_fetcher.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'data.format': 'parquet',
                'data.compression': 'snappy',
                'data.start_date': '2020-01-01',
                'data.end_date': '2023-12-31'
            }.get(key, default)
            mock_config_instance.get_data_dir.return_value = tempfile.mkdtemp()
            mock_config.return_value = mock_config_instance
            
            fetcher = StockDataFetcher()
            assert fetcher is not None
            assert fetcher.data_format == 'parquet'
    
    @patch('src.data.stock_data_fetcher.create_jquants_client')
    def test_fetch_stock_prices_cached(self, mock_create_client):
        """Test fetching stock prices with cache"""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Code': ['7203', '7203'],
            'Date': ['2023-01-01', '2023-01-02'],
            'Close': [1000, 1050],
            'Volume': [100000, 110000]
        })
        
        with patch('src.data.stock_data_fetcher.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'data.format': 'parquet',
                'data.compression': 'snappy',
                'data.start_date': '2020-01-01',
                'data.end_date': '2023-12-31'
            }.get(key, default)
            mock_config_instance.get_data_dir.return_value = tempfile.mkdtemp()
            mock_config.return_value = mock_config_instance
            
            fetcher = StockDataFetcher()
            
            # Mock the file loading to return cached data
            with patch.object(fetcher, '_load_dataframe', return_value=sample_data):
                result = fetcher.fetch_stock_prices(['7203'])
                assert len(result) == 2
                assert '7203' in result['Code'].values


class TestExternalDataFetcher:
    """Test external data fetcher (mocked)"""
    
    @patch('yfinance.Ticker')
    def test_external_data_fetcher_yahoo_finance(self, mock_ticker_class):
        """Test fetching data from Yahoo Finance (mocked)"""
        # Mock yfinance response
        mock_ticker = MagicMock()
        mock_history_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_ticker.history.return_value = mock_history_data
        mock_ticker_class.return_value = mock_ticker
        
        with patch('src.data.external_data_fetcher.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'data.format': 'parquet',
                'data.compression': 'snappy',
                'data.start_date': '2020-01-01',
                'data.end_date': '2023-12-31'
            }.get(key, default)
            mock_config_instance.get_data_dir.return_value = tempfile.mkdtemp()
            mock_config.return_value = mock_config_instance
            
            fetcher = ExternalDataFetcher()
            result = fetcher.fetch_yahoo_finance_data('USDJPY=X', '2023-01-01', '2023-01-02')
            
            assert len(result) == 2
            assert 'Close' in result.columns
            assert 'symbol' in result.columns


class TestDataPreprocessor:
    """Test data preprocessor"""
    
    def test_data_preprocessor_initialization(self):
        """Test data preprocessor initialization"""
        with patch('src.data.data_preprocessor.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'features.fill_method': 'ffill',
                'features.outlier_method': 'quantile',
                'features.outlier_lower': 0.01,
                'features.outlier_upper': 0.99
            }.get(key, default)
            mock_config.return_value = mock_config_instance
            
            preprocessor = DataPreprocessor()
            assert preprocessor is not None
            assert preprocessor.fill_method == 'ffill'
    
    def test_clean_stock_price_data(self):
        """Test cleaning stock price data"""
        with patch('src.data.data_preprocessor.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'features.fill_method': 'ffill',
                'features.outlier_method': 'quantile',
                'features.outlier_lower': 0.01,
                'features.outlier_upper': 0.99
            }.get(key, default)
            mock_config.return_value = mock_config_instance
            
            preprocessor = DataPreprocessor()
            
            # Create sample data with issues
            sample_data = pd.DataFrame({
                'Code': ['7203', '7203', '7203', '7203'],
                'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
                'Open': [1000, 1010, np.nan, 1020],
                'High': [1100, 1110, 1050, 1120],
                'Low': [950, 960, 990, 970],
                'Close': [1050, 1070, 1040, 1080],
                'Volume': [100000, 110000, 0, 120000]
            })
            
            result = preprocessor.clean_stock_price_data(sample_data)
            assert len(result) <= len(sample_data)  # Some records may be removed
            assert 'Code' in result.columns
            assert 'Close' in result.columns
    
    def test_normalize_features(self):
        """Test feature normalization"""
        with patch('src.data.data_preprocessor.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            preprocessor = DataPreprocessor()
            
            # Create sample data
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [10, 20, 30, 40, 50],
                'other_col': ['a', 'b', 'c', 'd', 'e']
            })
            
            normalized_df, scaler = preprocessor.normalize_features(
                sample_data, 
                ['feature1', 'feature2'],
                method='standard'
            )
            
            assert scaler is not None
            # Normalized features should have approximately mean=0, std=1
            assert abs(normalized_df['feature1'].mean()) < 0.1
            assert abs(normalized_df['feature2'].mean()) < 0.1
    
    def test_create_lag_features(self):
        """Test creating lag features"""
        with patch('src.data.data_preprocessor.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            preprocessor = DataPreprocessor()
            
            # Create sample time series data
            sample_data = pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=5),
                'Code': ['7203'] * 5,
                'Close': [100, 102, 101, 103, 105]
            })
            
            result = preprocessor.create_lag_features(
                sample_data,
                columns=['Close'],
                lags=[1, 2],
                group_by='Code'
            )
            
            assert 'Close_lag_1' in result.columns
            assert 'Close_lag_2' in result.columns
            assert result['Close_lag_1'].iloc[1] == 100  # First lag


class TestDataValidator:
    """Test data validator"""
    
    def test_data_validator_initialization(self):
        """Test data validator initialization"""
        with patch('src.data.data_validator.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            validator = DataValidator()
            assert validator is not None
    
    def test_validate_stock_price_data_good(self):
        """Test validating good stock price data"""
        with patch('src.data.data_validator.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            validator = DataValidator()
            
            # Create good sample data
            good_data = pd.DataFrame({
                'Code': ['7203', '7203', '7203'],
                'Date': pd.date_range('2023-01-01', periods=3),
                'Open': [1000, 1010, 1020],
                'High': [1100, 1110, 1120],
                'Low': [950, 960, 970],
                'Close': [1050, 1070, 1080],
                'Volume': [100000, 110000, 120000]
            })
            
            result = validator.validate_stock_price_data(good_data)
            assert result['status'] in ['passed', 'passed_with_warnings']
            assert result['total_rows'] == 3
    
    def test_validate_stock_price_data_bad(self):
        """Test validating bad stock price data"""
        with patch('src.data.data_validator.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            validator = DataValidator()
            
            # Create bad sample data
            bad_data = pd.DataFrame({
                'Code': ['7203', '7203'],
                'Date': pd.date_range('2023-01-01', periods=2),
                'Open': [-100, 1010],  # Negative price
                'High': [1100, 900],   # High < Low in second row
                'Low': [950, 960],
                'Close': [1050, 920],
                'Volume': [-1000, 110000]  # Negative volume
            })
            
            result = validator.validate_stock_price_data(bad_data, strict=True)
            assert result['status'] == 'failed'
            assert len(result['issues']) > 0
    
    def test_validate_feature_matrix(self):
        """Test validating feature matrix"""
        with patch('src.data.data_validator.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.get.return_value = None
            mock_config.return_value = mock_config_instance
            
            validator = DataValidator()
            
            # Create sample feature matrix
            features_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [10, 20, 30, 40, 50],
                'feature3': [1, 1, 1, 1, 1],  # Constant feature
                'feature4': [1, 2, np.nan, 4, 5]  # Missing values
            })
            
            result = validator.validate_feature_matrix(features_data)
            assert result['status'] in ['passed', 'passed_with_warnings']
            assert result['metrics']['total_features'] == 4
            assert result['metrics']['total_samples'] == 5


class TestIntegration:
    """Integration tests for data modules"""
    
    def test_factory_functions(self):
        """Test factory functions work"""
        with patch('src.data.jquants_client.get_config'):
            client = create_jquants_client()
            assert isinstance(client, JQuantsClient)
        
        with patch('src.data.stock_data_fetcher.get_config'):
            with patch('src.data.stock_data_fetcher.create_jquants_client'):
                fetcher = create_stock_data_fetcher()
                assert isinstance(fetcher, StockDataFetcher)
        
        with patch('src.data.external_data_fetcher.get_config'):
            ext_fetcher = create_external_data_fetcher()
            assert isinstance(ext_fetcher, ExternalDataFetcher)
        
        with patch('src.data.data_preprocessor.get_config'):
            preprocessor = create_data_preprocessor()
            assert isinstance(preprocessor, DataPreprocessor)
        
        with patch('src.data.data_validator.get_config'):
            validator = create_data_validator()
            assert isinstance(validator, DataValidator)