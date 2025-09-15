"""
Tests for configuration management
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from utils.config import Config, get_config, reload_config


class TestConfig:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test basic config initialization"""
        config = Config()
        assert config is not None
        assert hasattr(config, 'config_data')
        assert isinstance(config.config_data, dict)
    
    def test_config_get_default(self):
        """Test getting configuration values with defaults"""
        config = Config()
        
        # Test with default value
        value = config.get('nonexistent.key', 'default_value')
        assert value == 'default_value'
        
        # Test nested key with default
        value = config.get('deeply.nested.nonexistent', 42)
        assert value == 42
    
    def test_config_set_and_get(self):
        """Test setting and getting configuration values"""
        config = Config()
        
        # Set a simple value
        config.set('test.key', 'test_value')
        value = config.get('test.key')
        assert value == 'test_value'
        
        # Set nested value
        config.set('nested.deep.value', 123)
        value = config.get('nested.deep.value')
        assert value == 123
    
    def test_config_with_yaml_file(self):
        """Test config with YAML file"""
        # Create temporary YAML file
        config_data = {
            'test_section': {
                'test_key': 'test_value',
                'numeric_key': 42
            },
            'simple_key': 'simple_value'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            config = Config(config_path=temp_config_path)
            
            # Test loaded values
            assert config.get('test_section.test_key') == 'test_value'
            assert config.get('test_section.numeric_key') == 42
            assert config.get('simple_key') == 'simple_value'
        
        finally:
            os.unlink(temp_config_path)
    
    def test_config_with_env_override(self):
        """Test config with environment variable override"""
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG',
            'RANDOM_STATE': '42'
        }):
            config = Config()
            
            # Test boolean conversion
            assert config.get('development.debug') is True
            
            # Test string value
            assert config.get('logging.level') == 'DEBUG'
            
            # Test integer conversion
            assert config.get('models.random_state') == 42
    
    def test_get_data_dir(self):
        """Test data directory creation"""
        config = Config()
        
        # Test raw data directory
        raw_dir = config.get_data_dir('raw')
        assert isinstance(raw_dir, Path)
        assert raw_dir.exists()
        assert raw_dir.is_dir()
        
        # Test other directories
        for dir_type in ['feature', 'model', 'signals', 'logs']:
            data_dir = config.get_data_dir(dir_type)
            assert isinstance(data_dir, Path)
            assert data_dir.exists()
    
    def test_validate_required_settings(self):
        """Test validation of required settings"""
        config = Config()
        
        # Should fail validation initially (missing API credentials)
        is_valid = config.validate_required_settings()
        assert is_valid is False  # Expected to fail without real credentials
    
    def test_global_config_instance(self):
        """Test global config instance"""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        assert config1 is config2
        
        # Test reload creates new instance
        config3 = reload_config()
        assert config3 is not config1
        
        # New get_config should return the reloaded instance
        config4 = get_config()
        assert config4 is config3