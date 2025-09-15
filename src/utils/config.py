"""
Configuration utilities for the AI stock prediction system
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager with hierarchical key access"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize config with optional dictionary"""
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation like 'model.lightgbm.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final key
        config[keys[-1]] = value
    
    def update(self, other_config: Dict) -> None:
        """Update config with another dictionary"""
        self._deep_update(self._config, other_config)
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Deep update dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return self._config.copy()


# Global configuration instance
_global_config = Config()

# Default configuration
_default_config = {
    'data': {
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'output_dir': 'data/output',
        'temp_dir': 'data/temp'
    },
    'model': {
        'ensemble': {
            'weights': {
                'lightgbm': 0.45,
                'catboost': 0.45,
                'logistic_regression': 0.10
            },
            'calibration_method': 'isotonic',
            'use_calibration': True
        },
        'lightgbm': {
            'early_stopping_rounds': 200,
            'verbose_eval': False
        },
        'catboost': {
            'early_stopping_rounds': 200,
            'verbose': False
        },
        'logistic': {
            'use_feature_selection': True,
            'max_features': 100,
            'standardize': True
        },
        'optimization': {
            'n_trials': 200,
            'timeout': None,
            'n_jobs': 1,
            'primary_metric': 'precision',
            'secondary_metric': 'f1',
            'metric_weight': 0.8,
            'min_precision_threshold': 0.7,
            'target_precision': 0.75
        },
        'calibration': {
            'target_precision': 0.75,
            'min_precision': 0.70,
            'max_daily_selections': 3
        }
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

# Initialize global config with defaults
_global_config.update(_default_config)


def get_config() -> Config:
    """Get global configuration instance"""
    return _global_config


def load_config_file(file_path: str) -> None:
    """
    Load configuration from JSON file
    
    Args:
        file_path: Path to configuration file
    """
    config_path = Path(file_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    _global_config.update(config_data)


def save_config_file(file_path: str) -> None:
    """
    Save current configuration to JSON file
    
    Args:
        file_path: Path to save configuration
    """
    config_path = Path(file_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(_global_config.to_dict(), f, indent=2, ensure_ascii=False)


def get_data_dir(dir_type: str = 'processed') -> Path:
    """
    Get data directory path
    
    Args:
        dir_type: Type of directory ('raw', 'processed', 'output', 'temp')
        
    Returns:
        Path to directory
    """
    base_path = Path(__file__).parent.parent.parent
    
    if dir_type == 'raw':
        return base_path / _global_config.get('data.raw_dir', 'data/raw')
    elif dir_type == 'processed':
        return base_path / _global_config.get('data.processed_dir', 'data/processed')
    elif dir_type == 'output':
        return base_path / _global_config.get('data.output_dir', 'data/output')
    elif dir_type == 'temp':
        return base_path / _global_config.get('data.temp_dir', 'data/temp')
    else:
        raise ValueError(f"Unknown directory type: {dir_type}")


def ensure_directories():
    """Ensure all required directories exist"""
    for dir_type in ['raw', 'processed', 'output', 'temp']:
        dir_path = get_data_dir(dir_type)
        dir_path.mkdir(parents=True, exist_ok=True)