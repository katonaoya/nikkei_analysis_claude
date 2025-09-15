"""
Configuration management utilities
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager that combines YAML config and environment variables"""
    
    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file
            env_path: Path to .env file
        """
        self.project_root = Path(__file__).parent.parent
        
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            # Try to load .env from project root
            env_file = self.project_root / '.env'
            if env_file.exists():
                load_dotenv(env_file)
        
        # Load YAML config
        if config_path is None:
            config_path = self.project_root / 'config' / 'config.yaml'
        
        self.config_data = self._load_yaml_config(config_path)
        
        # Override with environment variables
        self._override_with_env_vars()
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return {}
    
    def _override_with_env_vars(self):
        """Override config values with environment variables"""
        # API credentials
        if os.getenv('JQUANTS_MAIL_ADDRESS'):
            self.config_data.setdefault('api', {}).setdefault('jquants', {})['mail_address'] = os.getenv('JQUANTS_MAIL_ADDRESS')
        if os.getenv('JQUANTS_PASSWORD'):
            self.config_data.setdefault('api', {}).setdefault('jquants', {})['password'] = os.getenv('JQUANTS_PASSWORD')
        if os.getenv('JQUANTS_REFRESH_TOKEN'):
            self.config_data.setdefault('api', {}).setdefault('jquants', {})['refresh_token'] = os.getenv('JQUANTS_REFRESH_TOKEN')
        
        if os.getenv('RUNPOD_API_KEY'):
            self.config_data.setdefault('api', {}).setdefault('runpod', {})['api_key'] = os.getenv('RUNPOD_API_KEY')
        if os.getenv('RUNPOD_ENDPOINT_ID'):
            self.config_data.setdefault('api', {}).setdefault('runpod', {})['endpoint_id'] = os.getenv('RUNPOD_ENDPOINT_ID')
        
        # Model parameters
        if os.getenv('PRECISION_THRESHOLD'):
            self.config_data.setdefault('prediction', {})['base_threshold'] = float(os.getenv('PRECISION_THRESHOLD'))
        if os.getenv('MAX_DAILY_PICKS'):
            self.config_data.setdefault('prediction', {})['max_daily_picks'] = int(os.getenv('MAX_DAILY_PICKS'))
        
        # Data settings
        if os.getenv('DATA_START_DATE'):
            self.config_data.setdefault('data', {})['start_date'] = os.getenv('DATA_START_DATE')
        if os.getenv('DATA_END_DATE'):
            self.config_data.setdefault('data', {})['end_date'] = os.getenv('DATA_END_DATE')
        
        # Other environment overrides
        env_mappings = {
            'ENV': ('environment',),
            'DEBUG': ('development', 'debug'),
            'LOG_LEVEL': ('logging', 'level'),
            'RANDOM_STATE': ('models', 'random_state'),
            'N_JOBS': ('models', 'n_jobs'),
            'OPTUNA_TRIALS': ('models', 'optuna_trials'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to nested config
                config_section = self.config_data
                for key in config_path[:-1]:
                    config_section = config_section.setdefault(key, {})
                
                # Convert value to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                config_section[config_path[-1]] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.start_date')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config_section = self.config_data
        
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        config_section[keys[-1]] = value
    
    def get_data_dir(self, dir_type: str) -> Path:
        """
        Get data directory path
        
        Args:
            dir_type: Type of directory ('raw', 'feature', 'model', 'signals', 'logs')
            
        Returns:
            Path to directory
        """
        dir_map = {
            'raw': self.get('data.raw_dir', 'data/raw'),
            'feature': self.get('data.feature_dir', 'data/feature'),
            'model': self.get('data.model_dir', 'data/model'),
            'signals': self.get('data.signals_dir', 'signals'),
            'logs': self.get('data.logs_dir', 'logs'),
        }
        
        dir_path = self.project_root / dir_map.get(dir_type, f'data/{dir_type}')
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def validate_required_settings(self) -> bool:
        """
        Validate that all required settings are present
        
        Returns:
            True if all required settings are present
        """
        required_settings = [
            'api.jquants.mail_address',
            'api.jquants.password',
            'api.runpod.api_key',
            'api.runpod.endpoint_id',
            'data.start_date',
            'data.end_date',
        ]
        
        missing_settings = []
        for setting in required_settings:
            if self.get(setting) is None:
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"Missing required settings: {missing_settings}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config_data.copy()


# Global config instance
_config_instance = None


def get_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to YAML config file
        env_path: Path to .env file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path, env_path)
    
    return _config_instance


def reload_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Config:
    """
    Reload configuration from files
    
    Args:
        config_path: Path to YAML config file
        env_path: Path to .env file
        
    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path, env_path)
    return _config_instance