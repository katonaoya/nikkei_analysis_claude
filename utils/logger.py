"""
Logging utilities with structured logging support
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger

from .config import get_config


class StructuredLogger:
    """Structured logger with JSON formatting and file rotation"""
    
    def __init__(self, name: str = "stock_analysis", config_override: Optional[Dict] = None):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            config_override: Configuration overrides
        """
        self.name = name
        self.config = get_config()
        
        if config_override:
            for key, value in config_override.items():
                self.config.set(f'logging.{key}', value)
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup loguru logger with configuration"""
        # Remove default handler
        logger.remove()
        
        # Get logging configuration
        log_level = self.config.get('logging.level', 'INFO')
        log_format = self.config.get('logging.format', 'json')
        log_to_console = self.config.get('logging.log_to_console', True)
        log_to_file = self.config.get('logging.log_to_file', True)
        rotation = self.config.get('logging.rotation', '1 day')
        retention = self.config.get('logging.retention', '30 days')
        
        # Console handler
        if log_to_console:
            if log_format == 'json':
                logger.add(
                    sys.stdout,
                    format="{time} | {level} | {name}:{function}:{line} - {message}",
                    level=log_level,
                    serialize=True
                )
            else:
                logger.add(
                    sys.stdout,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                    level=log_level,
                    colorize=True
                )
        
        # File handler
        if log_to_file:
            log_dir = self.config.get_data_dir('logs')
            log_file = log_dir / f"{self.name}.log"
            
            if log_format == 'json':
                logger.add(
                    log_file,
                    format="{time} | {level} | {name}:{function}:{line} - {message}",
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    serialize=True,
                    enqueue=True
                )
            else:
                logger.add(
                    log_file,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    enqueue=True
                )
    
    
    def bind(self, **kwargs):
        """Bind additional context to logger"""
        return logger.bind(**kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        logger.bind(**kwargs).info(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        logger.bind(**kwargs).debug(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        logger.bind(**kwargs).warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        logger.bind(**kwargs).error(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context"""
        logger.bind(**kwargs).critical(message)
    
    def log_execution_start(self, process_name: str, **context):
        """Log execution start with context"""
        self.info(
            f"Starting {process_name}",
            process=process_name,
            status="started",
            **context
        )
    
    def log_execution_end(self, process_name: str, duration: Optional[float] = None, **context):
        """Log execution end with context"""
        log_data = {
            "process": process_name,
            "status": "completed",
            **context
        }
        
        if duration is not None:
            log_data["duration_seconds"] = duration
        
        self.info(f"Completed {process_name}", **log_data)
    
    def log_execution_error(self, process_name: str, error: Exception, **context):
        """Log execution error with context"""
        self.error(
            f"Error in {process_name}: {str(error)}",
            process=process_name,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )
    
    def log_data_info(self, operation: str, **metrics):
        """Log data operation metrics"""
        self.info(
            f"Data operation: {operation}",
            operation=operation,
            category="data",
            **metrics
        )
    
    def log_model_info(self, operation: str, **metrics):
        """Log model operation metrics"""
        self.info(
            f"Model operation: {operation}",
            operation=operation,
            category="model",
            **metrics
        )
    
    def log_prediction_info(self, **metrics):
        """Log prediction metrics"""
        self.info(
            "Prediction completed",
            category="prediction",
            **metrics
        )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.info(
            "Performance metrics",
            category="metrics",
            **metrics
        )


# Global logger instances
_loggers = {}


def get_logger(name: str = "stock_analysis", config_override: Optional[Dict] = None) -> StructuredLogger:
    """
    Get or create logger instance
    
    Args:
        name: Logger name
        config_override: Configuration overrides
        
    Returns:
        StructuredLogger instance
    """
    global _loggers
    
    logger_key = f"{name}_{hash(str(config_override))}" if config_override else name
    
    if logger_key not in _loggers:
        _loggers[logger_key] = StructuredLogger(name, config_override)
    
    return _loggers[logger_key]


def log_function_call(func_name: str, **kwargs):
    """Decorator-friendly function call logger"""
    logger = get_logger()
    logger.debug(f"Function call: {func_name}", function=func_name, **kwargs)


def log_exception(exc: Exception, context: Optional[str] = None, **kwargs):
    """Log exception with context"""
    logger = get_logger()
    context_msg = f" in {context}" if context else ""
    logger.error(f"Exception{context_msg}: {str(exc)}", exception_type=type(exc).__name__, **kwargs)