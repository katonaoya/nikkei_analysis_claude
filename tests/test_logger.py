"""
Tests for logging utilities
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from utils.logger import StructuredLogger, get_logger, log_exception


class TestStructuredLogger:
    """Test structured logging functionality"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = StructuredLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.config is not None
    
    def test_logger_basic_logging(self):
        """Test basic logging methods"""
        logger = StructuredLogger("test_logger")
        
        # Test that these don't raise exceptions
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")
    
    def test_logger_with_context(self):
        """Test logging with additional context"""
        logger = StructuredLogger("test_logger")
        
        # Test logging with context
        logger.info("Test message with context", user_id=123, action="test_action")
        logger.error("Test error with context", error_code="E001", component="test")
    
    def test_logger_bind(self):
        """Test logger binding"""
        logger = StructuredLogger("test_logger")
        
        # Test binding returns logger-like object
        bound_logger = logger.bind(request_id="12345", user="test_user")
        assert bound_logger is not None
    
    def test_execution_logging(self):
        """Test execution lifecycle logging"""
        logger = StructuredLogger("test_logger")
        
        # Test execution start/end logging
        logger.log_execution_start("test_process", component="test")
        logger.log_execution_end("test_process", duration=1.5)
        logger.log_execution_error("test_process", ValueError("Test error"))
    
    def test_specialized_logging(self):
        """Test specialized logging methods"""
        logger = StructuredLogger("test_logger")
        
        # Test data operation logging
        logger.log_data_info("data_fetch", records=1000, source="test_api")
        
        # Test model operation logging
        logger.log_model_info("model_training", accuracy=0.85, features=50)
        
        # Test prediction logging
        logger.log_prediction_info(predictions=100, threshold=0.8)
        
        # Test performance metrics
        logger.log_performance_metrics({
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81
        })
    
    def test_global_logger_instance(self):
        """Test global logger instance management"""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        # Should return same instance for same name
        assert logger1 is logger2
        
        # Different names should return different instances
        logger3 = get_logger("different_logger")
        assert logger3 is not logger1
    
    def test_log_exception_function(self):
        """Test standalone exception logging function"""
        test_exception = ValueError("Test exception")
        
        # Should not raise any exceptions
        log_exception(test_exception, context="test_context", additional_info="test")


class TestLoggerConfiguration:
    """Test logger configuration handling"""
    
    def test_logger_with_config_override(self):
        """Test logger with configuration overrides"""
        config_override = {
            "level": "DEBUG",
            "log_to_console": True,
            "log_to_file": False
        }
        
        logger = StructuredLogger("test_logger", config_override)
        assert logger.config.get('logging.level') == 'DEBUG'
        assert logger.config.get('logging.log_to_console') is True
        assert logger.config.get('logging.log_to_file') is False
    
    def test_json_formatter(self):
        """Test JSON formatter functionality"""
        logger = StructuredLogger("test_logger")
        
        # Create a mock record
        class MockRecord:
            def __init__(self):
                self.time = "2023-01-01T00:00:00"
                self.level = type('Level', (), {'name': 'INFO'})()
                self.name = "test_logger"
                self.module = "test_module"
                self.function = "test_function"
                self.line = 123
                self.message = "Test message"
                self.extra = {"key": "value"}
                self.exception = None
        
        record = MockRecord()
        
        # Test JSON formatter (should not raise exception)
        try:
            formatted = logger._json_formatter(record)
            # Should be valid JSON
            parsed = json.loads(formatted)
            assert parsed["message"] == "Test message"
            assert parsed["level"] == "INFO"
            assert parsed["service"] == "test_logger"
        except Exception as e:
            pytest.fail(f"JSON formatter failed: {e}")


# Integration test to ensure logger works with file system
def test_logger_file_integration():
    """Integration test for file logging"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_override = {
            "log_to_file": True,
            "logs_dir": temp_dir
        }
        
        logger = StructuredLogger("integration_test", config_override)
        logger.info("Integration test message")
        
        # Check that log file was created
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) > 0, "No log files were created"