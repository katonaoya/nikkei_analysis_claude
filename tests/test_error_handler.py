"""
Tests for error handling utilities
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from utils.error_handler import (
    StockAnalysisError, DataError, ModelError, APIError, ConfigurationError, ValidationError,
    retry_on_exception, safe_execute, ErrorHandler, get_error_handler, with_error_context,
    validate_not_none, validate_positive, validate_in_range, validate_list_not_empty
)


class TestCustomExceptions:
    """Test custom exception classes"""
    
    def test_custom_exceptions_inheritance(self):
        """Test that custom exceptions inherit properly"""
        assert issubclass(DataError, StockAnalysisError)
        assert issubclass(ModelError, StockAnalysisError)
        assert issubclass(APIError, StockAnalysisError)
        assert issubclass(ConfigurationError, StockAnalysisError)
        assert issubclass(ValidationError, StockAnalysisError)
    
    def test_custom_exceptions_creation(self):
        """Test creating custom exceptions"""
        data_error = DataError("Test data error")
        assert str(data_error) == "Test data error"
        
        model_error = ModelError("Test model error")
        assert str(model_error) == "Test model error"


class TestRetryDecorator:
    """Test retry decorator functionality"""
    
    def test_successful_function_no_retry(self):
        """Test function that succeeds on first try"""
        call_count = 0
        
        @retry_on_exception(max_retries=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        assert result == "success"
        assert call_count == 1
    
    def test_function_succeeds_after_retries(self):
        """Test function that fails then succeeds"""
        call_count = 0
        
        @retry_on_exception(max_retries=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_function_fails_all_retries(self):
        """Test function that fails all retries"""
        call_count = 0
        
        @retry_on_exception(max_retries=2, delay=0.01)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            always_failing_function()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_retry_specific_exceptions(self):
        """Test retry only on specific exceptions"""
        call_count = 0
        
        @retry_on_exception(max_retries=2, delay=0.01, exceptions=ValueError)
        def selective_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise TypeError("Non-retryable error")
            return "success"
        
        with pytest.raises(TypeError, match="Non-retryable error"):
            selective_retry_function()
        
        assert call_count == 2


class TestSafeExecute:
    """Test safe execution utility"""
    
    def test_safe_execute_success(self):
        """Test safe execute with successful function"""
        def successful_function():
            return "success"
        
        result = safe_execute(successful_function, default_value="default")
        assert result == "success"
    
    def test_safe_execute_failure(self):
        """Test safe execute with failing function"""
        def failing_function():
            raise ValueError("Test error")
        
        result = safe_execute(failing_function, default_value="default")
        assert result == "default"
    
    def test_safe_execute_with_logging(self):
        """Test safe execute with error logging"""
        def failing_function():
            raise ValueError("Test error")
        
        with patch('utils.error_handler.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = safe_execute(
                failing_function,
                default_value="default",
                log_errors=True,
                context="test_context"
            )
            
            assert result == "default"
            mock_logger.error.assert_called_once()


class TestErrorHandler:
    """Test ErrorHandler class"""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        handler = ErrorHandler("test_handler")
        assert handler.error_counts == {}
        assert handler.context_stack == []
    
    def test_context_management(self):
        """Test error context management"""
        handler = ErrorHandler("test_handler")
        
        # Test context stack
        handler.push_context("context1")
        assert handler.get_current_context() == "context1"
        
        handler.push_context("context2")
        assert handler.get_current_context() == "context1 -> context2"
        
        popped = handler.pop_context()
        assert popped == "context2"
        assert handler.get_current_context() == "context1"
        
        handler.pop_context()
        assert handler.get_current_context() == "global"
    
    def test_handle_error(self):
        """Test error handling"""
        handler = ErrorHandler("test_handler")
        
        with patch.object(handler, 'logger') as mock_logger:
            test_error = ValueError("Test error")
            
            # Test handle error without re-raising
            handler.handle_error(test_error, "test_operation", reraise=False)
            
            mock_logger.error.assert_called_once()
            assert len(handler.error_counts) == 1
    
    def test_handle_warning(self):
        """Test warning handling"""
        handler = ErrorHandler("test_handler")
        
        with patch.object(handler, 'logger') as mock_logger:
            handler.handle_warning("Test warning", "test_operation", extra_data="test")
            mock_logger.warning.assert_called_once()
    
    def test_error_summary(self):
        """Test error summary functionality"""
        handler = ErrorHandler("test_handler")
        
        # Simulate some errors
        handler.push_context("test_context")
        with patch.object(handler, 'logger'):
            handler.handle_error(ValueError("Error 1"), "operation1", reraise=False)
            handler.handle_error(ValueError("Error 2"), "operation1", reraise=False)
            handler.handle_error(TypeError("Error 3"), "operation2", reraise=False)
        
        summary = handler.get_error_summary()
        assert len(summary) >= 2  # At least 2 different error types/operations
        
        # Test reset
        handler.reset_error_counts()
        assert len(handler.get_error_summary()) == 0


class TestContextManager:
    """Test error context manager"""
    
    def test_context_manager_success(self):
        """Test context manager with successful operation"""
        handler = ErrorHandler("test_handler")
        
        with with_error_context("test_operation"):
            # Simulate successful operation
            pass
        
        # Should not have any errors
        assert len(handler.get_error_summary()) == 0
    
    def test_context_manager_with_exception(self):
        """Test context manager with exception"""
        handler = get_error_handler("test_handler")
        
        with patch.object(handler, 'logger'):
            try:
                with with_error_context("test_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass  # Exception should propagate
        
        # Should have logged the error
        assert len(handler.get_error_summary()) > 0


class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_not_none(self):
        """Test not-none validation"""
        # Should pass
        result = validate_not_none("value", "test_param")
        assert result == "value"
        
        # Should fail
        with pytest.raises(ValidationError, match="test_param cannot be None"):
            validate_not_none(None, "test_param")
    
    def test_validate_positive(self):
        """Test positive number validation"""
        # Should pass
        result = validate_positive(5, "test_param")
        assert result == 5
        
        result = validate_positive(0.1, "test_param")
        assert result == 0.1
        
        # Should fail
        with pytest.raises(ValidationError, match="test_param must be positive"):
            validate_positive(0, "test_param")
        
        with pytest.raises(ValidationError, match="test_param must be positive"):
            validate_positive(-1, "test_param")
    
    def test_validate_in_range(self):
        """Test range validation"""
        # Should pass
        result = validate_in_range(5, 0, 10, "test_param")
        assert result == 5
        
        result = validate_in_range(0, 0, 10, "test_param")  # Edge case
        assert result == 0
        
        result = validate_in_range(10, 0, 10, "test_param")  # Edge case
        assert result == 10
        
        # Should fail
        with pytest.raises(ValidationError, match="test_param must be between"):
            validate_in_range(-1, 0, 10, "test_param")
        
        with pytest.raises(ValidationError, match="test_param must be between"):
            validate_in_range(11, 0, 10, "test_param")
    
    def test_validate_list_not_empty(self):
        """Test non-empty list validation"""
        # Should pass
        result = validate_list_not_empty([1, 2, 3], "test_param")
        assert result == [1, 2, 3]
        
        # Should fail
        with pytest.raises(ValidationError, match="test_param cannot be empty"):
            validate_list_not_empty([], "test_param")


class TestGlobalErrorHandler:
    """Test global error handler instance"""
    
    def test_global_error_handler_singleton(self):
        """Test that global error handler returns same instance"""
        handler1 = get_error_handler("test")
        handler2 = get_error_handler("test")
        
        # Should be the same instance
        assert handler1 is handler2