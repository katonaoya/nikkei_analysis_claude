"""
Error handling utilities with retry mechanisms
"""

import time
import functools
from typing import Callable, Type, Optional, Union, Tuple, Any
from .logger import get_logger


class StockAnalysisError(Exception):
    """Base exception for stock analysis system"""
    pass


class DataError(StockAnalysisError):
    """Exception for data-related errors"""
    pass


class ModelError(StockAnalysisError):
    """Exception for model-related errors"""
    pass


class APIError(StockAnalysisError):
    """Exception for API-related errors"""
    pass


class ConfigurationError(StockAnalysisError):
    """Exception for configuration-related errors"""
    pass


class ValidationError(StockAnalysisError):
    """Exception for data validation errors"""
    pass


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger_name: Optional[str] = None
):
    """
    Decorator to retry function on specified exceptions
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Exception type(s) to retry on
        logger_name: Logger name for logging retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or "error_handler")
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            function=func.__name__,
                            attempts=attempt + 1,
                            final_error=str(e)
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {current_delay}s",
                        function=func.__name__,
                        attempt=attempt + 1,
                        error=str(e),
                        retry_delay=current_delay
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_value: Any = None,
    log_errors: bool = True,
    logger_name: Optional[str] = None,
    context: Optional[str] = None
) -> Any:
    """
    Safely execute a function and return default value on error
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        log_errors: Whether to log errors
        logger_name: Logger name for logging
        context: Additional context for logging
        
    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger = get_logger(logger_name or "error_handler")
            context_msg = f" in {context}" if context else ""
            logger.error(
                f"Safe execution failed{context_msg}: {str(e)}",
                function=func.__name__ if hasattr(func, '__name__') else str(func),
                error_type=type(e).__name__,
                context=context
            )
        return default_value


class ErrorHandler:
    """Centralized error handling with context management"""
    
    def __init__(self, logger_name: str = "error_handler"):
        self.logger = get_logger(logger_name)
        self.error_counts = {}
        self.context_stack = []
    
    def push_context(self, context: str):
        """Push error context onto stack"""
        self.context_stack.append(context)
    
    def pop_context(self):
        """Pop error context from stack"""
        if self.context_stack:
            return self.context_stack.pop()
        return None
    
    def get_current_context(self) -> str:
        """Get current error context"""
        return " -> ".join(self.context_stack) if self.context_stack else "global"
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        reraise: bool = True,
        **context_data
    ):
        """
        Handle error with logging and optional re-raising
        
        Args:
            error: Exception that occurred
            operation: Operation description
            reraise: Whether to re-raise the exception
            **context_data: Additional context data
        """
        error_type = type(error).__name__
        current_context = self.get_current_context()
        
        # Count errors
        error_key = f"{current_context}:{operation}:{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            operation=operation,
            context=current_context,
            error_type=error_type,
            error_count=self.error_counts[error_key],
            **context_data
        )
        
        if reraise:
            raise error
    
    def handle_warning(
        self,
        message: str,
        operation: str,
        **context_data
    ):
        """
        Handle warning with logging
        
        Args:
            message: Warning message
            operation: Operation description
            **context_data: Additional context data
        """
        current_context = self.get_current_context()
        
        self.logger.warning(
            f"Warning in {operation}: {message}",
            operation=operation,
            context=current_context,
            **context_data
        )
    
    def get_error_summary(self) -> dict:
        """Get summary of error counts"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts.clear()


class ContextManager:
    """Context manager for error handling"""
    
    def __init__(self, error_handler: ErrorHandler, context: str):
        self.error_handler = error_handler
        self.context = context
    
    def __enter__(self):
        self.error_handler.push_context(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.error_handler.pop_context()
        
        if exc_type is not None:
            self.error_handler.handle_error(
                exc_val,
                self.context,
                reraise=False  # Don't re-raise in context manager
            )
        
        # Don't suppress exceptions
        return False


# Global error handler
_global_error_handler = None


def get_error_handler(logger_name: str = "error_handler") -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(logger_name)
    
    return _global_error_handler


def with_error_context(context: str):
    """Context manager for error handling"""
    error_handler = get_error_handler()
    return ContextManager(error_handler, context)


def validate_not_none(value: Any, name: str) -> Any:
    """Validate that value is not None"""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: Union[int, float], name: str) -> Union[int, float]:
    """Validate that value is positive"""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_in_range(value: Union[int, float], min_val: Union[int, float], 
                     max_val: Union[int, float], name: str) -> Union[int, float]:
    """Validate that value is in specified range"""
    if not min_val <= value <= max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


def validate_list_not_empty(value: list, name: str) -> list:
    """Validate that list is not empty"""
    if not value:
        raise ValidationError(f"{name} cannot be empty")
    return value