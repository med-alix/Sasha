"""
Logging utilities for optimization with styled output.
"""
import logging
import sys
from typing import Dict, Any

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored logging output."""
    
    # ANSI color codes
    COLORS = {
        'TITLE': '\033[1;36m',    # Cyan
        'SECTION': '\033[1;34m',  # Blue
        'KEY': '\033[1;33m',      # Yellow
        'VALUE': '\033[1;35m',    # Magenta
        'SUCCESS': '\033[1;32m',  # Green
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors based on extra attributes."""
        if hasattr(record, 'color'):
            color = self.COLORS.get(record.color, '')
            reset = self.COLORS['RESET']
            
            if hasattr(record, 'key_value'):
                key, value = record.key_value
                record.msg = f"{self.COLORS['KEY']}{key}: {self.COLORS['VALUE']}{value}{reset}"
            else:
                record.msg = f"{color}{record.msg}{reset}"
                
            if hasattr(record, 'style') and record.style == 'title':
                line = '-' * len(record.msg)
                record.msg = f"{color}{line}\n{record.msg};{line}{reset}"
                
        return super().format(record)

class OptimizationLogger:
    """Logger class for optimization processes with styled output."""
    
    def __init__(self, name: str = "optimization_logger", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        # Create console handler with custom formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter('%(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(level)
    
    def title(self, message: str, color: str = 'TITLE'):
        """Log a title with surrounding lines."""
        extra = {'color': color, 'style': 'title'}
        self.logger.info(message, extra=extra)
    
    def section(self, message: str):
        """Log a section header."""
        self.logger.info(message, extra={'color': 'SECTION'})
    
    def key_value(self, key: str, value: Any):
        """Log a key-value pair."""
        self.logger.info('', extra={'color': 'KEY', 'key_value': (key, value)})
    
    def optimization_results(self, optimized_params: Dict[str, float], fun_value: float, success: bool):
        """Log optimization results in a formatted manner."""
        self.title("Optimization Results", color='SUCCESS')
        for key, value in optimized_params.items():
            self.key_value(key, f"{value:.4f}")
        self.key_value("Optimized Function Value", f"{-fun_value:.4f}")
        self.key_value("Optimization Success", str(success))