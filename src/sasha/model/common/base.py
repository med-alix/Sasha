from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional
import logging
from colorlog import ColoredFormatter


class BaseLogger(BaseModel):

    """Base class for logging functionality that can be inherited by any class"""
    logger_name: str = Field(default="BaseLogger")
    log_level: int = Field(default=logging.INFO)
    _logger: Optional[logging.Logger] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup a colored logger with consistent formatting"""
        logger = logging.getLogger(self.logger_name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        logger.setLevel(self.log_level)
        
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s%(reset)s - %(levelname)s - %(bold_white)s%(name)s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @property
    def logger(self) -> logging.Logger:
        """Access the logger instance"""
        if self._logger is None:
            self._logger = self._setup_logger()
        return self._logger
