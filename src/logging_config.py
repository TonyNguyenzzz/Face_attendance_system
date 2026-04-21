"""
Logging configuration module for Face Recognition Attendance System.
Provides centralized logging setup with configurable levels and handlers.
"""
import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path


def get_logger(name: str, level: int = logging.ERROR) -> logging.Logger:
    """
    Get a logger instance with the specified name and level.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure the logging system based on configuration settings.
    
    Args:
        config: Configuration dictionary with logging_level and disable_console_logging
    """
    log_level_str = config.get("logging_level", "ERROR").upper()
    log_level = getattr(logging, log_level_str, logging.ERROR)
    disable_console = config.get("disable_console_logging", False)
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure root logger
    if not disable_console:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Set levels for specific loggers
    loggers = ["GUI", "FaceEmbedding", "AttendanceSystem", "DatabaseHandler", 
               "FaceDetector", "CameraHandler", "TrainWorker"]
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = not disable_console
    
    if not disable_console:
        logging.info(f"Logging configured with level: {log_level_str}")
