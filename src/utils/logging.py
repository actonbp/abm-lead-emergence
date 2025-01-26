"""Logging utilities for leadership emergence model."""

import logging
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    """Setup logging configuration.
    
    Args:
        output_dir: Directory to save log file
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create log file path
    log_file = output_dir / f'sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler) 