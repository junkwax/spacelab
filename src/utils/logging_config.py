import logging
from logging.handlers import RotatingFileHandler

def setup_logging(level=logging.INFO, log_file="spacelab.log"):
    """Configure logging with file rotation and stream handling."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # File handler with rotation (10 MB per file, max 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)