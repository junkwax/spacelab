import logging

def setup_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("spacelab.log"),
            logging.StreamHandler()
        ]
    )