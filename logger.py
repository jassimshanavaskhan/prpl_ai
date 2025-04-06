import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jaz.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
