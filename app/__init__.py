import logging

__version__ = "1.0.0"
__author__ = "Oleh Kuzmenko"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.info(f"Initializing Retrosynthesis App v{__version__}")
