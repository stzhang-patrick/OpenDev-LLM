from .utils.logging import get_logger

logger_path = '/mnt/proj/workspace/FrogEngine/src/test.log'
logger = get_logger(__name__)

def test_logging():
    logger.info("This is from test.py")