from runner.utils.logging import get_logger

logger_path = '/mnt/proj/workspace/FrogEngine/src/test.log'
logger = get_logger(__name__, logger_path=logger_path)


from termcolor import colored  
  
  
# 要记录的信息  
info = "this is a test"  

info = colored(info, "red")

# 记录彩色文本到日志  
logger.info(f"this should be plain: {info}")