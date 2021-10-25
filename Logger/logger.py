# -*- coding: UTF-8 -*-
import logging

class Logger(object):

    """
    fmt option:
    属性名称             格式                描述
    args           不需要用户格式化         合并到msg产生message的包含参数的元组
    asctime         %(asctime)s           表示日志记录的时间
    created         %(created)f           表示日志创建的时间
    exc_info        不需要格式化             异常元组（sys.exc_info）未发生异常则为None
    funcName        %(funcName)s           函数名包括调用日志记录
    levelname       %(levelname)s          消息文本记录级别（‘DEBUG’, 'INFO', 'WARNING', 'ERROR', 'CRITICAL'）
    levelno         %(levelno)s            消息数字的记录级别
    lineno          %(lineno)d             发出日志调用所在的源行号
    module--模块     %（modules）            模块
    msecs           %(msecs)d              logger创建的时间的毫秒部分
    msg              不需要格式化
    名称             %（name）s              用于记录调用的日志记录器的名称
    pathname         %(pathname)s          发出日志记录调用的源文件的完整路径名
    process          %(process)d            进程ID
    processName      %(processName)s        进程名
    relativeCreated  %(relativeCreated)d    以毫秒数表示的LogRecord被创建的时间， 即相对于Logging模块被加载时间的插值
    stack_info        不需要格式化
    thread           %(thread)d             线程ID
    threadName       %(threadName)s         线程名（如果可用）

    """

    level_options = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_dir, fmt="%(asctime)s - %(pathname)s/%(filename)s [line: %(lineno)d] - [%(levelname)s] %(message)s", level='info'):
        self.logging = logging.getLogger("test_Logger")
        self.logging.setLevel(self.level_options[level])
        formatter = logging.Formatter(fmt)
        # steam handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logging.addHandler(sh)
        # filehandler
        fh = logging.FileHandler(filename=log_dir, mode="w", encoding='utf-8')
        fh.setFormatter(formatter)
        self.logging.addHandler(fh)


if __name__ == "__main__":
    logger = Logger(log_dir="./test.log", level="debug")
    logger.logging.debug("debug")
    logger.logging.info("info")
    logger.logging.warning("warn")
    logger.logging.error("error")
    logger.logging.critical("crit")
