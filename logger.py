
import time
import logging



formatter = logging.Formatter('%(levelname)s-%(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# first file logger
logger = setup_logger('infoLogger', 'completed.log')
logger.info('New Info Session')

# second file logger
errorLogger = setup_logger('errorLogger', 'errors.log', logging.ERROR)
errorLogger.error('New Error Session')


