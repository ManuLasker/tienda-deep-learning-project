import logging

BASIC_LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s : %(message)s"


def get_logger(name: str, _format: str = BASIC_LOGGING_FORMAT):
    handler = logging.StreamHandler()
    formatter = logging.Formatter(_format)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

