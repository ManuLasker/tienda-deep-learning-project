import logging

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.DEBUG)

def get_logger(name):
  logger = logging.getLogger(name)
  return logger