import tensorflow as tf
import logging

def _setup_logging():
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = logging.getLogger('tensorflow')
    logger.propagate = False
    logger.handlers[0].formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_setup_logging()


def info(msg):
     logging.info(msg)


def warning(msg):
    logging.warning(msg)

