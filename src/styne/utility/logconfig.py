import logging


def enable_logging(level=logging.INFO):
    """Enable stdout logging for the styne library."""
    logger = logging.getLogger("styne")
    logger.setLevel(level)

    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return

    handler = logging.StreamHandler()
    if level <= logging.DEBUG:
        formatString = "[%(name)s] %(levelname)s %(message)s"
    else:
        formatString = "[styne] %(message)s"

    handler.setFormatter(logging.Formatter(formatString))
    logger.addHandler(handler)
