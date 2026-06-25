import logging

from styne.utility.logconfig import enable_logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["enable_logging"]
