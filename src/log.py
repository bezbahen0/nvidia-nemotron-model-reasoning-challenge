import logging
import sys
import os

class OriginFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'origin'):
            record.origin = record.name
        return True

def setup_logging():
    root_logger = logging.getLogger()
    
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(OriginFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(origin)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

setup_logging()

class Logger:
    def __init__(self):
        self._logger = logging.getLogger("App")
    
    def _get_caller_info(self):
        try:
            frame = sys._getframe(3) 
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            return f"{filename}:{lineno}"
        except ValueError:
            return "unknown"

    def _log(self, level_method, message, *args, **kwargs):
        origin = self._get_caller_info()
        level_method(message, *args, extra={'origin': origin}, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log(self._logger.info, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log(self._logger.error, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._log(self._logger.warning, message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        self._log(self._logger.debug, message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        self._log(self._logger.exception, message, *args, **kwargs)

logger = Logger()
