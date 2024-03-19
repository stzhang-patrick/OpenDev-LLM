import logging
import sys
from typing import Optional
import os
from datetime import datetime, timezone, timedelta


class TimeZoneFormatter(logging.Formatter):
    r"""
    Custom logging formatter that converts UTC time to local time.
    'tz_offset' is the time zone offset in hours.
    e.g. UTC+8:00 -> `tz_offset` = 8
    """
    def __init__(self, fmt=None, datefmt=None, tz_offset=0):
        super().__init__(fmt, datefmt)
        self.tz_offset = tz_offset

    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        dt = dt.astimezone(timezone(timedelta(hours=self.tz_offset)))
        return dt

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat(timespec='milliseconds')


def get_logger(name: str,
               logger_path: Optional[str] = None,
               tz_offset: Optional[int] = None,
               ) -> logging.Logger:
    r"""
    Return a standard logger with a stream handler to stdout and a file 
    handler to a log file if 'logger_path' is specified.
    'tz_offset' is the time zone offset in hours, default to 0, which 
    corresponds to UTC.
    """

    if tz_offset is None:
        tz_offset = int(os.getenv('TZ_OFFSET', 0))

    formatter = TimeZoneFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
        datefmt="%m/%d/%Y %H:%M:%S",
        tz_offset=tz_offset
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if logger_path is not None:
        # Create the log file if it does not exist
        if not os.path.exists(logger_path):
            with open(logger_path, "w"):
                pass
        
        file_handler = logging.FileHandler(logger_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger