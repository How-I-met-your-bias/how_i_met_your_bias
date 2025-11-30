"""
Nathan Roos

Misceallaneous utilities
"""

import time
import datetime
import logging


def get_date_now() -> str:
    """
    Return a string with the current date and time "dd_mm_yy_Hh_Mmin_Ss"
    """
    return time.strftime("%d_%m_%y_%Hh_%Mmin_%Ss", time.localtime())


def sec_to_dhms(seconds: int) -> str:
    """
    Convert seconds to days, hours, minutes and seconds.
    """
    return str(datetime.timedelta(seconds=seconds))


class DisableImshowWarning:
    """Context manager to disable the warning of imshow
    'Clipping input data to the valid range ...'

    Usage:
    with DisableImshowWarning():
        plt.imshow(data)
        ...
    """

    def __init__(self):
        self.logger = logging.getLogger()
        self.old_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(100)

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.setLevel(self.old_level)
