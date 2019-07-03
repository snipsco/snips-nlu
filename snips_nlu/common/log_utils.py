from __future__ import unicode_literals

from builtins import str
from datetime import datetime
from functools import wraps

from snips_nlu.common.utils import json_debug_string


class DifferedLoggingMessage(object):

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fn(*self.args, **self.kwargs))


def log_elapsed_time(logger, level, output_msg=None):
    if output_msg is None:
        output_msg = "Elapsed time ->:\n{elapsed_time}"

    def get_wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            start = datetime.now()
            msg_fmt = dict()
            res = fn(*args, **kwargs)
            if "elapsed_time" in output_msg:
                msg_fmt["elapsed_time"] = datetime.now() - start
            logger.log(level, output_msg.format(**msg_fmt))
            return res

        return wrapped

    return get_wrapper


def log_result(logger, level, output_msg=None):
    if output_msg is None:
        output_msg = "Result ->:\n{result}"

    def get_wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            msg_fmt = dict()
            res = fn(*args, **kwargs)
            if "result" in output_msg:
                try:
                    res_debug_string = json_debug_string(res)
                except TypeError:
                    res_debug_string = str(res)
                msg_fmt["result"] = res_debug_string
            logger.log(level, output_msg.format(**msg_fmt))
            return res

        return wrapped

    return get_wrapper
