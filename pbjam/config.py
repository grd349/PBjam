# Configures the pbjam package upon import
import logging, sys

HANDLER_FMT = logging.Formatter("%(asctime)-15s : %(levelname)-8s : %(name)-17s : %(message)s")


class info_filter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


class stdout_handler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)
        self.setFormatter(HANDLER_FMT)
        self.addFilter(info_filter())


class stderr_handler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stderr)
        self.setFormatter(HANDLER_FMT)
        self.setLevel('WARNING')
