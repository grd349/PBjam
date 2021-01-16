#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import logging
# from .config import stdout_handler, stderr_handler
from .config import console_handler

# Setup global pbjam logger
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')  # <--- minimum possible level for global pbjam logger

# logger.addHandler(stdout_handler())
# logger.addHandler(stderr_handler())
logger.addHandler(console_handler())

logger.debug(f'Initializing {__name__}')

from .version import __version__
logger.debug(f'version == {__version__}')

from .priors import kde
from .session import session
from .asy_peakbag import asymp_spec_model, asymptotic_fit
from .peakbag import peakbag
from .ellone import ellone
from .star import star
from .mcmc import mcmc
from .mcmc import nested

logger.debug(f'Initialized {__name__}')
