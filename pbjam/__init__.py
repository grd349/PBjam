#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# Setup global pbjam logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')  # <--- minimum level for global pbjam package logger

# Setup console handler
from .jar import stream_handler
console_handler = stream_handler(level='INFO')
logger.addHandler(console_handler)
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
