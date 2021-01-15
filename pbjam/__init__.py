#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import logging
HANDLER_FMT = logging.Formatter("%(asctime)-15s : %(levelname)-8s : %(name)-17s : %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

# Add a stream handler at level=='INFO' - should we do this?
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(HANDLER_FMT)
_stream_handler.setLevel('INFO')

logger.addHandler(_stream_handler)
logger.debug('Importing PBjam')

from .version import __version__
from .priors import kde
from .session import session
from .asy_peakbag import asymp_spec_model, asymptotic_fit
from .peakbag import peakbag
from .ellone import ellone
from .star import star
from .mcmc import mcmc
from .mcmc import nested
