#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel('DEBUG')

# if len(_logger.handlers) == 0:
# Don't add a stream handler if any handler already exists, i.e. user knows what they're doing
_handler = logging.StreamHandler()
_FMT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d, %H:%M:%S',)
_handler.setFormatter(_FMT)
_handler.setLevel('INFO')
_logger.addHandler(_handler)

_logger.info('Importing PBjam')

from .version import __version__
from .priors import kde
from .session import session
from .asy_peakbag import asymp_spec_model, asymptotic_fit
from .peakbag import peakbag
from .ellone import ellone
from .star import star
from .mcmc import mcmc
from .mcmc import nested
