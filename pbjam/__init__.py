#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler())
# TODO: format stream handler if need be.

from .version import __version__
from .priors import kde
from .session import session
from .asy_peakbag import asymp_spec_model, asymptotic_fit
from .peakbag import peakbag
from .ellone import ellone
from .star import star
from .mcmc import mcmc
from .mcmc import nested
