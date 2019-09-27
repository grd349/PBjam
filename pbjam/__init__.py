#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__

from .guess_epsilon import epsilon
from .priors import kde
from .jar import session
from .asy_peakbag import asymp_spec_model
from .peakbag import peakbag
from .star import star
from .mcmc import mcmc
