#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .guess_epsilon import epsilon
from .asy_peakbag import model, Prior, mcmc
