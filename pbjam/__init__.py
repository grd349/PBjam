#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .peakbagging import DynestyPeakbag, NumPyroPeakbag
from .core import star
from .modeID import modeIDsampler

#from .session import session
 
