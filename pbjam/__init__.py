#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .peakbagging import basePeakbag
from .core import star, session
from .modeID import modeID
 
