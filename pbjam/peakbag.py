"""A class that performs high fidelity peakbagging of all identified modes,
using the asypeakbag output as priors, and taking into account rotational
and inclincational splitting."""

import numpy as np
from pbjam import epsilon, mcmc
import lightkurve as lk

"empty for now"
