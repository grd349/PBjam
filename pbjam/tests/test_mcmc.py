import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                            assert_allclose)
from ..mcmc import mcmc, nested

def like(p):
    return -0.5 * (p['x']**2 + p['y']**2) - np.log(2*np.pi)

def prior(p):
    return 0.0

def test_nest_inst():
    ''' Test instance of nest '''
    names = ['x', 'y']
    bounds = [[-1, 1], [-1, 1]]
    nest = nested(names, bounds, like, prior)

def test_mcmc_inst():
    ''' Test instance of mcmc ''' 
    start = np.zeros(2)
    mc = mcmc(start, like, prior)
