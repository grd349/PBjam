from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import asy_peakbag

params = ['large separation', 'numax', 'epsilon', 'd02', 'alpha',
          'p-mode envelope height', 'p-mode envelope width',
          'mode width (log10)', 'Teff', 'bp_rp']
generic_params = [10**1.23, 10**2.34, 1.36,
                  10**0.3, 10**-2.4,
                  1.4, 1.35, -0.75,
                  10**3.69, 1.3]
generic_bounds = [(1, 600), (1, 6000), (0.4, 1.6),
                  (0.0, 100.0), (0.0, 0.1),
                  (0.1, 1e6), (0.1, 5000), (-3, 2),
                  (2000, 7500), (-1, 2.5)]
generic_gaussian = [(0, 0) for n in params]

def test_model():
    """Can I initialize a model instance"""
    model = asy_peakbag.asymp_spec_model(np.linspace(1,2,3), 2)

def test_model_lorentzian():
    ''' Very simple lorentzian test '''
    model = asy_peakbag.asymp_spec_model(np.linspace(1,200,3), 2)
    lor = model.lor(1, 10.0, np.log10(1.0))
    assert_almost_equal(lor[0], 10.0, 0.1)
    assert_almost_equal(lor[1], 0.0, 0.1)

def test_model_pair():
    """Can I define a pair of modes in a model instance"""
    model = asy_peakbag.asymp_spec_model(np.linspace(1,10,10), 2)
    pair = model.pair(6, 10, 0.1, 2.0)
    assert_almost_equal(pair[5], 10.0, 0.1)

def test_model_asy():
    """Can I generate a complete model realization"""
    model = asy_peakbag.asymp_spec_model(np.linspace(1,10,10), 2)

    asy = model.model(*generic_params)

def test_model_call():
    model = asy_peakbag.asymp_spec_model(np.linspace(1,10,10), 2)
    asy = model(generic_params)

def test_prior():
    prior = asy_peakbag.Prior([], [])

def test_prior_bounds():
    prior = asy_peakbag.Prior([(10, 5000), (1, 200), (-3, 2)],
                              [(0, 0), (0,0), (0,0)])
    lp = prior.pbound([120.0, 10.0, 1.0])
    assert_almost_equal(lp, 0.0, 2)
    lp = prior.pbound([1.0, 10.0, 1.0])
    assert_almost_equal(lp, -np.inf, 1)
    lp = prior.pbound([100.0, 1000.0, 1.0])
    assert_almost_equal(lp, -np.inf, 1)

def test_prior_call():
    prior = asy_peakbag.Prior(generic_bounds, generic_gaussian)
    lp_bound = prior.pbound(generic_params)
    assert lp_bound == 0
    lp_gauss = prior.pgaussian(generic_params)
    assert lp_gauss >= 0
    p = generic_params
    lp = prior(generic_params)
    assert lp > 1e-3

def test_mcmc():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    prior = asy_peakbag.Prior(generic_bounds, generic_gaussian)
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, prior)

def test_mcmc_likelihood():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    prior = asy_peakbag.Prior(generic_bounds, generic_gaussian)
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, prior)
    ll = mcmc.likelihood(x0)

def test_mcmc_call():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = model(x0)
    prior = asy_peakbag.Prior(generic_bounds, generic_gaussian)
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, prior)
    samples = mcmc(10, 20)
