from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import asy_peakbag

params = ['numax', 'large separation', 'epsilon', 'alpha', 'd02',
          'p-mode envelope height', 'p-mode envelope width',
          'mode width (log10)', 'Teff', 'bp_rp']
generic_params = [120.0, 10.0, 0.08, 1.0, 0.03, 10.0, 18.0, -1.0, 4500.0, 1.25]
generic_bounds = [(1, 6000), (1, 200), (0.03, 0.2), (0.5, 2.5), (0.0, 0.1),
                  (0.1, 1e6), (0.1, 5000),
                  (-3, 2), (2000, 7500), (-1, 2.5)]

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
    prior = asy_peakbag.Prior([(10, 5000), (1, 200)], [(0, 0), (0,0)])
    lp = prior.pbound([120.0, 10.0])
    assert_almost_equal(lp, 0.0, 2)
    lp = prior.pbound([1.0, 10.0])
    assert_almost_equal(lp, -np.inf, 1)
    lp = prior.pbound([100.0, 1000.0])
    assert_almost_equal(lp, -np.inf, 1)

def test_prior_call():
    # numax, dnu, d02, eps, alpha, hmax, Envwidth, w, seff
    gaussian = []
    prior = asy_peakbag.Prior(generic_bounds, gaussian)
    lp = prior(generic_params)
    assert_almost_equal(lp, 0.1, 0.1)

def test_mcmc():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)

def test_mcmc_likelihood():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)
    ll = mcmc.likelihood(x0)

def test_mcmc_call():
    x0 = generic_params
    bounds =  generic_bounds
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = model(x0)
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)
    samples = mcmc(10, 20)
    assert_allclose(x0, samples.mean(axis=0), 1)
