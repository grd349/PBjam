from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import asy_peakbag

def test_model():
    model = asy_peakbag.model(np.linspace(1,2,3))

def test_model_lorentzian():
    ''' Very simple lorentzian test '''
    model = asy_peakbag.model(np.linspace(1,200,3))
    lor = model.lor(1, 10.0, np.log10(1.0))
    assert_almost_equal(lor[0], 10.0, 0.1)
    assert_almost_equal(lor[1], 0.0, 0.1)

def test_model_pair():
    model = asy_peakbag.model(np.linspace(1,10,10))
    pair = model.pair(6, 10, 0.1, 2.0)
    assert_almost_equal(pair[5], 10.0, 0.1)

def test_model_asy():
    model = asy_peakbag.model(np.linspace(1,10,10))
    asy = model.asy(120.0, 10.0, 0.08, 1.0, 0.03, 10.0, 18.0, 0.2)

def test_model_call():
    model = asy_peakbag.model(np.linspace(1,10,10))
    asy = model([120.0, 10.0, 0.08, 1.0, 0.03, 10.0, 18.0, 0.2])

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
    bounds = [(1, 6000), (1, 200), (0.03, 0.2), (0.5, 2.5),
              (0.0, 0.1), (0.1, 1e6), (0.1, 5000), (-3, 2), (2, 4)]
    gaussian = []
    prior = asy_peakbag.Prior(bounds, gaussian)
    p = [120.0, 10.0, 0.1, 1.0, 0.03, 100.0, 20.0, -1, 2.6]
    lp = prior(p)
    assert_almost_equal(lp, 0.1, 0.1)

def test_mcmc():
    x0 = np.array([120.0, 10.0, 0.1, 1.0, 0.03, 100.0, 20.0, -1, 2.6])
    f = np.linspace(0, 288, 100)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, x0)

def test_mcmc_likelihood():
    x0 = np.array([120.0, 10.0, 0.1, 1.0, 0.03, 100.0, 20.0, -1, 2.6])
    f = np.linspace(0, 288, 100)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, x0)
    ll = mcmc.likelihood(x0)

def test_mcmc_call():
    x0 = np.array([120.0, 10.0, 0.1, 1.0, 0.03, 100.0, 20.0, -1, 2.6])
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.model(f)
    snr = model(x0[:-1])
    mcmc = asy_peakbag.mcmc(f, snr, x0)
    samples = mcmc(x0)
    assert_allclose(x0, samples.mean(axis=0), 1)
