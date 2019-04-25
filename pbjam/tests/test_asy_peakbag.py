from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import asy_peakbag

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
    asy = model.model(120.0, 10.0, 0.08, 1.0, 0.03, 10.0, 18.0, 0.2)

def test_model_call():
    model = asy_peakbag.asymp_spec_model(np.linspace(1,10,10), 2)
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
    bounds =  [[x0[0]*0.9, x0[0]*1.1], # numax
              [x0[1]*0.9, x0[1]*1.1], # Dnu
              [0.5, 2.5], # eps
              [0, 1], # alpha
              [0.04, 0.2], # d02
              [x0[5]*0.5, x0[5]*1.5], #hmax
              [x0[6]*0.9, x0[6]*1.1], #Ewidth
              [-2, 1.0], # mode width (log10)
              [1e2, 1e4]] # seff    
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)

def test_mcmc_likelihood():
    x0 = np.array([120.0, 10.0, 0.1, 1.0, 0.03, 100.0, 20.0, -1, 2.6])
    bounds =  [[x0[0]*0.9, x0[0]*1.1], # numax
          [x0[1]*0.9, x0[1]*1.1], # Dnu
          [0.5, 2.5], # eps
          [0, 1], # alpha
          [0.04, 0.2], # d02
          [x0[5]*0.5, x0[5]*1.5], #hmax
          [x0[6]*0.9, x0[6]*1.1], #Ewidth
          [-2, 1.0], # mode width (log10)
          [1e2, 1e4]] # seff    
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = np.ones(len(f))
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)
    ll = mcmc.likelihood(x0)

def test_mcmc_call():
    x0 = np.array([120.0, # numax
                   10.0, # Dnu
                   1.0, # eps
                   0.001, # alpha
                   0.1, # d02/dnu
                   10.0,  #hmax
                   20.0, #Ewidth
                   -1, # mode width (log10)
                   4000]) # seff  
          
    bounds =  [[x0[0]*0.9, x0[0]*1.1], # numax
               [x0[1]*0.9, x0[1]*1.1], # Dnu
               [0.5, 2.5], # eps
               [0, 1], # alpha
               [0.04, 0.2], # d02/dnu
               [x0[5]*0.5, x0[5]*1.5], #hmax
               [x0[6]*0.9, x0[6]*1.1], #Ewidth
               [-2, 1.0], # mode width (log10)
               [1e2, 1e4]] # seff    
    
    f = np.linspace(0, 288, 100)
    model = asy_peakbag.asymp_spec_model(f, 2)
    snr = model(x0[:-1])
    mcmc = asy_peakbag.mcmc(f, snr, model, x0, bounds)
    samples = mcmc(10, 20)
    assert_allclose(x0, samples.mean(axis=0), 1)
