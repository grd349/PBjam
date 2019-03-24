from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import guess_epsilon

def test_epsilon():
    ''' Test init '''
    eps = guess_epsilon.epsilon()

def test_epsilon_vrard():
    ''' Test a simple red giant result '''
    eps = guess_epsilon.epsilon(method='Vrard')
    result = eps.vrard(4.5)
    assert_almost_equal(result[0], 1.0, 0.1)
    assert_almost_equal(result[1], 0.1, 0.01)

def test_epsilon_vrard_giant():
    ''' Test a simple red giant result '''
    eps = guess_epsilon.epsilon(method='Vrard')
    result = eps(4.5)
    assert_almost_equal(result[0], 1.0, 0.1)
    assert_almost_equal(result[1], 0.1, 0.01)

def test_epsilon_vrard_dwarf():
    ''' Check Vrard unc high for the Sun '''
    eps = guess_epsilon.epsilon(method='Vrard')
    result = eps(135.0, 3050.0)
    assert_almost_equal(result[1], 1.0, 0.01)

def test_read_prior_data():
    ''' Check we can read in the prior data '''
    eps = guess_epsilon.epsilon()
    eps.read_prior_data()

def test_make_kde():
    ''' Check we can read in the prior data '''
    eps = guess_epsilon.epsilon()
    eps.read_prior_data()
    eps.make_kde()

def test_normal():
    eps = guess_epsilon.epsilon()
    assert_almost_equal(eps.normal(0.0, 0.00001, 1.0), 0.0, 0.01)

def test_likelihood():
    pass

def test_kde_sampler():
    pass

def test_application():
    eps = guess_epsilon.epsilon(method='kde')
    res = eps(dnu = 10.0, numax = 120.0, teff=4800.0,
              dnu_err = 0.1, numax_err = 1.0, teff_err = 70.0)
    assert_almost_equal(res[0], 1.2, 0.05)
    assert_almost_equal(res[1], 0.12, 0.02)

def test_application_sparse():
    eps = guess_epsilon.epsilon(method='kde')
    res = eps(dnu = 10.0, dnu_err = 0.1)
    assert_almost_equal(res[0], 1.2, 0.2)
