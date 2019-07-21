from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import guess_epsilon

def test_epsilon():
    ''' Test init '''
    eps = guess_epsilon.epsilon()

def test_read_prior_data():
    ''' Check we can read in the prior data '''
    eps = guess_epsilon.epsilon()
    eps.read_prior_data()

def test_make_kde():
    '''
    Check we can read in the prior data
    and make a kde.
    '''
    eps = guess_epsilon.epsilon()
    eps.read_prior_data()
    eps.make_kde()

def test_normal():
    eps = guess_epsilon.epsilon()
    assert_almost_equal(eps.normal(0.0, 0.00001, 1.0), 0.0, 0.01)

def test_to_log10():
    eps = guess_epsilon.epsilon()
    ret = eps.to_log10(10, 0.1)
    assert_almost_equal(ret[0], 1.0, 0.01)
    assert_almost_equal(ret[1], 0.01, 0.001)

def test_obs_to_log():
    eps = guess_epsilon.epsilon()
    obs = {'dnu': [10.0, 0.1],
           'numax': [100.0, 1.0],
           'teff': [4000.0, 100.0],
           'bp_rp': [1.3, 0.05]}
    eps.obs_to_log(obs)
    assert_almost_equal(eps.log_obs['dnu'][0], 1.0, 0.01)

def test_likelihood():
    # TODO add test here
    pass

def test_kde_sampler():
    # TODO add test here
    pass

@pytest.mark.slow
def test_application():
    eps = guess_epsilon.epsilon()
    res = eps(dnu = [10.0, 0.1],
              numax = [120.0, 1.0],
              teff = [4800.0, 70.0])
    assert_almost_equal(res[0], 1.2, 0.2)

@pytest.mark.slow
def test_application_sparse():
    eps = guess_epsilon.epsilon()
    res = eps(dnu = [10.0, 0.1])
    assert_almost_equal(res[0], 1.2, 0.2)

@pytest.mark.slow
def test_application_sparse_numax():
    eps = guess_epsilon.epsilon()
    res = eps(numax = [120.0, 1.0])
    assert_almost_equal(res[0], 1.2, 0.2)

@pytest.mark.slow
def test_application_sparse_teff():
    eps = guess_epsilon.epsilon()
    res = eps(teff = [4800.0, 70.0])
    assert_almost_equal(res[0], 1.2, 0.2)
