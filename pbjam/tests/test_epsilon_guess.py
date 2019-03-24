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
