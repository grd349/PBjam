from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)

from pbjam import guess_epsilon

def test_epsilon():
    eps = guess_epsilon.epsilon()

def test_epsilon_vrard():
    eps = guess_epsilon.epsilon(method='Vrard')
    # Test a simple red giant result
    result = eps.vrard(4.5)
    assert_almost_equal(result[0], 1.0, 0.1)
    assert_almost_equal(result[1], 0.1, 0.01)

def test_epsilon_vrard_giant():
    eps = guess_epsilon.epsilon(method='Vrard')
    # Test a simple red giant result
    result = eps(4.5)
    assert_almost_equal(result[0], 1.0, 0.1)
    assert_almost_equal(result[1], 0.1, 0.01)

def test_epsilon_vrard_dwarf():
    eps = guess_epsilon.epsilon(method='Vrard')
    # Check unc high for the Sun
    result = eps(135.0, 3050.0)
    assert_almost_equal(result[1], 1.0, 0.01)
