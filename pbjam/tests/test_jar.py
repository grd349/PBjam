"""Tests for the jar module"""

from pbjam.jar import normal, to_log10, get_priorpath, get_percentiles
import pbjam.tests.pbjam_tests as pbt
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

def test_normal():
    """Test for the log of a normal distribution"""
    
    # setup
    func = normal
    inp = [0, 0, 1]
    inp_arr = [np.linspace(-10,10,100), 0, 1]

    # simple tests    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)
    pbt.right_shape(func, inp_arr, (len(inp_arr[0]),))
    
    # check height at mean is correct
    assert(10**func(*inp)==1)
    
def test_to_log10():
    """Test for the log of a normal distribution"""
    
    # setup
    func = to_log10
    inp = [10,1]
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, list)
    
    # check some inputs
    assert(func(*inp)[0]==1)
    assert_almost_equal(func(*inp)[1], 0.043429, 5)
    
    inp = [1e5,1e4]
    assert(func(*inp)[0]==5)

def test_get_priorpath():
    """Tests the function for getting the default prior_data filepath"""
    
    # setup
    func = get_priorpath
    
    # simple tests
    pbt.does_it_run(func, None)
    pbt.does_it_return(func, None)
    pbt.right_type(func, None, str)
    
    # check some inputs
    assert('prior_data.csv' in func())
    assert('data' in func())
    
def test_get_percentiles():
    """Tests the function for getting the percentiless of a distribution"""
    
    # setup
    func = get_percentiles
    inp = [np.random.normal(0,1, size = 10), 3]
    
    #print(func(*inp)[])
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, (2*inp[1]+1,))
    
    # check some different inputs
    inp = [np.random.normal(0,1, size = 30000), 5]
    pbt.right_shape(func, inp, (2*inp[1]+1,))
    
    inp = [[0,0,0,1,1], 1]
    assert_array_equal(func(*inp), [0., 0., 1.])

    