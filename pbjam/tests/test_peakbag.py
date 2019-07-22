from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)
import pandas as pd
import pbjam

def test_peakbag():
    ''' Test init '''
    f = np.linspace(10, 200, 1000)
    snr = np.ones(len(f))
    asy_result = []
    pb = pbjam.peakbag(f, snr, asy_result, init=False)

def test_make_start():
    ''' Test make_start '''
    f = np.linspace(10, 200, 1000)
    snr = np.ones(len(f))
    asy_result = {'modeID': pd.DataFrame(data=[[0, 100],[0, 110],
                                                [2, 98], [2, 108]],
                                         columns=['ell', 'nu_med']),
                  'summary': pd.DataFrame(data=[[10, 105, 50, 30, 0.1, 2.0]],
                  index=['mean'],
                  columns=['dnu', 'numax', 'env_height',
                            'env_width', 'mode_width', 'd02'])
                  }
    pb = pbjam.peakbag(f, snr, asy_result, init=False)
    pb.make_start()
    assert len(pb.start['l0']) == 2

def test_trim_ladder():
    ''' Test trim ladder '''
    f = np.linspace(10, 200, 10000)
    snr = np.ones(len(f))
    asy_result = {'modeID': pd.DataFrame(data=[[0, 100],[0, 110],
                                                [2, 98], [2, 108]],
                                         columns=['ell', 'nu_med']),
                  'summary': pd.DataFrame(data=[[10, 105, 50, 30, 0.1, 2.0]],
                  index=['mean'],
                  columns=['dnu', 'numax', 'env_height',
                            'env_width', 'mode_width', 'd02'])
                  }
    pb = pbjam.peakbag(f, snr, asy_result, init=False)
    pb.make_start()
    pb.trim_ladder()

def test_simple_compile():
    ''' Test compile of simple model '''
    f = np.linspace(10, 200, 10000)
    snr = np.ones(len(f))
    asy_result = {'modeID': pd.DataFrame(data=[[0, 100],[0, 110],
                                                [2, 98], [2, 108]],
                                         columns=['ell', 'nu_med']),
                  'summary': pd.DataFrame(data=[[10, 105, 50, 30, 0.1, 2.0]],
                  index=['mean'],
                  columns=['dnu', 'numax', 'env_height',
                            'env_width', 'mode_width', 'd02'])
                  }
    pb = pbjam.peakbag(f, snr, asy_result)
    pb.simple()

def test_model_gp_compile():
    ''' Test compile of gp model '''
    f = np.linspace(10, 200, 10000)
    snr = np.ones(len(f))
    asy_result = {'modeID': pd.DataFrame(data=[[0, 100],[0, 110],
                                                [2, 98], [2, 108]],
                                         columns=['ell', 'nu_med']),
                  'summary': pd.DataFrame(data=[[10, 105, 50, 30, 0.1, 2.0]],
                  index=['mean'],
                  columns=['dnu', 'numax', 'env_height',
                            'env_width', 'mode_width', 'd02'])
                  }
    pb = pbjam.peakbag(f, snr, asy_result)
    pb.model_gp()

@pytest.mark.slow
def test_sample():
    pass # TODO write a test here.
