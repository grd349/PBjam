#import pbjam
#from pbjam import asy_peakbag
#from pbjam.asy_peakbag import get_nmax
#import pytest


## Function
# does it run
# does it return something if it should? Or set an attribute
# is the output the type of thing you want
# is the output the shape you want

## Reasonable/unreasonable?
# is the output something reasonable if I give it something reasonable?
# is the output something unreasonable if I give it something unreasonable

from ..asy_peakbag import asymp_spec_model

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_allclose

def load_reasonable():
    reas = {}
    reas['solar'] = [np.log10(135.0), # dnu
                     np.log10(3050.0), # numax 
                     1.25, #eps
                     0.5, # d02
                     -2.5, # alpha
                     1.5, # envheight
                     2.2, # envwidth
                     0.0, # modwidth
                     3.77, #teff
                     0.8] # bp_rp
    
    reas['boeing'] = [1.23072917362692,
                      2.34640449860884,
                      1.33476843739203,
                      0.347104562462547,
                      -2.17266900058679,
                      1.47003574648704,
                      1.31305213708824,
                      -0.668935969601942,
                      3.69363755249007,
                      1.22605501887223]
    
    reas['SC'] = np.vstack((np.linspace(1e-20,8500,10000),
                            np.linspace(1e-20,8500,10000)))
    
    reas['norders'] = 7
    
    return reas


def test_asymp_spec_model_init():
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['norders'])
    
    init_attributes = ['_asymptotic_relation', '_get_enns', '_get_nmax', 
                       '_lor', '_pair', 'f', 'model', 'norders']
    
    for attr in init_attributes:
        assert(hasattr(mod, attr))
    

def test_get_nmax_function(): 
    
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['norders'])
    dnu, numax, eps = 10**R['solar'][0], 10**R['solar'][1], R['solar'][2]
    
    # does it run?
    mod._get_nmax(dnu, numax, eps) 
    
    # does it return something if it should? Or set an attribute
    assert(mod._get_nmax(dnu, numax, eps) is not None) 
    
    # is the output the type of thing you want
    out = mod._get_nmax(dnu, numax, eps) 
    assert(isinstance(out, float))
    
    # is the output the shape you want
    assert(np.shape(out) == ())
    
    n = 10
    out = mod._get_nmax(np.float64(dnu).repeat(n), 
                        np.float64(numax).repeat(n),
                        np.float64(eps).repeat(n))
    assert(np.shape(out) == (n,))
    
    
def test_get_nmax_reasonable():

    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0],  R['norders'])
    
    # is the output something reasonable if I give it something reasonable?
    # Solar values
    assert_almost_equal(mod._get_nmax(10**R['solar'][0], 
                                      10**R['solar'][1], 
                                      R['solar'][2]), 21, decimal = 0)
    # KIC4448777
    assert_almost_equal(mod._get_nmax(10**R['boeing'][0], 
                                      10**R['boeing'][1], 
                                      R['boeing'][2]), 11, decimal = 0)
    
def test_get_enns_function():
    
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['norders'])
    dnu, numax, eps = 10**R['solar'][0], 10**R['solar'][1], R['solar'][2]
    
    norders = R['norders']
    nmax = mod._get_nmax(dnu, numax, eps)
    
    # does it run
    out = mod._get_enns(nmax, norders)

    # does it return something if it should? Or set an attribute
    assert(out is not None)

    # is the output the type of thing you want 
    assert(isinstance(out, np.ndarray))

    # is the output of the shape you want        
    assert(np.shape(out)==(norders,))
    
    nsamples = 10
    out = mod._get_enns(np.zeros(nsamples)+nmax, norders)
    assert(isinstance(out, np.ndarray))
    assert(np.shape(out)==(nsamples,norders))
    
    
def test_get_enns_reasonable():
    
    R = load_reasonable()    
    norders = R['norders']   
    mod = asymp_spec_model(R['SC'][0]  , norders)
    
    # Solar    
    dnu, numax, eps = 10**R['solar'][0], 10**R['solar'][1], R['solar'][2]
    nmax = mod._get_nmax(dnu, numax, eps)
    assert_array_equal(mod._get_enns(nmax,norders), np.arange(18,25))
    
    # Boeing
    dnu, numax, eps = 10**R['boeing'][0], 10**R['boeing'][1], R['boeing'][2]
    nmax = mod._get_nmax(dnu, numax, eps)
    assert_array_equal(mod._get_enns(nmax,norders), np.arange(8,15))
    

def test_asymptotic_relation_function():
    R = load_reasonable()    
    norders = R['norders']   
    mod = asymp_spec_model(R['SC'][0]  , norders)
    dnu, numax, eps, alpha = 10**R['solar'][0], 10**R['solar'][1], R['solar'][2], R['solar'][4]
    
    # does it run
    out = mod._asymptotic_relation(numax, dnu, eps, alpha, norders)

    # does it return something if it should? Or set an attribute
    assert(out is not None)
    
    # is the output the type of thing you want
    assert(isinstance(out, np.ndarray))
        
    # is the output the shape you want
    out = mod._asymptotic_relation(numax, dnu, eps, alpha, norders)
    assert(np.shape(out)==(norders,))
    
    n = 10
    out = mod._asymptotic_relation(np.float64(numax).repeat(n), 
                                   np.float64(dnu).repeat(n), 
                                   np.float64(eps).repeat(n), 
                                   np.float64(alpha).repeat(n), norders)
    assert(np.shape(out)==(norders,n))
    

def test_asymptotic_relation_reasonable():
    R = load_reasonable()    
    norders = R['norders']   
    mod = asymp_spec_model(R['SC'][0]  , norders)
    
    # solar
    dnu, numax, eps, alpha = 10**R['solar'][0], 10**R['solar'][1], R['solar'][2], 10**R['solar'][4]   
    nu_expected = [2601.134, 2734.921, 2869.134, 3003.775, 3138.842, 3274.336, 
                   3410.257]
    assert_allclose(mod._asymptotic_relation(numax, dnu, eps, alpha, norders),
                    nu_expected, atol = 0.001)
    
    # boeing
    dnu, numax, eps, alpha = 10**R['boeing'][0], 10**R['boeing'][1], R['boeing'][2], 10**R['boeing'][4]   
    nu_expected = [159.583, 176.226, 192.983, 209.855, 226.841, 243.942, 261.157]
    assert_allclose(mod._asymptotic_relation(numax, dnu, eps, alpha, norders),
                    nu_expected, atol = 0.001)    
    
#def test_P_envelope_function():
#def test_P_envelope_reasonable():
#    
#def test_lor_function():
#def test_lor_reasonable():
#
#def test_pair_function():
#def test_pair_reasonable():
#
#def test_model_function():
#def test_model_reasonable():