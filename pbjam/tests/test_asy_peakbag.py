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
    reas['solar'] = {'pars': [np.log10(135.0), # dnu
                              np.log10(3050.0), # numax 
                              1.25, #eps
                              0.5, # d02
                              -2.5, # alpha
                              1.5, # envheight
                              2.2, # envwidth
                              0.0, # modwidth
                              3.77, #teff
                              0.8],
                        'nmax': 21, # bp_rp
                        'freqs' : np.array([2601.134, 2734.921, 2869.134, 3003.775, 3138.842, 3274.336, 3410.257])}
        
    reas['boeing'] = {'pars': [1.23072917362692,
                               2.34640449860884,
                               1.33476843739203,
                               0.347104562462547,
                               -2.17266900058679,
                               1.47003574648704,
                               1.31305213708824,
                               -0.668935969601942,
                               3.69363755249007,
                               1.22605501887223],
                        'nmax': 11,
                        'freqs': np.array([159.583, 176.226, 192.983, 209.855, 226.841, 243.942, 261.157])}
    
    for key in ['solar', 'boeing']:
        reas[key]['norders'] = len(reas[key]['freqs'])
    
    reas['SC'] = np.vstack((np.linspace(1e-20,8500,10000),
                            np.linspace(1e-20,8500,10000)))
    
    reas['nsamples'] = 10
    return reas

def does_it_run(func, args):
    func(*args)

def does_it_return(func, args):
    assert(func(*args) is not None)

def right_type(func, args, expected):
    out = func(*args)
    assert(isinstance(out, expected))

def right_shape(func, args, expected):
    out = func(*args)
    assert(np.shape(out)==expected)

def assert_positive(x):
    assert(all(x) >= 0)

def test_asymp_spec_model_init():
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['solar']['norders'])
    
    init_attributes = ['_asymptotic_relation', '_get_enns', '_get_nmax', 
                       '_lor', '_pair', 'f', 'model', 'norders']
    
    for attr in init_attributes:
        assert(hasattr(mod, attr))
    

def test_get_nmax_function(): 
    
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['solar']['norders'])
    dnu, numax, eps = 10**R['solar']['pars'][0], 10**R['solar']['pars'][1], R['solar']['pars'][2]
    n = R['nsamples']

    does_it_run(mod._get_nmax, [dnu, numax, eps]) 
    does_it_return(mod._get_nmax, [dnu, numax, eps])
    right_type(mod._get_nmax, [dnu, numax, eps], float)
    right_type(mod._get_nmax, [np.float64(dnu).repeat(n), 
                               np.float64(numax).repeat(n),
                               np.float64(eps).repeat(n)], np.ndarray)   
    right_shape(mod._get_nmax, [dnu, numax, eps], ())    
    right_shape(mod._get_nmax, [np.float64(dnu).repeat(n), 
                               np.float64(numax).repeat(n),
                               np.float64(eps).repeat(n)], (n,))    
    
def test_get_nmax_reasonable():

    R = load_reasonable()    
     
    expected = {'solar': R['solar']['nmax'],
                'boeing': R['boeing']['nmax']}
    
    for key in expected.keys():
        mod = asymp_spec_model(R['SC'][0], R[key]['norders'])
        out = mod._get_nmax(10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2])
        assert_almost_equal(out, expected[key], decimal = 0)
    
def test_get_enns_function():
    
    R = load_reasonable()    
    mod = asymp_spec_model(R['SC'][0]  , R['solar']['norders'])
        
    norders = R['solar']['norders']
    nmax = R['solar']['nmax']
    nsamples = R['nsamples']

    does_it_run(mod._get_enns, [nmax, norders])
    does_it_return(mod._get_enns, [nmax, norders])
    right_type(mod._get_enns, [nmax, norders], np.ndarray)
    right_shape(mod._get_enns, [nmax, norders], (norders,))
    right_shape(mod._get_enns, [np.array([nmax]).repeat(nsamples), norders], (nsamples, norders))
        
    
def test_get_enns_reasonable():
    
    R = load_reasonable()    
   
    # Solar
    expected = {'solar': np.arange(18,25), 
                'boeing': np.arange(8,15)} 
    
    for key in expected.keys():
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        dnu, numax, eps = 10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2]
        nmax = mod._get_nmax(dnu, numax, eps)
        assert_array_equal(mod._get_enns(nmax,norders), expected[key])
        

def test_asymptotic_relation_function():
    R = load_reasonable()    
    norders = R['solar']['norders']
    nsamples = R['nsamples']

    mod = asymp_spec_model(R['SC'][0]  , norders)
    dnu, numax, eps, alpha = 10**R['solar']['pars'][0], 10**R['solar']['pars'][1], R['solar']['pars'][2], R['solar']['pars'][4]
    
    does_it_run(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders])
    does_it_return(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders])
    right_type(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders], np.ndarray)
    right_shape(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders], (norders,))    
    right_shape(mod._asymptotic_relation, [np.float64(numax).repeat(nsamples), 
                                           np.float64(dnu).repeat(nsamples), 
                                           np.float64(eps).repeat(nsamples), 
                                           np.float64(alpha).repeat(nsamples), norders], (norders, nsamples))

def test_asymptotic_relation_reasonable():
    R = load_reasonable()    
       
    expected = {'solar': R['solar']['freqs'],
                'boeing': R['boeing']['freqs']}
    
    for key in expected.keys():
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        dnu, numax, eps, alpha = 10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2], 10**R[key]['pars'][4]   
        out = mod._asymptotic_relation(numax, dnu, eps, alpha, norders)
        assert_allclose(out, expected[key], atol = 0.001)    
    

    
def test_P_envelope_function():
    R = load_reasonable()    
    norders = R['solar']['norders']   
    mod = asymp_spec_model(R['SC'][0], norders)

    envheight, numax, envwidth = 10**R['solar']['pars'][5], 10**R['solar']['pars'][1],10**R['solar']['pars'][6]

    nus = R['solar']['freqs']
    
    does_it_run(mod._P_envelope, [nus[0], envheight, numax, envwidth])
    does_it_return(mod._P_envelope, [nus[0], envheight, numax, envwidth])
    right_type(mod._P_envelope, [nus[0], envheight, numax, envwidth], float)
    right_type(mod._P_envelope, [nus, envheight, numax, envwidth], np.ndarray)
    right_shape(mod._P_envelope, [nus[0], envheight, numax, envwidth], ())
    right_shape(mod._P_envelope, [nus, envheight, numax, envwidth], np.shape(nus))
    

def test_P_envelope_reasonable():
    R = load_reasonable()    
    
    expected = {'solar': [ 0.57311571,  4.38319375, 16.48949317, 30.30597392, 27.02511492, 11.6127985, 2.38800568],
                'boeing': [ 0.29329611,  2.46950433, 10.88381593, 24.77113388, 28.71635548, 16.72404827, 4.82571522]}
    
    for key in expected.keys():
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        envheight, numax, envwidth = 10**R[key]['pars'][5], 10**R[key]['pars'][1],10**R[key]['pars'][6]
        nus = R[key]['freqs']
        
        out = mod._P_envelope(*[nus, envheight, numax, envwidth])    
        assert_allclose(out, expected[key], atol = 0.001)    
        assert_positive(out)
    

def test_lor_function():
    R = load_reasonable()    
 
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])
    
    h, freq, w = 10**R['solar']['pars'][5], 10**R['solar']['pars'][1],10**R['solar']['pars'][6]
        
    does_it_run(mod._lor, [freq, h, w])
    does_it_return(mod._lor, [freq, h, w])
    right_type(mod._lor, [freq, h, w], np.ndarray)
    right_shape(mod._lor, [freq, h, w], (np.shape(R['SC'][0])))
    

def test_lor_reasonable():
    R = load_reasonable()    
    
    for key in ['solar', 'boeing']:
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        
        h, freq, w = 10**R[key]['pars'][5], 10**R[key]['pars'][1],10**R[key]['pars'][6]
    
        out = mod._lor(*[freq, h, w])
        assert_positive(out)   

#def test_pair_function():
#def test_pair_reasonable():
#
#def test_model_function():
#def test_model_reasonable():
    
#def test_asymp_spec_model_call():
    