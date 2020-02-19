''' 
## Function
# does it run
# does it return something if it should? Or set an attribute
# is the output the type of thing you want
# is the output the shape you want

## Reasonable/unreasonable?
# is the output something reasonable if I give it something reasonable?
# is the output something unreasonable if I give it something unreasonable

from ..asy_peakbag import asymp_spec_model, asymptotic_fit
from ..star import star
from ..jar import to_log10
import lightkurve as lk
import astropy.units as units
import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy.testing import assert_almost_equal, assert_array_equal, assert_allclose

def load_example():
    stars = {}
    
    # A silly example
    stars['silly'] = {'pars': [10, 10, 1, 10, 10, 1, 1, 1, 10, 1],
                      'obs': {'dnu': (1, 1), 'numax': (1, 1), 'teff': (1, 1), 'bp_rp': (1, 1)},
                      'nmax': 1, 
                      'freqs' : np.array([1,10])}
    
    # A more realistic example
    stars['solar'] = {'pars': [np.log10(135.0), # dnu
                              np.log10(3050.0), # numax 
                              1.25, #eps
                              0.5, # d02
                              -2.5, # alpha
                              1.5, # envheight
                              2.2, # envwidth
                              0.0, # modwidth
                              3.77, #teff
                              0.8], # bp_rp
                        'obs': {'dnu': (135, 1.35), 'numax': (3050, 30), 'teff': (5777, 60), 'bp_rp': (0.8, 0.01)},
                        'nmax': 21, 
                        'freqs' : np.array([2601.134, 2734.921, 2869.134, 3003.775, 3138.842, 3274.336, 3410.257])}
    
    # Another more realistic example
    stars['boeing'] = {'pars': [1.23072917362692,
                               2.34640449860884,
                               1.33476843739203,
                               0.347104562462547,
                               -2.17266900058679,
                               1.47003574648704,
                               1.31305213708824,
                               -0.668935969601942,
                               3.69363755249007,
                               1.22605501887223],
                        'obs' : {'dnu': (16.97, 0.05), 'numax': (220.0, 3.0), 'teff': (4750, 100), 'bp_rp': (1.34, 0.1)}, 
                        'nmax': 11,
                        'freqs': np.array([159.583, 176.226, 192.983, 209.855, 226.841, 243.942, 261.157])}
    
    for key in ['solar', 'boeing', 'silly']:
        stars[key]['norders'] = len(stars[key]['freqs'])
        
        stars[key]['log_obs'] = {x: to_log10(*stars[key]['obs'][x]) for x in stars[key]['obs'].keys() if x != 'bp_rp'}
    
    stars['SC'] = np.vstack((np.linspace(1e-20,8500,10000),
                            np.linspace(1e-20,8500,10000)))
    
    stars['nsamples'] = 10
    return stars

def does_it_run(func, args):
    if args is None:
        func()
    else:
        func(*args)

def does_it_return(func, args):
    if args is None:
        assert(func() is not None)
    else:
        assert(func(*args) is not None)

def right_type(func, args, expected):
    if args is None:
        assert(isinstance(func(), expected))
    else:
        assert(isinstance(func(*args), expected))

def right_shape(func, args, expected):
    if args is None:
        assert(np.shape(func())==expected)
    else:
        assert(np.shape(func(*args))==expected)

def assert_positive(x):
    assert(all(x) >= 0)

def assert_hasattributes(object, attributes):
    for attr in attributes:
        assert(hasattr(object, attr))


def init_dummy_star(ID = 'silly'):
    R = load_example()
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))

    st = star(ID, pg, *[R[ID]['obs'][x] for x in R[ID]['obs'].keys()])
    
    st.kde = type('kde', (object,), {})()
    st.kde.samples = np.ones((2,10))
    
    data = np.array(R['boeing']['pars']).repeat(20).reshape((10,-1))
    st.kde.kde = sm.nonparametric.KDEMultivariate(data=data, var_type='cccccccccc', bw='scott')
    
    return st

def load_dummy_asy_fit(st):
    R = load_example()    
    asymptotic_fit(st, norders=R['solar']['norders'])
    st.asy_fit.fit = type('fit', (object,), {})()
    st.asy_fit.fit.flatchain = np.ones((100, 10))
    st.asy_fit.fit.flatchain[:,1] = 2
    st.asy_fit.fit.flatchain[:,3] = 0
    st.asy_fit.fit.flatchain[:,4] = -2
    st.asy_fit.fit.flatlnlike = np.ones(100)
    st.asy_fit.fit.flatlnlike[0] = 2
    st.asy_fit.log_obs = R[st.ID]['log_obs']

def test_asymptotic_fit_init():
    R = load_example()    
    st = init_dummy_star()
    norders = R['solar']['norders']
   
    asymptotic_fit(st, norders=norders)
    
    attributes = ['_asymptotic_relation', '_get_asy_start', '_get_enns', 
                  '_get_freq_range', '_get_nmax', '_get_summary_stats', 
                  '_lor', '_pair', '_start_init', 'f', 'get_modeIDs', 'kde', 
                  'likelihood', 'model', 'norders', 'par_names', 'pg', 
                  'plot_corner', 'plot_echelle', 'plot_spectrum', 'plot_start',
                  'prior', 's', 'sel', 'start', 'start_samples']
    
    assert_hasattributes(st.asy_fit, attributes)
    assert(hasattr(st, 'asy_fit'))

def test_get_asy_start():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
   
    st.asy_fit._get_asy_start()
    does_it_return(st.asy_fit._get_asy_start, None)
    right_type(st.asy_fit._get_asy_start, None, list)
    right_shape(st.asy_fit._get_asy_start, None, (10,))
    assert_allclose(st.asy_fit._get_asy_start(), [10, 10, 1, 10, 10, 1, 1, 1, 10, 1])
    
def test_get_freq_range():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
    
    st.asy_fit._get_freq_range()
    does_it_return(st.asy_fit._get_freq_range, None)
    right_type(st.asy_fit._get_freq_range, None, tuple)
    right_shape(st.asy_fit._get_freq_range, None, (2,))
    assert_allclose(st.asy_fit._get_freq_range(), [-32.5, 52.5])
    
def test_get_modeIDs():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
    R = load_example()    
    norders = R['solar']['norders']
    
    st.asy_fit.get_modeIDs(st.asy_fit.fit, norders)
    does_it_return(st.asy_fit.get_modeIDs, [st.asy_fit.fit, norders])
    right_type(st.asy_fit.get_modeIDs, [st.asy_fit.fit, norders], pd.DataFrame)
    df = st.asy_fit.get_modeIDs(st.asy_fit.fit, norders)
    assert(all(df['nu_mad'].values == np.zeros(2*norders)))
    assert_allclose(df['nu_med'], np.array([69.45, 70.45, 79.2, 80.2, 89.05, 90.05, 99, 100, 109.05, 110.05, 119.2, 120.2, 129.45, 130.45]))
    
def test_get_summary_stats():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
    
    st.asy_fit._get_summary_stats(st.asy_fit.fit)
    does_it_return(st.asy_fit._get_summary_stats, [st.asy_fit.fit])
    right_type(st.asy_fit._get_summary_stats, [st.asy_fit.fit], pd.DataFrame)
    right_shape(st.asy_fit._get_summary_stats, [st.asy_fit.fit], (10,10))
    
    out = st.asy_fit._get_summary_stats(st.asy_fit.fit)
    for key in ['std', 'skew', 'MAD']:
        assert_array_equal(out.loc[:, key], 0)

    for key in ['mle', 'mean', '2nd', '16th', '50th', '84th', '97th']:
        assert_array_equal(out.loc[:,'mle'], np.array([ 1.,  2.,  1.,  0., -2.,  1.,  1.,  1.,  1.,  1.]))
    
    
def test_prior_function():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
    R = load_example()
    
    p = R['boeing']['pars']
    
    st.asy_fit.prior(p)
    does_it_return(st.asy_fit.prior, [p])
    right_type(st.asy_fit.prior, [p], float)
    
    unreasonable0 = np.ones(10)
    assert(st.asy_fit.prior(unreasonable0) == -np.inf)
    
    
def test_likelihood_function():
    st = init_dummy_star()
    load_dummy_asy_fit(st)
    R = load_example()
    
    p = R['boeing']['pars']

    st.asy_fit.likelihood(p)
    
    #assert(st.asy_fit.likelihood(R['silly']['pars'])==-np.inf)
    
#def test_asymptotic_fit_call():












def test_asymp_spec_model_init():
    R = load_example()    
    mod = asymp_spec_model(R['SC'][0]  , R['solar']['norders'])
    
    attributes = ['_asymptotic_relation', '_get_enns', '_get_nmax', '_lor', 
                  '_pair', 'f', 'model', 'norders']
    
    assert_hasattributes(mod, attributes)
    
def test_get_nmax(): 
    
    R = load_example()    
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
         
    expected = {'solar': R['solar']['nmax'],
                'boeing': R['boeing']['nmax']}
    
    for key in expected.keys():
        mod = asymp_spec_model(R['SC'][0], R[key]['norders'])
        out = mod._get_nmax(10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2])
        assert_almost_equal(out, expected[key], decimal = 0)
    
    assert(mod._get_nmax(1,1,1) == 0.0)
    
def test_get_enns():
    
    R = load_example()    
    mod = asymp_spec_model(R['SC'][0]  , R['solar']['norders'])
        
    norders = R['solar']['norders']
    nmax = R['solar']['nmax']
    nsamples = R['nsamples']

    does_it_run(mod._get_enns, [nmax, norders])
    does_it_return(mod._get_enns, [nmax, norders])
    right_type(mod._get_enns, [nmax, norders], np.ndarray)
    right_shape(mod._get_enns, [nmax, norders], (norders,))
    right_shape(mod._get_enns, [np.array([nmax]).repeat(nsamples), norders], (nsamples, norders))
              
    expected = {'solar': np.arange(18,25), 
                'boeing': np.arange(8,15)} 
    
    for key in expected.keys():
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        dnu, numax, eps = 10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2]
        nmax = mod._get_nmax(dnu, numax, eps)
        assert_array_equal(mod._get_enns(nmax,norders), expected[key])
        
    assert_allclose(mod._get_enns(0, norders)-min(mod._get_enns(0, norders)), range(norders))
        
def test_asymptotic_relation():
    R = load_example()    
    norders = R['solar']['norders']
    nsamples = R['nsamples']

    mod = asymp_spec_model(R['SC'][0]  , norders)
    dnu, numax, eps, alpha = 10**R['solar']['pars'][0], 10**R['solar']['pars'][1], R['solar']['pars'][2], 10**R['solar']['pars'][4]
    
    does_it_run(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders])
    does_it_return(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders])
    right_type(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders], np.ndarray)
    right_shape(mod._asymptotic_relation, [numax, dnu, eps, alpha, norders], (norders,))    
    right_shape(mod._asymptotic_relation, [np.float64(numax).repeat(nsamples), 
                                           np.float64(dnu).repeat(nsamples), 
                                           np.float64(eps).repeat(nsamples), 
                                           np.float64(alpha).repeat(nsamples), norders], (norders, nsamples))

    expected = {'solar': R['solar']['freqs'],
                'boeing': R['boeing']['freqs']}
    
    for key in expected.keys():
        norders = R[key]['norders']   
        mod = asymp_spec_model(R['SC'][0]  , norders)
        dnu, numax, eps, alpha = 10**R[key]['pars'][0], 10**R[key]['pars'][1], R[key]['pars'][2], 10**R[key]['pars'][4]   
        out = mod._asymptotic_relation(numax, dnu, eps, alpha, norders)
        assert_allclose(out, expected[key], atol = 0.001)    
      
def test_P_envelope():
    R = load_example()    
 
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])

    envheight, numax, envwidth = 10**R['solar']['pars'][5], 10**R['solar']['pars'][1],10**R['solar']['pars'][6]

    nus = R['solar']['freqs']
    
    does_it_run(mod._P_envelope, [nus[0], envheight, numax, envwidth])
    does_it_return(mod._P_envelope, [nus[0], envheight, numax, envwidth])
    right_type(mod._P_envelope, [nus[0], envheight, numax, envwidth], float)
    right_type(mod._P_envelope, [nus, envheight, numax, envwidth], np.ndarray)
    right_shape(mod._P_envelope, [nus[0], envheight, numax, envwidth], ())
    right_shape(mod._P_envelope, [nus, envheight, numax, envwidth], np.shape(nus))
  
    
    expected = {'solar': [ 0.57311571,  4.38319375, 16.48949317, 30.30597392, 27.02511492, 11.6127985, 2.38800568],
                'boeing': [ 0.29329611,  2.46950433, 10.88381593, 24.77113388, 28.71635548, 16.72404827, 4.82571522]}
    
    for key in expected.keys(): 
        mod = asymp_spec_model(R['SC'][0]  , R[key]['norders'])
        envheight, numax, envwidth = 10**R[key]['pars'][5], 10**R[key]['pars'][1],10**R[key]['pars'][6]
        nus = R[key]['freqs']
        
        out = mod._P_envelope(*[nus, envheight, numax, envwidth])    
        assert_allclose(out, expected[key], atol = 0.001)    
        assert_positive(out)
    
def test_lor():
    R = load_example()    
 
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])
    
    h, freq, w = 10**R['solar']['pars'][5], 10**R['solar']['pars'][1],10**R['solar']['pars'][6]
        
    does_it_run(mod._lor, [freq, h, w])
    does_it_return(mod._lor, [freq, h, w])
    right_type(mod._lor, [freq, h, w], np.ndarray)
    right_shape(mod._lor, [freq, h, w], (np.shape(R['SC'][0])))
  
    for key in ['solar', 'boeing']:

        mod = asymp_spec_model(R['SC'][0], R[key]['norders'] )
        
        h, freq, w = 10**R[key]['pars'][5], 10**R[key]['pars'][1],10**R[key]['pars'][6]
    
        out = mod._lor(*[freq, h, w])
        assert_positive(out)   

def test_pair():
    R = load_example()    
 
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])
    
    h, freq, w, d02 = 10**R['solar']['pars'][5], 10**R['solar']['pars'][1], 10**R['solar']['pars'][6], 10**R['solar']['pars'][3]
    
    does_it_run(mod._pair, [freq, h, w, d02])
    does_it_return(mod._pair, [freq, h, w, d02])
    right_type(mod._pair, [freq, h, w, d02], np.ndarray)
    right_shape(mod._pair, [freq, h, w, d02], np.shape(R['SC'][0]))
     
    hfac = 0.7 
    for key in ['solar', 'boeing']:
        
        mod = asymp_spec_model(R['SC'][0], R[key]['norders'])
        
        h, freq, w, d02 = 10**R[key]['pars'][5], 10**R[key]['pars'][1], 10**R[key]['pars'][6], 10**R[key]['pars'][3]
    
        out = mod._pair(*[freq, h, w, d02], hfac)
        
        res = out - mod._lor(freq, h, w) - mod._lor(freq - d02, h*hfac, w)
        
        assert_almost_equal(res, np.zeros(len(R['SC'][0])))
        assert_positive(out) 
        
def test_model():
    R = load_example()
    
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])
 
    does_it_run(mod.model,  R['solar']['pars'])
    does_it_return(mod.model,  R['solar']['pars'])
    right_type(mod.model,  R['solar']['pars'], np.ndarray)
    right_shape(mod.model,  R['solar']['pars'], np.shape(R['SC'][0]))
    
    hfac = 0.7
    
    for key in ['solar', 'boeing']:
        
        norders = R[key]['norders']
        mod = asymp_spec_model(R['SC'][0], norders)
        
        out = mod.model(*R[key]['pars'])
        assert_positive(out)

        dnu = 10**R[key]['pars'][0]
        numax = 10**R[key]['pars'][1]
        eps =  R[key]['pars'][2]
        d02 = 10**R[key]['pars'][3]
        alpha =10**R[key]['pars'][4]
        envheight = 10**R[key]['pars'][5]
        envwidth =  10**R[key]['pars'][6]
        w = 10**R[key]['pars'][7]
        
        f0s = mod._asymptotic_relation(numax, dnu, eps, alpha, norders)
        
        Hs = mod._P_envelope(f0s, envheight, numax, envwidth)
        
        for i in range(len(f0s)):
            out -= mod._lor(f0s[i], Hs[i], w) 
            out -= mod._lor(f0s[i] - d02, Hs[i]*hfac, w)
            
        assert_almost_equal(out, np.ones(len(R['SC'][0])))
        
def test_asymp_spec_model_call():
    R = load_example()
    
    mod = asymp_spec_model(R['SC'][0], R['solar']['norders'])
    does_it_run(mod,  [R['solar']['pars']]) 
    does_it_return(mod,  [R['solar']['pars']])
    right_type(mod,  [R['solar']['pars']], np.ndarray)
    right_shape(mod,  [R['solar']['pars']], np.shape(R['SC'][0]))
    
    
    
    
    
    
    
    
    
    
#    ID = 'test_KIC4448777'
    #f, p = np.genfromtxt('./pbjam/tests/mypsd.asciifile').T
 
#    numax = (10**R['boeing']['pars'][1], 3)
#    dnu = (10**R['boeing']['pars'][0], 0.05)
#    teff = (10**R['boeing']['pars'][8], 100)
#    bp_rp = (R['boeing']['pars'][9], 0.1)
#    norders = R['boeing']['norders']
#
#    st0 = star(ID, pg, numax, dnu, teff, bp_rp)
#    st0.run_kde() 
''' 
