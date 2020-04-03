from ..asy_peakbag import asymp_spec_model, asymptotic_fit
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pbjam.tests.pbjam_tests as pbt

cs = pbt.case('silly')

def test_get_asy_start():
    func = cs.st.asy_fit._get_asy_start
    
    pbt.does_it_run(func, None)
    pbt.does_it_return(func, None)
    pbt.right_type(func, None, list)
    pbt.right_shape(func, None, (10,)) 
    assert_allclose(func(), [10, 10, 1, 10, 10, 1, 1, 1, 10, 1])
    
def test_get_freq_range():   
    func = cs.st.asy_fit._get_freq_range

    pbt.does_it_run(func, None)
    pbt.does_it_return(func, None)
    pbt.right_type(func, None, tuple)
    pbt.right_shape(func, None, (2,))
    assert_allclose(func(), [-12.5, 22.5])
    
def test_get_modeIDs():

    st = cs.st
    norders = cs.pars['norders']
    
    func = st.asy_fit.get_modeIDs 
    
    pbt.does_it_return(func, [st.asy_fit.fit, norders])
    pbt.right_type(func, [st.asy_fit.fit, norders], pd.DataFrame)
    
    df = st.asy_fit.get_modeIDs(st.asy_fit.fit, norders)
    assert(all(df['nu_mad'].values == np.zeros(2*norders)))
    assert_allclose(df['nu_med'], np.array([89.05, 90.05, 99., 100.]))
    
def test_get_summary_stats():

    st = cs.st

    func = st.asy_fit._get_summary_stats #(st.asy_fit.fit)
    inp = [st.asy_fit.fit]
    
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, pd.DataFrame)
    pbt.right_shape(func, inp, (10,10))

    out = func(*inp)

    for key in ['std', 'skew', 'MAD']:
        assert_array_equal(out.loc[:, key], 0)

    for key in ['mle', 'mean', '2nd', '16th', '50th', '84th', '97th']:
        assert_array_equal(out.loc[:,'mle'], np.array([ 1.,  2.,  1.,  0., -2.,  1.,  1.,  1.,  1.,  1.]))
  
def test_prior_function():
    st = cs.st
    func = st.asy_fit.prior
    inp = [cs.pars['asypars']]

    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)
        
    assert(func(*inp)==-np.inf)
    
def test_likelihood_function():
    st = cs.st
    func = st.asy_fit.likelihood
    inp = [cs.pars['asypars']]
    
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)

    assert(np.isnan(func(*inp)))
    
def test_asymptotic_fit_init():
    st = cs.st

    asymptotic_fit(st, norders=cs.pars['norders'])
    
    attributes = ['_asymptotic_relation', '_fill_diag', '_get_asy_start', 
                  '_get_enns', '_get_freq_range', '_get_nmax', 
                  '_get_summary_stats', '_log_obs', '_lor', 
                  '_make_prior_corner', '_obs', '_pair', 
                  '_plot_offdiag', '_save_my_fig', 'f', 'get_modeIDs', 'kde', 
                  'likelihood', 'model', 'norders', 
                  'par_names', 'pg', 'plot_corner', 'plot_echelle', 
                  'plot_prior', 'plot_spectrum', 'plot_start', 
                  'prior', 'prior_file', 's', 'sel', 'start', 'start_samples', 
                  '_P_envelope']
    
    pbt.assert_hasattributes(st.asy_fit, attributes)

def test_asymp_spec_model_init():
    
    mod = asymp_spec_model(cs.st.f, cs.pars['norders'])
    
    attributes = ['_P_envelope', '_asymptotic_relation', '_get_enns', 
                  '_get_nmax', '_lor', '_pair', 'f', 'model', 'norders']

    pbt.assert_hasattributes(mod, attributes)


def test_get_nmax():   
    nsamples = cs.pars['nsamples']
    norders = cs.pars['norders']

    mod = asymp_spec_model(cs.st.f, norders)
    
    dnu, numax, eps = 10**cs.pars['asypars'][0], 10**cs.pars['asypars'][1], cs.pars['asypars'][2]

    func = mod._get_nmax
    inp = [dnu, numax, eps]
    inp_arr = [np.float64(dnu).repeat(nsamples), 
               np.float64(numax).repeat(nsamples),
               np.float64(eps).repeat(nsamples)]
    
    pbt.does_it_run(func, inp) 
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)
    pbt.right_type(func, inp_arr, np.ndarray)
    
    pbt.right_shape(func, inp, ())
    pbt.right_shape(func, inp_arr, (nsamples,))    
     
    assert(func(*inp) == 0.0)    
    
def test_get_enns():
    norders = cs.pars['norders']
    nmax = cs.pars['nmax']
    nsamples = cs.pars['nsamples']
    
    mod = asymp_spec_model(cs.st.f, norders)

    func = mod._get_enns
    inp = [nmax, norders]
    inp_arr = [np.array([nmax]).repeat(nsamples), norders]
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, (norders,))
    pbt.right_shape(func, inp_arr, (nsamples, norders))
    
    assert_array_equal(func(*inp), [0, 1])
        
def test_asymptotic_relation():
    norders = cs.pars['norders']
    nsamples = cs.pars['nsamples']
    
    mod = asymp_spec_model(cs.st.f, norders)
    
    dnu, numax, eps, alpha = 3050.0, 135.0, 1.45, 10**-2.5 # roughly solar
    

    
    func = mod._asymptotic_relation

    inp = [dnu, numax, eps, alpha, norders]
    inp_arr = [np.float64(numax).repeat(nsamples), 
               np.float64(dnu).repeat(nsamples), 
               np.float64(eps).repeat(nsamples), 
               np.float64(alpha).repeat(nsamples), norders]
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, (norders,))    
    pbt.right_shape(func, inp_arr, (norders, nsamples))

    assert_allclose(func(*inp), [2896.028668, 3030.75434 ], atol = 0.001)  
      
def test_P_envelope():
    norders = cs.pars['norders']
    
    mod = asymp_spec_model(cs.st.f, norders)

    nus = cs.pars['freqs']

    envheight, numax, envwidth = 10**1.5, nus[0], 10**2.2 

    func = mod._P_envelope
    inp = [nus, envheight, numax, envwidth]
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, np.shape(nus))
  
    assert_allclose(func(*inp), [31.6227766, 31.5718312], atol = 0.001)
    
def test_lor():
    norders = cs.pars['norders']
    
    mod = asymp_spec_model(cs.st.f, norders)

    h, freq, w = 1, cs.st.f[0], 1

    func = mod._lor
    inp = [freq, h, w]
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, np.shape(cs.st.f))
    assert_array_equal(func(*inp), [1,1])
    
def test_pair():
    norders = cs.pars['norders']
        
    mod = asymp_spec_model([0.5, 1], norders)

    h, freq, w, d02 = 1, cs.st.f[0], 1, 0.5
    
    func = mod._pair
    inp = [h, freq, w, d02]
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, np.shape(cs.st.f))
    
    assert_array_equal(func(*inp), [1.2,  1.35])
   
def test_model():
    norders = cs.pars['norders']
    
    mod = asymp_spec_model([0.5, 1], norders)

    func = mod.model
    inp = cs.pars['asypars']
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, np.shape(cs.st.f))
    
    assert_allclose(func(*inp), [7.93069307, 7.73076923], atol = 0.001)

def test_asymp_spec_model_call():
    norders = cs.pars['norders']
    
    mod = asymp_spec_model([0.5, 1], norders)

    func = mod.model
    inp = cs.pars['asypars']
    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, np.shape(cs.st.f))
        
    assert_allclose(mod(inp), mod.model(*inp))

        
# The test functions below require longer runs and are not suitable for GitHub
# workflows. the mark.slow decorator doesn't seem to work with GitHub workflows.
    
# #def test_asymptotic_fit_call():