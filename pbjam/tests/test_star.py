
import lightkurve as lk
import astropy.units as units
import numpy as np
import pbjam.tests.pbjam_tests as pbt
import os, pytest
from ..star import star, _format_name


def test_star_init():
    """Test the init of the star class
    
    Notes
    -----
    If adding more star inits, remember to os.rmdir(st.path) to clean up after
    the test. 
    
    
    """
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), (4750, 250), (1.34, 0.1))
    
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), bp_rp = (1.34, 0.1))
    assert(np.all(np.array(st.teff) != np.array([None,None])))

    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), teff = (4750, 250))
    assert(np.all(np.array(st.bp_rp) != np.array([None,None])))
    
    with pytest.raises(TypeError):
        st = star('thisisatest', pg)

    # simple check to see if all the attributes are there compared to the time
    # of test creation.
    atts = ['ID', '_fill_diag', '_log_obs', '_make_prior_corner', '_obs', 
            '_get_outpath', '_plot_offdiag', '_save_my_fig', '_set_outpath', 
            'bp_rp', 'dnu', 'f', 'numax', 'path', 'pg', 'plot_corner', 
            'plot_echelle', 'plot_prior', 'plot_spectrum', 'plot_start', 
            'prior_file', 'run_asy_peakbag', 'run_kde', 'run_peakbag', 
            's', 'teff']

    pbt.assert_hasattributes(st, atts)

    # cleanup
    os.remove(st.log_file._filename)  # Remove log file
    os.rmdir(st.path)
    
def test_outpath():
    """Tests for the function that sets the output filename
    
    Notes
    -----
    If adding more star inits, remember to os.rmdir(st.path) to clean up after
    the test. 
    
    """
    
    # setup
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), (4750, 250), (1.34, 0.1))
    func = st._get_outpath
    inp = ['test.png']
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, str)
    
    # Some input tests
    assert(os.path.isdir(os.path.dirname(func(*inp))))
    
    # cleanup
    os.remove(st.log_file._filename)
    os.rmdir(st.path)    
    
def test_set_outpath():
    """Tests for the function that sets the output filename
    
    Notes
    -----
    If adding more star inits, remember to os.rmdir(st.path) to clean up after
    the test. 
    
    """
    
    # setup
    pth = f'{os.getcwd()}/pbjam/tests'
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), (4750, 250), (1.34, 0.1))
    func = st._set_outpath
    
    # Input tests and clean up
    assert(os.path.isdir(st.path))
    os.remove(st.log_file._filename)
    os.rmdir(st.path)
    
    inp = [pth]
    func(*inp)
    assert(os.path.isdir(st.path))
    
    # cleanup
    os.rmdir(st.path)


# The test functions below require longer runs and are not suitable for GitHub
# workflows. the mark.slow decorator doesn't seem to work with GitHub workflows.
    
def test_run_kde():
    """Tests that a KDE can be created and sampled for given prior data

    Notes
    -----
    This test is incomplete

    """
    # setup
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), (4750, 250), (1.34, 0.1))
    func = st.run_kde
    
    # simple tests
    pbt.does_it_run(func, None)
    
    # cleanup
    os.remove(st.log_file._filename)
    os.rmdir(st.path)

def test_format_name():

    ID_in = ['HD16417', 'HD 16417', 
             'TIC 176860064', 'TIC176860064', 
             'tess 176860064', 'tess176860064', 
             'kplr 8006161', 'KIC 8006161', 'kplr8006161', 'KIC8006161',
             'kplr 008006161', 'kplr008006161', 'KIC 008006161', 'KIC008006161']
    
    ID_out = ['hd16417', 'hd 16417', 
              'TIC 176860064', 'TIC 176860064', 
              'TIC 176860064', 'TIC 176860064',
              'KIC 8006161', 'KIC 8006161', 'KIC 8006161', 'KIC 8006161',
              'KIC 8006161','KIC 8006161','KIC 8006161','KIC 8006161']
    
    for i,name in enumerate(ID_in):
        assert(_format_name(name) == ID_out[i])
#def test_run_asy_peakbag():
#def test_run_peakbag():