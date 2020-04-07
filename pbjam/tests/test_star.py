
import lightkurve as lk
import astropy.units as units
import numpy as np
import pbjam.tests.pbjam_tests as pbt
import os
from ..star import star


def test_star_init():
    """Test the init of the star class
    
    Notes
    -----
    If adding more star inits, remember to os.rmdir(st.path) to clean up after
    the test. 
    
    
    """
    pg = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
    st = star('thisisatest', pg, (220.0, 3.0), (16.97, 0.05), (4750, 250), (1.34, 0.1))
    
    # simple check to see if all the attributes are there compared to the time
    # of test creation.
    atts = ['ID', '_fill_diag', '_log_obs', '_make_prior_corner', '_obs', 
            '_outpath', '_plot_offdiag', '_save_my_fig', '_set_outpath', 
            'bp_rp', 'dnu', 'f', 'numax', 'path', 'pg', 'plot_corner', 
            'plot_echelle', 'plot_prior', 'plot_spectrum', 'plot_start', 
            'prior_file', 'run_asy_peakbag', 'run_kde', 'run_peakbag', 
            's', 'teff']

    pbt.assert_hasattributes(st, atts)

    # cleanup
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
    func = st._outpath
    inp = ['test.png']
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, str)
    
    # Some input tests
    assert(os.path.isdir(os.path.dirname(func(*inp))))
    
    # cleanup
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
    os.rmdir(st.path)
    
    inp = [pth]
    func(*inp)
    assert(os.path.isdir(st.path))
    os.rmdir(st.path)


# The test functions below require longer runs and are not suitable for GitHub
# workflows. the mark.slow decorator doesn't seem to work with GitHub workflows.
    
#def test_run_kde():
#def test_run_asy_peakbag():
#def test_run_peakbag():