#import pbjam
#from pbjam import asy_peakbag
#from pbjam.asy_peakbag import get_nmax
#import pytest


# does it run
# does it return if it should? Or set an attribute
# does it output the type of thing you want
# does it output something reasonable if I give it something reasonable?
# does it output something unreasonable if I give it something unreasonable

from ..asy_peakbag import asymp_spec_model, get_nmax, get_enns

import numpy as np



def test_get_nmax():   
    get_nmax(100, 1000, 1.5) # does it run?
    
    assert(get_nmax(100, 1000, 1.5) is not None) # does it return?
    
    out = get_nmax(100, 1000, 1.5) # is the output type what is expected?
    assert(isinstance(out, float))

def test_get_enns():

    get_enns(10, 2) is not None
    
    assert(get_enns(10, 2) is not None)
    
    norders = 2
    nmax = 10
    
    out = get_enns(nmax, norders)
    assert(isinstance(out, np.ndarray))
    assert(np.shape(out)==(norders,))
    
    nsamples = 10
    out = get_enns(np.zeros(nsamples)+nmax, norders)
    assert(isinstance(out, np.ndarray))
    assert(np.shape(out)==(10,norders))
    
    
    



def test_init_asymp_spec_model():
    """ Test asymp_spec_model initializes
    """
    asy = asymp_spec_model(1, 2)