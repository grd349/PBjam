""" Setup jam sessions and perform mode ID and peakbagging

This jar contains the input layer for setting up jam sessions for peakbagging
solar-like oscillators. 

A jam session is started by initializing the session class instance with a
target ID, numax, large separation, and effective temperature, or lists of 
these if you are working on multiple targets. Alternatively a dataframe or 
dictionary can also be provided, with columns corresponding to the above
keywords. 

Specific quarters, campgains or sectors can be requested in a kwargs dictionary
with the relevant keyword (i.e., 'quarter' for KIC, etc.) 

Once initialized, the session class contains a list of star class instances
for each requested target, with corresponding SNR spectra for each. 

The next step is to perform a mode ID on the SNR spectra. At the moment PBjam 
only supports use of the asymptotic relation mode ID method. 

Note
----
Target IDs must be resolvable by Lightkurve
"""


import lightkurve as lk
from pbjam.asy_peakbag import asymptotic_fit
import numpy as np
import astropy.units as units


def bouncer(X):
    """ Turn elements of X into lists, and check their length
    
    Parameters
    ----------
    X : list
        List of objects to be turned into list of lists of objects (yeah...)
    
    Returns
    -------
    X : list
        List of lists of objects that was formerly just a list of objects
    """
    
    # TODO - this should probably be split into to, one to enforce list type 
    # and one to enforce all items in list of lists must have same length
    # right now ID, numax, dnu, teff are separate from kwarg arguments
    lens = []
    # Check that all elements of X are lists, and if not, make them so
    for i,x in enumerate(X):        
        if not isinstance(x, (list, np.ndarray, tuple)):
            X[i] = [x]                  
        lens.append(len(X[i]))
    # Check that all elements of X are the same length
    assert lens[1:] == lens[:-1], "Provided inputs must be same length"
    
    return X
   
def get_psd_from_lk(ID, lkargs):
    """ Use Lightkurve to get snr
    
    Querries MAST using Lightkurve, based on the provided target ID(s) and 
    observing season number. Then computes the periodogram based on the 
    downloaded time series. 
    
    Parameters
    ----------
    ID : str, list of strs
        String or list of strings of target IDs that Lightkurve can resolve. 
        (KIC, TIC, EPIC).
    lkargs : dict
        Dictionary of keywords for Lightkurve to get the correct observing
        season.
        quarter : for Kepler targets
        month : for Kepler targets, applies to short-cadence data
        sector : for TESS targets
        campaign : for K2 targets
        
    Returns
    -------
    PS_list : list of tuples
        List of tuples for each requested target. First column of a tuple is 
        frequency, second column is power. 
    """
    PS_list = []
    
    for i, id in enumerate(ID):

        lc = lk.search_lightcurvefile(target=id, 
                                      quarter=lkargs['quarter'][i], 
                                      campaign=lkargs['campaign'][i],
                                      sector=lkargs['sector'][i], 
                                      month = lkargs['month'][i]).download().PDCSAP_FLUX
         
        lc = lc.remove_nans().normalize().flatten().remove_outliers()

        p = lc.to_periodogram(freq_unit=units.microHertz,
                                      normalization='psd',).flatten()
        
        PS_list.append((np.array(p.frequency),np.array(p.power)))
        
    return PS_list


class star():
    def __init__(self, ID, f, s, numax, dnu, teff):
        self.ID = ID
        self.f = f
        self.s = s
        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.epsilon = None
        self.mode_ID = {}
        self.asy_model = None
    
    def asymptotic_modeid(self, d02=None, alpha=None, seff=None, 
                          mode_width=None, env_width=None, env_height=None, 
                          norders = 5):
        
        fit = asymptotic_fit(self, d02, alpha, seff, mode_width, env_width, env_height)
        
        fit.run(norders)
        
        self.mode_ID = fit.mode_ID
        self.asy_model = fit.asy_model

class session():

    def __init__(self, ID=None, numax=None, dnu=None, teff=None, 
                       path=None, timeseries=None, psd=None, dictionary=None, 
                       dataframe=None, kwargs = {}):
            
        physpars = [numax, dnu, teff]
        physchk = all(physpars)
        
        # Given ID will use LK to download
        if ID and not path and not timeseries and not psd:
            assert physchk, 'Must provide numax, dnu, and teff'
        
            ID, numax, dnu, teff = bouncer([ID, numax, dnu, teff])
             
            lkargs = {}        
            for key in ['cadence', 'month', 'quarter', 'campaign', 'sector']:
                if key in kwargs:
                    lkargs[key] = kwargs[key]
                else:
                    lkargs[key] = [None]*len(ID)
                    
                lkargs[key] = bouncer([lkargs[key]])[0]
            
            PS_list  = get_psd_from_lk(ID, lkargs)
        
        self.stars = [star(ID[i], PS_list[i][0],PS_list[i][1], numax[i], dnu[i], teff[i]) for i in range(len(ID))]
        


#        
#        # Given path will use genfromtxt to read ascii file, must be 2 column time and relative flux, must also provude  
#        # must also provide numax,dnu,teff
#        elif ID and path and not timeseries and not psd:
#            assert(physchk)
#
#        
#        # Given lc periodogram object, tuple, array, list of (frequency, psd) will compute psd
#        # must also have numax,dnu,teff
#        elif ID and timeseries and not psd and not path:            
#            assert(physchk)
#
#        
#            # Type check, is it tuple or lc object        
#            if lc:
#                'bla'
#            elif tup:
#                'bla'
#            elif array:
#                'bla'
#            elif list:
#                'bla'
#            else:
#                'Unhandled type'
#                sys.exit()
#                
#        # Given lc periodogram object, tuple, array, list of (frequency, psd)
#        # must also have numax,dnu,teff
#        elif ID and psd and not path and not timeseries:
#            assert(physchk)
#            
#            if lc:
#                'bla'
#            elif tup:
#                'bla'
#            elif array:
#                'bla'
#            elif list:
#                'bla'
#            else:
#                'Unhandled type'
#                sys.exit()
#            
#        # Given dataframe, must at least have ID, numax, dnu and teff keywords
#        # path column will override download
#        elif dataframe:
#            assert(any(x not in ['ID','numax','dnu','teff'] for x in dataframe.keys()))
#        
#        # Given dictionary, must at least have ID, numax, dnu and teff keywords
#        # path keyword will override download
#        elif dictionary:
#            assert(any(x not in ['ID','numax','dnu','teff'] for x in dictionary.keys()))
#                
#        # *throws hands in the air and sighs audibly*
#        else:
#            'Break'
#            sys.exit()
        
            
        


        
        
        

    