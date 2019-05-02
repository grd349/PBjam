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
for each requested target, with corresponding spectra for each.

The next step is to perform a mode ID on the spectra. At the moment PBjam
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
    """ Turn elements of X into lists, and check their lengths are the same

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
    for i, x in enumerate(X):
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
        tgt = lk.search_lightcurvefile(target=id,
                                       quarter=lkargs['quarter'][i],
                                       campaign=lkargs['campaign'][i],
                                       sector=lkargs['sector'][i],
                                       month=lkargs['month'][i])
        lc = tgt.download().PDCSAP_FLUX
        lc = lc.remove_nans().normalize().flatten().remove_outliers()
        p = lc.to_periodogram(freq_unit=units.microHertz,
                              normalization='psd').flatten()
        PS_list.append((np.array(p.frequency), np.array(p.power)))
    return PS_list


class star():
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Parameters
    ----------
    ID : str
        Identifier of the target. If no timeseries or powerspectrum files are
        provided the lightcurve will be downloaded by Lightkurve. ID must
        therefore be resolvable by Lightkurve.
    f : float, array
        Array of frequency bins of the spectrum (muHz)
    s : array
        The power at frequencies f
    numax : float
        Initial guess for numax. Frequency of maximum power of the p-mode
        envelope (muHz)
    dnu : float
        Initial guess for dnu. Large separation of l=0 modes (muHz)
    teff : float
        Temperature estimate for the star. Used to compute epsilon.
    """

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
                          norders=5):
        """ Called to perform mode ID using the asymptotic method

        Parameters
        ----------
    d02 : float, optional
        Initial guess for the small frequency separation (in muHz) between
        l=0 and l=2.
    alpha : float, optional
        Initial guess for the scale of the second order frequency term in the
        asymptotic relation
    seff : float, optional
        Normalized Teff
    mode_width : float, optional
        Initial guess for the mode width (in log10!) for all the modes that are
        fit.
    env_width : float, optional
        Initial guess for the p-mode envelope width (muHz)
    env_height : float, optional
        Initial guess for the p-mode envelope height
    norders : int, optional
        Number of radial orders to fit
        """

        fit = asymptotic_fit(self, d02, alpha, seff, mode_width, env_width,
                             env_height)

        fit.run(norders)

        self.mode_ID = fit.mode_ID
        self.asy_model = fit.asy_model


class session():
    """ Main class used to initiate peakbagging.

    Use this class to initialize the star class instance(s) based on the
    provided input. Data can be provided as ascii files, or will automatically
    be downloaded from MAST using Lightkurve.

    Note
    ----
    The physical parameters, numax, dnu, teff, must each be provided at least
    as a list of length 2 for each star. This should containg the parameter
    value and it's error. Setting the error to zero will effectively fix the
    corresponding parameter in the fit.

    Examples
    --------
    Single target, no ascii files. In this case PBjam will attempt to download
    the time series from MAST, using Lightkurve. In this case kwargs must be
    a dictionary with the required observing season number, e.g.
    kwargs = {'quarter': 5} for Kepler quarter 5. Cadence and quarter month
    keywords can also be passed to Lightkurve through kwargs.

    Multiple targets, no ascii files. Similar to the single target case, except
    all arguments must be in a list. kwargs must now be a dictionary of lists.

    Single or multiple targets, with ascii files. Similar to above each target
    must have an ID and associated physical parameters. Here the timeseries
    or power spectrum can be passed to the PBjam session as either a tuple
    or a local path to the ascii file, or a list of either. Note that if both
    a timeseries and spectrum are passed, the timeseries will be ignored.

    Dictionary or dataframe of targets. In this case, use the provided
    template .csv or pickled .dict file. Similar to above, a path can be
    provided, or just a target ID. In the latter case, the observing season
    must be provided in the relevant column as either an integer, or a string
    of zeros with 1 in the bit number equivalent to the requested observing
    seasons.
    """

    def __init__(self, ID=None, numax=None, dnu=None, teff=None,
                 timeseries=None, psd=None, dictionary=None,
                 dataframe=None, kwargs={}):

        physpars = [numax, dnu, teff]
        physchk = all(physpars)

        # Given ID will use LK to download
        if ID and not timeseries and not psd:
            assert physchk, 'Must provide numax, dnu, and teff'

            ID, numax, dnu, teff = bouncer([ID, numax, dnu, teff])

            lkargs = {}
            for key in ['cadence', 'month', 'quarter', 'campaign', 'sector']:
                if key in kwargs:
                    lkargs[key] = kwargs[key]
                else:
                    lkargs[key] = [None]*len(ID)

                lkargs[key] = bouncer([lkargs[key]])[0]

            PS_list = get_psd_from_lk(ID, lkargs)

        self.stars = [star(ID[i], PS_list[i][0], PS_list[i][1], numax[i],
                      dnu[i], teff[i]) for i in range(len(ID))]



      
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