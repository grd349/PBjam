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
import pandas as pd
import warnings
import matplotlib.pyplot as plt


def multiplier(x, N):
    if not x[0]:
        return [None]*N
    else:
        return x

def enforce_list(*X):
    # Check that all elements of X are lists, and if not, make them so
    Y = []
    for i, x in enumerate(X):
        if not isinstance(x, (list, np.ndarray, tuple)):
            Y.append([x])
        else:
            Y.append(x)
    return Y

def check_list_lengths(X):
    lens = []
    if type(X) == dict:
        for key in X.keys():
            lens.append(len(X[key]))
    elif type(X) == list:        
        for i, x in enumerate(X):
            lens.append(len(X[i]))    
    # Check that all elements of X are the same length
    assert lens[1:] == lens[:-1], "Provided inputs must be same length"
    
def download_lc(ID, lkargs):
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
        cadence : long or short

    Returns
    -------
    PS_list : list of tuples
        List of tuples for each requested target. First column of a tuple is
        frequency, second column is power.
    source_list : list
        List of fitsfile names for each target
    """

    lc_list = []
    source_list = []
    for i, id in enumerate(ID):
        tgt = lk.search_lightcurvefile(target=id,
                                       quarter=lkargs['quarter'][i],
                                       campaign=lkargs['campaign'][i],
                                       sector=lkargs['sector'][i],
                                       month=lkargs['month'][i],
                                       cadence=lkargs['cadence'][i])
        lc = tgt.download().PDCSAP_FLUX
        lc = lc.remove_nans().normalize().flatten().remove_outliers()
        lc_list.append(lc)
        source_list.append(tgt.table['productFilename'][0])
    return lc_list, source_list


def get_psd(arr, arr_type):
    """ Get psd from timeseries/psd arguments in session class

    Parameters
    ----------
    arr : list
        List of either tuples of time/flux, tuples of frequency/power, file
        paths, lightkurve.lightcurve objects, or lightkurve.periodogram
        objects.
    arr_type: str
        Definition of the type of data in arr, TS for time series, PS for
        power spectrum.

    Returns
    -------
    PS_list : list
        List of length 2 tuples, with frequency in first column and power in
        second column, for each star in the session list.
    """

    make_lk = lk.LightCurve
    make_lk_pg = lk.periodogram.Periodogram

    PS_list = []
    
    tinyoffset = 1e-20 # to avoid cases LC median = 0 (lk doesn't like it)
    for i, A in enumerate(arr):
        if arr_type == 'TS':
            if type(A) == str:
                inpt = np.genfromtxt(A, usecols=(0, 1))
                lk_lc = make_lk(time=inpt[:, 0], flux=inpt[:, 1]+tinyoffset)
            elif type(A) == tuple:
                assert len(A) >= 2, 'Tuple must be of length >=2 '
                lk_lc = make_lk(time=A[0], flux=A[1]+tinyoffset)
            elif A.__module__ == lk.lightcurve.__name__:
                lk_lc = A
            else:
                raise TypeError("Can't handle this type of time series object")
            lk_p = lk_lc.to_periodogram(freq_unit=units.microHertz,
                                        normalization='psd').flatten()

        if arr_type == 'PS':
            if type(A) == str:
                inpt = np.genfromtxt(A, usecols=(0, 1))
                lk_p = make_lk_pg(inpt[:, 0]*units.microhertz,
                                  units.Quantity(inpt[:, 1], None))
            elif type(A) == tuple:
                assert len(A) >= 2, 'Tuple must be of length >=2 '
                lk_p = make_lk_pg(A[0]*units.microhertz,
                                  units.Quantity(A[1], None))
            elif A.__module__ == lk.periodogram.__name__:
                lk_p = A.flatten()
            else:
                raise TypeError("Can't handle this type of spectrum object")

        PS_list.append((np.array(lk_p.frequency), np.array(lk_p.power)))
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
    source : str, optional
        Pathname of the file used to make the star class instance (timeseries
        or psd). If data is downloaded via Lightkurve the fits file name is 
        used. 
    """

    def __init__(self, ID, f, s, numax, dnu, teff = None, bp_rp = None,
                 epsilon = None, source = None):
        self.ID = ID
        self.f = f
        self.s = s
        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp
        self.epsilon = epsilon
        self.asy_modeID = {}
        self.asy_model = None
        self.asy_bestfit = {}
        self.source = source

    def asymptotic_modeid(self, d02=None, alpha=None, mode_width=None, 
                          env_width=None, env_height=None, norders=5):
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

        fit = asymptotic_fit(self, d02, alpha, mode_width, env_width, 
                             env_height)
        fit.run(norders)

        self.asy_modeID = fit.asy_modeID
        self.asy_model = fit.asy_model
        self.asy_bestfit = fit.asy_bestfit
        
        
    def plot_asyfit(self, model=None, ax=None, modeID=None):
        # Plot resulting spectrum model
        if not model:
            model = self.asy_model
        mod_f, mod_s = model        
        if not modeID:
            modeID = self.asy_modeID        
        if not ax:
            fig, ax = plt.subplots(figsize = (16,9))
        
        ax.set_xlim(min(mod_f), max(mod_f))
        idx = (mod_f[0] < self.f) & (self.f < mod_f[-1])
        ax.set_ylim(0, max(self.s[idx]) * 1.1)
        ax.plot(self.f[idx], self.s[idx], lw = 0.5, label = 'Spectrum')
        ax.plot(mod_f, mod_s, label = 'Model', lw = 4)
        ax.set_xlabel('Frequency [$\mu$Hz]')
        ax.set_ylabel('SNR')
        linestyles = ['-','--','-.','.']
        labels = ['$l=0$','$l=1$','$l=2$','$l=3$']
        for i in range(len(modeID)):
            ax.axvline(modeID['nu_mu'][i], color = 'C3', 
                       ls = linestyles[modeID['ell'][i]], alpha = 0.5)
        for i in np.unique(modeID['ell']):
            ax.plot([-100,-101],[-100,-101], ls = linestyles[i], color = 'C3',
                    label = labels[i])
        ax.axvline(self.numax[0], color = 'k', label = r'$\nu_{\mathrm{max}}$',alpha = 0.5)
        ax.legend()
        
        return fig, ax

    
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

    def __init__(self, ID=None, numax=None, dnu=None, teff=None, bp_rp=None,
                 epsilon=None, timeseries=None, psd=None, dictlike=None, 
                 kwargs={}):

        listchk = all([ID, numax, dnu])

        lk_kws = ['cadence', 'month', 'quarter', 'campaign', 'sector']

        # Given ID will use LK to download
        if listchk:
            ID, numax, dnu, teff, bp_rp, epsilon = enforce_list(ID, numax, dnu,
                                                                teff, bp_rp,
                                                                epsilon)
            
            teff = multiplier(teff, len(ID))
            bp_rp = multiplier(bp_rp, len(ID))
            epsilon = multiplier(epsilon, len(ID))

            check_list_lengths([ID, numax, dnu, teff, bp_rp, epsilon])
    
            if not timeseries and not psd:    
                for key in lk_kws:
                    if key not in kwargs:
                        kwargs[key] = [None]*len(ID)
                    kwargs[key] = enforce_list(kwargs[key])[0] 
                check_list_lengths(kwargs)
                lc_list, source_list = download_lc(ID, kwargs)
                PS_list = get_psd(lc_list, arr_type='TS')

        # Given time series as lk object, tuple or path
            elif timeseries:
                timeseries = enforce_list(timeseries)[0]
                check_list_lengths([timeseries])
                PS_list = get_psd(timeseries, arr_type='TS')
                source_list = [x if type(x) == str else None for x in timeseries]

        # Given power spectrum as lk object, tuple or path
            elif psd:
                psd = enforce_list(psd)[0]
                check_list_lengths([psd])           
                PS_list = get_psd(psd, arr_type='PS')
                source_list = [x if type(x) == str else None for x in psd]
            
        # Given dataframe or dictionary
        elif isinstance(dictlike, (dict, np.recarray, pd.DataFrame)):
            try:
                df = pd.DataFrame.from_records(dictlike)
            except TypeError:
                print('Unrecognized type in dictlike. Must be convertable to dataframe through pandas.DataFrame.from_records()')

            if any([ID, numax, dnu, teff, bp_rp]):
                warnings.warn('Dictlike provided as input, ignoring other inputs.')

            # Check if required keywords are present
            dfkeys = ['ID', 'numax', 'dnu', 'numax_error', 'dnu_error']
            dfkeychk = any(x not in dfkeys for x in df.keys())
            if not dfkeychk:
                raise(KeyError, 'Some of the required keywords were missing.')
                
            ID = list(df['ID'])
            numax = [[df['numax'][i], df['numax_error'][i]] for i in range(len(ID))]
            dnu = [[df['dnu'][i], df['dnu_error'][i]] for i in range(len(ID))]
            
            if ('teff' in df.keys) and ('teff_error' in df.keys):
                teff = [[df['teff'][i], df['teff_error'][i]] for i in range(len(ID))]
            else:
                teff = [None for i in range(len(ID))]

            if ('bp_rp' in df.keys): # No provided errors on bp_rp
                bp_rp = [[df['bp_rp'][i]] for i in range(len(ID))]
            else:
                bp_rp = [None for i in range(len(ID))]


            # if timeseries/psd columns exist, assume its a list of paths
            if 'timeseries' in df.keys():
                PS_list = get_psd(list(df['timeseries']), arr_type='TS')
                source_list = df['timeseries']
            elif 'psd' in df.keys():
                PS_list = get_psd(list(df['psd']), arr_type='PS')
                source_list = df['psd']
            else:  # Else try to get data from Lightkurve
                if not any(x in df.keys() for x in lk_kws):
                    raise(KeyError, 'Must provide dataframe with observing season keywords and optionally cadence and/or month.')

                for key in lk_kws:
                    if key in df.keys():
                        kwargs[key] = list(df[key])
                    else:
                        kwargs[key] = [None]*len(ID)
                check_list_lengths(kwargs)

                lc_list, source_list = download_lc(ID, kwargs)
                PS_list = get_psd(lc_list, arr_type='TS')

        else:
            raise NotImplementedError("Magic not implemented, please give PBjam some input")
            
        self.stars = [star(ID=ID[i], f=PS_list[i][0], s=PS_list[i][1],
                           numax=numax[i], dnu=dnu[i], teff=teff[i],
                           bp_rp=bp_rp[i], epsilon=epsilon[i],
                           source=source_list[i]) for i in range(len(ID))]
