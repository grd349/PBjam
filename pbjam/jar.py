""" Setup jam sessions and perform mode ID and peakbagging

This jar contains the input layer for setting up jam sessions for peakbagging
solar-like oscillators. This is the easiest way to handle targets in PBjam.

It's possible to manually initiate star class instances and do all the fitting
that way, but it's simpler to just use the session class, which handles 
everything, including formatting of the inputs.

A jam session is started by initializing the session class instance with a
target ID, numax, and a large separation. Additional parameters like the 
effective surface temperature of the star, are optional but help convergence. 

Lists of the above can be provided for multiple targets, but it's often simpler
to just provide PBjam with a dictionary or Pandas dataframe. See mytgts.csv
for a template.

Custom timeseries or periodogram can be provided as either file pathnames,
numpy arrays, or lightkurve.LightCurve/lightkurve.periodogram objects. If 
nothing is provided PBjam will download the data automatically.

Specific quarters, campgains or sectors can be requested with the relevant 
keyword (i.e., 'quarter' for KIC, etc.). If none of these are provided, PBjam
will download all available data, picking the long cadence versions by default.

Once initialized, the session class contains a list of star class instances
for each requested target, with associated spectra for each.

The next step is to perform a mode ID on the spectra. At the moment PBjam
only supports use of the asymptotic relation mode ID method.

Finally the peakbagging method takes the output from the modeID and performs
a proper HMC peakbagging run.

Plotting the results of each stage is also possible.

Note
----
Target IDs must be resolvable by Lightkurve

For automatic download the long cadence data set is used by default, so set
the cadence to 'short' for main-sequence targets.
"""

import lightkurve as lk
from pbjam.asy_peakbag import asymptotic_fit
import numpy as np
import astropy.units as units
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os, glob, warnings, psutil

from . import PACKAGEDIR


def organize_sess_dataframe(vardf):
    """ Takes input dataframe and organizes

    Checks to see if required columns are present in the input dataframe,
    and adds optional columns if they don't exists, containing None values.

    Parameters
    ----------
    vardf : Pandas.DataFrame
        Input dataframe
    """
    keys = ['ID', 'numax', 'dnu', 'numax_err', 'dnu_err']
    if not any(x not in keys for x in vardf.keys()):
        raise(KeyError, 'Some of the required keywords were missing.')

    N = len(vardf)

    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter']
    doubles = ['eps', 'teff', 'bp_rp']

    for key in singles:
        if key not in vardf.keys():
            vardf[key] = np.array([None]*N)

    for key in doubles:
        if key not in vardf.keys():
            vardf[key] = np.array([None]*N)
            vardf[key+'_err'] = np.array([None]*N)

    if 'timeseries' not in vardf.keys():
        format_col(vardf, None, 'timeseries')
    if 'psd' not in vardf.keys():
        format_col(vardf, None, 'psd')


def organize_sess_input(**vardct):
    """ Takes input and organizes them in a dataframe

    Checks to see if required inputs are present and inserts them into a
    dataframe. Any optional columns that are not included in the input are
    added as None columns.

    Parameters
    ----------
    vardct : objects
        Variable inputs to Session class to be arranged into a dataframe

    Returns
    -------
    vardf : Pandas.DataFrame
        Dataframe containing the inputs from Session class call.

    """
    vardf = pd.DataFrame({'ID': np.array(vardct['ID']).reshape((-1, 1)).flatten()})

    N = len(vardf)
    doubles = ['numax', 'dnu', 'eps', 'teff', 'bp_rp']
    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter']

    for key in singles:
        if not vardct[key]:
            vardf[key] = np.array([None]*N)
        else:
            vardf[key] = vardct[key]

    for key in doubles:
        if not vardct[key]:
            vardf[key] = np.array([None]*N)
            vardf[key+'_err'] = np.array([None]*N)
        else:
            vardf[key] = np.array(vardct[key]).reshape((-1, 2))[:, 0].flatten()
            vardf[key+'_err'] = np.array(vardct[key]).reshape((-1, 2))[:, 1].flatten()
    return vardf


def query_mast(id, lkwargs):
    """ Search for target on MAST server

    Get all the lightcurves available for a target id, using options in kwargs
    dictionary.

    Parameters
    ----------
    id : string
        Target id, must be resolvable by LightKurve
    lkwargs : dictionary containing keywords for the LightKurve search.
        cadence, quarter, campaign, sector, month.

    Returns
    -------
    search_results : list
        List of fits files for the requested target
    """

    search_results = lk.search_lightcurvefile(target=id, **lkwargs)
    if len(search_results) == 0:
        warnings.warn('LightKurve did not return %s cadence data for %s' % (lkwargs['cadence'], id))
        return []
    else:
        return search_results.download_all()


def sort_lc(lc):
    """ Sort a lightcurve in LightKurve object

    LightKurve lightcurves are not necessarily sorted in time, which causes
    an error in periodogram.

    Parameters
    ----------
    lc : LightKurve.LightCurve instance
        LightKurve object to be modified

    Returns
    -------
    lc : LightKurve.LightCurve instance
        The sorted LightKurve object

    """

    sidx = np.argsort(lc.time)
    lc.time = lc.time[sidx]
    lc.flux = lc.flux[sidx]
    return lc


def clean_lc(lc):
    """ Perform LightKurve operations on object

    Performes basic cleaning of a light curve, removing nans, outliers,
    median filtering etc.

    Parameters
    ----------
    lc : LightKurve.LightCurve instance
        LightKurve object to be cleaned

    Returns
    -------
    lc : LightKurve.LightCurve instance
        The cleaned LightKurve object
    """

    lc = lc.remove_nans().normalize().flatten().remove_outliers()
    return lc


def query_lightkurve(id, lkwargs, use_cached):
    """ Check cache for fits file, or download it

    Based on use_cached flag, will look in the cache for fits file
    corresponding to request id star. If nothing is found in cached it will be
    downloaded from the MAST server.

    Parameters
    ----------
    id : string
        Identifier for the requested star. Must be resolvable by LightKurve
    lkwargs : dictionary containing keywords for the LightKurve search.
        cadence, quarter, campaign, sector, month.
    use_cached: bool
        Whether or not to used data in the LightKurve cache.

    Note:
    -----
    Prioritizes long cadence over short cadence unless otherwise specified.

    """
    lk_cache = os.path.join(*[os.path.expanduser('~'),
                              '.lightkurve-cache',
                              'mastDownload/*/'])
    if not lkwargs['cadence']:
        lkwargs['cadence'] = 'long'
    if lkwargs['cadence'] == 'short':
        tgtfiles = glob.glob(lk_cache + f'*{str(int(id))}*/*_slc.fits')
    elif lkwargs['cadence'] == 'long':
        tgtfiles = glob.glob(lk_cache + f'*{str(int(id))}*/*_llc.fits')
    else:
        raise TypeError('Unrecognized cadence input for %s' % (id))

    if (not use_cached) or (use_cached and (len(tgtfiles) == 0)):
        if ((len(tgtfiles) == 0) and use_cached):
            warnings.warn('Could not find %s cadence data for %s in cache, checking MAST...' % (lkwargs['cadence'], id))
        print('Querying MAST')
        lc_col = query_mast(id, lkwargs)
        if len(lc_col) == 0:
            raise ValueError("Could not find %s cadence data for %s in cache or on MAST" % (lkwargs['cadence'], id))

    elif (use_cached and (len(tgtfiles) != 0)):
        lc_col = [lk.open(n) for n in tgtfiles]
    else:
        raise ValueError('Unhandled Exception')
    lc0 = clean_lc(lc_col[0].PDCSAP_FLUX)
    for i, lc in enumerate(lc_col[1:]):
        lc0 = lc0.append(clean_lc(lc.PDCSAP_FLUX))
    return lc0


def arr_to_lk(x, y, name, typ):
    """ LightKurve object from input
    
    Creates either a lightkurve.LightCurve or lightkurve.periodogram object 
    from the input arrays.
    
    Parameters
    ----------
    x : list-like
        First column of timeseries or periodogram (time/frequency).
    y : list-like
        Second column of timeseries or periodogram (flux/power).
    name : string
        Target ID
    typ : string
        Either timeseries or periodogram.
    
    Returns
    -------
    lkobj : object
        Either lightkurve.LightCurve or lightkurve.periodogram object  
        depending on typ.
    
    """
    if typ == 'timeseries':
        return lk.LightCurve(time=x, flux=y, targetid=name)
    elif typ == 'psd':
        return lk.periodogram.Periodogram(x*units.microhertz,
                                          units.Quantity(y, None),
                                          targetid=name)
    else:
        raise KeyError("Don't modify anything but psd and timeseries cols")


def format_col(vardf, col, key):
    """ Add timeseries or psd column to dataframe based on input

    Based on the contents of col, will try to format col and add it as a column
    to vardf with column name key. col can be many things, so the decision is 
    based mainly on the dimensionality of col. 
    
    If dim = 0, it's assumed that col is either None, or a string, (for the 
    latter it assumes there is then only one target). 
    
    If dim = 1, it's assumed that col is a list-like object, consisting of 
    either None or strings, these are passed along without modification. 
    
    If dim = 2, col is assumed to be either a time series or power spectrum 
    of shape (2,M), with time/frequency in 1st row and flux/power in the 
    second. 
    
    If dim = 3, it is assumed to be list of (2,M) arrays. 
    
    In both of the latter cases col is converted to LightKurve object(s).

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Pandas Dataframe instance to which either a timeseries or psd column
        will be added.
    col : object
        Input from Session call, corresponding to key
    key : name of column to add to H

    """
    N = np.shape(vardf['ID'])[0]

    col = np.array(col, dtype=object)

    # If dim = 0, it's either none or a string
    if col.ndim == 0:
        if not col:
            # If none, then multiply up to length of ID
            vardf[key] = np.array([None]*N)
        else:
            # If string, then single target
            vardf[key] = np.array(col).reshape((-1, 1)).flatten()

    # If dim = 1, it's either a list of nones, strings or lightkurve objects
    elif col.ndim == 1:
        vardf[key] = col

    # if dim = 2, it's an array or tuple, with time and flux
    elif col.ndim == 2:
        x = np.array(col[0, :], dtype=float)
        y = np.array(col[1, :], dtype=float)
        vardf[key] = np.array([arr_to_lk(x, y, vardf['ID'][0])], key)

    # If dim = 3, it's a list of arrays or tuples
    elif col.ndim == 3:
        temp = np.array([], dtype=object)
        for i in range(N):
            x = np.array(col[i, 0, :], dtype=float)
            y = np.array(col[i, 1, :], dtype=float)
            temp = np.append(temp,
                             np.array([arr_to_lk(x, y, vardf.loc[i, 'ID'], key)]))
        vardf[key] = temp
    else:
        print('Unhandled exception')


def lc_to_lk(vardf, use_cached=True):
    """ Convert time series column in dataframe to lk.LightCurve object

    Goes through the timeseries column in the dataframe and tries to convert
    it to a LightKurve.LightCurve object. If string, it's assumed to be a file
    path name, if None, it will query the LightCurve cache locally or MAST if
    nothing is found. Skips column entries which are already LightKurve objects
    or if a psd for the star in question exists in the psd column.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'timeseries' column consisting of None, strings
        or LightKurver.LightCurve objects.
    Returns
    -------

    """

    tinyoffset = 1e-20  # to avoid cases LC median = 0 (lk doesn't like it)
    key = 'timeseries'
    for i, id in enumerate(vardf['ID']):

        if isinstance(vardf.loc[i, key], str):
            t, d = np.genfromtxt(vardf.loc[i, key], usecols=(0, 1)).T
            d += tinyoffset
            vardf.loc[i, key] = arr_to_lk(t, d, vardf.loc[i, 'ID'], key)
        elif not vardf.loc[i, key]:
            if vardf.loc[i, 'psd']:
                pass
            else:
                D = {x: vardf.loc[i, x] for x in ['cadence', 'month', 'sector',
                                              'campaign', 'quarter']}
                lk_lc = query_lightkurve(id, D, use_cached)
                vardf.loc[i, key] = lk_lc
        elif vardf.loc[i, key].__module__ == lk.lightcurve.__name__:
            pass
        else:
            raise TypeError("Can't handle this type of time series object")

        if vardf.loc[i, key]:
            sort_lc(vardf.loc[i, key])


def lk_to_pg(vardf):
    """ Convert psd column in dataframe to lk periodgram object list

    Takes whatever is in the psd column of a dataframe and tries to turn it
    into a LightKurve.periodogram object. If column entry is a string, it
    assumes it's a path name, if None, it will try to take what's in the
    timeseries column and compute periodogram based on that. If a periodogram
    object is already present, it'll just flatten it (to be safe) and then
    continue on.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'psd' column consisting of None, strings or
        LightKurver.periodogram objects.

    """

    key = 'psd'
    for i, id in enumerate(vardf['ID']):
        if isinstance(vardf.loc[i, key], str):
            f, s = np.genfromtxt(vardf.loc[i, key], usecols=(0, 1)).T
            vardf.loc[i, key] = arr_to_lk(f, s, vardf.loc[i, 'ID'], key)
        elif not vardf.loc[i, key]:
            lk_lc = vardf.loc[i, 'timeseries']
            vardf.loc[i, key] = lk_lc.to_periodogram(freq_unit=units.microHertz, normalization='psd').flatten()

        elif vardf.loc[i, key].__module__ == lk.periodogram.__name__:
            vardf.loc[i, key] = vardf.loc[i, key].flatten()
        else:
            raise TypeError("Can't handle this type of time series object")

def print_memusage(pre='', post=''):
    process = psutil.Process(os.getpid())
    print(pre, process.memory_info().rss, 'bytes', post)  # in bytes 


class session():
    """ Main class used to initiate peakbagging.

    Use this class to initialize a star class instance for one or more targets.
    
    Once initialized, calling this class instance will execute a complete
    peakbagging run. 

    Data can be provided in multiple different ways, the simplest of which is
    just to let PBjam query the MAST server. Otherwise arrays of 
    timeseries/power spectra, lightkurve.LightCurve/lightkurve.periodogram, 
    or just path names as strings, is also possible.

    The physical parameters, numax, dnu, teff, must each be provided at least
    as a list of length 2 for each star. This should contain the parameter
    value and it's error. 

    For multiple target all the above can be provided as lists, but the 
    easiest way is to simply provide a dataframe from csv file.

    Examples
    --------
    Peakbagging run for a single target: 
   
    jam_sess = pbjam.session(ID =  '4448777',  numax = [220.0, 3.0],
                             dnu = [16.97, 0.01], teff = [4750, 100],
                             bp_rp = [1.34, 0.01], cadence = 'short')
    jam_sess()
    
    Peakbagging run for multiple targets:
    jam_sess = pbjam.session(dictlike = mydataframe)
    jam_sess()    
    
    By default, PBjam will download all the available data, favoring long 
    cadence. Cadence and specific observing seasons (quarter, month, campagin,
    sector) can be specified for more detailed control.
    
    Parameters
    ----------
    ID : string, int
        Target identifier, if custom timeseries/periodogram is provided, it 
        must be resolvable by LightKurve (KIC, TIC, EPIC, HD, etc.)
    numax : list
        List of the form [numax, numax_error], list of lists for multiple 
        targets 
    dnu : list
        List of the form [dnu, dnu_error], list of lists for multiple targets
    teff : list, optional
        List of the form [teff, teff_error], list of lists for multiple targets
    bp_rp : list, optional
        List of the form [bp_rp, bp_rp_error], list of lists for multiple 
        targets
    epsilon : list, optional
        List of the form [epsilon, epsilon_error], list of lists for multiple 
        targets
    timeseries : object, optional
        Timeseries input. Leave as None for PBjam to download it automatically.
        Otherwise, arrays of shape (2,N), lightkurve.LightCurve objects, or
        strings for pathnames are accepted.
    psd : object, optional
        Periodogram input. Leave as None for PBjam to use Timeseries to compute
        it for you. Otherwise, arrays of shape (2,N), lightkurve.periodogram
        objects, or strings for pathnames are accepted.
    dictlike : pandas.DataFrame or dictionary, optional
        DataFrame, dictionary, record array with a list of targets, and their 
        properties. If string, PBjam will assume it's a pathname to a csv file.
        Specify timeseries and psd columns with file pathnames to use manually 
        reduced data. 
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run. 
        Warning, if running multiple targets, make sure you have enough memory.
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead. 
        Untested on short cadence. 
    use_cached : bool
        Flag for using cached data. If fitting the same targets multiple times,
        use to this to not download the data every time.
    cadence : string
        Argument for lightkurve to download correct data type. Can be 'short' 
        or 'long'. 'long' is default setting, so if you're looking at main
        sequence stars, make sure to manually set 'short'.
    month : int
        Argument for lightkurve when requesting Kepler short cadence data.
    quarter : int
        Argument for lightkurve when requesting Kepler data.
    campaign : int
        Argument for lightkurve when requesting K2 data. 
    sector : int
        Argument for lightkurve when requesting TESS data.
    
    Attributes
    ----------
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead. 
        Untested on short cadence. 
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run. 
        Warning, if running multiple targets, make sure you have enough memory.
    stars : list
        Session will store star class instances in this list, based on the 
        requested targets.
    """

    def __init__(self, ID=None, numax=None, dnu=None,
                 teff=None, bp_rp=None, epsilon=None,
                 timeseries=None, psd=None, dictlike=None, store_chains=False,
                 nthreads=1, use_cached=False, cadence=None, campaign=None,
                 sector=None, month=None, quarter=None):

        self.nthreads = nthreads
        self.store_chains = store_chains
        self.stars = []
        
        #print_memusage(pre = 'Session init start')

        if isinstance(dictlike, (dict, np.recarray, pd.DataFrame, str)):
            if isinstance(dictlike, str):    
                vardf = pd.read_csv(dictlike)
            else:                
                try:
                    vardf = pd.DataFrame.from_records(dictlike)
                except TypeError:
                    print('Unrecognized type in dictlike. Must be able to convert to dataframe through pandas.DataFrame.from_records()')

            if any([ID, numax, dnu, teff, bp_rp, epsilon]):
                warnings.warn('Dictlike provided as input, ignoring other input fit parameters.')

            organize_sess_dataframe(vardf)

        elif ID and numax and dnu:
            vardf = organize_sess_input(ID=ID, numax=numax, dnu=dnu, teff=teff,
                                        bp_rp=bp_rp, eps=epsilon,
                                        cadence=cadence, campaign=campaign,
                                        sector=sector, month=month,
                                        quarter=quarter)
            format_col(vardf, timeseries, 'timeseries')
            format_col(vardf, psd, 'psd')

        lc_to_lk(vardf, use_cached=use_cached)
        lk_to_pg(vardf)
        
        #print_memusage(pre = 'df setup')
        
        for i in range(len(vardf)):
            #print_memusage(pre = f'Initializing star {i}')

            self.stars.append(star(ID=vardf.loc[i, 'ID'],
                                   f=np.array(vardf.loc[i, 'psd'].frequency),
                                   s=np.array(vardf.loc[i, 'psd'].power),
                                   numax=vardf.loc[i, ['numax', 'numax_err']].values,
                                   dnu=vardf.loc[i, ['dnu', 'dnu_err']].values,
                                   teff=vardf.loc[i, ['teff', 'teff_err']].values,
                                   bp_rp=vardf.loc[i, ['bp_rp', 'bp_rp_err']].values,
                                   epsilon=vardf.loc[i, ['eps', 'eps_err']].values,
                                   store_chains=self.store_chains,
                                   nthreads=self.nthreads))

        for i, st in enumerate(self.stars):
            if st.numax[0] > st.f[-1]:
                warnings.warn("Input numax is greater than Nyquist frequeny for %s" % (st.ID))
                
    def __call__(self, step = None, norders = 8, plots = True):
        """ The doitall script
        
        Calling session will by default do asymptotic mode ID and peakbagging
        for all stars in the session.
        
        Parameters
        ----------
        step : string
            Which step to perform. Can currently be 'asymptotic_modeid' and 
            'peakbag'. asymptotic_modeid must be run before peakbag. 
        norders : int
            Number of orders to include in the fits
        plots : bool
            Flag for whether or not to generate plots too. By default PBjam 
            will only plot the summary figure, but if flatchain is large, it
            will also try to make a corner plot.
        """
        from tqdm import tqdm
        
        #print_memusage(pre = f'Call do it all')
        
        for star in tqdm(self.stars):
            
            if not step or (step == 'asymptotic_modeid'):
                star.asymptotic_modeid(norders = 9)
            
            if not step or (step == 'peakbag'):
                pass  # TODO - add peakbagging option
                
            if not step or plots:
                star.plot_asyfit()
                if np.shape(star.asy_result.flatchain)[0] > 200: 
                    star.corner()
            
            #print_memusage(pre = f'Star {star.ID} finished')

            
    def record(self, path = None):
        """ The recordall script
        
        Stores the various results for all the star class instances in the 
        session. These include figures, a csv with the mode ID, a csv with
        the summary statistics of the marginalized posterior distributions, and
        a pickles of the star class instances. 
        
        Parameters
        ----------
        path : str
            Dictory pathname to place the results        
        """
        
        import pickle
        
        if not path:
            raise ValueError('Specify path for recording your session')
        
        for star in self.stars:
            
            if not star.recorded:
            
                if isinstance(star.figures, dict):
                    for key in star.figures.keys():
                        fig = star.figures[key]                   
                        fig.savefig(os.path.join(*[path, f'{star.ID}_{key}.png']))
                star.figures = None  # TODO - can't pickle fig instances?
                                
                with open(os.path.join(*[path, f'{star.ID}.p']), "wb") as f: 
                    pickle.dump(star, f)
                
                star.asy_result.modeID.to_csv(os.path.join(*[path, f'{star.ID}_modeID.csv']))
                star.asy_result.summary.to_csv(os.path.join(*[path, f'{star.ID}_summary.csv']))
            
                star.recorded = True
            elif star.recorded:
                print(f'{star.ID} has already been recorded.')
            else:
                raise ValueError('Unrecognized value in star.recorded.')
class star():
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Parameters
    ----------
    ID : string, int
        Target identifier, if custom timeseries/periodogram is provided, it 
        must be resolvable by LightKurve (KIC, TIC, EPIC, HD, etc.)
    f : float, array
        Array of frequency bins of the spectrum (muHz)
    s : array
        The power at frequencies f
    numax : list
        List of the form [numax, numax_error], list of lists for multiple 
        targets 
    dnu : list
        List of the form [dnu, dnu_error], list of lists for multiple targets
    teff : list, optional
        List of the form [teff, teff_error], list of lists for multiple targets
    bp_rp : list, optional
        List of the form [bp_rp, bp_rp_error], list of lists for multiple 
        targets
    epsilon : list, optional
        List of the form [epsilon, epsilon_error], list of lists for multiple 
        targets   
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run. 
        Warning, if running multiple targets, make sure you have enough memory.
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead. 
        Untested on short cadence. 

    Attributes
    ----------
    data_file : str
        Filepath to file containing prior data on the fit parameters, either
        from literature or previous fits.
    figures : list
        List of figures objects that are created if plotting fit outputs. These
        will be saved if session.record is called.
    asy_result : pbjam.asy_peakbag.asymptotic_fit class instance
        Contains the result of the modeID stage of the peakbagging process from
        fitting the asymptotic relation to the provided data.
    """

    def __init__(self, ID, f, s, numax, dnu, teff=None, bp_rp=None,
                 epsilon=None, source=None, store_chains=False, nthreads=1):
        self.ID = ID
        self.f = f
        self.s = s
        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp
        self.epsilon = epsilon
        self.asy_result = None
        self.source = source
        self.nthreads = nthreads
        self.store_chains = store_chains
        self.data_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])
        self.figures = {}
        self.recorded = False

    def asymptotic_modeid(self, d02=None, alpha=None, mode_width=None,
                          env_width=None, env_height=None, norders=8):
        """ Perform mode ID using the asymptotic method.

        Calls the asymptotic_fit method from asy_peakbag and does an MCMC fit
        to the asymptotic relation for l=2,0 pairs in the spectrum, with a
        multivariate KDE as a prior on all the parameters.

        Results are stored in the star.asy_result attribute.

        Parameters
        ----------
        d02 : float, optional
            Initial guess for the small frequency separation (in muHz) between
            l=0 and l=2.
        alpha : float, optional
            Initial guess for the scale of the second order frequency term in
            the asymptotic relation
        mode_width : float, optional
            Initial guess for the mode width (in log10!) for all the modes that
            are fit.
        env_width : float, optional
            Initial guess for the p-mode envelope width (muHz)
        env_height : float, optional
            Initial guess for the p-mode envelope height
        norders : int, optional
            Number of radial orders to fit
        store_chains : bool, optional
            Flag for storing all the full set of samples from the MCMC run. 
            Warning, if running multiple targets, make sure you have enough 
            memory.
        nthreads : int, optional
            Number of multiprocessing threads to use to perform the fit. For 
            long cadence data 1 is best, more will just add parallelization 
            overhead. Untested on short cadence. 
        """

        fit = asymptotic_fit(self, d02, alpha, mode_width, env_width,
                             env_height, store_chains=self.store_chains,
                             nthreads=self.nthreads, norders=norders)
        fit.run()

        self.asy_result = fit

    def make_spectrum_plot(self, ax, sel, model, modeID, best):
        """ Plot the spectrum and model

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        sel : boolean array
            Used to select the range of frequency bins that the model is
            computed on.
        model : asy_peakbag.model.model instance
            Function for computing a spectrum model given a set of parameters.
        modeID : pandas.DataFrame instance
            Dataframe containing the mode angular degree and frequency of the
            fit modes.
        best : list of floats
            The parameters of the maximum likelihood estimate from the fit.
        """
        ax.plot(self.f[sel], self.s[sel], lw=0.5, label='Spectrum',
                color='C0', alpha=0.5)

        ax.plot(self.f[sel], model, lw=3, color='C3', alpha=1)

        linestyles = ['-', '--', '-.', '.']
        labels = ['$l=0$', '$l=1$', '$l=2$', '$l=3$']
        for i in range(len(modeID)):
            ax.axvline(modeID['nu_mu'][i], color='C3',
                       ls=linestyles[modeID['ell'][i]], alpha=0.5)

        for i in np.unique(modeID['ell']):
            ax.plot([-100, -101], [-100, -101], ls=linestyles[i],
                    color='C3', label=labels[i])
        ax.plot([-100, -101], [-100, -101], label='Model', lw=3,
                color='C3')
        ax.axvline(best['numax'], color='k', alpha=0.75, lw=3,
                   label=r'$\nu_{\mathrm{max}}$')

        ax.set_ylim(0, min([best['env_height'] * 5, max(self.s[sel])]))
        ax.set_ylabel('SNR')
        ax.set_xticks([])
        ax.set_xlim(self.f[sel][0],self.f[sel][-1])
        ax.legend()

    def make_residual_plot(self, ax, sel):
        """ Make residual plot

        Plot the ratio (residual) of the spectrum and the best-fit model

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        sel : boolean array
            Used to select the range of frequency bins that the model is
            computed on.
        """
        ax.plot(self.f[sel], self.residual)
        ax.set_xlabel(r'Frequency [$\mu$Hz]')
        ax.set_xlim(self.f[sel][0], self.f[sel][-1])
        ax.set_ylabel('SNR/Model')
        ax.set_yscale('log')
        ax.set_ylim(1e-1, max(self.residual))

    def make_residual_kde_plot(self, ax, res_lims):
        """ Make residual kde plot

        Plot the KDE of the residual along with a reference KDE based on a
        samples drawn from a pure exponential distribution.

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        res_lims : list of floats
            Axis limits from residual plot, so that the plot scales agree.
        """
        res = self.residual
        ref = np.random.exponential(scale=1, size=max(2000, len(res)))
        res_kde = gaussian_kde(res)
        ref_kde = gaussian_kde(ref)
        y = np.linspace(res_lims[0], res_lims[1], 5000)
        xlim = [min([min(res_kde(y)), min(ref_kde(y))]),
                max([max(res_kde(y)), max(ref_kde(y))])]
        cols = ['C0', 'C1']
        for i, kde in enumerate([res_kde(y), ref_kde(y)]):
            ax.plot(kde, y, lw=4, color=cols[i])
            ax.fill_betweenx(y, x2=xlim[0], x1=kde, color=cols[i], alpha=0.5)
        ax.plot(np.exp(-y), y, ls='dashed', color='k', lw=1)
        ax.set_ylim(y[0], y[-1])
        ax.set_xlim(max([1e-4, xlim[0]]), 1.1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def make_Teff_plot(self, ax, gs, percs, prior):
        """ Plot Teff results

        Plot the initial guess and best-fit Teff, with the prior as context.

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        gs : dictionary
            Dictionary containing initial guess values for the fit
        percs : pandas.Series
            Pandas series containing the percentile values of the marginalized
            posteriors from the fit.
        prior : pandas.DataFrame
            Pandas DataFrame with the prior values on the the fit parameters.
        """
        ax.errorbar(x=gs['dnu'][0], y=gs['teff'][0], xerr=gs['dnu'][1],
                    yerr=gs['teff'][1], fmt='o', color='C1')
        ax.errorbar(x=percs['dnu'][1], y=percs['teff'][1],
                    xerr=np.diff(percs['dnu']).reshape(2, 1),
                    yerr=np.diff(percs['teff']).reshape(2, 1),
                    fmt='o', color='C0')
        ax.scatter(prior['dnu'], prior['teff'], c='k', s=2, alpha=0.2)
        ax.set_xlabel(r'$\Delta\nu$ [$\mu$Hz]')
        ax.set_ylabel(r'$T_{\mathrm{eff}}$ [K]')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xscale('log')

    def make_epsilon_plot(self, ax, gs, percs, prior):
        """ Plot epsilon results

        Plot the initial guess and best-fit epsilon, with the prior as context.

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        gs : dictionary
            Dictionary containing initial guess values for the fit
        percs : pandas.Series
            Pandas series containing the percentile values of the marginalized
            posteriors from the fit.
        prior : pandas.DataFrame
            Pandas DataFrame with the prior values on the the fit parameters.
        """
        ax.errorbar(x=percs['dnu'][1], y=percs['eps'][1],
                    xerr=np.diff(percs['dnu']).reshape(2, 1),
                    yerr=np.diff(percs['eps']).reshape(2, 1),
                    fmt='o', color='C0')
        ax.scatter(prior['dnu'], prior['eps'], c='k', s=2, alpha=0.2)
        ax.set_ylabel(r'$\epsilon$')
        ax.set_ylim(0.4, 1.6)
        ax.set_xscale('log')
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def make_numax_plot(self, ax, gs, percs, prior):
        """ Plot numax results

        Plot the initial guess and best-fit numax, with the prior as context.

        Parameters
        ----------
        ax : matplotlib axis instance
            Axis instance to plot in.
        gs : dictionary
            Dictionary containing initial guess values for the fit
        percs : pandas.Series
            Pandas series containing the percentile values of the marginalized
            posteriors from the fit.
        prior : pandas.DataFrame
            Pandas DataFrame with the prior values on the the fit parameters.
        """

        ax.errorbar(x=gs['dnu'][0], y=gs['numax'][0], xerr=gs['dnu'][1],
                    yerr=gs['numax'][1], fmt='o', color='C1')
        ax.errorbar(x=percs['dnu'][1], y=percs['numax'][1],
                    xerr=np.diff(percs['dnu']).reshape(2, 1),
                    yerr=np.diff(percs['numax']).reshape(2, 1),
                    fmt='o', color='C0')
        ax.scatter(prior['dnu'], prior['numax'], c='k', s=2, alpha=0.2)
        ax.set_ylabel(r'$\nu_{\mathrm{max}}$ [$\mu$Hz]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def plot_asyfit(self, fig=None, model=None, modeID=None):
        """ Make diagnostic plot of the fit.

        Plot various diagnostics of a fit, including the best-fit model,
        spectrum residuals, residual histogram (KDE), and numax, epsilon, Teff
        context plots.

        Parameters
        ----------
        fig : matplotlib figure instance
            Figure instance to plot in.
        model : array of floats
            Spectrum model to overplot. By default this is the best-fit model
            from the latest fit.
        modeID : pandas.DataFrame
            Dataframe of angular degrees and frequencies from the fit
        """

        if not model:
            model = self.asy_result.best_model
        if not modeID:
            modeID = self.asy_result.modeID
        if not fig:
            fig = plt.figure(figsize=(12, 7))

        prior = pd.read_csv('pbjam/data/prior_data.csv')
        smry = self.asy_result.summary
        gs = self.asy_result.guess
        sel = self.asy_result.sel
        percs = smry.loc[['16th', '50th', '84th']]

        self.residual = self.s[sel]/model

        # Main plot
        ax_main = fig.add_axes([0.05, 0.23, 0.69, 0.76])
        self.make_spectrum_plot(ax_main, sel, model, modeID, smry.loc['best'])

        # Residual plot
        ax_res = fig.add_axes([0.05, 0.07, 0.69, 0.15])
        self.make_residual_plot(ax_res, sel)

        # KDE plot
        ax_kde = fig.add_axes([0.75, 0.07, 0.19, 0.15])
        self.make_residual_kde_plot(ax_kde, ax_res.get_ylim())

        # Teff plot
        ax_teff = fig.add_axes([0.75, 0.30, 0.19, 0.226])
        self.make_Teff_plot(ax_teff, gs, percs, prior)

        # epsilon plot
        ax_eps = fig.add_axes([0.75, 0.53, 0.19, 0.226])
        self.make_epsilon_plot(ax_eps, gs, percs, prior)

        # nu_max plot
        ax_numax = fig.add_axes([0.75, 0.76, 0.19, 0.23])
        self.make_numax_plot(ax_numax, gs, percs, prior)

        self.figures['summary'] = fig
        
        return fig

    def corner(self, chains = None, labels = None):
        """ Make a corner plot for the MCMC chains
        
        Returns
        -------
        fig : matplotlib figure instance
            Figure instance with corner plot
        
        """
        
        import corner
        
        if not chains:
            chains = self.asy_result.flatchain
        if not labels:
            labels = self.asy_result.pars_names

        fig = corner.corner(xs = chains, labels = labels, plot_density = False)

        self.figures['corner'] = fig

        return fig 
