""" Setup PBjam sessions and perform mode ID and peakbagging

This module contains the input layer for setting up PBjam sessions for 
peakbagging solar-like oscillators. This is the easiest way to handle targets 
in PBjam.

It's possible to manually initiate star class instances and do all the fitting
that way, but it's simpler to just use the session class, which handles
everything, including formatting of the inputs.

A PBjam session is started by initializing the session class instance with a
target ID, $\nu_{max}$, a large separation, effective temperature and Gaia bp_rp 
color. The class instance is the called to run through all the peakbagging 
steps automatically. See the Session class documentation for an example.

Lists of the above can be provided for multiple targets, but it's often simpler
to just provide PBjam with a dictionary or Pandas dataframe. See mytgts.csv
for a template.

Custom timeseries or periodograms can be provided as either file pathnames,
`numpy' arrays, or lightkurve.LightCurve/lightkurve.periodogram objects. If
nothing is provided PBjam will download the data automatically using 
`LightKurve'.

Specific quarters, campgains or sectors can be requested with the relevant
keyword (i.e., 'quarter' for KIC, etc.). If none of these are provided, PBjam
will download all available data, using the long cadence versions by default.

Once initialized, the session class contains a list of star class instances
for each requested target, with associated spectra for each.

The next step is to perform a mode ID on the spectra. At the moment PBjam
only supports use of the asymptotic relation mode ID method. Additional methods
can be added in future.

Finally the peakbagging method takes the output from the modeID and performs
a proper HMC peakbagging run to get the unparameterized mode frequencies.

Plotting the results of each stage is also possible.

Note
----
For automatic download the long cadence data set is used by default, so set
the cadence to `short' for main-sequence targets.

"""

import lightkurve as lk
import numpy as np
import astropy.units as units
import pandas as pd
import os, pickle, warnings
from .star import star
from datetime import datetime

        


def _organize_sess_dataframe(vardf):
    """ Takes input dataframe and tidies it up.

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
    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter', 'mission']
    doubles = ['teff', 'bp_rp']

    for key in singles:
        if key not in vardf.keys():
            vardf[key] = np.array([None]*N)

    for key in doubles:
        if key not in vardf.keys():
            vardf[key] = np.array([None]*N)
            vardf[key+'_err'] = np.array([None]*N)

    if 'timeseries' not in vardf.keys():
        _format_col(vardf, None, 'timeseries')
    if 'spectrum' not in vardf.keys():
        _format_col(vardf, None, 'spectrum')


def _organize_sess_input(**vardct):
    """ Takes input and organizes them in a dataframe.

    Checks to see if required inputs are present and inserts them into a
    dataframe. Any optional columns that are not included in the input are
    added as empty columns.

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
    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter', 'mission']
    doubles = ['numax', 'dnu', 'teff', 'bp_rp']

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

def _sort_lc(lc):
    """ Sort a lightcurve in Lightkurve object.

    Lightkurve lightcurves are not necessarily sorted in time, which causes
    an error in periodogram.

    Parameters
    ----------
    lc : Lightkurve.LightCurve instance
        Lightkurve object to be modified

    Returns
    -------
    lc : Lightkurve.LightCurve instance
        The sorted Lightkurve object
        
    """

    sidx = np.argsort(lc.time)
    lc.time = lc.time[sidx]
    lc.flux = lc.flux[sidx]
    return lc

#def _launch_query(id, download_dir, lkwargs):
#    """ Search for target on MAST server.
#
#    Get all the lightcurves available for a target id, using options in kwargs
#    dictionary. The lightcurves are downloaded using the lightkurve API, and
#    the target ID must therefore be parseable by lightkurve.
#
#    Parameters
#    ----------
#    id : string
#        Target id, must be resolvable by Lightkurve.
#
#    download_dir : str
#        Directory to download the lightcurves into.
#
#    lkwargs : dictionary containing keywords for the LightKurve search. For 
#        example cadence, quarter, campaign, sector, month.
#
#    Returns
#    -------
#    search_results : list
#        List of fits files for the requested target
#        
#    """
#
#    print(f'Querying MAST for {id} light curve.')
#    search_results = lk.search_lightcurvefile(target=id, **lkwargs)
#    if len(search_results) == 0:
#        warnings.warn('LightKurve did not return %s cadence data for %s' % (lkwargs['cadence'], id))
#        return []
#    else:
#        return search_results.download_all(download_dir=download_dir)







#def _set_cadence(lkwargs):
#    """ Select the cadence of the data to download
#    
#    Determines the extension to use later in the lookup of cached fits files,
#    to be passed to LightKurve for online lookup.
#    
#    If no cadence argument is passed it will default to long cadence. 
#    
#    Parameters
#    ----------
#    lkwargs : dict
#        Dictionary to be passed to LightKurve.
#        
#    Returns
#    -------
#    ext : str
#        Fits file short/long cadence extension.
#        
#    """
#    if not lkwargs['cadence']:
#        lkwargs['cadence'] = 'long'
#    
#    if lkwargs['cadence'] == 'short':
#        ext = '*_slc.fits'
#    elif lkwargs['cadence'] == 'long':
#        ext = '*_lc.fits'
#    else:
#        raise TypeError('Unrecognized cadence input for %s' % (id))
#    return ext

#def _set_cache_dir(download_dir):
#    """ Determine which directory to use as cache
#    
#    Parameters
#    ----------
#    download_dir : str
#        None or path to store results from a star
#    
#    Returns
#    -------
#    cache_dir : str
#        Path to store results from a star
#        
#    """
#    
#    if not download_dir:
#        download_dir = os.path.join(*[os.path.expanduser('~'), 
#                                     '.lightkurve-cache'])
#    return download_dir    

#def _set_mission(ID, lkwargs):
#    """ Set mission keyword in lkwargs
#    
#    If no mission is selected will attempt to figure it out based on any
#    prefixes in the ID string, and add this to the LightKurve keywords 
#    arguments dictionary.
#    
#    Parameters
#    ----------
#    ID : str
#        ID string of the target
#    lkwargs : dict
#        Dictionary to be passed to LightKurve
#        
#    """
#    if lkwargs['mission'] is None:
#        if ('kic' in ID.lower()) or ('kplr' in ID.lower()) :
#            lkwargs['mission'] = 'kepler'
#        elif ('ktwo' in ID.lower()) or ('epic' in ID.lower()):
#            lkwargs['mission'] = 'k2'
#        elif ('tic' in ID.lower()) or ('tess' in ID.lower()):
#            lkwargs['mission'] = 'tess'
#        else:
#            warnings.warn('Unknown mission selected. MAST might not understand.')

#def _lookup_cached_files(ID, download_dir, ext):
#    """ Look through the local cache directory for target files
#    
#    Looks through the local cache directory for any files matching the ID of 
#    the target, and with the requested extension.
#    
#    Parameters
#    ----------
#    id : str
#        Input ID for the target
#    download_dir : str
#        Path to the cache directory
#    ext : str
#        Fits file short/long cadence extension
#    
#    Returns
#    -------
#    tgtfiles : list
#        List of file names matching the search criteria
#        
#    """
#    
#    if isinstance(ID, str):
#        baseid = ID.lower()
#        for prefix in ['kic','epic','tic','kplr']:
#            baseid = baseid.replace(prefix, '')
#        baseid = str(int(baseid))
#      
#    tgtfiles = glob.glob(os.path.join(*[download_dir, 'mastDownload', '*', 
#                                        f'*{baseid}*', ext]))
#    return tgtfiles

def _query_lightkurve(ID, download_dir, use_cached, lkwargs):
    
    ID = _format_name(ID)
    
    _set_mission(ID, lkwargs)
    
    ID = check_ID(ID, lkwargs)
    
    search = perform_search(ID, lkwargs)
    
    fitsFiles = check_lc_cache(search, lkwargs)
    
    lc = load_fits(fitsFiles, lkwargs)
    
    lc = _clean_lc(lc)
    
    return lc
    
#    """ Check cache for fits file, or download it.
#
#    Based on use_cached flag, will look in the cache for fits file
#    corresponding to request id star. If nothing is found in cached it will be
#    downloaded from the MAST server.
#
#    Parameters
#    ----------
#    id : string
#        Identifier for the requested star. Must be resolvable by Lightkurve.
#    download_dir : str
#        Path to the cache directory    
#    use_cached: bool
#        Whether or not to used data in the Lightkurve cache.
#    lkwargs : dict
#        Dictionary containing keywords for the Lightkurve search.
#        cadence, quarter, campaign, sector, month.
#
#    Note:
#    -----
#    Prioritizes long cadence over short cadence unless otherwise specified.
#
#    """
#    
#    cache_dir = _set_cache_dir(download_dir)
#    
#    _set_mission(ID, lkwargs)
#    
#    ext = _set_cadence(lkwargs)
#
#    tgtfiles = _lookup_cached_files(ID, cache_dir, ext)
#
#    if (use_cached and (len(tgtfiles) != 0)):
#        lc_col = [lk.open(n) for n in tgtfiles]
#        
#    elif (not use_cached) or (use_cached and (len(tgtfiles) == 0)):
#        if (use_cached and (len(tgtfiles) == 0)):
#            warnings.warn('Could not find %s cadence data for %s in cache, checking MAST...' % (lkwargs['cadence'], ID))
#
#        lc_col = _launch_query(ID, cache_dir, lkwargs)
#
#        if len(lc_col) == 0:
#            raise ValueError("Could not find %s cadence data for %s in cache or on MAST" % (lkwargs['cadence'], ID))
#    else:
#        raise ValueError('Could not find any cached data, and failed to access MAST')
#    
#    # Perform reduction on first lc of the lc collection and append the rest
#    lc0 = _clean_lc(lc_col[0].PDCSAP_FLUX)
#    for i, lc in enumerate(lc_col[1:]):
#        lc0 = lc0.append(_clean_lc(lc.PDCSAP_FLUX))
#        
#    return lc0

      
def _arr_to_lk(x, y, name, typ):
    """ LightKurve object from input.

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
    elif typ == 'spectrum':
        return lk.periodogram.Periodogram(x*units.microhertz,
                                          units.Quantity(y, None),
                                          targetid=name)
    else:
        raise KeyError("Don't modify anything but spectrum and timeseries cols")


def _format_col(vardf, col, key):
    """ Add timeseries or spectrum column to dataframe based on input

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

    In both of the latter cases col is converted to Lightkurve object(s).

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Pandas Dataframe instance to which either a timeseries or spectrum column
        will be added.
    col : object
        Input from Session call, corresponding to key
    key : str
        Name of column to add to vardf

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
        vardf[key] = [_arr_to_lk(x, y, vardf['ID'][0], key)]

    # If dim = 3, it's a list of arrays or tuples
    elif col.ndim == 3:
        temp = np.array([], dtype=object)
        for i in range(N):
            x = np.array(col[i, 0, :], dtype=float)
            y = np.array(col[i, 1, :], dtype=float)
            temp = np.append(temp,
                             np.array([_arr_to_lk(x, y, vardf.loc[i, 'ID'], key)]))
        vardf[key] = temp
    else:
        print('Unhandled exception')



def _lc_to_lk(ID, tsIn, specIn, download_dir, use_cached, lkwargs):
    """ Convert time series column in dataframe to lk.LightCurve object

    Goes through the timeseries column in the dataframe and tries to convert
    it to a Lightkurve.LightCurve object. If string, it's assumed to be a file
    path name, if None, it will query the LightCurve cache locally or MAST if
    nothing is found. Skips column entries which are already LightKurve objects
    or if a spectrum for the star in question exists in the spectrum column.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'timeseries' column consisting of None, strings
        or Lightkurve.LightCurve objects.

    """

    tinyoffset = 1  # to avoid cases LC median = 0 (lk doesn't like it) This may no longer be necessary.
    
        
    if isinstance(tsIn, str):
        try:
            t, d = np.genfromtxt(tsIn, usecols=(0, 1), delimiter = ',').T 
        except:
            try:
                t, d = np.genfromtxt(tsIn, usecols=(0, 1), delimiter = ' ').T
            except:
                raise IOError('Failed to read the provided ascii files. Please check that they have the required 2-column format, and they are use either comma or white-space delimiters.')
        
        d += tinyoffset
        tsOut = _arr_to_lk(t, d, ID, 'timeseries')
        
    elif not tsIn:
        if specIn:
            pass
        else:
            tsOut = _query_lightkurve(ID, download_dir, use_cached, lkwargs)

    elif tsIn.__module__ == lk.lightcurve.__name__:
        pass
    else:
        raise TypeError("Can't handle this type of time series object")

    if tsOut:
        _sort_lc(tsOut)

    return tsOut


def _lk_to_pg(ID, tsIn, specIn):
    """ Convert spectrum column in dataframe to Lightkurve periodgram object list

    Takes whatever is in the spectrum column of a dataframe and tries to turn it
    into a Lightkurve.periodogram object. If column entry is a string, it
    assumes it's a path name, if None, it will try to take what's in the
    timeseries column and compute periodogram based on that. If a periodogram
    object is already present, it'll just flatten it (to be safe) and then
    continue on.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'spectrum' column consisting of None, strings or
        Lightkurve.periodogram objects.

    """
    
    if isinstance(specIn, str):
        f, s = np.genfromtxt(specIn, usecols=(0, 1)).T
        specOut = _arr_to_lk(f, s, ID, 'spectrum')
        
    elif not specIn:
        specOut = tsIn.to_periodogram(freq_unit=units.microHertz, normalization='psd').flatten()
        
    elif specIn.__module__ == lk.periodogram.__name__:
        specOut = specIn.flatten()
        
    else:
        raise TypeError("Can't handle this type of time series object")

    return specOut



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
    value and it's error estimate.

    For multiple targets all the above can be provided as lists, but the
    easiest way is to simply provide a dataframe from a csv file.

    By default, PBjam will download all the available data, favoring long
    cadence. Cadence and specific observing seasons (quarter, month, campagin,
    sector) can be specified for more detailed control. The data is downloaded
    using the Lightkurve API, and cleaned on the timeseries level by removing
    NaNs, and outliers, and flattening the lightcurve (again using LightKurve's
    API). Note that the flattening process may impact the appearance of the
    granulation background.
    
    Note
    ----
    If you have a large directory of cached lightcurves or power spectra, it 
    is best to supply the filenames to the files.

    Examples
    --------
    Peakbagging run for a single target:

    >>> jam_sess = pbjam.session(ID =  '4448777',  numax = [220.0, 3.0],
                             dnu = [16.97, 0.01], teff = [4750, 100],
                             bp_rp = [1.34, 0.01], cadence = 'short')
    >>> jam_sess()

    Peakbagging run for multiple targets:
    >>> jam_sess = pbjam.session(dictlike = mydataframe)
    >>> jam_sess()

    Parameters
    ----------
    ID : string, optional
        Target identifier, if custom timeseries/periodogram is provided it can
        be anything. Otherwise it must be resolvable by LightKurve so that it 
        can be download (KIC, TIC, EPIC, HD, etc.)
    numax : list, optional
        List of the form [numax, numax_error], list of lists for multiple
        targets
    dnu : list, optional
        List of the form [dnu, dnu_error], list of lists for multiple targets
    teff : list, optional
        List of the form [teff, teff_error], list of lists for multiple targets
    bp_rp : list, optional
        List of the form [bp_rp, bp_rp_error], list of lists for multiple
        targets
    timeseries : object, optional
        Timeseries input. Leave as None for PBjam to download it automatically.
        Otherwise, arrays of shape (2,N), lightkurve.LightCurve objects, or
        strings for pathnames are accepted.
    spectrum : object, optional
        Periodogram input. Leave as None for PBjam to use Timeseries to compute
        it for you. Otherwise, arrays of shape (2,N), lightkurve.periodogram
        objects, or strings for pathnames are accepted.
    dictlike : pandas.DataFrame or dictionary, optional
        DataFrame, dictionary, record array with a list of targets, and their
        properties. If string, PBjam will assume it's a pathname to a csv file.
        Specify timeseries and spectrum columns with file pathnames to use manually
        reduced data.
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run.
        Warning, if running multiple targets, make sure you have enough memory.
    nthreads : int
        Number of multiprocessing threads to use to perform the HMC peakbag 
        fit. 
    use_cached : bool, optional
        Flag for using cached data. If fitting the same targets multiple times,
        use to this to not download the data every time.
    cadence : string, optional
        Argument for lightkurve to download correct data type. Can be 'short'
        or 'long'. 'long' is default setting, so if you're looking at main
        sequence stars, make sure to manually set 'short'.    
    campaign : int, optional
        Argument for lightkurve when requesting K2 data.
    sector : int, optional
        Argument for lightkurve when requesting TESS data.
    month : int, optional
        Argument for lightkurve when requesting Kepler short cadence data.
    quarter : int, optional
        Argument for lightkurve when requesting Kepler data.    
    mission : str, optional
        Which mission to use data from. Default is all, so if your target has
        been observed by, e.g., both Kepler and TESS, LightKurve will throw
        and error.
    make_plots : bool, optional
        Whether or not to automatically generate diagnostic plots for the 
        different stages of the peakbagging process.    
    path : str, optional
        Path to store the plots and results for the various stages of the 
        peakbagging process.    
    download_dir : str, optional
        Directory to cache lightkurve downloads. Lightkurve will place the fits
        files in the default lightkurve cache path in your home directory.     
    model_type : str, optional
        Argument passed to peakbag, defines which model type to be used to 
        represent the mode linewidths. Options are 'simple' or 'model_gp'.
           
    Attributes
    ----------

    stars : list
        Session will store star class instances in this list, based on the
        requested targets.   
    pb_model_type : str
        Model to use for the mode widths in peakbag.
       
    """

    def __init__(self, ID=None, numax=None, dnu=None, teff=None, bp_rp=None,
                 timeseries=None, spectrum=None, dictlike=None, use_cached=False, 
                 cadence=None, campaign=None, sector=None, month=None, 
                 quarter=None, mission=None, path=None, download_dir=None):

        self.stars = []
        
        if isinstance(dictlike, (dict, np.recarray, pd.DataFrame, str)):
            if isinstance(dictlike, str):
                vardf = pd.read_csv(dictlike)
            else:
                try:
                    vardf = pd.DataFrame.from_records(dictlike)
                except TypeError:
                    print('Unrecognized type in dictlike. Must be able to convert to dataframe through pandas.DataFrame.from_records()')

            if any([ID, numax, dnu, teff, bp_rp]):
                warnings.warn('Dictlike provided as input, ignoring other input fit parameters.')

            _organize_sess_dataframe(vardf)

        elif ID:
            vardf = _organize_sess_input(ID=ID, numax=numax, dnu=dnu, teff=teff,
                                         bp_rp=bp_rp, cadence=cadence,
                                         campaign=campaign, sector=sector,
                                         month=month, quarter=quarter, 
                                         mission=mission)
            
            _format_col(vardf, timeseries, 'timeseries')
            _format_col(vardf, spectrum, 'spectrum')

        for i in vardf.index:
            
            lkwargs = {x: vardf.loc[i, x] for x in ['cadence', 'month', 
                                                    'sector', 'campaign',
                                                    'quarter', 'mission']}
    
            vardf.at[i, 'timeseries'] = _lc_to_lk(vardf.loc[i, 'ID'], 
                                                  vardf.loc[i, 'timeseries'], 
                                                  vardf.loc[i, 'spectrum'],
                                                  download_dir, 
                                                  use_cached,
                                                  lkwargs)
            
            vardf.at[i,'spectrum'] = _lk_to_pg(vardf.loc[i,'ID'], 
                                               vardf.loc[i, 'timeseries'], 
                                               vardf.loc[i, 'spectrum'])
            
            self.stars.append(star(ID=vardf.loc[i, 'ID'],
                                   pg=vardf.loc[i, 'spectrum'],
                                   numax=vardf.loc[i, ['numax', 'numax_err']].values,
                                   dnu=vardf.loc[i, ['dnu', 'dnu_err']].values,
                                   teff=vardf.loc[i, ['teff', 'teff_err']].values,
                                   bp_rp=vardf.loc[i, ['bp_rp', 'bp_rp_err']].values,
                                   path=path))

        for i, st in enumerate(self.stars):
            if st.numax[0] > st.f[-1]:
                warnings.warn("Input numax is greater than Nyquist frequeny for %s" % (st.ID))

    def __call__(self, bw_fac=1, norders=8, model_type='simple', tune=1500, 
                 nthreads=1, verbose=False, make_plots=False, store_chains=False, 
                 asy_sampling='mcmc', developer_mode=False):
        """ Call all the star class instances

        Once initialized, calling the session class instance will loop through
        all the stars that it contains, and call each one. This performs a full
        peakbagging run on each star in the session.

        Parameters
        ----------
        bw_fac : float, optional.
            Scaling factor for the KDE bandwidth. By default the bandwidth is
            automatically set, but may be scaled to adjust for sparsity of the 
            prior sample. Default is 1.            
        norders : int, optional.
            Number of orders to include in the fits. Default is 8.            
        model_type : str, optional.
            Can be either 'simple' or 'model_gp' which sets the type of mode 
            width model. Defaults is 'simple'.             
        tune : int, optional
            Numer of tuning steps passed to pm.sample. Default is 1500.         
        nthreads : int, optional.
            Number of processes to spin up in pymc3. Default is 1.    
        verbose : bool, optional.
            Should PBjam say anything? Default is False.
        make_plots : bool, optional.
            Whether or not to produce plots of the results. Default is False.            
        store_chains : bool, optional.
            Whether or not to store MCMC chains on disk. Default is False.
        developer_mode : bool
            Run asy_peakbag in developer mode. Currently just retains the input 
            value of dnu and numax as priors, for the purposes of expanding
            the prior sample. Important: This is not good practice for getting 
            science results!    
            
        """
        
        self.pb_model_type = model_type

        for i, st in enumerate(self.stars):
            try:
                st(bw_fac=bw_fac, tune=tune, norders=norders, 
                   model_type=self.pb_model_type, make_plots=make_plots, 
                   store_chains=store_chains, nthreads=nthreads, 
                   asy_sampling=asy_sampling, developer_mode=developer_mode)
                
                self.stars[i] = None
            
            # Crude way to send error messages that occur in star up to Session 
            # without ending the session. Is there a better way?
            except Exception as ex:
                 message = "Star {0} produced an exception of type {1} occurred. Arguments:\n{2!r}".format(st.ID, type(ex).__name__, ex.args)
                 print(message)
            














def _format_name(name):
    """ Format input ID
    
    Users tend to be inconsistent in naming targets, which is an issue for 
    looking stuff up on, e.g., Simbad. 
    
    This function formats the name so that Simbad doesn't throw a fit.
    
    If the name doesn't look like anything in the variant list it will only be 
    changed to a lower-case string.
    
    Parameters
    ----------
    name : str
        Name to be formatted.
    
    Returns
    -------
    name : str
        Formatted name
        
    """
    import re
    name = str(name)
    name = name.lower()
    
    # Add naming exceptions here
    variants = {'KIC': ['kic', 'kplr', 'KIC'],
                'Gaia DR2': ['gaia dr2', 'gdr2', 'dr2', 'Gaia DR2'],
                'Gaia DR1': ['gaia dr1', 'gdr1', 'dr1', 'Gaia DR1'], 
                'EPIC': ['epic', 'ktwo', 'EPIC'],
                'TIC': ['tic', 'tess', 'TIC']
               }
    
    fname = None
    for key in variants:   
        for x in variants[key]:
            if x in name:
                fname = name.replace(x,'')
                fname = re.sub(r"\s+", "", fname, flags=re.UNICODE)
                fname = key+' '+str(int(fname))
                return fname
            
    return name

def load_fits(files, lkwargs):
    """ Read fitsfiles into a Lightkurve object
    
    Parameters
    ----------
    files : list
        List of pathnames to fits files
    
    Returns
    -------
    lc : lightkurve.lightcurve.KeplerLightCurve object
        Lightkurve light curve object containing the concatenated set of 
        quarters.
        
    """
    if lkwargs['mission'] in ['Kepler', 'K2']:
        lcs = [lk.lightcurvefile.KeplerLightCurveFile(file) for file in files]
        lccol = lk.collections.LightCurveFileCollection(lcs)
        lc = lccol.PDCSAP_FLUX.stitch()
    elif lkwargs['mission'] in ['TESS']:
        lcs = [lk.lightcurvefile.TessLightCurveFile(file) for file in files]
        lccol = lk.collections.LightCurveFileCollection(lcs)
        lc = lccol.PDCSAP_FLUX.stitch()
    return lc

def _set_mission(ID, lkwargs):
    """ Set mission keyword in lkwargs
    
    If no mission is selected will attempt to figure it out based on any
    prefixes in the ID string, and add this to the LightKurve keywords 
    arguments dictionary.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    lkwargs : dict
        Dictionary to be passed to LightKurve
        
    """

    if lkwargs['mission'] is None:
        if ('kic' in ID.lower()):
            lkwargs['mission'] = 'Kepler'
        elif ('epic' in ID.lower()) :
            lkwargs['mission'] = 'K2'
        elif ('tic' in ID.lower()):
            lkwargs['mission'] = 'TESS'
        else:
            lkwargs['mission'] = ('Kepler', 'K2', 'TESS')
            
def search_and_dump(ID, cadence, mission, search_cache):
    """ Get search result and store it in dictionary
    
    """
    current_date = datetime.now().isoformat()
    store_date = current_date[:current_date.index('T')].replace('-','')
       
    search = lk.search_lightcurvefile(ID, cadence=cadence, mission=mission)

    resultDict = {'result': search,
                  'timestamp': store_date}
    
    fname = os.path.join(*[search_cache, f"{ID}_{cadence}.lksearchresult"])

    pickle.dump(resultDict, open(fname, "wb"))
    
    return resultDict   

def check_ID(ID, lkwargs):
    
    if not any([x in ID for x in ['KIC', 'TIC', 'EPIC']]):
        
        search = lk.search_lightcurvefile(ID, cadence=lkwargs['cadence'], mission=lkwargs['mission'])

        if len(search) == 0:
            raise ValueError(f'No results for {ID} found on MAST')

        maxFreqName = max(set(list(search.table['target_name'])), key = list(search.table['target_name']).count)
        maxFreqObsCol = max(set(list(search.table['obs_collection'])), key = list(search.table['obs_collection']).count)

        if maxFreqObsCol == 'TESS':
            prefix = 'TIC'
        else:
            prefix = ''

        temp_id = prefix + maxFreqName

        ID = _format_name(temp_id).replace(' ', '')
        lkwargs['mission'] = maxFreqObsCol
    else:
        ID = ID.replace(' ', '')
    return ID

def perform_search(ID, lkwargs, download_dir=None, cache_expire = 30):
    """ Find filenames related to target
    
    Preferentially accesses cached search results, otherwise searches the 
    MAST archive.
    
    Parameters
    ----------
    ID : str
        Target ID (must be KIC, TIC, or ktwo prefixed)
    cadence : str
        Cadence of the observations, 'short' or 'long'
    mission : str
        Mission 
    download_dir : str
        Directory for fits file and search results caches. Default is ~/.lightkurve-cache. 
    cache_expire : int
        Expiration time for the search cache results. Files older than this will be 
        
    Returns
    -------
    search : lightkurve.search.SearchResult
        Search result from MAST. 
    
    """
       
    # Set default lightkurve cache directory if nothing else is given
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
    
    # Make the search cache dir if it doesn't exist
    cachepath = os.path.join(*[download_dir, 'searchResults', lkwargs['mission']])
    if not os.path.isdir(cachepath):
        os.makedirs(cachepath)

    filepath = os.path.join(*[cachepath, f"{ID}_{lkwargs['cadence']}.lksearchresult"])
    
    if os.path.exists(filepath):  
        
        resultDict = pickle.load(open(filepath, "rb"))
        fdate = resultDict['timestamp'] 
        ddate = datetime.now() - datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:]))
        
        # If file is saved more than cache_expire days ago, a new search is performed
        if ddate.days > cache_expire:   
            resultDict = search_and_dump(ID, lkwargs['cadence'], lkwargs['mission'], cachepath)
            
    else:
        resultDict = search_and_dump(ID, lkwargs['cadence'], lkwargs['mission'], cachepath)
        
    return resultDict['result']

def check_lc_cache(search, lkwargs, download_dir=None):
    """ Query cache directory or download fits files.
    
    Searches the Lightkurve cache directory set by download_dir for fits files
    matching the search query, and returns a list of path names of the fits
    files.
    
    If not cache either doesn't exist or doesn't contain all the files in the
    search, all the fits files will be downloaded again.
    
    Parameters
    ----------
    search : lightkurve.search.SearchResult
        Search result from MAST. 
    download_dir : str
        Top level of the Lightkurve cache directory. default is 
        ~/.lightkurve-cache
        
    Returns
    -------
    files_in_cache : list
        List of path names to the fits files in the cache directory
    
    """
        
    if download_dir is None:
        download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve-cache'])
     
    files_in_cache = []

    for i, row in enumerate(search.table):
        fname = os.path.join(*[download_dir, 'mastDownload', lkwargs['mission'], row['obs_id'], row['productFilename']])
        if os.path.exists(fname):
            files_in_cache.append(fname)
    
    if len(files_in_cache) != len(search):       
        search.download_all(download_dir = download_dir)
        files_in_cache = [os.path.join(*[download_dir, 'mastDownload', lkwargs['mission'], row['obs_id'], row['productFilename']]) for row in search.table]

    return files_in_cache

def _clean_lc(lc):
    """ Perform Lightkurve operations on object.

    Performes basic cleaning of a light curve, removing nans, outliers,
    median filtering etc.

    Parameters
    ----------
    lc : Lightkurve.LightCurve instance
        Lightkurve object to be cleaned

    Returns
    -------
    lc : Lightkurve.LightCurve instance
        The cleaned Lightkurve object
        
    """

    lc = lc.remove_nans().flatten().remove_outliers()
    return lc