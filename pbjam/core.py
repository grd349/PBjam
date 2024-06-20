"""

The `star' class is the core of PBjam and refers to a single target that is to 
be peakbagged. Each `star' instance is assigned an ID and physical input 
parameters, as well as a time series or power spectrum. 

The different steps in the peakbagging process are then passed the `star' 
instance, updating it with the results of each step. The outputs of each step
are stored in a dedicated directory created with the star ID.

The `session' class wraps one or more star class instances and peakbags them all
sequentially. The recommended use of PBjam is the use the `session' class, and
only use the `star' class for more granular control of the peakbagging process.

"""




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
#from pbjam import IO
from pbjam.plotting import plotting
from pbjam import IO
import lightkurve as lk
from lightkurve.periodogram import Periodogram
import numpy as np
import astropy.units as units
import pandas as pd
import os, pickle, warnings
from datetime import datetime
from pbjam.modeID import modeID
from pbjam.peakbagging import peakbag

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
    singles = ['exptime', 'campaign', 'sector', 'month', 'quarter', 'mission']
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
    singles = ['exptime', 'campaign', 'sector', 'month', 'quarter', 'mission']
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

def _query_lightkurve(ID, download_dir, use_cached, lkwargs):
    """ Get time series using LightKurve
    
    Performs a search for available fits files on MAST and then downloads them
    if nessary.
    
    The search results are cached with an expiration of 30 days. If a search
    result is found, the fits file cache is searched for a matching file list
    which is then used.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    download_dir : str
        Directory for fits file and search results caches. 
    use_cached : bool, optional
        Whether or not to use the cached time series. Default is True.
    lkwargs : dict
        Dictionary to be passed to LightKurve  
    
    Returns
    -------
    lc : Lightkurve.LightCurve instance
        The concatenated time series for the target.
    """
    
    ID = _format_name(ID)
    
    _set_mission(ID, lkwargs)
    
    ID = _getMASTidentifier(ID, lkwargs)
    
    search = _perform_search(ID, lkwargs, use_cached)
    
    fitsFiles = _check_lc_cache(search, lkwargs['mission'])
    
    lc = _load_fits(fitsFiles, lkwargs['mission'])
    
    lc = _clean_lc(lc)
    
    return lc
    
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
        return Periodogram(x*units.microhertz,
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
     ID : str
        ID string of the target
    tsIn : Lightkurve.LightCurve object
        Lightkurve.LightCurve with the time series of the target. If specIn is
        None, this will be used to compute the spectrum.
    specIn : str or Lightkurve.periodogram object
        Lightkurve.periodogram object containing the spectrum of the target.
    download_dir : str
        Directory for fits file and search results caches. 
    use_cached : bool
        Whether or not to use the cached time series. 
    lkwargs : dict
        Dictionary with arguments to be passed to lightkurve. 
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
        
        tsOut = lk.LightCurve(time=t, flux=d, targetid=ID)
        
    elif tsIn is None:
        if specIn:
            tsOut = None
            pass
        else:
            tsOut = _query_lightkurve(ID, download_dir, use_cached, lkwargs)

    elif tsIn.__module__ == lk.lightcurve.__name__:
        tsOut = tsIn
    else:
        raise TypeError("Can't handle this type of time series object")

    if tsOut:
        _sort_lc(tsOut)

    return tsOut

def _lk_to_pg(ID, tsIn, specIn):
    """ Convert spectrum column in dataframe to Lightkurve.periodgram objects

    Takes whatever is in the spectrum column of a dataframe and tries to turn it
    into a Lightkurve.periodogram object. If column entry is a string, it
    assumes it's a path name, if None, it will try to take what's in the
    timeseries column and compute periodogram based on that. If a periodogram
    object is already present, it'll just flatten it (to be safe) and then
    continue on.

    Parameters
    ----------
    ID : str
        ID string of the target
    tsIn : Lightkurve.LightCurve object
        Lightkurve.LightCurve with the time series of the target. If specIn is
        None, this will be used to compute the spectrum.
    specIn : str or Lightkurve.periodogram object
        Lightkurve.periodogram object containing the spectrum of the target.

    Returns
    -------
    specOut : Lightkurve.periodogram object
        Lightkurve.periodogram object containing the spectrum of the target.
        Note this is now flattened so that it corresponds to the SNR spectrum.
    """
    
    if isinstance(specIn, str):
        f, s = np.genfromtxt(specIn, usecols=(0, 1)).T
        specOut = lk.periodogram.Periodogram(f*units.microhertz,
                                          units.Quantity(s, None),
                                          targetid=ID).flatten() 
        
    elif not specIn:
        specOut = tsIn.to_periodogram(freq_unit=units.microHertz, normalization='psd').flatten()
        
    elif specIn.__module__ == lk.periodogram.__name__:
        specOut = specIn.flatten()
        
    else:
        raise TypeError("Can't handle this type of time series object")

    return specOut

def _load_fits(files, mission):
    """ Read fitsfiles into a Lightkurve object
    
    Parameters
    ----------
    files : list
        List of pathnames to fits files
    mission : str
        Which mission to download the data from.
    
    Returns
    -------
    lc : lightkurve.lightcurve.KeplerLightCurve object
        Lightkurve light curve object containing the concatenated set of 
        quarters.
        
    """
    if mission in ['Kepler', 'K2']:
        lcs = [lk.lightcurvefile.KeplerLightCurveFile(file) for file in files]
        lcCol = lk.collections.LightCurveCollection(lcs)
        lc = lcCol.stitch()
    elif mission in ['TESS']:
        lcs = [lk.lightcurvefile.TessLightCurveFile(file) for file in files]
        lcCol = lk.collections.LightCurveCollection(lcs)
        lc = lcCol.stitch()
    return lc

def _set_mission(ID, lkwargs):
    """ Set mission keyword in lkwargs.
    
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
            
def _search_and_dump(ID, lkwargs, search_cache):
    """ Get lightkurve search result online.
    
    Uses the lightkurve search_lightcurve to find the list of available
    data for a target ID. 
    
    Stores the result in the ~/.lightkurve-cache/searchResult directory as a 
    dictionary with the search result object and a timestamp.
    
    Parameters
    ----------
    ID : str
        ID string of the target
    lkwargs : dict
        Dictionary to be passed to LightKurve
    search_cache : str
        Directory to store the search results in. 
        
    Returns
    -------
    resultDict : dict
        Dictionary with the search result object and timestamp.    
    """
    
    current_date = datetime.now().isoformat()
    store_date = current_date[:current_date.index('T')].replace('-','')
       
    search = lk.search_lightcurve(ID, exptime=lkwargs['exptime'], 
                                  mission=lkwargs['mission'])

    resultDict = {'result': search,
                  'timestamp': store_date}
    
    fname = os.path.join(*[search_cache, f"{ID}_{lkwargs['exptime']}.lksearchresult"])
    
    pickle.dump(resultDict, open(fname, "wb"))
    
    return resultDict   

def _getMASTidentifier(ID, lkwargs):
    """ return KIC/TIC/EPIC for given ID.
    
    If input ID is not a KIC/TIC/EPIC identifier then the target is looked up
    on MAST and the identifier is retried. If a mission is not specified the 
    set of observations with the most quarters/sectors etc. will be used. 
    
    Parameters
    ----------
    ID : str
        Target ID
    lkwargs : dict
        Dictionary with arguments to be passed to lightkurve. In this case
        mission and exptime.
    
    Returns
    -------
    ID : str
        The KIC/TIC/EPIC ID of the target.    
    """
    
    if not any([x in ID for x in ['KIC', 'TIC', 'EPIC']]):
        
        search = lk.search_lightcurve(ID, exptime=lkwargs['exptime'], mission=lkwargs['mission'])

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

def _perform_search(ID, lkwargs, use_cached, download_dir=None, 
                    cache_expire=30):
    """ Find filenames related to target
    
    Preferentially accesses cached search results, otherwise searches the 
    MAST archive.
    
    Parameters
    ----------
    ID : str
        Target ID (must be KIC, TIC, or ktwo prefixed)
    lkwargs : dict
        Dictionary with arguments to be passed to lightkurve. In this case
        mission and exptime.
    use_cached : bool, optional
        Whether or not to use the cached time series. Default is True.
    download_dir : str, optional.
        Directory for fits file and search results caches. Default is 
        ~/.lightkurve-cache. 
    cache_expire : int, optional.
        Expiration time for the search cache results. Files older than this 
        will be. The default is 30 days.
        
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

    filepath = os.path.join(*[cachepath, f"{ID}_{lkwargs['exptime']}.lksearchresult"])
    
    if os.path.exists(filepath) and use_cached:  
        
        resultDict = pickle.load(open(filepath, "rb"))
        fdate = resultDict['timestamp'] 
        ddate = datetime.now() - datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:]))
        
        # If file is saved more than cache_expire days ago, a new search is performed
        if ddate.days > cache_expire:   
            resultDict = _search_and_dump(ID, lkwargs, cachepath)
            
    else:
        resultDict = _search_and_dump(ID, lkwargs, cachepath)
        
    return resultDict['result']

def _check_lc_cache(search, mission, download_dir=None):
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
    mission : str
        Which mission to download the data from.
    download_dir : str, optional.
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
        fname = os.path.join(*[download_dir, 'mastDownload', mission, row['obs_id'], row['productFilename']])
        if os.path.exists(fname):
            files_in_cache.append(fname)
    
    if len(files_in_cache) != len(search):       
        search.download_all(download_dir = download_dir)
        files_in_cache = [os.path.join(*[download_dir, 'mastDownload', mission, row['obs_id'], row['productFilename']]) for row in search.table]

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

class session():
    """ Main class used to initiate peakbagging.

    Use this class to initialize a star class instance for one or more targets.

    Once initialized, calling the session class instance will execute a complete
    peakbagging run.

    Data can be provided in multiple different ways, the simplest of which is
    just to let PBjam query the MAST server. Otherwise arrays of
    timeseries/power spectra, lightkurve.LightCurve/lightkurve.periodogram,
    or just path names as strings, is also possible.

    The physical parameters, such numax, dnu, teff, bp_rp, must each be provided
    at least as a list of length 2 for each star. This should contain the 
    parameter value and it's error estimate.
        
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
    1. If you have a large directory of cached lightcurves or power spectra, it 
    is best to supply the filenames to the files.
    
    2. Teff and bp_rp provide much of the same information, so providing both is 
    not strictly necessary. However, for best results you should try to provide
    both. If bp_rp is omitted though, PBjam will attempt to find this value in 
    online catalogs.
    
    3. PBjam will not combine time series from different missions.
    
    Examples
    --------
    Peakbagging run for a single target:

    >>> jam_sess = pbjam.session(ID =  '4448777',  numax = [220.0, 3.0],
                                 dnu = [16.97, 0.01], teff = [4750, 100],
                                 bp_rp = [1.34, 0.01], exptime = 1800)
    >>> jam_sess()

    Peakbagging run for multiple targets:
    >>> jam_sess = pbjam.session(dictlike = mydataframe)
    >>> jam_sess()

    Parameters
    ----------
    ID : string, optional
        Target identifier, most commonly used identifiers can be used if you 
        want PBjam to download the data (KIC, TIC, HD, Bayer etc.). If you 
        provide data yourself the name can be any string.
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
        Spectrum input. Leave as None for PBjam to use Timeseries to compute
        it for you. Otherwise, arrays of shape (2,N), lightkurve.periodogram
        objects, or strings for pathnames are accepted.
    dictlike : pandas.DataFrame or dictionary, optional
        DataFrame, dictionary, record array with a list of targets, and their
        properties. If string, PBjam will assume it's a pathname to a csv file.
        Specify timeseries and spectrum columns with file pathnames to use 
        manually reduced data.
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run.
        Warning, if running multiple targets, make sure you have enough memory.
    use_cached : bool, optional
        Flag for using cached data. If fitting the same targets multiple times,
        use to this to not download the data every time.
    cadence : string, deprecated
        This was used to specify the observation cadence for downloading via
        LightKurve. This was changed to exptime in Lightkurve v. 2
    exptime : string, optional
        Exposure time of the observations in seconds. Argument for lightkurve to
        download correct data type. Can be 1800 or Kepler 'long' cadence. 1800 
        is default setting, so if you're looking at main sequence stars, make 
        sure to manually set 60.    
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
                 exptime=None, campaign=None, sector=None, month=None, 
                 quarter=None, mission=None, path=None, download_dir=None):
      
        self.stars = []

        # Case 1: ID or list of IDs
        # - numax, dnu, teff, bp_rp must match shape
        # - Case 1.0 If neither TS or PSD is supplied, download using lk_kwargs
        # - Case 1.1 If TS is also supplised compute PSD
        # - Case 1.2 If PSD is supplied continue
        # Case 2: csvfile path or dataframe is supplied
        # - must contain ID, numax, dnu, teff, bp_rp columns
        # - 1.0 if no TS or PSD column is supplied - download each ID using lk_kwargs columns, if no lk_kwargs columns assume default settings
        # - 1.1 if TS column - must be absolute path strings to point to TS - then compute PSD - then continue
        # - 1.2 if PSD column - must be absolute path strings to point to PSD - then continue
       
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
                                         bp_rp=bp_rp, exptime=exptime,
                                         campaign=campaign, sector=sector,
                                         month=month, quarter=quarter, 
                                         mission=mission)
            
            _format_col(vardf, timeseries, 'timeseries')
            _format_col(vardf, spectrum, 'spectrum')
        else:
            raise TypeError('session.__init__ requires either ID or dictlike')

        if any(np.isnan(vardf['exptime'])) and any((vardf['timeseries'])) and any(vardf['spectrum']):
            warnings.warn('If the target may have been observed at more than one cadence, it is recommended that you set exptime')
 
        for i in vardf.index:
            
            lkwargs = {x: vardf.loc[i, x] for x in ['exptime', 'month', 
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

    def __call__(self,  ):
        """ Sequentially call all the star class instances

        Calling the session class instance will loop through all the stars that it contains, and call each one. 
        
        This performs a full peakbagging run on each star in the session.

        Parameters
        ----------
         
        """
        
         

        for i, st in enumerate(self.stars):
             
            st()
 
            # self.stars[i] = None
            
            # Crude way to send error messages that occur in star up to Session 
            # without ending the session. Is there a better way?
#            except Exception as ex:
#                 message = "Star {0} produced an exception of type {1} occurred. Arguments:\n{2!r}".format(st.ID, type(ex).__name__, ex.args)
#                 print(message)
            



class star(plotting):

    def __init__(self, ID, f, s, addObs, outpath=None, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        self.outpath = IO._set_outpath(ID, self.outpath)

        if priorpath is None:
            self.priorpath = IO.get_priorpath()

    def modeID(self, addPriors={}, N_p=7, N_pca=100, PCAdims=8, **kwargs):
        
        self.modeID = modeID(self.f, self.s, self.addObs, addPriors, N_p=N_p, Npca=N_pca, PCAdims=PCAdims, priorpath=self.priorpath)

        self.modeID()
  
        return self.modeID.result         
     
    def peakbag(self, modeIDDict=None, **kwargs):
 
        if modeIDDict is None:
            modeIDDIct = self.modeID.result
  
        self.peakbag = peakbag(self.f, self.s, modeIDDIct['ell'], modeIDDIct['zeta'], **modeIDDIct['summary'], Nslices=4)

        self.peakbag()

        return self.peakbag.result
 
    def __call__(self, modeID_kwargs={}, peakbag_kwargs={}):
 
        modeIDResult = self.modeID(**modeID_kwargs)
 
        peakbagResult = self.peakbag(self.modeID.result, **peakbag_kwargs)

        return modeIDResult, peakbagResult