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
import numpy as np
import astropy.units as units
import pandas as pd
import os, glob, warnings, psutil

from pbjam.star import star

def organize_sess_dataframe(vardf):
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

    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter']
    doubles = ['teff', 'bp_rp']

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
    """ Takes input and organizes them in a dataframe.

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
    doubles = ['numax', 'dnu', 'teff', 'bp_rp']
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
    """ Search for target on MAST server.

    Get all the lightcurves available for a target id, using options in kwargs
    dictionary. The lightcurves are downloaded using the lightkurve API, and
    the target ID must therefore be parseable by lightkurve.

    Parameters
    ----------
    id : string
        Target id, must be resolvable by Lightkurve.

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


def clean_lc(lc):
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

    lc = lc.remove_nans().normalize().flatten().remove_outliers()
    return lc


def query_lightkurve(id, lkwargs, use_cached):
    """ Check cache for fits file, or download it.

    Based on use_cached flag, will look in the cache for fits file
    corresponding to request id star. If nothing is found in cached it will be
    downloaded from the MAST server.

    Parameters
    ----------
    id : string
        Identifier for the requested star. Must be resolvable by Lightkurve
    lkwargs : dict
        Dictionary containing keywords for the Lightkurve search.
        cadence, quarter, campaign, sector, month.
    use_cached: bool
        Whether or not to used data in the Lightkurve cache.

    Note:
    -----
    Prioritizes long cadence over short cadence unless otherwise specified.

    """
    lk_cache = os.path.join(*[os.path.expanduser('~'),
                              '.lightkurve-cache',
                              'mastDownload/*/'])

    # Remove
    if isinstance(id, str):
        for prefix in ['KIC','EPIC','TIC','kplr','tic']:
            id = id.strip(prefix)

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
        print(f'Querying MAST for {id}')
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

    In both of the latter cases col is converted to Lightkurve object(s).

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Pandas Dataframe instance to which either a timeseries or psd column
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
    it to a Lightkurve.LightCurve object. If string, it's assumed to be a file
    path name, if None, it will query the LightCurve cache locally or MAST if
    nothing is found. Skips column entries which are already LightKurve objects
    or if a psd for the star in question exists in the psd column.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'timeseries' column consisting of None, strings
        or Lightkurve.LightCurve objects.
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
    """ Convert psd column in dataframe to Lightkurve periodgram object list

    Takes whatever is in the psd column of a dataframe and tries to turn it
    into a Lightkurve.periodogram object. If column entry is a string, it
    assumes it's a path name, if None, it will try to take what's in the
    timeseries column and compute periodogram based on that. If a periodogram
    object is already present, it'll just flatten it (to be safe) and then
    continue on.

    Parameters
    ----------
    vardf : pandas.DataFrame instance
        Dataframe containing a 'psd' column consisting of None, strings or
        Lightkurve.periodogram objects.

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
    print(pre, process.memory_info().rss // 1000, 'Kbytes', post)  # in bytes


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

    >>> jam_sess = pbjam.session(ID =  '4448777',  numax = [220.0, 3.0],
                             dnu = [16.97, 0.01], teff = [4750, 100],
                             bp_rp = [1.34, 0.01], cadence = 'short')
    >>> jam_sess()

    Peakbagging run for multiple targets:
    jam_sess = pbjam.session(dictlike = mydataframe)
    jam_sess()

    By default, PBjam will download all the available data, favoring long
    cadence. Cadence and specific observing seasons (quarter, month, campagin,
    sector) can be specified for more detailed control. The data is downloaded
    using the Lightkurve API, and cleaned on the timeseries level by removing
    NaNs and outliers and flattening the lightcurve (again using LightKurve's
    API). Note that the flattening process may impact the appearance of the
    granulation background.

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

    def __init__(self, ID=None, numax=None, dnu=None, teff=None, bp_rp=None,
                 timeseries=None, psd=None, dictlike=None, store_chains=True,
                 nthreads=1, use_cached=False, cadence=None, campaign=None,
                 sector=None, month=None, quarter=None, make_plots=False,
                 path=None, model_type='simple'):

        self.nthreads = nthreads
        self.store_chains = store_chains
        self.stars = []
        self.pb_model_type = model_type

        #print_memusage(pre = 'Session init start')

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

            organize_sess_dataframe(vardf)

        elif ID:
            vardf = organize_sess_input(ID=ID, numax=numax, dnu=dnu, teff=teff,
                                        bp_rp=bp_rp, cadence=cadence,
                                        campaign=campaign, sector=sector,
                                        month=month, quarter=quarter)
            format_col(vardf, timeseries, 'timeseries')
            format_col(vardf, psd, 'psd')

        lc_to_lk(vardf, use_cached=use_cached)
        lk_to_pg(vardf)

        #print_memusage(pre = 'df setup')

        for i in range(len(vardf)):
            #print_memusage(pre = f'Initializing star {i}')

            self.stars.append(star(ID=vardf.loc[i, 'ID'],
                                   periodogram=vardf.loc[i, 'psd'],
                                   numax=vardf.loc[i, ['numax', 'numax_err']].values,
                                   dnu=vardf.loc[i, ['dnu', 'dnu_err']].values,
                                   teff=vardf.loc[i, ['teff', 'teff_err']].values,
                                   bp_rp=vardf.loc[i, ['bp_rp', 'bp_rp_err']].values,
                                   store_chains=store_chains,
                                   nthreads=self.nthreads,
                                   make_plots=make_plots,
                                   path=path))

        for i, st in enumerate(self.stars):
            if st.numax[0] > st.f[-1]:
                warnings.warn("Input numax is greater than Nyquist frequeny for %s" % (st.ID))

    def __call__(self, norders=8):
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

        """

        for i, st in enumerate(self.stars):
            #try:
            st(norders=norders, model_type=self.pb_model_type)
            self.stars[i] = None
            #except:
            #    warnings.warn(f'Failed on star {st.ID}')
