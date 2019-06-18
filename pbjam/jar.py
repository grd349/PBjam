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
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os, glob, warnings

from . import PACKAGEDIR


def organize_sess_dataframe(H):
    """ Takes input dataframe and organizes

    Checks to see if required columns are present in the input dataframe,
    and adds optional columns if they don't exists, containing None values.

    Parameters
    ----------
    H : Pandas.DataFrame
        Input dataframe
    """
    keys = ['ID', 'numax', 'dnu', 'numax_error', 'dnu_error']
    if not any(x not in keys for x in H.keys()):
        raise(KeyError, 'Some of the required keywords were missing.')

    N = len(H)

    doubles = ['epsilon', 'teff', 'bp_rp']
    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter']

    for key in singles:
        if key not in H.keys():
            H[key] = np.array([None]*N)

    for key in doubles:
        if key not in H.keys():
            H[key] = np.array([None]*N)
            H[key+'_error'] = np.array([None]*N)

    if 'timeseries' not in H.keys():
        format_col(H, None, 'timeseries')
    if 'psd' not in H.keys():
        format_col(H, None, 'psd')


def organize_sess_input(**X):
    """ Takes input and organizes them in a dataframe

    Checks to see if required inputs are present and inserts them into a
    dataframe. Any optional columns that are not included in the input are
    added as None columns.

    Parameters
    ----------
    X : objects
        Variable inputs to Session class to be arranged into a dataframe

    Returns
    -------
    H : Pandas.DataFrame
        Dataframe containing the inputs from Session class call.

    """
    H = pd.DataFrame({'ID': np.array(X['ID']).reshape((-1, 1)).flatten()})

    N = len(H)
    doubles = ['numax', 'dnu', 'epsilon', 'teff', 'bp_rp']
    singles = ['cadence', 'campaign', 'sector', 'month', 'quarter']

    for key in singles:
        if not X[key]:
            H[key] = np.array([None]*N)
        else:
            H[key] = X[key]

    for key in doubles:
        if not X[key]:
            H[key] = np.array([None]*N)
            H[key+'_error'] = np.array([None]*N)
        else:
            H[key] = np.array(X[key]).reshape((-1, 2))[:, 0].flatten()
            H[key+'_error'] = np.array(X[key]).reshape((-1, 2))[:, 1].flatten()
    return H


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
        LightKurve object to be modified

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
    if typ == 'timeseries':
        return lk.LightCurve(time=x, flux=y, targetid=name)
    elif typ == 'psd':
        return lk.periodogram.Periodogram(x*units.microhertz,
                                          units.Quantity(y, None),
                                          targetid=name)
    else:
        raise KeyError("Don't modify anything but psd and timeseries cols")


def format_col(H, col, key):
    """ Add timeseries or psd column to dataframe based on input

    Based on the contents of col, will try to format col and add it as a column
    to H with column name key. col can be many things, so the decision is based
    mainly on the dimensionality of col. If dim = 0, it's assumed that col is
    either None, or a string,  (for the latter it assumes there is then only
    one target). If dim = 1, it's assumed that col is a list-like object,
    consisting of either None or strings, these are passed along without
    modification. If dim = 2, col is assumed to be either a time series or
    power spectrum of shape (2,M), with time/frequency in 1st row and
    flux/power in the second. If dim = 3, it is assumed to be list of (2,M)
    arrays. In both of the latter cases col is converted to
    LightKurve object(s).

    Parameters
    ----------
    H : pandas.DataFrame instance
        Pandas Dataframe instance to which either a timeseries or psd column
        will be added.
    col : object
        Input from Session call, corresponding to key
    key : name of column to add to H

    """
    N = np.shape(H['ID'])[0]

    col = np.array(col, dtype=object)

    # If dim = 0, it's either none or a string
    if col.ndim == 0:
        if not col:
            # If none, then multiply up to length of ID
            H[key] = np.array([None]*N)
        else:
            # If string, then single target
            H[key] = np.array(col).reshape((-1, 1)).flatten()

    # If dim = 1, it's either a list of nones, strings or lightkurve objects
    elif col.ndim == 1:
        H[key] = col

    # if dim = 2, it's an array or tuple, with time and flux
    elif col.ndim == 2:
        x = np.array(col[0, :], dtype=float)
        y = np.array(col[1, :], dtype=float)
        H[key] = np.array([arr_to_lk(x, y, H['ID'][0])], key)

    # If dim = 3, it's a list of arrays or tuples
    elif col.ndim == 3:
        temp = np.array([], dtype=object)
        for i in range(N):
            x = np.array(col[i, 0, :], dtype=float)
            y = np.array(col[i, 1, :], dtype=float)
            temp = np.append(temp,
                             np.array([arr_to_lk(x, y, H.loc[i, 'ID'], key)]))
        H[key] = temp
    else:
        print('Unhandled exception')


def lc_to_lk(H, use_cached=True):
    """ Convert time series column in dataframe to lk.LightCurve object

    Goes through the timeseries column in the dataframe and tries to convert
    it to a LightKurve.LightCurve object. If string, it's assumed to be a file
    path name, if None, it will query the LightCurve cache locally or MAST if
    nothing is found. Skips column entries which are already LightKurve objects
    or if a psd for the star in question exists in the psd column.

    Parameters
    ----------
    H : pandas.DataFrame instance
        Dataframe containing a 'timeseries' column consisting of None, strings
        or LightKurver.LightCurve objects.
    Returns
    -------

    """

    tinyoffset = 1e-20  # to avoid cases LC median = 0 (lk doesn't like it)
    key = 'timeseries'
    for i, id in enumerate(H['ID']):

        if isinstance(H.loc[i, key], str):
            t, d = np.genfromtxt(H.loc[i, key], usecols=(0, 1)).T
            d += tinyoffset
            H.loc[i, key] = arr_to_lk(t, d, H.loc[i, 'ID'], key)
        elif not H.loc[i, key]:
            if H.loc[i, 'psd']:
                pass
            else:
                D = {x: H.loc[i, x] for x in ['cadence', 'month', 'sector',
                                              'campaign', 'quarter']}
                lk_lc = query_lightkurve(id, D, use_cached)
                H.loc[i, key] = lk_lc
        elif H.loc[i, key].__module__ == lk.lightcurve.__name__:
            pass
        else:
            raise TypeError("Can't handle this type of time series object")

        if H.loc[i, key]:
            sort_lc(H.loc[i, key])


def lk_to_pg(H):
    """ Convert psd column in dataframe to lk periodgram object list

    Takes whatever is in the psd column of a dataframe and tries to turn it
    into a LightKurve.periodogram object. If column entry is a string, it
    assumes it's a path name, if None, it will try to take what's in the
    timeseries column and compute periodogram based on that. If a periodogram
    obect is already present, it'll just flatten it (to be safe) and then
    continue on.

    Parameters
    ----------
    H : pandas.DataFrame instance
        Dataframe containing a 'psd' column consisting of None, strings or
        LightKurver.periodogram objects.

    """

    key = 'psd'
    for i, id in enumerate(H['ID']):
        if isinstance(H.loc[i, key], str):
            f, s = np.genfromtxt(H.loc[i, key], usecols=(0, 1)).T
            H.loc[i, key] = arr_to_lk(f, s, H.loc[i, 'ID'], key)
        elif not H.loc[i, key]:
            lk_lc = H.loc[i, 'timeseries']
            H.loc[i, key] = lk_lc.to_periodogram(freq_unit=units.microHertz, normalization='psd').flatten()

        elif H.loc[i, key].__module__ == lk.periodogram.__name__:
            H.loc[i, key] = H.loc[i, key].flatten()
        else:
            raise TypeError("Can't handle this type of time series object")


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

    def __init__(self, ID=None, numax=None, dnu=None,
                 teff=None, bp_rp=None, epsilon=None,
                 timeseries=None, psd=None, dictlike=None, store_chains=False,
                 nthreads=1, use_cached=False, cadence=None, campaign=None,
                 sector=None, month=None, quarter=None,  kwargs={}):

        self.nthreads = nthreads
        self.store_chains = store_chains
        self.stars = []

        if isinstance(dictlike, (dict, np.recarray, pd.DataFrame)):
            try:
                DF = pd.DataFrame.from_records(dictlike)
            except TypeError:
                print('Unrecognized type in dictlike. Must be able to convert to dataframe through pandas.DataFrame.from_records()')

            if any([ID, numax, dnu, teff, bp_rp, epsilon]):
                warnings.warn('Dictlike provided as input, ignoring other input fit parameters.')

            organize_sess_dataframe(DF)

        elif ID and numax and dnu:
            DF = organize_sess_input(ID=ID, numax=numax, dnu=dnu, teff=teff,
                                     bp_rp=bp_rp, epsilon=epsilon,
                                     cadence=cadence, campaign=campaign,
                                     sector=sector, month=month,
                                     quarter=quarter)
            format_col(DF, timeseries, 'timeseries')
            format_col(DF, psd, 'psd')

        lc_to_lk(DF, use_cached=use_cached)
        lk_to_pg(DF)

        for i in range(len(DF)):
            self.stars.append(star(ID=DF.loc[i, 'ID'],
                                   f=np.array(DF.loc[i, 'psd'].frequency),
                                   s=np.array(DF.loc[i, 'psd'].power),
                                   numax=DF.loc[i,['numax', 'numax_error']].values,
                                   dnu=DF.loc[i,['dnu', 'dnu_error']].values,
                                   teff=DF.loc[i,['teff', 'teff_error']].values,
                                   bp_rp=DF.loc[i,['bp_rp', 'bp_rp_error']].values,
                                   epsilon=DF.loc[i,['epsilon', 'epsilon_error']].values,
                                   store_chains=self.store_chains,
                                   nthreads=self.nthreads))

        for i, st in enumerate(self.stars):
            if st.numax[0] > st.f[-1]:
                warnings.warn("Input numax is greater than Nyquist frequeny for %s" % (st.ID))


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
    numax : list of floats, length 2
        Initial guess for the frequency of maximum power (muHz), numax, and its
    dnu : list of floats, length 2
        Initial guess for the large freqeuncy separation (muHz), dnu, and its
        error.
    teff : list of floats, length 2, optional
        Initial guess for the effective surface temperature (K), Teff, and its
        error.
    bp_rp : list of floats, length 2, optional
        Initial guess for the Gaia Gbp-Grp color, and its error. Errors on
        Gbp_Grp are not available in the Gaia archive. A coservative suggestion
        is approximately 0.05-0.1.
    epsilon : list of floats, length 2, optional
        Initial guess for the scaling relation phase term, epsilon, and its
        error.
    source : str, optional
        Pathname of the file used to make the star class instance (timeseries
        or psd). If data is downloaded via Lightkurve the fits file name is
        used.
    store_chains : bool
        Flag for storing all the chains of the MCMC walkers. Warning: if you
        are running many stars in a single session this may cause memory
        issues.
    nthreads : int
        Number of multi-processing threads to use when doing the MCMC. Best to
        leave this at 1 for long cadence targets.

    Attributes
    ----------
    data_file : str
        Filepath to file containing prior data on the fit parameters, either
        from literature or previous fits.
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

    def asymptotic_modeid(self, d02=None, alpha=None, mode_width=None,
                          env_width=None, env_height=None, norders=8):
        """ Perform mode ID using the asymptotic method.

        Calls the asymptotic_fit method from asy_peakbag and does an MCMC fit
        of the asymptotic relation for l=2,0 pairs to the spectrum, with a
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
        """

        fit = asymptotic_fit(self, d02, alpha, mode_width, env_width,
                             env_height, store_chains=self.store_chains,
                             nthreads=self.nthreads, nrads=norders)
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
        ax.set_xlim(min(self.f[sel]), max(self.f[sel]))
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
        ax.scatter(prior['dnu'], prior['Teff'], c='k', s=2, alpha=0.2)
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

        return fig

    def corner(self):
        import corner

        xs = self.asy_result.flatchain
        labels = self.asy_result.pars_names

        return corner.corner(xs = xs, labels = labels, plot_density = False)
