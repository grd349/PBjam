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

import os, warnings
from .asy_peakbag import asymptotic_fit
from .priors import kde
from .peakbag import peakbag
from .jar import get_priorpath, to_log10
from .plotting import plotting
import pandas as pd
import numpy as np

class star(plotting):
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Note spectrum is flattened (background divided out.)

    Examples
    --------
    Peakbag using the star class. Note that the star class only takes Lightkurve
    periodograms, pg, as spectrum input. 

    >>> st = pbjam.star(ID='KIC4448777', pg=pg, numax=[220.0, 3.0], 
                           dnu=[16.97, 0.01], teff=[4750, 100],
                           bp_rp = [1.34, 0.01])
    >>> st(make_plots=True)

    Parameters
    ----------
    ID : string, int
        Target identifier. If custom timeseries/periodogram is provided, it
        must be resolvable by LightKurve (KIC, TIC, EPIC, HD, etc.).
    pg : lightkurve.periodogram.Periodogram object
        A lightkurve periodogram object containing frequencies in units of
        microhertz and power (in arbitrary units).
    numax : list
        List of the form [numax, numax_error]. For multiple targets, use a list
        of lists.
    dnu : list
        List of the form [dnu, dnu_error]. For multiple targets, use a list
        of lists.
    teff : list
        List of the form [teff, teff_error]. For multiple targets, use a list
        of lists.
    bp_rp : list
        List of the form [bp_rp, bp_rp_error]. For multiple targets, use a list
        of lists.
    path : str, optional
        The path at which to store output. If no path is set but make_plots is
        True, output will be saved in the current working directory. Default is
        the current working directory.
    prior_file : str, optional
        Path to the csv file containing the prior data. Default is
        pbjam/data/prior_data.csv

    Attributes
    ----------
    f : array
        Array of power spectrum frequencies
    s : array
        power spectrum

    """

    def __init__(self, ID, pg, numax, dnu, teff=[None,None], bp_rp=[None,None], 
                 path=None, prior_file=None):

        self.ID = ID
        self.pg = pg.flatten()  # in case user supplies unormalized spectrum

        if numax[0] < 25:
            warnings.warn('The input numax is less than 25. The prior is not well defined here, so be careful with the result.')
        self.numax = numax
        self.dnu = dnu

        teff, bp_rp = self._checkTeffBpRp(teff, bp_rp)
        self.teff = teff
        self.bp_rp = bp_rp

        self.f = self.pg.frequency.value
        self.s = self.pg.power.value

        self._obs = {'dnu': self.dnu, 'numax': self.numax, 'teff': self.teff,
                     'bp_rp': self.bp_rp}

        self._log_obs = {x: to_log10(*self._obs[x]) for x in self._obs.keys() if x != 'bp_rp'}

        self._set_outpath(path)

        if prior_file is None:
            self.prior_file = get_priorpath()
        else:
            self.prior_file = prior_file

    def _checkTeffBpRp(self, teff, bp_rp):
        # Teff and Gbp-Grp provide a lot of the same information, so only one of
        # them need to be provided to start with. If one is not provided, PBjam
        # will assume a wide prior on it.
        teff_bad = np.all(np.array(teff) == [None,None]) or np.isnan(teff[0])
        bp_rp_bad = np.all(np.array(bp_rp) == [None,None]) or np.isnan(bp_rp[0])
        
        if teff_bad and bp_rp_bad:
            raise ValueError('Must provide either teff or bp_rp arguments when initializing the star class.')
        elif teff_bad :
            teff = [4889, 1500] # these are rough esimates from the prior
        elif bp_rp_bad:
            bp_rp = [1.2927, 0.5] # these are rough esimates from the prior

        return teff, bp_rp

    def _outpath(self, x):
        """ Shorthand for setting the full output path

        TODO: optionally could be generallized to all of pbjam and do more 
        advanced checks to see if the dir exists etc.

        Parameters
        ----------
        x : str
            Base filename

        Returns
        -------
        outpath : str
            Full output pathname
        """

        return os.path.join(*[self.path, x])

    def _set_outpath(self, path):
        """ Sets the path attribute for star

        If path is a string it is assumed to be a path name, if not the
        current working directory will be used.

        Attempts to create an output directory for all the results that PBjam
        produces. A directory is created when a star class instance is
        initialized, so a session might create multiple directories.

        Parameters
        ----------
        path : str
            Directory to place the star subdirectory.

        """

        if isinstance(path, str):
            # If path is str, presume user wants to override self.path
            self.path = os.path.join(*[path, f'{self.ID}'])
        else:
            self.path = os.path.join(*[os.getcwd(), f'{self.ID}'])

        # Check if self.path exists, if not try to create it
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except Exception as ex:
                message = "Star {0} produced an exception of type {1} occurred. Arguments:\n{2!r}".format(self.ID, type(ex).__name__, ex.args)
                print(message)

    def run_kde(self, bw_fac=1.0, make_plots=False):
        """ Run all steps involving KDE.

        Starts by creating a KDE based on the prior data sample. Then samples
        this KDE for initial starting positions for asy_peakbag.

        Parameters
        ----------
        bw_fac : float
            Scaling factor for the KDE bandwidth. By default the bandwidth is
            automatically set, but may be scaled to adjust for sparsity of the
            prior sample.
        make_plots : bool, optional
            Whether or not to produce plots of the results. Default is False.

        """

        print('Starting KDE estimation')
        # Init
        kde(self)

        # Call
        self.kde(dnu=self.dnu, numax=self.numax, teff=self.teff,
                 bp_rp=self.bp_rp, bw_fac=bw_fac)

        # Store
        if make_plots:
            self.kde.plot_corner(path=self.path, ID=self.ID,
                                 savefig=make_plots)
            self.kde.plot_spectrum(pg=self.pg, path=self.path, ID=self.ID,
                                   savefig=make_plots)
            self.kde.plot_echelle(path=self.path, ID=self.ID,
                                  savefig=make_plots)

    def run_asy_peakbag(self, norders, make_plots=False,
                        store_chains=False, method='mcmc', 
                        developer_mode=False):
        """ Run all steps involving asy_peakbag.

        Performs a fit of the asymptotic relation to the spectrum (l=2,0 only),
        using initial guesses and prior for the fit parameters from KDE.

        Parameters
        ----------
        norders : int
            Number of orders to include in the fits.
        make_plots : bool, optional
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional
            Whether or not to store MCMC chains on disk. Default is False.
        method : string
            Method to be used for sampling the posterior. Options are 'mcmc' or
            'nested. Default method is 'mcmc' that will call emcee, alternative
            is 'nested' to call nested sampling with CPnest.
        developer_mode : bool
            Run asy_peakbag in developer mode. Currently just retains the input 
            value of dnu and numax as priors, for the purposes of expanding
            the prior sample. Important: This is not good practice for getting 
            science results!
            
        """

        print('Starting asymptotic peakbagging')
        # Init
        asymptotic_fit(self, norders=norders)

        # Call
        self.asy_fit(method, developer_mode)

        # Store
        self.asy_fit.summary.to_csv(self._outpath(f'asymptotic_fit_summary_{self.ID}.csv'),
                                    index=True, index_label='name')
        self.asy_fit.modeID.to_csv(self._outpath(f'asymptotic_fit_modeID_{self.ID}.csv'),
                                   index=False)
        if make_plots:
            self.asy_fit.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.asy_fit.plot_corner(path=self.path, ID=self.ID,
                                     savefig=make_plots)
            self.asy_fit.plot_echelle(path=self.path, ID=self.ID,
                                      savefig=make_plots)

        if store_chains:
            pd.DataFrame(self.asy_fit.samples, columns=self.asy_fit.par_names).to_csv(self._outpath(f'asymptotic_fit_chains_{self.ID}.csv'), index=False)

    def run_peakbag(self, model_type='simple', tune=1500, nthreads=1,
                    make_plots=False, store_chains=False):
        """  Run all steps involving peakbag.

        Performs fit using simple Lorentzian profile pairs to subsections of 
        the power spectrum, based on results from asy_peakbag.

        Parameters
        ----------
        model_type : str
            Can be either 'simple' or 'model_gp' which sets the type of mode
            width model. Defaults is 'simple'.
        tune : int, optional
            Numer of tuning steps passed to pm.sample. Default is 1500.
        nthreads : int, optional.
            Number of processes to spin up in pymc3. Default is 1.
        make_plots : bool, optional.
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional.
            Whether or not to store MCMC chains on disk. Default is False.

        """

        print('Starting peakbagging')
        # Init
        peakbag(self, self.asy_fit)

        # Call
        self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)

        # Store
        self.peakbag.summary.to_csv(self._outpath(f'peakbag_summary_{self.ID}.csv'),
                                    index_label='name')

        if store_chains:
            pass  # TODO need to pickle the samples if requested.
        if make_plots:
            self.peakbag.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.peakbag.plot_echelle(path=self.path, ID=self.ID, 
                                      savefig=make_plots)


    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500,
                 nthreads=1, make_plots=True, store_chains=True, 
                 asy_sampling='mcmc', developer_mode=False):
        """ Perform all the PBjam steps

        Starts by running KDE, followed by Asy_peakbag and then finally peakbag.
        
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
        make_plots : bool, optional.
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional.
            Whether or not to store MCMC chains on disk. Default is False.
        asy_sampling : string
            Method to be used for sampling the posterior in asy_peakbag. Options
            are 'mcmc' or 'nested. Default method is 'mcmc' that will call 
            emcee, alternative is 'nested' to call nested sampling with CPnest.
        developer_mode : bool
            Run asy_peakbag in developer mode. Currently just retains the input 
            value of dnu and numax as priors, for the purposes of expanding
            the prior sample. Important: This is not good practice for getting 
            science results!    
        """

        self.run_kde(bw_fac=bw_fac, make_plots=make_plots)

        self.run_asy_peakbag(norders=norders, make_plots=make_plots,
                             store_chains=store_chains, method=asy_sampling,
                             developer_mode=developer_mode)

        self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads,
                         make_plots=make_plots, store_chains=store_chains)
