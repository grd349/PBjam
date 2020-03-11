import os
from .asy_peakbag import asymptotic_fit
from .priors import kde
from .peakbag import peakbag
from .jar import get_priorpath, to_log10
from .plotting import plotting
import pandas as pd

class star(plotting):
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Note spectrum is flattened (background divided out.)

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

    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run.
        Warning, if running multiple targets, make sure you have enough memory.

    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead.
        Untested on short cadence.

    make_plots : bool, optional
        If True, will save figures when calling methods in `star`.

    path : str, optional
        The path at which to store output. If no path is set but make_plots is
        True, output will be saved in the current working directory.

    verbose : bool, optional
        If True, will show error messages on the users terminal if they occur.

    Attributes
    ----------
    f : array
        Array of power spectrum frequencies
    s : array
        power spectrum
    data_file : str
        Path to the csv file containing the prior data

    """

    def __init__(self, ID, pg, numax, dnu, teff, bp_rp, path=None,
                 prior_file = None):

        self.ID = ID
        self.pg = pg.flatten() # in case user supplies unormalized spectrum
        self.f = self.pg.frequency.value
        self.s = self.pg.power.value

        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp
        self._obs = {'dnu': self.dnu, 'numax': self.numax, 'teff': self.teff, 'bp_rp': self.bp_rp}
        self._log_obs = {x: to_log10(*self._obs[x]) for x in self._obs.keys() if x != 'bp_rp'}

        self._set_path(path)
        self._make_output_dir()

        if prior_file is None:
            self.prior_file = get_priorpath()
        else:
            self.prior_file = prior_file

    def _set_path(self, path):
        """ Sets the path attribute for star
        
        If path is a string it is assumed to be a path name, if not the 
        current working directory will be used. 
        
        Parameters
        ----------
        path : str
            Directory to store peakbagging output.
        
        """

        if isinstance(path, str):
            # If path is str, presume user wants to override self.path
            self.path = os.path.join(*[path, f'{self.ID}'])
        else:
            self.path = os.path.join(*[os.getcwd(), f'{self.ID}'])
            
    def _make_output_dir(self):
        """ Make output directory for star

        Attempts to create an output directory for all the results that PBjam
        produces. A directory is created when a star class instance is
        initialized, so a session might create multiple directories.

        """

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

        Also generates plots of the results and stores them in the star
        directory.

        Parameters
        ----------
        bw_fac : float
            Scaling factor for the KDE bandwidth. The bandwidth is
            automatically, but may be scaled to adjust for, .e.g, sparsity of
            the prior sample.
        make_plots : bool
            Whether or not to produce plots of the results.

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

    def run_asy_peakbag(self, norders=None, make_plots=False,
                        store_chains=False):
        """ Run all stesps involving asy_peakbag.

        Performs a fit of the asymptotic relation to the spectrum (l=2,0 only),
        and outputs result plots and a summary of the fit results.

        Parameters
        ----------
        norders : int
            Number of orders to include in the fits
        make_plots : bool
            Whether or not to produce plots of the results.
        store_chains : bool
            Whether or not to store MCMC chains on disk.

        """

        print('Starting asymptotic peakbagging')
        # Init
        asymptotic_fit(self, norders=norders)

        # Call
        self.asy_fit()

        # Store
        outpath = lambda x: os.path.join(*[self.path, x])
        self.asy_fit.summary.to_csv(outpath(f'asy_fit_summary_{self.ID}.csv'),
                                    index=True, index_label='name')
        self.asy_fit.modeID.to_csv(outpath(f'asy_fit_modeID_{self.ID}.csv'),
                                   index=False)

        if make_plots:
            self.asy_fit.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.asy_fit.plot_corner(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.asy_fit.plot_echelle(path=self.path, ID=self.ID, 
                                      savefig=make_plots)

        if store_chains:
            pd.DataFrame(self.asy_fit.samples, columns=self.asy_fit.par_names).to_csv(outpath(f'asy_peakbag_chains_{self.ID}.csv'), index=False)



    def run_peakbag(self, model_type='simple', tune=1500, nthreads=1,
                    make_plots=False, store_chains=False):
        """  Run all stesps involving peakbag.

        Performs fit using simple lorentzian pairs two subsections of the
        power spectrum based on results from asy_peakbag.

        Parameters
        ----------
        model_type : str
            Defaults to 'simple'.
            Can be either 'simple' or 'model_gp' which sets the type of model
            to be fitted to the data.
        tune : int
            Numer of tuning steps passed to pm.sample
        make_plots : bool
            Whether or not to produce plots of the results.
        store_chains : bool
            Whether or not to store MCMC chains on disk.
        nthreads : int
            Number of processes to spin up in pymc3

        """

        print('Starting peakbagging')
        # Init
        peakbag(self, self.asy_fit)

        # Call
        self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)

        # Store
        outpath = lambda x: os.path.join(*[self.path, x])
        self.peakbag.summary.to_csv(outpath(f'peakbag_summary_{self.ID}.csv'),
                                    index_label='name')

        if store_chains:
            pass # TODO need to pickle the samples if requested.
        if make_plots:
            self.peakbag.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.peakbag.plot_echelle(path=self.path, ID=self.ID, 
                                      savefig=make_plots)


    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500,
                 verbose=False, make_plots=True, store_chains=True, nthreads=1):
        """ Perform all the PBjam steps


        Parameters
        ----------
        bw_fac : float
            Scaling factor for the KDE bandwidth. The bandwidth is
            automatically, but may be scaled to adjust for, .e.g, sparsity of
            the prior sample.
        norders : int
            Number of orders to include in the fits
        model_type : str
            Defaults to 'simple'. Can be either 'simple' or 'model_gp' which
            sets the type of model to be fit for the mode linewidths.
        tune : int
            Numer of tuning steps passed to pm.sample
        verbose : bool
            Should I say anything?
        make_plots : bool
            Whether or not to produce plots of the results.
        store_chains : bool
            Whether or not to store MCMC chains on disk.

        """

        self.run_kde(bw_fac=bw_fac, make_plots=make_plots)

        self.run_asy_peakbag(norders=norders, make_plots=make_plots,
                             store_chains=store_chains)

        self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads,
                         make_plots=make_plots, store_chains=store_chains)
