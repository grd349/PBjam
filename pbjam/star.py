import os
from .asy_peakbag import asymptotic_fit
from .priors import kde
from .peakbag import peakbag
from .jar import get_priorpath
from .plotting import plotting
import pandas as pd

class star(plotting):
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

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
        self.pg = pg
        self.f = pg.frequency.value
        self.s = pg.power.value

        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp

        self.make_output_dir(path)

        if prior_file is None:
            self.prior_file = get_priorpath() 
        else:
            self.prior_file = prior_file
    
    def make_output_dir(self, path):
        
        # Check if self has path attribute
        if not hasattr(self, 'path'):
            self.path = None
            
        if isinstance(self.path, str):  # If yes, do nothing
            pass
        else:  # If not, assume path should be cwd+ID
            self.path = os.path.join(*[os.getcwd(), f'{self.ID}'])
        
        # If path is str, presume user wants to override self.path
        if isinstance(path, str):
            self.path =os.path.join(*[path, f'{self.ID}'])

        # Check if self.path exists, if not try to create it
        if os.path.isdir(self.path) is None:
            try:
                os.makedirs(self.path)
            except Exception as ex:
                message = "Star {0} produced an exception of type {1} occurred. Arguments:\n{2!r}".format(self.ID, type(ex).__name__, ex.args)
                print(message)
          
    def run_kde(self, bw_fac=1.0, make_plots=False):
        """
        Runs the kde code and makes plots if self.make_plots is set.
        """
        print('Starting KDE estimation')
        # Init
        kde(self, bw_fac=bw_fac)
        
        # Call
        self.kde(dnu=self.dnu, numax=self.numax, teff=self.teff, 
                 bp_rp=self.bp_rp)
        
        # Store
        if make_plots:
            self.kde.plot_corner(path=self.path, ID=self.ID, 
                                 savefig=make_plots)
            self.kde.plot_spectrum(pg=self.pg, path=self.path, ID=self.ID, 
                                       savefig=make_plots)
             
            
    def run_asy_peakbag(self, norders=None, make_plots=False, 
                        store_chains=False, nthreads=1):
        """
        Runs the asy_peakbag code.
        """
        print('Starting Asy_peakbag')
        # Init
        asymptotic_fit(self, self.kde, norders=norders, 
                       store_chains=store_chains, nthreads=nthreads)
        
        # Call
        self.asy_fit(dnu=self.dnu, numax=self.numax, teff=self.teff, 
                     bp_rp=self.bp_rp)
        
        # Store
        outpath = lambda x: os.path.join(*[self.path, x])
        self.asy_fit.summary.to_csv(outpath(f'asy_fit_summary_{self.ID}.csv'),
                                    index=True)
        self.asy_fit.modeID.to_csv(outpath(f'asy_fit_modeID_{self.ID}.csv'),
                                   index=False)
        
        if make_plots:
            self.asy_fit.plot_spectrum(path=self.path, ID=self.ID, 
                                       savefig=make_plots)
            self.asy_fit.plot_corner(path=self.path, ID=self.ID, 
                                       savefig=make_plots)
        
        if store_chains:
            pd.DataFrame(self.asy_fit.samples, columns=self.asy_fit.par_names).to_csv(outpath(f'asy_peakbag_chains_{self.ID}.csv'), index=False) 
        
            

    def run_peakbag(self, model_type='simple', tune=1500, nthreads=1, make_plots=False, store_chains=False):
        """
        Runs peakbag on the given star.
        """
        
        print('Starting peakbagging run')
        # Init
        self.peakbag = peakbag(self, self.asy_fit)
        
        # Call
        self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)
        
        # Store
        outpath = lambda x: os.path.join(*[self.path, x])
        self.peakbag.summary.to_csv(outpath(f'peakbag_summary_{self.ID}.csv'))
        
        if store_chains:
            pass # TODO need to pickle the samples if requested.
        if make_plots:
            self.peakbag.plot_spectrum(path=self.path, ID=self.ID, 
                                       savefig=make_plots)


    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500, 
                 verbose=False, make_plots=True, store_chains=True, nthreads=1):
        """ Instead of a _call_ we should just make this a function maybe? Whats wrong with __call__?"""
        
        self.run_kde(bw_fac=bw_fac, make_plots=make_plots)          
   
        self.run_asy_peakbag(norders=norders, make_plots=make_plots, 
                             store_chains=store_chains, nthreads=nthreads)
        
        self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads, 
                         make_plots=make_plots, store_chains=store_chains)
