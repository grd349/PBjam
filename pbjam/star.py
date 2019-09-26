import os, warnings, sys
from pbjam.asy_peakbag import asymptotic_fit
from pbjam.guess_epsilon import epsilon
from pbjam.peakbag import peakbag
from . import PACKAGEDIR
import pymc3 as pm
from .plotting import plot_corner, plot_spectrum


class star():
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Parameters
    ----------
    ID : string, int
        Target identifier. If custom timeseries/periodogram is provided, it
        must be resolvable by LightKurve (KIC, TIC, EPIC, HD, etc.).

    periodogram : lightkurve.periodogram.Periodogram object
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

    def __init__(self, ID, periodogram, numax, dnu, teff, bp_rp, path=None, 
                 prior_file = None):
          
        self.ID = ID
        self.pg = periodogram
        self.f = periodogram.frequency.value
        self.s = periodogram.power.value

        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp

        self.path = path
        
        if not prior_file:
            self.prior_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])
        else:
            self.prior_file = prior_file
        
        self.plot_corner = plot_corner
        self.plot_spectrum = plot_spectrum


    def make_output_dir(self, path, verbose):
        if path == None:
            path = os.getcwd()
        self.path = os.path.join(*[path, f'{self.ID}'])

        try:
            os.mkdir(self.path)
        except OSError:
            if verbose:
                warnings.warn(f'Path {self.path} already exists - I will try to overwrite ... ')
          
          
    def run_epsilon(self, bw_fac=1.0, make_plots=False):
        """
        Runs the epsilon code and makes plots if self.make_plots is set.
        """
        epsilon(self, bw_fac=bw_fac)
        
        self.epsilon(dnu=self.dnu, numax=self.numax, teff=self.teff, 
                     bp_rp=self.bp_rp)
        
        if make_plots:
            self.plot_corner(self, self.epsilon, make_plots)
            self.plot_spectrum(self, self.epsilon, make_plots)
             
            
    def run_asy_peakbag(self, norders=None, make_plots=False, 
                        store_chains=False, nthreads=1):
        """
        Runs the asy_peakbag code.
        """
        
        asymptotic_fit(self, self.epsilon, norders=norders, 
                       store_chains=store_chains, nthreads=nthreads)
        
        self.asy_fit(dnu=self.dnu, numax=self.numax, teff=self.teff, 
                     bp_rp=self.bp_rp)
        
        outpath = lambda x: os.path.join(*[self.path, x])
        self.asy_fit.summary.to_csv(outpath(f'{type(self).__name__}_summary_{self.ID}.csv'),
                                    index=True)
        self.asy_fit.modeID.to_csv(outpath(f'{type(self).__name__}_modeID_{self.ID}.csv'),
                                   index=False)

        if store_chains:
            pass # TODO need to pickle the chains if requested.
        if make_plots:
            self.plot_spectrum(self, self.asy_fit, make_plots)#.savefig(outpath + f'_{self.ID}.png')
            self.plot_corner(self, self.asy_fit, make_plots)#.savefig(outpath + f'_corner_{self.ID}.png')
            

    def run_peakbag(self, model_type='simple', tune=1500, nthreads=1, make_plots=False, store_chains=False):
        """
        Runs peakbag on the given star.
        """
        self.peakbag = peakbag(self, self.asy_fit)
        self.peakbag.sample(model_type=model_type, tune=tune, nthreads=nthreads)
        
        outpath = lambda x: os.path.join(*[self.path, x])
        pm.summary(self.peakbag.samples).to_csv(outpath(f'{type(self).__name__}_summary_{self.ID}.csv'))
        if store_chains:
            pass # TODO need to pickle the samples if requested.
        if make_plots:
            self.plot_spectrum(self, self.peakbag, make_plots)
            #self.peakbag.plot_flat_fit().savefig(outpath + f'_{self.ID}.png')


    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500, 
                 verbose=False, make_plots=True, store_chains=True, nthreads=1):
        """ Instead of a _call_ we should just make this a function maybe? Whats wrong with __call__?"""
        
        self.make_output_dir(self.path, verbose) 
        print('Running epsilon')
        self.run_epsilon(bw_fac=bw_fac, make_plots=make_plots)          
   
        print('epsilon complete')
        self.run_asy_peakbag(norders=norders, make_plots=make_plots, 
                             store_chains=store_chains, nthreads=nthreads)
        
        print('asy complete')
        self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads, make_plots=make_plots, store_chains=store_chains)
