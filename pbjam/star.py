import os, warnings
from pbjam.asy_peakbag import asymptotic_fit
from pbjam.guess_epsilon import epsilon
from pbjam.peakbag import peakbag
from . import PACKAGEDIR
import pymc3 as pm


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

    def __init__(self, ID, periodogram, numax, dnu, teff, bp_rp, nthreads=1, 
                 path=None, store_chains=True, make_plots=False, 
                 verbose=False):
          
        self.ID = ID
        self.pg = periodogram
        self.f = periodogram.frequency.value
        self.s = periodogram.power.value

        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.bp_rp = bp_rp

        self.nthreads = nthreads
        self.store_chains = store_chains
        self.make_plots = make_plots

        self.data_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])

        self.make_output_dir(path, verbose)


    def make_output_dir(self, path, verbose):
        if path == None:
            path = os.getcwd()
        self.path = os.path.join(*[path, f'{self.ID}'])

        try:
            os.mkdir(self.path)
        except OSError:
            if verbose:
                warnings.warn(f'Path {self.path} already exists - I will try to overwrite ... ')
          
          
    def run_epsilon(self, bw_fac=1.0):
        """
        Runs the epsilon code and makes plots if self.make_plots is set.
        """
        self.epsilon = epsilon(bw_fac=bw_fac)
        self.epsilon_result = self.epsilon(dnu=self.dnu,
                                           numax=self.numax,
                                           teff=self.teff,
                                           bp_rp=self.bp_rp)
        
        outpath = os.path.join(*[self.path, 'epsilon'])
        if self.make_plots:
            self.epsilon.plot(self.pg).savefig(outpath + f'_{self.ID}.png')
            self.epsilon.plot_corner().savefig(outpath + f'_corner_{self.ID}.png')
            
            
    def run_asy_peakbag(self, norders=6):
        """
        Runs the asy_peakbag code.
        """
        self.asy_fit = asymptotic_fit(self.f, self.s, self.epsilon.samples,
                                      self.teff, self.bp_rp, nthreads=1,
                                      store_chains=self.store_chains,
                                      norders=norders)
        
        self.asy_result = self.asy_fit.run(dnu=self.dnu, numax=self.numax,
                                           teff=self.teff, bp_rp=self.bp_rp)

        outpath = os.path.join(*[self.path, 'asy'])
        self.asy_result['summary'].to_csv(outpath + f'_summary_{self.ID}.csv',
                                          index=True)
        self.asy_result['modeID'].to_csv(outpath + f'_modeID_{self.ID}.csv',
                                          index=False)

        if self.store_chains:
            pass # TODO need to pickle the chains if requested.
        if self.make_plots:
            self.asy_fit.plot().savefig(outpath + f'_{self.ID}.png')
            self.asy_fit.plot_corner().savefig(outpath + f'_corner_{self.ID}.png')
            

    def run_peakbag(self, model_type='simple', tune=1500):
        """
        Runs peakbag on the given star.
        """
        self.peakbag = peakbag(self.f, self.s, self.asy_result)
        self.peakbag.sample(model_type=model_type, tune=tune,
                            cores=self.nthreads)
        
        outpath = os.path.join(*[self.path, 'peakbag'])
        pm.summary(self.peakbag.samples).to_csv(outpath + f'_summary_{self.ID}.csv')
        if self.store_chains:
            pass # TODO need to pickle the samples if requested.
        if self.make_plots:
            self.peakbag.plot_flat_fit().savefig(outpath + f'_{self.ID}.png')


    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500):
        """ Instead of a _call_ we should just make this a function maybe? """
        self.run_epsilon(bw_fac=bw_fac)
        self.run_asy_peakbag(norders=norders)
        self.run_peakbag(model_type=model_type, tune=tune)
