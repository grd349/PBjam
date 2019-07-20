import os
import lightkurve as lk
from pbjam.asy_peakbag import asymptotic_fit, envelope_width
from pbjam.guess_epsilon import epsilon
from pbjam.peakbag import peakbag
import numpy as np
import astropy.units as units
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, warnings, psutil, pickle
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

    teff : list, optional
        List of the form [teff, teff_error]. For multiple targets, use a list
        of lists.

    bp_rp : list, optional
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
    """

    def __init__(self, ID, periodogram,
                 numax, dnu, teff=None, bp_rp=None,
                 store_chains=True, nthreads=1,
                 make_plots=False,
                 path=None, verbose=False):
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

        if path == None:
            path = os.getcwd()
        self.bpath = os.path.join(*[path, f'{self.ID}'])
        try:
            os.mkdir(self.bpath)
        except OSError:
            if verbose:
                warnings.warn(f'Path {self.bpath} already exists - I will try to overwrite ... ')

        self.data_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])

    def run_epsilon(self, bw_fac=1.0):
        """
        Runs the epsilon code and makes plots if self.make_plots is set.
        """
        self.epsilon = epsilon()
        self.epsilon_result = self.epsilon(dnu=self.dnu,
                                           numax=self.numax,
                                           teff=self.teff,
                                           bp_rp=self.bp_rp,
                                           bw_fac=bw_fac)
        if self.make_plots:
            self.epsilon.plot(self.pg).savefig(self.bpath + os.sep + f'epsilon_{self.ID}.png')
            self.epsilon.plot_corner().savefig(self.bpath + os.sep + f'epsilon_corner_{self.ID}.png')

    def run_asy_peakbag(self, norders=6, burnin=3000):
        self.asy_fit = asymptotic_fit(self.f, self.s, self.epsilon.samples,
                                      self.teff, self.bp_rp,
                                      store_chains=self.store_chains,
                                      nthreads=1, norders=norders)
        self.asy_result = self.asy_fit.run(burnin=burnin)
        if self.store_chains:
            pass # TODO need to pickle the chains if requested.
        if self.make_plots:
            self.asy_fit.plot_corner().savefig(self.bpath + os.sep + f'asy_corner_{self.ID}.png')
            self.asy_fit.plot().savefig(self.bpath + os.sep + f'asy_{self.ID}.png')


    def run_peakbag(self, model_type='simple', tune=1500):
        """
        Runs peakbag on the given star.
        """
        self.peakbag = peakbag(self.f, self.s, self.asy_result)
        self.peakbag.sample(model_type=model_type, tune=tune,
                            cores=self.nthreads)
        pm.summary(self.peakbag.samples).to_csv(self.bpath + os.sep + f'peakbag_summary_{self.ID}.csv')
        if self.store_chains:
            pass # TODO need to pickle the samples if requested.
        if self.make_plots:
            self.peakbag.plot_fit().savefig(self.bpath + os.sep + f'peakbag_{self.ID}.png')


    def __call__(self, bw_fac=1.0, norders=8,
                 model_type='simple', tune=1500):
        """ Instead of a _call_ we should just make this a function maybe? """
        self.run_epsilon(bw_fac=bw_fac)
        self.run_asy_peakbag(norders=norders)
        self.run_peakbag(model_type=model_type, tune=tune)
