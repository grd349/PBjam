"""Fitting the asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max, the central frequency of the seismic mode envelope,
in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 modes are ignored.
"""

import numpy as np
import pbjam as pb
import os
import pandas as pd
from collections import OrderedDict
from . import PACKAGEDIR
import scipy.stats as scist
import astropy.convolution as conv
import matplotlib.pyplot as plt
import corner
import emcee

def get_nmax(numax, dnu, eps):
    """Compute radial order at numax.

    Compute the radial order at numax, which in this implimentation of the
    asymptotic relation is not necessarily integer.

    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope (muHz).
    dnu : float
        Large separation of l=0 modes (muHz).
    eps : float
        Epsilon phase term in asymptotic relation (muHz).

    Returns
    -------
        nmax : float
            non-integer radial order of maximum power of the p-mode envelope
    """

    return numax / dnu - eps


def get_enns(nmax, norders):
    """Compute radial order numbers.

    Get the enns that will be included in the asymptotic relation fit. These
    are all integer.

    Parameters
    ----------
    nmax : float
        Frequency of maximum power of the p-mode envelope
    norders : int
        Total number of radial orders to consider

    Returns
    -------
    enns : ndarray
            Numpy array of norders radial orders (integers) around numax (nmax).
    """

    below = np.floor(nmax - np.floor(norders/2)).astype(int)
    above = np.floor(nmax + np.ceil(norders/2)).astype(int)
    if type(below) == np.int64:
        return np.arange(below, above)
    else:
        return np.concatenate([np.arange(x, y) for x, y in zip(below, above)]).reshape(-1, norders)


def asymptotic_relation(numax, dnu, eps, alpha, norders):
    """ Compute the l=0 mode frequencies from the asymptotic relation for
    p-modes

    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope (muHz).
    dnu : float
        Large separation of l=0 modes (muHz).
    eps : float
        Epsilon phase term in asymptotic relation (unitless).
    alpha : float
        Curvature factor of l=0 ridge (second order term, unitless).
    norders : int
        Number of desired radial orders to calculate frequncies for, centered
        around numax.

    Returns
    -------
    nu0s : ndarray
        Array of l=0 mode frequencies from the asymptotic relation (muHz).

    """
    nmax = get_nmax(numax, dnu, eps)
    enns = get_enns(nmax, norders)
    return (enns.T + eps + alpha/2*(enns.T - nmax)**2) * dnu


def P_envelope(nu, hmax, numax, width):
    """ Power of the seismic p-mode envelope

    Computes the power at frequency nu in the p-mode envelope from a Gaussian
    distribution. Used for computing mode heights.

    Parameters
    ----------
    nu : float
        Frequency (in muHz).

    hmax : float
        Height of p-mode envelope (in SNR).

    numax : float
        Frequency of maximum power of the p-mode envelope (in muHz).

    width : float
        Width of the p-mode envelope (in muHz).

    Returns
    -------
    h : float
        Power at frequency nu (in SNR)
    """
    hmax = 10**hmax
    width = 10**width
    return hmax * np.exp(- 0.5 * (nu - numax)**2 / width**2)


def get_summary_stats(fit, model, pnames):
    """ Make dataframe with fit summary statistics

    Creates a dataframe that contains various quantities that summarize the
    fit. Note, these are predominantly derived from the marginalized posteriors.

    Parameters
    ----------
    fit : asy_peakbag.mcmc instance
        asy_peakbag.mcmc that was used to fit the spectrum, containing the
        log-likelihoods and MCMC chains.
    model : the asymp_spec_model.model instance that defines the model used to
        fit the spectrum.
    pnames : list
       List of names of each of the parameters in the fit.

    Returns
    -------
    summary : pandas.DataFrame
        Dataframe with the summary statistics.
    mle_model : ndarray
        Numpy array with the model spectrum corresponding to the maximum
        likelihood solution.
    """

    summary = pd.DataFrame()
    smry_stats = ['mle','mean','std', 'skew', '2nd', '16th', '50th', '84th',
                  '97th', 'MAD']
    idx = np.argmax(fit.flatlnlike)
    means = np.mean(fit.flatchain, axis = 0)
    stds = np.std(fit.flatchain, axis = 0)
    skewness = scist.skew(fit.flatchain, axis = 0)
    pars_percs = np.percentile(fit.flatchain, [50-95.4499736104/2,
                                               50-68.2689492137/2,
                                               50,
                                               50+68.2689492137/2,
                                               50+95.4499736104/2], axis=0)
    mads =  scist.median_absolute_deviation(fit.flatchain, axis=0)
    mle = fit.flatchain[idx,:]
    for i, par in enumerate(pnames):
        z = [mle[i], means[i], stds[i], skewness[i],  pars_percs[0,i],
             pars_percs[1,i], pars_percs[2,i], pars_percs[3,i],
             pars_percs[4,i], mads[i]]
        A = {key: z[i] for i, key in enumerate(smry_stats)}
        summary[par] = pd.Series(A)
    mle_model = model(mle)
    return summary, mle_model


class asymp_spec_model():
    """Class for spectrum model using asymptotic relation.

    Parameters
    ---------_
    f : float, ndarray
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.
    norders : int
        Number of radial order to fit

    Attributes
    ----------
    f : float, ndarray
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.

    norders : int
        Number of radial order to fit
    """

    def __init__(self, f, norders):
        self.f = f
        self.norders = norders


    def lor(self, freq, h, w):
        """ Lorentzian to describe a mode.

        Parameters
        ----------
        freq : float
            Frequency of lorentzian (muHz).
        h : float
            Height of the lorentizan (SNR).
        w : float
            Full width of the lorentzian (log10(muHz)).

        Returns
        -------
        mode : ndarray
            The SNR as a function frequency for a lorentzian.
        """

        w = 10**(w)
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)


    def pair(self, freq0, h, w, d02, hfac=0.7):
        """Define a pair as the sum of two Lorentzians.

        A pair is assumed to consist of an l=0 and an l=2 mode. The widths are
        assumed to be identical, and the height of the l=2 mode is scaled
        relative to that of the l=0 mode. The frequency of the l=2 mode is the
        l=0 frequency minus the small separation.

        Parameters
        ----------
        freq0 : float
            Frequency of the l=0 (muHz).
        h : float
            Height of the l=0 (SNR).
        w : float
            The mode width (identical for l=2 and l=0) (log10(muHz)).
        d02 : float
            The small separation (muHz).
        hfac : float, optional
            Ratio of the l=2 height to that of l=0 (unitless).

        Returns
        -------
        pair_model : array
            The SNR as a function of frequency of a mode pair.
        """

        pair_model = self.lor(freq0, h, w)
        pair_model += self.lor(freq0 - d02, h*hfac, w)
        return pair_model

    def model(self, dnu, numax, eps, d02, alpha, hmax, envwidth, modewidth,
              *args):
        """ Constructs a spectrum model from the asymptotic relation.

        The asymptotic relation for p-modes with angular degree, l=0, is
        defined as:

        nu_nl = (n + epsilon + alpha/2(n - nmax)**2) * log(dnu) ,

        where nmax = numax / dnu - epsilon.

        We separate the l=0 and l=2 modes by the small separation d02.

        Parameters
        ----------
        dnu : float
            Large separation log10(muHz)
        lognumax : float
            Frequency of maximum power of the p-mode envelope log10(muHz)

        eps : float
            Phase term of the asymptotic relation (unitless)
        alpha : float
            Curvature of the asymptotic relation log10(unitless)
        d02 : float
            Small separation log10(muHz)
        loghmax : float
            Gaussian height of p-mode envelope log10(SNR)
        logenvwidth : float
            Gaussian width of the p-mode envelope log10(muHz)
        logmodewidth : float
            Width of the modes (log10(muHz))
        *args : array-like
            List of additional parameters (Teff, bp_rp) that aren't actually
            used to construct the spectrum model, but just for evaluating the
            prior.

        Returns
        -------
        model : ndarray
            spectrum model around the p-mode envelope
        """

        f0s = asymptotic_relation(10**numax, 10**dnu, eps, 10**alpha, self.norders)
        Hs = P_envelope(f0s, hmax, 10**numax, envwidth)
        mod = np.ones(len(self.f))
        for n in range(len(f0s)):
            mod += self.pair(f0s[n], Hs[n], modewidth, 10**d02)
        return mod

    def __call__(self, p):
        """ Produce model of the asymptotic relation

        Parameters
        ----------
        p : list
            list of model parameters

        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
        """

        return self.model(*p)


class asymptotic_fit(pb.epsilon):
    """ Class for fitting a spectrum based on the asymptotic relation.

    Parameters
    ----------
    f : ndarray
        Numpy array of frequency bins of the spectrum (muHz).
    s : ndarray
        Numpy array of power in each frequency bin (SNR).
    start_samples: ndarray
        Samples representing the starting guess for the asymptotic
        peakbagging.
    teff : [real, real]
        Stellar effective temperature and uncertainty
    bp_rp : [real, real]
        The Gaia Gbp - Grp color value and uncertainty (probably ~< 0.01 dex)
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run.
        Warning, if running multiple targets, make sure you have enough memory.
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead.
        Untested on short cadence.
    norders : int, optional
        Number of radial orders to fit

    Attributes
    ----------
    f : ndarray
        Numpy array of frequency bins of the spectrum (muHz).
    s : ndarray
        Numpy array of power in each frequency bin (SNR).
    sel : ndarray, bool
        Numpy array of boolean values specifying the frequency range to be
        considered in the asymptotic relation fit.
    model : asy_peakbag.model.model instance
        Function for computing a spectrum model given a set of parameters.
    bounds : ndarray
        Numpy array of upper and lower boundaries for the asymptotic relation
        fit. These limits truncate the likelihood function.
    gaussian : ndarray
        Numpy array of tuples of mean and sigma for Gaussian
        priors on each of the fit parameters (To be removed when full
        KDE is implimented).
    """

    def __init__(self, f, snr, start_samples,
                 teff, bp_rp,
                 store_chains=True, nthreads=1, norders=8):

        self.start_samples = start_samples
        self.store_chains = store_chains
        self.nthreads = nthreads
        self.norders = norders
        self.f = f
        self.s = snr
        self.pars_names = ['dnu', 'numax', 'eps',
                           'd02', 'alpha', 'env_height',
                           'env_width', 'mode_width', 'teff',
                           'bp_rp']

        summary = start_samples.mean(axis=0)
        start = [10**(summary[0]), 10**(summary[1]), summary[2],
                10**(summary[3]), 10**(summary[4]), summary[5],
                summary[6], summary[7], 10**(summary[8]),
                summary[9]]
        self.start = start

        nmax = get_nmax(start[1], start[0], start[2])
        lower_n = nmax - self.norders/2 - 1.25 + start[2]
        upper_n = nmax + self.norders/2 + 0.25 + start[2]
        lower_frequency = lower_n * start[0]
        upper_frequency = upper_n * start[0]
        self.sel = np.where((self.f > lower_frequency) & (self.f < upper_frequency))
        self.model = asymp_spec_model(self.f[self.sel], self.norders)


        self.modeID = None
        self.summary = None
        self.flatchain = None
        self.lnlike_fin = None
        self.lnprior_fin = None
        self.mle_model = None
        self.acceptance = None


    def get_modeIDs(self, fit, N):
        """ Set mode ID in a dataframe

        Evaluates the asymptotic relation for each walker position from the
        MCMC fit. The median values of the resulting set of frequencies are
        then returned in a pandas.DataFrame

        Parameters
        ----------
        fit : asy_peakbag.mcmc class instance
            mcmc class instances used in the fit
        N : int
            Number of radial orders to output. Note that doesn't have to be
            the same as that used int he fit itself.

        Returns
        -------
        modeID : pandas.DataFrame
            Dataframe of radial order, n (best guess), angular degree, l,
            frequency and frequency error.
        """

        flatchain = fit.flatchain

        nsamps = np.shape(flatchain)[0]

        nu0_samps, nu2_samps = np.empty((nsamps, N)), np.empty((nsamps, N))

        nu0_samps = asymptotic_relation(10**flatchain[:, 1], 10**flatchain[:, 0],
                                        flatchain[:, 2], 10**flatchain[:, 4], N)
        nu2_samps = nu0_samps - 10**flatchain[:, 3]

        nus_med = np.median(np.array([nu0_samps, nu2_samps]), axis=2)
        nus_mad = scist.median_absolute_deviation(np.array([nu0_samps, nu2_samps]), axis=2)

        #nus_std = np.std(np.array([nu0_samps, nu2_samps]), axis=2)

        ells = [0 if i % 2 else 2 for i in range(2*N)]

        nus_med_out = []
        nus_mad_out = []

        for i in range(N):
            nus_med_out += [nus_med[1, i], nus_med[0, i]]
            nus_mad_out += [nus_mad[1, i], nus_mad[0, i]]

        modeID = pd.DataFrame({'ell': ells,
                               'nu_med': nus_med_out,
                               'nu_mad': nus_mad_out})
        return modeID

    def plot_start(self):
        '''
        Plots the starting model as a diagnotstic.
        '''
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(self.f, self.s, 'k-', label='Data', alpha=0.2)
        smoo = self.start[0] * 0.005 / (self.f[1] - self.f[0])
        kernel = conv.Gaussian1DKernel(stddev=smoo)
        smoothed = conv.convolve(self.s, kernel)
        ax.plot(self.f, smoothed, 'k-',
                label='Smoothed', lw=3, alpha=0.6)
        ax.plot(self.f[self.sel], self.model(self.start_samples.mean(axis=0)), 'r-',
                label='Start model', alpha=0.7)
        ax.set_ylim([0, smoothed.max()*1.5])
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend()
        return fig

    def plot(self, thin=100):
        '''
        Plots the data and some models generated from the samples
        from the posteriod distribution.

        Parameters
        ----------
        thin: int
            Thins the samples of the posterior in order to speed up plotting.

        Returns
        -------
        fig: Figure
            The figure element containing the plots.
        '''
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(self.f, self.s, 'k-', label='Data', alpha=0.2)
        smoo = self.start[0] * 0.005 / (self.f[1] - self.f[0])
        kernel = conv.Gaussian1DKernel(stddev=smoo)
        smoothed = conv.convolve(self.s, kernel)
        ax.plot(self.f, smoothed, 'k-',
                label='Smoothed', lw=3, alpha=0.6)
        ax.plot(self.f[self.sel], self.model(self.flatchain[0, :]), 'r-',
                label='Model', alpha=0.2)
        for i in np.arange(thin, len(self.flatchain), thin):
            ax.plot(self.f[self.sel], self.model(self.flatchain[i, :]), 'r-',
                    alpha=0.2)
        freqs = self.modeID['nu_med']
        for f in freqs:
            ax.axvline(f, c='k', linestyle='--')
        ax.set_ylim([0, smoothed.max()*1.5])
        ax.set_xlim([self.f[self.sel].min(), self.f[self.sel].max()])
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend(loc=1)
        return fig

    def plot_corner(self):
        '''
        Calls corner.corner on the sampling results
        '''
        fig = corner.corner(self.fit.flatchain, labels=self.pars_names,
                            quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
        return fig

    def likelihood(self, p):
        """ Likelihood function for set of model parameters

        Evaluates the likelihood function for a set of
        model parameters.  This includes the constraint from
        the observed variables.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        like : float
            likelihood function at p

        Note:
        log_dnu, log_numax, eps, log_d02, log_alpha, \
            log_env_height, log_env_width, log_mode_width, \
            log_teff, bp_rp = p
        """
        # Constraint from input obs
        ld = 0.0
        ld += self.normal(p[-2], *self.log_obs['teff'])
        ld += self.normal(p[-1], *self.log_obs['bp_rp'])
        # Constraint from the periodogram
        mod = self.model(p)
        like = -1.0 * np.sum(np.log(mod) + self.s[self.sel] / mod)
        return like + ld

    def run(self,
            dnu=[1, -1],
            numax=[1, -1],
            teff=[1, -1],
            bp_rp=[1, -1]):
        """ Setup, run and parse the asymptotic relation fit using EMCEE.

        Parameters
        ----------
        dnu : [real, real]
            Large frequency spacing and uncertainty
        numax : [real, real]
            Frequency of maximum power and uncertainty
        teff : [real, real]
            Stellar effective temperature and uncertainty
        bp_rp : [real, real]
            The Gaia Gbp - Grp color value and uncertainty
            (probably ~< 0.01 dex).

        Returns
        -------
        asy_result : Dict
            A dictionary of the modeID DataFrame and the summary DataFrame.
        """
        self.obs = {'dnu': dnu,
                    'numax': numax,
                    'teff': teff,
                    'bp_rp': bp_rp}
        self.obs_to_log(self.obs)

        self.prior = pb.epsilon().prior
        self.fit = pb.mcmc(self.start_samples.mean(axis=0), self.likelihood,
                           self.prior, nthreads=self.nthreads)

        self.fit(start_samples=self.start_samples)

        self.modeID = self.get_modeIDs(self.fit, self.norders)

        self.summary, self.mle_model = get_summary_stats(self.fit,
                                            self.model, self.pars_names)

        if self.store_chains:
            self.flatchain = self.fit.flatchain
            self.lnlike_fin = self.fit.flatlnlike
        else:
            self.flatchain = self.fit.chain[:,-1,:]
            self.lnlike_fin = np.array([self.fit.likelihood(self.fit.chain[i,-1,:]) for i in range(self.fit.nwalkers)])
            self.lnprior_fin = np.array([self.fit.lp(self.fit.chain[i,-1,:]) for i in range(self.fit.nwalkers)])

        self.acceptance = self.fit.acceptance
        return {'modeID': self.modeID, 'summary': self.summary}
