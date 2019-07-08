""" Fitting asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 modes
are ignored.
"""

import numpy as np
import pbjam as pb
import os
import pandas as pd
from collections import OrderedDict
from . import PACKAGEDIR
import scipy.stats as scist
import matplotlib.pyplot as plt

def mad(x, axis=0, scale=1.4826):
    """ Compute median absolute deviation

    Grabbed from Scipy version 1.3.0.

    TODO - To be removed once PBjam works with Scipy v > 1.2.0 and replaced
    with scipy.stats.median_absolute_deviation.

    Parameters
    ----------
    x : array
        Array to compute the mad for.
    axis : int
        Axis to compute the mad over
    scale : float
        Scale factor for the MAD, 1.4826 to match standard deviation for
        Gaussian.

    Returns
    -------
    mad : float
        The median absolute deviation(s) of the array
    """

    x = np.asarray(x)
    if axis is None:
        med = np.median(x)
        mabsdev = np.median(np.abs(x - med))
    else:
        med = np.apply_over_axes(np.median, x, axis)
        mabsdev = np.median(np.abs(x - med), axis=axis)
    return scale * mabsdev

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
        nmax : float
            non-integer radial order of maximum power of the p-mode envelope
    -------
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
    enns : array
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
    nu0s : array()
        Array of l=0 mode frequencies from the asymptotic relation (muHz).

    """
    nmax = get_nmax(numax, dnu, eps)
    enns = get_enns(nmax, norders)
    return (enns.T + eps + alpha/2*(enns.T - nmax)**2) * dnu

def P_envelope(nu, hmax, numax, width):
    """ Power of the p-mode envelope

    Computes the power at frequency nu in the p-mode envelope from a Gaussian
    distribution. Used for computing mode heights

    Parameters
    ----------
    nu : float
        Frequency (muHz).
    hmax : float
        Height of p-mode envelope (SNR).
    numax : float
        Frequency of maximum power of the p-mode envelope (muHz).
    width : float
        Width of the p-mode envelope (muHz).

    Returns
    -------
    h : float
        Power at frequency nu (SNR)
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
    mle_model : 1d array
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
    mads =  mad(fit.flatchain, axis=0)
    mle = fit.flatchain[idx,:]
    for i, par in enumerate(pnames):
        z = [mle[i], means[i], stds[i], skewness[i],  pars_percs[0,i],
             pars_percs[1,i], pars_percs[2,i], pars_percs[3,i],
             pars_percs[4,i], mads[i]]
        A = {key: z[i] for i, key in enumerate(smry_stats)}
        summary[par] = pd.Series(A)
    mle_model = model(mle)
    return summary, mle_model

def envelope_width(numax):
    return 0.66 * numax ** 0.88

class asymp_spec_model():
    """ Class for spectrum model using asymptotic relation

    Parameters
    ---------_
    f : float, array
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.
    norders : int
        Number of radial order to fit

    Attributes
    ----------
    f : float, array
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
            Frequency of lorentzian (muHz)
        h : float
            Height of the lorentizan (SNR)
        w : float
            Full width of the lorentzian (log10(muHz))

        Returns
        -------
        mode : array
            The SNR as a function frequency for a lorentzian
        """

        w = 10**(w)
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)

    def pair(self, freq0, h, w, d02, hfac=0.7):
        """ Define a pair as the sum of two Lorentzians

        A pair is assumed to consist of an l=0 and an l=2 mode. The widths are
        assumed to be identical, and the height of the l=2 mode is scaled
        relative to that of the l=0 mode. The frequency of the l=2 mode is the
        l=0 frequency minus the small separation.

        Parameters
        ----------
        freq0 : float
            Frequency of the l=0 (muHz)
        h : float
            Height of the l=0 (SNR)
        w : float
            The mode width (identical for l=2 and l=0) (log10(muHz))
        d02 : float
            The small separation (muHz)
        hfac : float, optional
            Ratio of the l=2 height to that of l=0 (unitless)

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
        """ Constructs a spectrum model from the asymptotic relation

        The asymptotic relation for p-modes with angular degree, l=0, is
        defined as:
        nu_nl = (n + epsilon + alpha/2(n - nmax)**2) * log_dnu,

        where nmax = numax / dnu - eps.

        We separate the l=0 and l=2 modes by d02.

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz)
        dnu : float
            Large separation (muHz)
        eps : float
            Phase term of the asymptotic relation (unitless)
        alpha : float
            Curvature of the asymptotic relation (unitless)
        d02 : float
            Small separation (muHz)
        hmax : float
            Gaussian height of p-mode envelope (SNR)
        envwidth : float
            Gaussian width of the p-mode envelope (muHz)
        modewidth : float
            Width of the modes (log10(muHz))
        *args : array-like
            List of additional parameters (Teff, bp_rp) that aren't actually
            used to construct the spectrum model, but just for evaluating the
            prior.

        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
        """

        f0s = asymptotic_relation(numax, dnu, eps, alpha, self.norders)
        Hs = P_envelope(f0s, hmax, numax, envwidth)
        mod = np.ones(len(self.f))
        for n in range(len(f0s)):
            mod += self.pair(f0s[n], Hs[n], modewidth, d02)
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

class asymptotic_fit():
    """ Class for fitting a spectrum based on the asymptotic relation

    Parameters
    ----------
    TODO - don't pass star pass attributes ... star : class instance
        Star class instance to perform the fit on. This contains the required
        attributes for the fit, f, s, numax, dnu, teff. All others are derived
        from these, or can optionally be set.
    teff :
           TODO
    bp_rp :
        TODO
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
    f : array
        Numpy array of frequency bins of the spectrum (muHz).
    s : array
        Numpy array of power in each frequency bin (SNR).
    sel : array, bool
        Numpy array of boolean values specifying the frequency range to be
        considered in the asymptotic relation fit.
    model : asy_peakbag.model.model instance
        Function for computing a spectrum model given a set of parameters.
    bounds : array
        Numpy array of upper and lower boundaries for the asymptotic relation
        fit. These limits truncate the likelihood function.
    gaussian : array
        Numpy array of tuples of mean and sigma for Gaussian
        priors on each of the fit parameters (To be removed when full
        KDE is implimented).

        Note:['dnu', 'numax', 'eps',
                     'd02', 'alpha', 'env_height',
                     'env_width', 'mode_width', 'teff',
                     'bp_rp']
        key: [log, log, lin, log, log, log, log, log, log, lin]
    """

    def __init__(self, f, snr, start_samples,
                 teff, bp_rp,
                 store_chains = False, nthreads=1, norders = 8):

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

        self.bounds = self.set_bounds(start)
        self.gaussian = self.set_gaussian_pars(teff, bp_rp)
        self.lp = Prior(self.bounds, self.gaussian)

        self.modeID = None
        self.summary = None
        self.flatchain = None
        self.lnlike_fin = None
        self.lnprior_fin = None
        self.mle_model = None
        self.acceptance = None

    def set_bounds(self, start):
        """ Set parameter bounds for asymptotic relation fit

        Parameters
        ----------
        start: arraylike
            Note:['dnu', 'numax', 'eps',
                         'd02', 'alpha', 'env_height',
                         'env_width', 'mode_width', 'teff',
                         'bp_rp']
        Returns
        -------
        bounds : array
            Numpy array of upper and lower boundaries for the asymptotic relation
            fit. These limits truncate the likelihood function.

        ['dnu', 'numax', 'eps',
                     'd02', 'alpha', 'env_height',
                     'env_width', 'mode_width', 'teff',
                     'bp_rp']
        """

        bounds = [[start[0]*0.9, start[0]*1.1], #dnu
                  [start[1]*0.9, start[1]*1.1], # numax
                  [0.4 , 1.6],  #eps
                  [0.05*start[0], 0.25*start[0]],  # d02
                  [1e-20, 0.1],  # alpha
                  [0, 4],  # hmax (log10)
                  [-1, 6],  # Ewidth (log10)
                  [-3, 3],  # mode width (log10)
                  [start[8] - 2000, start[8] + 2000], # teff
                  [start[9] - 1.0, start[9] + 1.0]  # Gaia bp-rp
                 ]
        return bounds

    def set_gaussian_pars(self, teff, bp_rp):
        """ Parameters of the Gaussian priors

        Used to define the mean and standard deviation of the Gaussian priors
        on each of the parameters.

        This will be deprecated in forthcoming versions of PBjam. To be
        replaced by a full KDE on all fit parameters.

        Returns
        -------
        gaussian : array
            Numpy array of tuples of mean and sigma for Gaussian
            priors on each of the fit parameters (To be removed when full
            KDE is implimented).
        """

        gaussian = [(0, 0),  # dnu
                    (0, 0),  # numax
                    (0, 0),  # eps
                    (0, 0),  # d02
                    (0, 0),  # alpha
                    (0, 0),  # env_height
                    (0, 0),  # env_width
                    (0, 0),  # mode width (log10)
                    (teff[0], teff[1]),  # Teff
                    (bp_rp[0], bp_rp[1]),  # Gaia bp-rp
                    ]
        return gaussian

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

        # TODO - is there a better/neater way to do this?

        flatchain = fit.flatchain

        nsamps = np.shape(flatchain)[0]

        nu0_samps, nu2_samps = np.empty((nsamps, N)), np.empty((nsamps, N))

        nu0_samps = asymptotic_relation(flatchain[:, 1], flatchain[:, 0],
                                        flatchain[:, 2], flatchain[:, 4], N)
        nu2_samps = nu0_samps - flatchain[:, 3]

        nus_med = np.median(np.array([nu0_samps, nu2_samps]), axis=2)
        nus_mad = mad(np.array([nu0_samps, nu2_samps]), axis=2)

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

    def run(self, burnin=1000, niter=1000):
        """ Setup, run and parse the asymptotic relation fit using EMCEE

        Returns
        -------
        modeID : pandas.DataFrame
            Dataframe of radial order, n (best guess), angular degree, l,
            frequency and frequency error.
        """

        self.fit = mcmc(self.f[self.sel], self.s[self.sel], self.model,
                   self.start, self.lp, nthreads=self.nthreads)

        self.fit(burnin=burnin, niter=niter)  # do the fit with default settings

        self.modeID = self.get_modeIDs(self.fit, self.norders)

        self.summary, self.mle_model = get_summary_stats(self.fit, self.model, self.pars_names)

        if self.store_chains:
            self.flatchain = self.fit.flatchain
            self.lnlike_fin = self.fit.flatlnlike
        else:
            self.flatchain = self.fit.chain[:,-1,:]
            self.lnlike_fin = np.array([self.fit.likelihood(self.fit.chain[i,-1,:]) for i in range(self.fit.nwalkers)])
            self.lnprior_fin = np.array([self.fit.lp(self.fit.chain[i,-1,:]) for i in range(self.fit.nwalkers)])

        self.acceptance = self.fit.acceptance
        return {'modeID': self.modeID, 'summary': self.summary}

    def plot(self, thin=100):
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(self.f, self.s, 'k-', label='Data')
        ax.plot(self.model.f, self.model(self.flatchain[0, :]),
                'r-', label='fit', alpha=0.3)
        for i in np.arange(thin, len(self.flatchain[:, 0]), thin):
            ax.plot(self.model.f, self.model(self.flatchain[i, :]),
                    alpha=0.3)
        ax.plot(self.model.f, self.mle_model, 'b-', alpha=0.7, lw=2)


class Prior(pb.epsilon):
    """ Evaluate the proirs on the provided model parameters

    Parameters
    ----------
    bounds : array
        Numpy array of upper and lower boundaries for the asymptotic relation
        fit. These limits truncate the likelihood function.
    gaussian : array
        Numpy array of tuples of mean and sigma for Gaussian
        priors on each of the fit parameters (To be removed when full
        KDE is implimented)

    Attributes
    ----------
    data_file : str
        Pathname to the file containing stellar parameters to make the KDE
        prior.
    prior_data : pandas.DataFrame instance
        Dataframe with all the prior data. Read from prior_data.csv.
    kde : sm.nonparametric.KDEMultivariate instance
        KDE based on the prior data file.
    """

    def __init__(self, bounds, gaussian):
        self.bounds = bounds
        self.gaussian = gaussian
        self.data_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])
        self.read_prior_data()  # Inherited from epsilon
        self.make_kde(bw_fac=1.0)  # Inherited from epsilon

    def pbound(self, p):
        ''' Check if parameter set is out of bounds

        Truncates posterior beyond the supplied bounds.

        Parameters
        ----------
        p : array
            Array containing model parameters

        Returns
        -------
        prior : float or inf
            If within bounds returns 0, -inf if not
        '''

        for idx, i in enumerate(p):
            if np.all(self.bounds[idx] != 0):
                if ((i < self.bounds[idx][0]) | (i > self.bounds[idx][1])):
                    return -np.inf
        return 0

    def pgaussian(self, p):
        """ Guassian priors

        Function for setting Gaussian priors

        Parameters
        ----------
        p : array
            Array containing mcmc proposals

        Returns
        -------
        lnprior : float
            Sum of Guassian priors evaluted at respective values of p
        """

        lnprior = 0.0
        for idx, x in enumerate(p):
            if self.gaussian[idx][1] != 0:
                lnprior += -0.5*(np.log(2*np.pi*self.gaussian[idx][1]**2) +
                                 (x - self.gaussian[idx][0])**2 / self.gaussian[idx][1]**2)
        return lnprior

    def __call__(self, p):
        """ Evaluate the priors for a set of parameters

        The prior is estimated by a KDE of a set of previous Kepler
        observations. This is truncated at some reasonable values to keep
        the MCMC sampler in check.

        Currently a Gaussian prior is added to some of the variables that are
        not included in the current version of the KDE. In future the KDE will
        be extended to include all the fit variables.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        lp : float
            The prior at p

        ['dnu', 'numax', 'eps',
                     'd02', 'alpha', 'env_height',
                     'env_width', 'mode_width', 'teff',
                     'bp_rp']

        """

        if self.pbound(p) == -np.inf:
            return -np.inf

        lp = np.log(self.kde.pdf([np.log10(p[0]), np.log10(p[1]), p[2],
                                  np.log10(p[3]), np.log10(p[4]),
                                  p[5], p[6], p[7],
                                  np.log10(p[8]), p[9]]))
        lp += self.pgaussian(p)
        return lp

class mcmc():
    """ Class for MCMC sampling

    Use EMCEE to fit a provided model to a spectrum.

    Parameters
    ----------
    f : float, array
        Array of frequency bins of the spectrum (muHz)
    s : array
        The power at frequencies f
    model : asy_peakbag.model.model instance
        Function for computing a spectrum model given a set of parameters.
    pars_names : list
        List of parameter names in the asymptotic fit
    guess : dictionary
        Dictionary for organizing the initial guess for the asymptotic fir
        parameters.
    bounds : array
        Numpy array of upper and lower boundaries for the asymptotic relation
        fit. These limits truncate the likelihood function.
    gaussian : array
        Numpy array of tuples of mean and sigma for Gaussian
        priors on each of the fit parameters (To be removed when full
        KDE is implimented)
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead.

    Attributes
    ----------
    ndim : int
        Number of fit variables
    lp : Prior class instance
        Prior class initialized using model parameter limits, Gaussian
        paramaters, and the KDE
    burnin : int
        Number of steps to take as the burn-in phase. These are discarded at
        the end of the MCMC run.
    niter : int
        Number of steps to take after the burn-in phase. These will be tested
        for convergence. Once the test is satisfied these steps will will be
        assumed to be samples of the posterior distribution.
    nwalkers : int
        Number of walkers to use in the MCMC fit
    prior_only : bool
        Flag for whether or not to just sample the prior, or to sample both
        sum of the prior and likelihood function. Use to draw initial positions
        for the walkers.
    chain : array
        Numpy array of shape (niter, nwalkers, ndim) with all the MCMC samples
        drawn after the burn-in phase.
    flatchain : array
        A flattened version of the chain array of shape (niter*nwalkers, ndim).
    lnlike : array
        Numpy array of shape (niter, nwalkers) of the likelihood at each step
        of each walker.
    flatlnlike : array
        A flattened version of the lnlike array of shape (niter*walkers)
    """

    def __init__(self, f, s, model, start, lp,
                 nthreads=1):
        self.f = f
        self.s = s
        self.model = model
        self.start = start
        self.nthreads = nthreads
        self.ndim = len(start)
        self.lp = lp

        self.niter = None
        self.nwalkers = None
        self.burnin = None
        self.prior_only = None
        self.chain = None
        self.flatchain = None
        self.lnlike = None
        self.flatlnlike = None
        self.acceptance = None


    def likelihood(self, p):
        """ Likelihood function for set of model parameters

        Evaluates the likelihood function and applies any priors for a set of
        model parameters.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        like : float
            likelihood function at p
        """
        logp = self.lp(p)

        if logp == -np.inf:
            return -np.inf

        mod = self.model(p)
        like = -1.0 * np.sum(np.log(mod) + self.s / mod)
        return like + logp

    def convergence_test(self):
        """ TBD
        """
        return True

    def __call__(self, niter=1000, nwalkers=50, burnin=1000, spread=1e-4,
                 prior_only = False):
        """ Initialize and run the EMCEE afine invariant sampler

        Parameters
        ----------
        niter : int
            Number of steps for the walkers to take (both burn-in and
            sampling).
        nwalkers : int
            Number of walkers to use in the EMCEE run.
        spread : float
            Percent spread around the intial position of the walkers. Small
            value starts the walkers in a tight ball, large value fills out
            the range set by parameter bounds.

        Returns
        -------
        sampler.flatchain : array
            The chain of (nwalkers, niter, ndim) flattened to
            (nwalkers*niter, ndim).
        """

        # Save these for later
        self.niter = niter
        self.nwalkers = nwalkers
        self.burnin = burnin
        self.prior_only = True

        import emcee

        # Start walkers in a tight random ball
        p0 = np.array([self.start + (np.random.randn(self.ndim) * spread) for i in range(self.nwalkers)])

        sampler_prior = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                              self.lp, threads=self.nthreads)
        pos, prob, state = sampler_prior.run_mcmc(p0, self.burnin) # Burningham
        pos = self.fold(sampler_prior, pos, spread)
        sampler_prior.reset()

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                        self.likelihood, threads=self.nthreads)
        pos, prob, state = sampler.run_mcmc(p0, self.burnin) # Burningham
        pos = self.fold(sampler, pos, spread)
        sampler.reset()

        converged = False
        while not converged:
            pos, prob, state = sampler.run_mcmc(pos, self.niter) # Sampling
            converged = self.convergence_test()

        self.chain = sampler.chain.copy()
        self.flatchain = sampler.flatchain
        self.lnlike = sampler.lnprobability
        self.flatlnlike = sampler.flatlnprobability
        self.acceptance = sampler.acceptance_fraction

        sampler.reset()  # This hopefully minimizes emcee memory leak


    def fold(self, sampler, pos, spread, accept_lim = 0.2):
        """ Fold low acceptance walkers into main distribution

        At the end of the burn-in, some walkers appear stuck with low
        acceptance fraction. These can be selected using a threshold, and
        folded back into the main distribution, estimated based on the median
        of the walkers with an acceptance fraction above the threshold.

        The stuck walkers are relocated with multivariate Gaussian, with mean
        equal to the median of the high acceptancew walkers, and a standard
        deviation equal to the median absolute deviation of these, with a
        small scaling factor.

        Parameters
        ----------
        sampler : emcee sampler object
            The sampler used in the fit
        pos : array
            The final position of the walkers after the burn-in phase
        spread : float
            The factor to apply to the walkers that are adjusted

        """
        idx = sampler.acceptance_fraction < accept_lim
        nbad = np.shape(pos[idx, :])[0]
        if nbad > 0:
            flatchains = sampler.chain[~idx, :, :].reshape((-1, self.ndim))
            good_med = np.median(flatchains, axis = 0)
            good_mad = mad(flatchains, axis = 0) * spread
            pos[idx, :] = np.array([[np.random.uniform(max(self.lp.bounds[j][0], good_med[j]-good_mad[j]),
                                                       min(self.lp.bounds[j][1], good_med[j]+good_mad[j])
                                                       ) for j in range(self.ndim)] for n in range(nbad)])
        return pos
