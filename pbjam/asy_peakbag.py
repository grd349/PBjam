""" Fitting asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 modes
are ignored.
"""

import numpy as np
import pbjam as pb
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

from . import PACKAGEDIR


def get_nmax(numax, dnu, eps):
    """Compute radial order at numax.

    Note this is not necessarily integer

    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope
    dnu : float
        Large separation of l=0 modes
    eps : float
        Epsilon phase term in asymptotic relation

    Returns
        nmax : float
            non-integer radial order of maximum power of the p-mode envelope
    -------
    """

    return numax / dnu - eps


def get_enns(nmax, nrads):
    """Compute radial orders to include in asymptotic relation.

    These are all integer

    Parameters
    ----------
    nmax : float
        Frequency of maximum power of the p-mode envelope
    nrads : int
        Total number of radial orders to consider

    Returns
    -------
    enns : array
            array of nrads radial orders (integers) around numax (nmax)
    """

    below = np.floor(nmax - np.floor(nrads/2))
    above = np.floor(nmax + np.ceil(nrads/2))
    return np.arange(below, above).astype(int)


def asymptotic_relation(numax, dnu, eps, alpha, nrads):
    """ Compute the l=0 mode frequencies from the asymptotic relation for
    p-modes

    Parameters
    ----------
    numax : float
        Frequency of maximum power of the p-mode envelope.
    dnu : float
        Large separation of l=0 modes.
    eps : float
        Epsilon phase term in asymptotic relation.
    alpha : float
        Curvature factor of l=0 ridge (second order term).
    nrads : int
        Number of desired radial orders to calculate frequncies for, centered
        around numax.

    Returns
    -------
    nu0s : array()
        Array of l=0 mode frequencies from the asymptotic relation

    """
    nmax = get_nmax(numax, dnu, eps)
    enns = get_enns(nmax, nrads)
    return (enns + eps + alpha/2*(enns - nmax)**2) * dnu


def P_envelope(nu, hmax, numax, width):
    """ Power of the p-mode envelope

    Computes the power at frequency nu in the p-mode envelope from a Gaussian
    distribution. Used for computing mode heights

    Parameters
    ----------
    nu : float
        Frequency
    hmax : float
        Height of p-mode envelope
    numax : float
        Frequency of maximum power of the p-mode envelope.
    width : float
        Width of the p-mode envelope

    Returns
    -------
    h : float
        Power at frequency nu
    """

    return hmax * np.exp(- 0.5 * (nu - numax)**2 / width**2)


class asymp_spec_model():
    """ Class for spectrum model using asymptotic relation

    Parameters
    ---------_
    f : float, array
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.
    nrads : int
        Number of radial order to fit

    Attributes
    ----------
    f : float, array
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.
    nrads : int
        Number of radial order to fit
    """

    def __init__(self, f, nrads):
        self.f = f
        self.nrads = nrads

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

    def model(self, numax, dnu, eps, alpha, d02,
                    hmax, envwidth, modewidth):
        """ Constructs a spectrum model from the asymptotic relation

        The asymptotic relation for p-modes in red giants is defined as:
        nu_nl = (n + epsilon + alpha/2(n - nmax)**2) * log_dnu
        where,
        nmax = numax / dnu - eps.
        We separate the l=0 and l=2 modes by d02*dnu.

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
            Small separation (units of dnu)
        hmax : float
            Gaussian height of p-mode envelope (SNR)
        envwidth : float
            Gaussian width of the p-mode envelope (muHz)
        modewidth : float
            Width of the modes (log_10(muHz))

        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
        """

        f0s = asymptotic_relation(numax, dnu, eps, alpha, self.nrads)
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
    ---------_
    star : class instance
        Star class instance to perform the fit on. This contains the required
        attributes for the fit, f, s, numax, dnu, teff. All others are derived
        from these, or can optionally be set.
    d02 : float, optional
        Initial guess for the small frequency separation (in muHz) between
        l=0 and l=2.
    alpha : float, optional
        Initial guess for the scale of the second order frequency term in the
        asymptotic relation
    seff : float, optional
        Normalized Teff
    mode_width : float, optional
        Initial guess for the mode width (in log10!) for all the modes that are
        fit.
    env_width : float, optional
        Initial guess for the p-mode envelope width (muHz)
    env_height : float, optional
        Initial guess for the p-mode envelope height

    Attributes
    ----------
    f : array
            Array of frequency bins of the spectrum (muHz)
    s : array
        The power at frequencies f
    numax : float
        Initial guess for numax. Frequency of maximum power of the p-mode
        envelope
    dnu : float
        Initial guess for dnu. Large separation of l=0 modes
    eps : float
        Initial guess for epsilon. Epsilon phase term in asymptotic relation
    d02 : float, optional
        Initial guess for the small frequency separation (in muHz) between
        l=0 and l=2.
    alpha : float, optional
        Initial guess for the scale of the second order frequency term in the
        asymptotic relation
    seff : float, optional
        Normalized Teff
    mode_width : float, optional
        Initial guess for the mode width (in log10!) for all the modes that are
        fit.
    env_width : float, optional
        Initial guess for the p-mode envelope width (muHz)
    env_height : float, optional
        Initial guess for the p-mode envelope height
    mode_ID : dataframe
        Pandas dataframe of the radial order, angular degree and mode frequency
        and error for the modes fit in the asymptotic relation.
    asy_model : tuple
        Tuple containing the frequency (first column) and best-fit spectrum
        model (second colunm). Frequency is truncated to a range around numax
        that contains the requested number of radial orders.
    """

    def __init__(self, star, d02, alpha, mode_width, env_width, env_height,
                 verbose=False):
        self.f = star.f
        self.s = star.s
        self.numax = star.numax
        self.dnu = star.dnu
        self.teff = star.teff
        self.bp_rp = star.bp_rp
        self.epsilon = star.epsilon
        self.d02 = d02
        self.alpha = alpha
        self.mode_width = mode_width
        self.env_width = env_width
        self.env_height = env_height
        self.asy_modeID = {}
        self.asy_model = None
        self.asy_bestfit = {}
        self.verbose = verbose

    def parse_asy_pars(self, verbose=False):
        """ Parse input and initial guesses for the asymptotic relation fit

        Parameters
        ----------
        verbose : bool
            Print the values of the initial guesses that will be used in the
            asymptotic relation fit.

        Returns
        -------
        pars : list
            List of initial guesses for the parameters in the asymptotic
            relation fit.
        """

        if not self.teff:
            self.teff = [4931, 4931] # TODO - hardcode, bad!
        
        if not self.bp_rp:
            self.bp_rp = [1.26, 1.26] # TODO - hardcode, bad!
    
        if not self.epsilon:
            ge_vrard = pb.epsilon()
            self.epsilon = ge_vrard(self.dnu, self.numax, self.teff)

        if not self.d02:
            self.d02 = 0.1*self.dnu[0]

        if not self.alpha:
            self.alpha = 1e-3

        if not self.mode_width:
            self.mode_width = 1e-20  # must be non-zero for walkers' start pos

        if not self.env_width:
            self.env_width = 0.66 * self.numax[0]**0.88

        if not self.env_height:
            df = np.median(np.diff(self.f))
            a = int(np.floor(self.dnu[0]/df))
            b = int(len(self.s) / a)
            smoo = self.s[:a*b].reshape((b, a)).mean(1)
            self.env_height = max(smoo)

        pars = [self.numax[0], self.dnu[0], self.epsilon[0], self.alpha,
                self.d02, self.env_height, self.env_width, self.mode_width,
                self.teff[0], self.bp_rp[0]]

        parsnames = ['numax', 'large separation', 'epsilon', 'alpha', 'd02',
                     'p-mode envelope height', 'p-mode envelope width',
                     'mode width (log10)', 'Teff', 'bp_rp']
        
        if verbose or self.verbose:
            for i in range(len(pars)):
                print('%s: %f' % (parsnames[i], pars[i]))
        return pars

    def run(self, N):
        """ Setup, run and parse the asymptotic relation fit using EMCEE

        Parameters
        ----------
        N : int
            Number of radial orders to fit

        Returns
        -------
        mode_ID : dataframe
            Pandas dataframe of the radial order, angular degree and mode
            frequency and error for the modes fit in the asymptotic relation.
        """
        
        x0 = self.parse_asy_pars()

        # select range around numax to fit
        sel = np.where(np.abs(self.f - self.numax[0]) < N/1.5*self.dnu[0])

        model = asymp_spec_model(self.f[sel], N)

        nsig = 5
        
        bounds = [[self.numax[0]-nsig*self.numax[1], self.numax[0]+nsig*self.numax[1]],  # numax
                  [self.dnu[0]-nsig*self.dnu[1], self.dnu[0]+nsig*self.dnu[1]],  # Dnu
                  [self.epsilon[0]-nsig*self.epsilon[1], self.epsilon[0]+nsig*self.epsilon[1]],  # eps
                  [-1, 1],  # alpha
                  [0.01*self.dnu[0], 0.2*self.dnu[0]],  # d02
                  [self.env_height*0.5, self.env_height*1.5],  # hmax
                  [self.env_width*0.9, self.env_width*1.1],  # Ewidth
                  [-2, 1.0],  # mode width (log10)
                  [0, self.teff[0] + nsig*self.teff[1]], # Teff
                  [self.bp_rp[0] - nsig*self.bp_rp[1], self.bp_rp[0] + nsig*self.bp_rp[1]] # Gaia bp-rp
                  ]  

        gaussian = [(0,0),
                    (0,0),
                    (0,0),
                    (0,0),
                    (0,0),
                    (0,0),
                    (0,0),
                    (0,0),
                    (self.teff[0], self.teff[1]),
                    (self.bp_rp[0], self.bp_rp[1]),
                    ]

        fit = mcmc(self.f[sel], self.s[sel], model, x0, bounds, gaussian)

        self.flatchain = fit()  # do the fit with default settings
        
        self.fit_pars = np.percentile(self.flatchain, [16, 50, 84], axis=0)

        self.asy_model = (model.f, model.model(*self.fit_pars[1,:-2]))

        # Get mode ID and frequency list
        # TODO - is there a better/neater way to do this?
        nu0s = np.empty((fit.niter*fit.nwalkers, N))
        for j in range(fit.niter*fit.nwalkers):
            nu0s[j, :] = asymptotic_relation(*self.flatchain[j, :4], N)

        nu2s = np.array([nu0s[:, i] - self.flatchain[:, 4] for i in range(len(nu0s[0, :]))]).T

        nus_mu = np.median(np.array([nu0s, nu2s]), axis=1)
        nus_std = np.std(np.array([nu0s, nu2s]), axis=1)

        ells = [0 if i % 2 else 2 for i in range(2*len(nus_mu[0, :]))]

        nus_mu_out = []
        nus_std_out = []

        for i in range(len(nus_mu[0, :])):
            nus_mu_out += [nus_mu[1, i], nus_mu[0, i]]
            nus_std_out += [nus_std[1, i], nus_std[0, i]]

        self.asy_modeID = pd.DataFrame({'ell': ells,
                                        'nu_mu': nus_mu_out,
                                        'nu_std': nus_std_out})

        var_names = ['numax','dnu','eps','alpha','d02','env_height', 
                     'env_width','mode_width','teff','bp_rp']
        for j,key in enumerate(var_names):
            self.asy_bestfit[key] = self.fit_pars[:,j]

        return self.asy_modeID


class Prior(pb.epsilon):
    """ Evaluate the proirs on the provided model parameters

    Attributes
    ----------
    bounds : array
        Boundary values for model parameters, beyond which the likelihood
        is -inf
    gaussian : ???
    data_file : str
        File containing stellar parameters for make the prior KDE
    seff_offset : int
        Normalized Teff so that the KDE can have roughly
        the same bandwidth along each axis.
    read_prior_data : function
        Function to read the prior data file. Inherited from epsilon
    make_kde : function
        Function to get a KDE from the prior data file. Inherited from epsilon

    Parameters
    ----------
    bounds : list
        list of upper and lower bounds for the priors on the model parameters
    gaussian : ??? # TODO ... if at all
        ???
    """

    def __init__(self, bounds, gaussian, verbose=False):
        self.bounds = bounds
        self.gaussian = gaussian
        self.verbose = verbose
        self.data_file = os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])
        self.read_prior_data()  # Inherited from epsilon
        self.make_kde()  # Inherited from epsilon

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
        """ Guassian priors - not yet implemented!!!

        Function for setting Gaussian priors (???)

        Parameters
        ----------
        p : array
            Array containing mcmc proposal

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
        observations.

        Parameters
        ----------
        p : array
            Array of model parameters
        Teff: float
        bp_rp: float   

        Returns
        -------
        lp : float
            The prior at p
        """

        if self.pbound(p) == -np.inf:
            return -np.inf

        # Evaluate the prior, defined by a KDE
        # log10(Dnu), log10(numax), log10(Teff), bp_rp, eps
        lp = self.kde.pdf([np.log10(p[1]), np.log10(p[0]), np.log10(p[8]), 
                           p[9], p[3]])
    
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
    model: class instance
        Initialized instance of the model class to use in the fit
    x0 : array
        Initial positions for the MCMC walkers
    bounds: array
        Array of shape (len(x0), 2) of boundary values to use in the fit. Lower
        and upper limits are in the first and second columns respectively. This
        enforces a log-likelihood = -inf if any parameter exceeds these limits.

    Attributes
    ----------
    f : array
        Array of frequency bins of the spectrum (muHz)
    s : array
        The power at frequencies f
    model : class instance
        Initialized instance of the model class to use in the fit
    x0 : array
        Initial positions for the MCMC walkers
    lp : class instance
        Prior class initialized using model parameter limits
    """

    def __init__(self, f, s, model, x0, bounds, gaussian):
        self.f = f
        self.s = s
        self.model = model
        self.x0 = x0
        self.bounds = bounds
        self.ndim = len(x0)
        self.gaussian = gaussian
        self.lp = Prior(self.bounds, self.gaussian)

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
        
        model_pars = p[:-2] # -2 and -1 are Teff and bp_rp hyperparameters
        
        mod = self.model(model_pars)
        like = -1.0 * np.sum(np.log(mod) + self.s / mod)
        return like + logp

    def __call__(self, niter=1000, nwalkers=50, burnin=2000, spread=0.01):
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

        import emcee
        
        # Start walkers in a tight random ball
        p0 = np.array([[np.random.uniform(max(self.bounds[i][0],
                                              self.x0[i]*(1-spread)),
                                          min(self.bounds[i][1],
                                              self.x0[i]*(1+spread))) for i in range(self.ndim)] for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                        self.likelihood)
        print('Burningham')
        sampler.run_mcmc(p0, self.burnin)
        pb = sampler.chain[:, -1, :].copy()
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(pb, self.niter)
        out = sampler.flatchain.copy()
        sampler.reset() # This hopefully minimizes emcee memory leak
        return out
