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

def env_width_pl(numax):
    return 0.66 * numax ** 0.88

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

    below = np.floor(nmax - np.floor(nrads/2)).astype(int)
    above = np.floor(nmax + np.ceil(nrads/2)).astype(int)
    if type(below) == np.int64:
        return np.arange(below, above)
    else:
        return np.concatenate([np.arange(x, y) for x, y in zip(below, above)]).reshape(-1,nrads)    



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
    return (enns.T + eps + alpha/2*(enns.T - nmax)**2) * dnu
    

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

def get_summary_stats(fit, model, pnames):
        summary = pd.DataFrame()
        smry_stats = ['best','mean','std', 'skew', '2nd', '16th', '50th', '84th', 
                      '97th']
        idx = np.argmax(fit.flatlnlike)       
        means = np.mean(fit.flatchain, axis = 0)
        stds = np.std(fit.flatchain, axis = 0)
        skewness = scist.skew(fit.flatchain, axis = 0)
        pars_percs = np.percentile(fit.flatchain, [0.50-0.954499736104/2,
                                                 0.50-0.682689492137/2,
                                                 0.50,
                                                 0.50+0.682689492137/2,
                                                 0.50+0.954499736104/2], axis=0)
        best = fit.flatchain[idx,:]
        for i, par in enumerate(pnames):
            z = [best[i], means[i], stds[i], skewness[i],  pars_percs[0,i],
                 pars_percs[1,i], pars_percs[2,i], pars_percs[3,i], pars_percs[4,i]]
            A = {key: z[i] for i, key in enumerate(smry_stats)}
            summary[par] = pd.Series(A)
        best_model = model(best)
        return summary, best_model


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

    def model(self, numax, dnu, eps, alpha, d02, hmax, envwidth, modewidth, 
              *args):
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
        *args : array-like
            List of additional parameters (Teff, bp_rp) that aren't actually
            used to construct the spectrum model, but just for evaluating the
            prior.

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
    model : tuple
        Tuple containing the frequency (first column) and best-fit spectrum
        model (second colunm). Frequency is truncated to a range around numax
        that contains the requested number of radial orders.
    """

    def __init__(self, star, d02, alpha, mode_width, env_width, env_height, 
                 store_chains = False, nthreads=1, verbose=False, nrads = 8):
        
        pars = [star.numax, star.dnu, star.epsilon, alpha, d02, env_height,
                env_width, mode_width, star.teff, star.bp_rp,]
        
        self.nthreads = nthreads
        self.verbose = verbose
        self.nrads = nrads
        self.f = star.f
        self.s = star.s
        self.pars_names = ['numax', 'dnu', 'eps', 'alpha', 'd02', 'env_height',
                           'env_width', 'mode_width', 'teff', 'bp_rp']
        self.guess = OrderedDict({pars_names: pars for pars_names, pars in zip(self.pars_names, pars)})
        self.parse_asy_pars() # interpret inputs and/or guess missing vals        
        self.sel = np.where(np.abs(self.f - self.guess['numax'][0]) < self.nrads/1.5*self.guess['dnu'][0])
        self.model = asymp_spec_model(self.f[self.sel], self.nrads)
        self.bounds = self.set_bounds()
        self.gaussian = self.set_gaussian_pars()
        
        self.store_chains = store_chains        
        self.modeID = None
        self.summary = None
        self.flatchains = None
        self.lnlike_fin = None
        self.lnprior_fin = None
        self.best_model = None
        
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
        
        if not self.guess['teff']:
            self.guess['teff'] = [4931, 4931]  # TODO - hardcode, bad!

        if not self.guess['bp_rp']:
            self.guess['bp_rp'] = [1.26, 1.26]  # TODO - hardcode, bad!

        if not self.guess['eps']:
            ge = pb.epsilon(nthreads=self.nthreads)
            self.guess['eps'] = ge(self.guess['dnu'], 
                                   self.guess['numax'], 
                                   self.guess['teff'])  
        if not self.guess['d02']:
            self.guess['d02'] = [0.14*self.guess['dnu'][0]]

        if not self.guess['alpha']:
            self.guess['alpha'] = [0.01]

        if not self.guess['mode_width']:
            self.guess['mode_width'] = [np.log10(0.05 + 0.64 * (self.guess['teff'][0]/5777.0)**17)]

        if not self.guess['env_width']:
            self.guess['env_width'] = [env_width_pl(self.guess['numax'][0])]

        if not self.guess['env_height']:
            df = np.median(np.diff(self.f))
            a = int(np.floor(self.guess['dnu'][0]/df))
            b = int(len(self.s) / a)
            smoo = self.s[:a*b].reshape((b, a)).mean(1)
            self.guess['env_height'] = [max(smoo)]

        if verbose or self.verbose:
            for key in self.pars_names:
                print('%s: %f' % (key, self.guess[key]))
                
    def set_bounds(self, nsig = 5):
        bounds = [[max(1e-20, self.guess['numax'][0]-nsig*self.guess['numax'][1]),  # numax
                   self.guess['numax'][0]+nsig*self.guess['numax'][1]],

                  [max(1e-20, self.guess['dnu'][0]-nsig*self.guess['dnu'][1]),  # Dnu
                   self.guess['dnu'][0]+nsig*self.guess['dnu'][1]],

                  [max(0.4, self.guess['eps'][0]-nsig*self.guess['eps'][1]),  # eps
                   min(1.6, self.guess['eps'][0]+nsig*self.guess['eps'][1])],

                  [1e-20, 0.1],  # alpha

                  [0.05*self.guess['dnu'][0], 0.25*self.guess['dnu'][0]],  # d02

                  [self.guess['env_height'][0]*0.5, self.guess['env_height'][0]*1.5],  # hmax

                  [self.guess['env_width'][0]*0.75, self.guess['env_width'][0]*1.25],  # Ewidth

                  [-2, 1.2],  # mode width (log10)

                  [max(3000.0, self.guess['teff'][0]-nsig*self.guess['teff'][1]),  # Teff
                   min(7800.0, self.guess['teff'][0]+nsig*self.guess['teff'][1])],

                  [self.guess['bp_rp'][0]-nsig*self.guess['bp_rp'][1],
                   self.guess['bp_rp'][0]+nsig*self.guess['bp_rp'][1]]  # Gaia bp-rp
                  ]
        return bounds

    def set_gaussian_pars(self):
        gaussian = [(0, 0),  # numax
                    (0, 0),  # Dnu
                    (0, 0),  # eps
                    (0.015*self.guess['dnu'][0]**-0.32, 0.01),  # alpha
                    (0.14*self.guess['dnu'][0], 0.3*self.guess['dnu'][0]),  # d02
                    (0, 0),  # hmax
                    (0, 0),  # Ewidth
                    (np.log10(0.05 + 0.64 * (self.guess['teff'][0]/5777.0)**17), 0.2),  # mode width (log10)
                    (self.guess['teff'][0], self.guess['teff'][1]),  # Teff
                    (self.guess['bp_rp'][0], self.guess['bp_rp'][1]),  # Gaia bp-rp
                    ]
        return gaussian
           
    def get_modeIDs(self, fit, N):
        # TODO - is there a better/neater way to do this?
        
        flatchain = fit.flatchain

        nsamps = np.shape(flatchain)[0]

        nu0_samps, nu2_samps = np.empty((nsamps, N)), np.empty((nsamps, N))

        nu0_samps = asymptotic_relation(*flatchain[:, :4].T, N)
        nu2_samps = nu0_samps - flatchain[:, 4]
                    
        nus_med = np.median(np.array([nu0_samps, nu2_samps]), axis=2)
        nus_std = np.std(np.array([nu0_samps, nu2_samps]), axis=2)

        ells = [0 if i % 2 else 2 for i in range(2*N)]

        nus_mu_out = []
        nus_std_out = []

        for i in range(N):
            nus_mu_out += [nus_med[1, i], nus_med[0, i]]
            nus_std_out += [nus_std[1, i], nus_std[0, i]]
        
        modeID = pd.DataFrame({'ell': ells,
                               'nu_mu': nus_mu_out,
                               'nu_std': nus_std_out})
        return modeID
    

    
    def run(self):
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
        fit = mcmc(self.f[self.sel], self.s[self.sel], self.model, self.guess, 
                   self.pars_names, self.bounds, self.gaussian, nthreads=self.nthreads)

        fit()  # do the fit with default settings

        self.modeID = self.get_modeIDs(fit, self.nrads)

        self.summary, self.best_model = get_summary_stats(fit, self.model, self.pars_names)

        if self.store_chains:
            self.flatchain = fit.flatchain
            self.lnlike_fin = fit.flatlnlike
        else:
            self.flatchain = fit.chain[:,-1,:]
            self.lnlike_fin = np.array([fit.likelihood(fit.chain[i,-1,:]) for i in range(fit.nwalkers)])
            self.lnprior_fin = np.array([fit.lp(fit.chain[i,-1,:]) for i in range(fit.nwalkers)])



        return self.modeID


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

        lp = np.log(self.kde.pdf([np.log10(p[1]), np.log10(p[0]), 
                                  np.log10(p[8]), p[9], p[3]]))
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

    def __init__(self, f, s, model, guess, pars_names, bounds, gaussian, 
                 nthreads=1):
        self.f = f
        self.s = s
        self.model = model
        self.ndim = len(pars_names)
        self.pars_names = pars_names
        self.guess = guess
        self.bounds = bounds
        self.gaussian = gaussian
        self.lp = Prior(self.bounds, self.gaussian)
        self.nthreads = nthreads        
        self.chains = None


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

    def __call__(self, niter=10, nwalkers=50, burnin=20, spread=0.01, 
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

#        mus    = [self.guess[key][0] for key in self.pars_names]
#        sigmas = [self.guess[key][1] for key in self.pars_names]

        # Start walkers in a tight random ball
        p0 = np.array([[np.random.uniform(max(self.bounds[i][0],
                                              self.guess[key][0]*(1-spread)),
                                          min(self.bounds[i][1],
                                              self.guess[key][0]*(1+spread))) for i, key in enumerate(self.guess.keys())] for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                        self.likelihood, threads=self.nthreads)
        print('Burningham')
        pos, prob, state = sampler.run_mcmc(p0, self.burnin)
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(pos, self.niter)
        self.chain = sampler.chain.copy()
        self.flatchain = sampler.flatchain
        self.lnlike = sampler.lnprobability
        self.flatlnlike = sampler.flatlnprobability
        sampler.reset()  # This hopefully minimizes emcee memory leak
        
        
