""" Fitting asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 is
ignored.
"""

import numpy as np
from pbjam import epsilon
import os
import pandas as pd

from . import PACKAGEDIR


def asymptotic_relation(numax, dnu, eps, alpha, nrads):
    nmax = get_nmax(numax, dnu, eps)    
    enns = get_enns(nmax, nrads)
    return (enns + eps + alpha/2*(enns - nmax)**2) * dnu

def P_envelope(f0, hmax, numax, width):
    return hmax * np.exp(- 0.5 * (f0 - numax)**2 / width**2)

def get_nmax(numax, dnu, eps):
    return numax / dnu - eps

def get_enns(nmax, nrads):
    # TODO - why +1?
    return np.arange(np.floor(nmax-np.floor(nrads/2)), np.floor(nmax+np.ceil(nrads/2)+1), 1)

class asymp_spec_model():
    """ Class for SNR spectrum model using asymptotic relation

    Parameters
    ---------_
    f : float, array
            Array of frequency bins of the SNR spectrum (muHz)

    Attributes
    ----------
    f : float, array
            Array of frequency bins of the SNR spectrum (muHz)
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
    
    def model(self, numax, dnu, eps, alpha, d02, hmax, Envwidth, w):
        """Constructs a spectrum model from the asymptotic relation

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
        d02 : float
            Small separation (units of dnu)
        eps : float
            Phase term of the asymptotic relation (unitless)
        alpha : float
            Curvature of the asymptotic relation (unitless)
        hmax : float
            Gaussian height of p-mode envelope (SNR)
        Envwidth : float
            Gaussian width of the p-mode envelope (muHz)
        w : float
            Width of the modes (log_10(muHz))

        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
        """

        f0s = asymptotic_relation(numax, dnu, eps, alpha, self.nrads)

        Hs = P_envelope(f0s, hmax, numax, Envwidth)
        
        mod = np.ones(len(self.f))
        for n in range(len(f0s)):
            mod += self.pair(f0s[n], Hs[n], w, d02*dnu)
            
        return mod

    def __call__(self, p):
        """Produce model of the asymptotic relation

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



def asymptotic_fit(f, snr, numax, dnu, epsilon, N = 5, d02 = None, mwidth = 1, alpha = 1e-3, eheight = 10, ewidth = None, seff = 4000):

    if not ewidth:
        ewidth = 0.66 * numax[0]**0.88                
        
    if not d02:
        d02 = 0.1 # In units of Dnu
    else:
        d02 = d02/dnu[0]
    
    x0 = [numax[0], dnu[0], epsilon[0], alpha, d02, eheight, ewidth, mwidth, seff]
        
    sel = np.where(np.abs(f - x0[0]) < N*x0[1]) # select range around numax to fit
            
    model = asymp_spec_model(f[sel], N)
    
    
    bounds = [[numax[0]-5*numax[1]    , numax[0]+5*numax[1]], # numax
              [dnu[0]-5*dnu[1]        , dnu[0]+5*dnu[1]], # Dnu
              [epsilon[0]-5*epsilon[1], epsilon[0]+5*epsilon[1]], # eps
              [0, 1], # alpha
              [0.04, 0.2], # d02
              [x0[5]*0.5, x0[5]*1.5], #hmax
              [x0[6]*0.9, x0[6]*1.1], #Ewidth
              [-2, 1.0], # mode width (log10)
              [1e2, 1e4]] # seff
    
    fit = mcmc(f[sel], snr[sel], model, x0, bounds)
    
    flatchain = fit() # do the fit with default settings    
    
    # Get mode ID and frequency list
    #TODO - is there a better way to do this?
    nu0s = np.empty((fit.niter*fit.nwalkers, N+1))
    for j in range(fit.niter*fit.nwalkers):
        nu0s[j,:] = asymptotic_relation(*flatchain[j,:4], N)
   
    nu2s = np.array([nu0s[:,i] - flatchain[:,4]*flatchain[:,1] for i in range(len(nu0s[0,:]))]).T
        
    nus_mu = np.median(np.array([nu0s, nu2s]), axis = 1)
    nus_std = np.std(np.array([nu0s, nu2s]), axis = 1)

    ells = [2 if i%2 else 0 for i in range(2*len(nus_mu[0,:]))]
    
    nus_mu_out = []
    nus_std_out = []    
    
    for i in range(len(nus_mu[0,:])):
        nus_mu_out  += [nus_mu[0,i] , nus_mu[1,i]]
        nus_std_out += [nus_std[0,i], nus_std[1,i]]
        
    return pd.DataFrame({'ell': ells, 'nu_mu': nus_mu_out, 'nu_std': nus_std_out})




class Prior(epsilon):
    """ Evaluate the proirs on the provided model parameters

    Attributes
    ----------
    bounds : array
        Boundary values for model parameters, beyond which the likelihood
        is -inf
    gaussian : ??
    data_file : str
        File containing stellar parameters for make the prior KDE
    seff_offset : int
        Normalization constant for Teff so that the KDE can have roughly
        the same bandwidth along each axis.
    read_prior_data : function
        Function to read the prior data file. Inherited from epsilon
    make_kde : function
        Function to get a KDE from the prior data file. Inherited from epsilon

    Parameters
    ----------
    bounds : list
        list of upper and lower bounds for the priors on the model parameters
    gaussian : ???
        ???
    """

    def __init__(self, bounds, gaussian):
        self.bounds = bounds
        self.gaussian = gaussian
        self.data_file = os.path.join(*[PACKAGEDIR,'data','rg_results.csv'])
        self.seff_offset = 4000.0 #(hard code)
        self.read_prior_data() # Inherited from epsilon
        self.make_kde() # Inherited from epsilon

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
            ???
        """

        lnprior = 0.0
        for idx, i in enumerate(p):
            if self.gaussian[idx][1] != 0:
                lnprior += -0.5 * (i - self.gaussian[idx][0])**2 / self.gaussian[idx][1]**2
        return lnprior

    def __call__(self, p):
        """ Evaluate the priors for a set of parameters

        The prior is estimated by a KDE of a set of previous Kepler
        observations.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        lp : float
            The prior at p
        """

        if self.pbound(p) == -np.inf:
            return -np.inf

        # Evaluate the prior, defined by a KDE
        # log10(Dnu), log10(numax), log10(Seff), eps
        lp = self.kde([np.log10(p[1]), np.log10(p[0]), np.log10(p[8]), p[3]])
        return lp
    
class mcmc():
    """ Class for MCMC sampling

    Fit the asymptotic relation around the p-mode envelope. The MCMC sampler
    burns-in and then proceeds to sample the posterior distribution.

    Parameters
    ----------
    f : float, array
        Array of frequency bins of the SNR spectrum (muHz)
    snr : array
        The power spectrum normalized to a background noise
        level of 1
    x0 : array
        Initial positions for the MCMC walkers

    Attributes
    ----------
    f : float, array
        Array of frequency bins of the SNR spectrum (muHz)
    snr : array
        The power spectrum normalized to a background noise
        level of 1
    sel : array
        Boolean array to select the frequency range immediately around the
        p-mode envelope (so we don't fit the whole damn spectrum)
    model : class instance
        Model class instance initalized with frequency range around the p-mode
        envelope
    seff_offset : int
        Normalization constant for Teff so that the KDE can have roughly
        the same bandwidth along each axis.
    lp : class instance
        Prior class initialized using model parameter limits
    """

    def __init__(self, f, s, model, x0, bounds):
        self.f = f
        self.s = s
        self.model = model
        self.x0 = x0
        self.bounds = bounds
        
        self.ndim = len(x0)
        self.lp = Prior(bounds, [(0,0) for n in range(len(x0))]) # init prior instance
       

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

        #TODO - change self.model(p[:-1]) to self.model(p), x0 needs to be changed
        logp = self.lp(p)
        if logp == -np.inf:
            return -np.inf
        mod = self.model(p[:-1]) # Last p is seff so ignore.
        like = -1.0 * np.sum(np.log(mod) + self.s / mod)
        return like + logp

    def __call__(self, niter=500, nwalkers=200, spread = 0.01):
        """ Initialize the EMCEE afine invariant sampler

        Parameters
        ----------
        x0 : array
            Initial starting location for MCMC walkers
        niter : int
            number of steps for the walkers to take (both burn-in and sampling)
        nwalkers : int
            number of walkers to use

        Returns
        -------
        sampler.flatchain : array
            the chain of (nwalkers, niter, ndim) flattened to
            (nwalkers*niter, ndim) (???)
        """

        self.niter = niter
        self.nwalkers = nwalkers

        import emcee
        
        # Start walkers in a tight random ball
        p0 = np.array([[np.random.uniform(max(self.bounds[i][0], self.x0[i]*(1-spread)), 
                                          min(self.bounds[i][1], self.x0[i]*(1+spread))) for i in range(self.ndim)] for i in range(nwalkers)])
        
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.likelihood)
        print('Burningham')
        sampler.run_mcmc(p0, niter)
        pb = sampler.chain[:,-1,:].copy()
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(pb, niter)
        return sampler.flatchain    
    
    