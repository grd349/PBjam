""" Fitting asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 is 
ignored.     
"""

import numpy as np
from pbjam import epsilon
import os

from . import PACKAGEDIR #I have no idea what this does

class model():
    """ Class for power spectrum model using asymptotic relation
    
    Parameters
    ---------_
    f : float, array 
            Array of frequency bins of the power spectrum (muHz)  
            
    Attributes
    ----------
    f : float, array 
            Array of frequency bins of the power spectrum (muHz)
    """
    
    def __init__(self, f):
        self.f = f

    def lor(self, freq, h, w):
        """ Lorentzian to describe a mode.
        
        Parameters
        ----------
        freq : float
            Frequency of lorentzian (muHz)
        h : float
            Height of the lorentizan (?, spectrum units) 
        w : float
            Full width of the lorentzian (log10(muHz))
        
        Returns
        ------- 
        mode : array
            The mode power distribution in frequency bins
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
            Height of the l=0 (?, spectrum units)
        w : float
            The mode width (identical for l=2 and l=0) (log10(muHz))
        d02 : float
            The small separation (muHz)
        hfac : float, optional
            Ratio of the l=2 height to that of l=0 (unitless)
            
        Returns
        -------
        pair_model : array
            The power distribution in frequency of a mode pair 
            (???, spectrum units)
        """
        
        pair_model = self.lor(freq0, h, w)
        pair_model += self.lor(freq0 - d02, h*hfac, w)
        return pair_model

    def asy(self, numax, dnu, d02, eps, alpha, hmax, Envwidth, w):
        """Constructs a spectrum model from the asymptotic relation
        
        The asymptotic relation for p-modes is defined as:
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
            Gaussian height of p-mode envelope (?, spectrum units) 
        Envwidth : float
            Gaussian width of the p-mode envelope (muHz)
        w : float
            Width of the modes (log_10(muHz))
        
        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
        """
        
        nmax = numax / dnu - eps
        nn = np.arange(np.floor(nmax-5), np.floor(nmax+6), 1) # (hard code)
        model = np.ones(len(self.f)) 
        for n in nn:
            # asymptotic relation
            f0 = (n + eps + alpha/2*(n - nmax)**2) * dnu 
            # Gaussian p-mode envelope
            h = hmax * np.exp(- 0.5 * (f0 - numax)**2 / Envwidth**2) 
            model += self.pair(f0, h, w, d02*dnu)
        return model

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
        
        return self.asy(*p)

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
        
        # ??? is this better/worse, slower/faster?
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
            if all(self.bounds[idx] != 0):
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
                lnprior += -0.5 * (i - self.gaussian[idx][0])**2 / \
                           self.gaussian[idx][1]**2
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
        lp = self.kde([np.log10(p[1]), np.log10(p[0]), p[8], p[3]])
        return lp

class mcmc():  
    """ Class for MCMC sampling 
    
    Fit the asymptotic relation around the p-mode envelope. The MCMC sampler 
    burns-in and then proceeds to sample the posterior distribution. 
    
    Parameters
    ----------
    f : float, array 
        Array of frequency bins of the power spectrum (muHz)        
    snr : array
        The power spectrum normalized to a background noise
        level of 1
    x0 : array
        Initial positions for the MCMC walkers
    
    Attributes
    ----------
    f : float, array 
        Array of frequency bins of the power spectrum (muHz)        
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
    
    def __init__(self, f, snr, x0):      
        self.f = f
        self.snr = snr
        self.sel = np.where(np.abs(self.f - x0[0]) < 3*x0[1])
        self.model = model(f[self.sel])
        # numax, dnu, d02, eps, alpha, hmax, Envwidth, w, seff
        # TODO - tidy this bit up!
        self.seff_offset = 4000.0 #(hard code)
        bounds = [[x0[0]*0.9, x0[0]*1.1], 
                  [x0[1]*0.9, x0[1]*1.1],
                  [0.04, 0.2], 
                  [0.5, 2.5], 
                  [0, 1], 
                  [x0[5]*0.5, x0[5]*1.5],
                  [x0[6]*0.9, x0[6]*1.1], 
                  [-2, 1.0], 
                  [2.0, 4.0]]
        self.lp = Prior(bounds, [(0,0) for n in range(len(x0))])

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
        mod = self.model(p[:-1]) # Last p is seff so ignore.
        like = -1.0 * np.sum(np.log(mod) + \
                          self.snr[self.sel]/mod)
        return like

    def __call__(self, x0, niter=1000, nwalkers=200):
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
        
        # TODO - Swap this with faster sampler?
        import emcee
        ndim = len(x0)
        # Start walkers in a tight random ball
        p0 = [np.array(x0) + np.random.rand(ndim)*1e-3 for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)
        print('Burningham')
        sampler.run_mcmc(p0, niter)
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(p0, niter)
        return sampler.flatchain
