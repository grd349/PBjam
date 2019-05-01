""" Fitting asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 is
ignored.
"""

import numpy as np
import pbjam as pb
import os
import pandas as pd

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
    """ Compute the l=0 mode frequencies from the asymptotic relation for p-modes
    
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
    
    def model(self, numax, dnu, eps, alpha, d02, hmax, envwidth, modewidth):
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

class star():
    
    def __init__(self, f, s, parameters):
        self.f = f
        self.s = s
        self.numax = parameters['numax']
        self.dnu = parameters['dnu']
        self.teff = parameters['teff']
        self.epsilon = None
        self.mode_ID = []
        self.fit_pars = []
        self.asy_model = []
        
        self._d02 = None
        self._alpha = None
        self._seff = None
        self._mode_width = None
        self._env_width = None
        self._env_height = None
        

    def parse_asy_pars(self, verbose = False):
        
        if not self.epsilon:
            ge_vrard = pb.epsilon()
            self.epsilon = ge_vrard(self.dnu, self.numax, self.teff)
        
        if not self._d02:
            self._d02 = 0.1*self.dnu[0]
            
        if not self._alpha:
            self._alpha = 1e-3
        
        if not self._seff:
            # TODO this needs to be done properly
            self._seff = 4000 
        
        if not self._mode_width:
            self._mode_width = 1e-20 # must be non-zero for walkers' start pos
        
        if not self._env_width:
            self._env_width = 0.66 * self.numax[0]**0.88      
        
        if not self._env_height:
            df = np.median(np.diff(self.f))
            a = int(np.floor(self.dnu[0]/df)) 
            b = int(len(self.s) / a)
            smoo = self.s[:a*b].reshape((b,a)).mean(1)
            self._env_height = max(smoo)
        
        pars = [self.numax[0], self.dnu[0], self.epsilon[0], self._alpha, 
                self._d02, self._env_height, self._env_width, self._mode_width,
                self._seff]
        
        parsnames = ['numax', 'large separation', 'epsilon', 'alpha', 'd02', 
                     'p-mode envelope height', 'p-mode envelope width',
                     'mode width (log10)', 'Seff (adjusted Teff)']
        if verbose:
            for i in range(len(pars)):
                print('%s: %f' % (parsnames[i], pars[i]))
                
        return pars
        
        
def asymptotic_fit(star, N = 5):
          
    x0 = star.parse_asy_pars()
            
    sel = np.where(np.abs(star.f - star.numax[0]) < N*star.dnu[0]) # select range around numax to fit
    
    model = asymp_spec_model(star.f[sel], N)
    
    # TODO this should ideally be handled in a neater way, factor 5 is arbitrary, and value errors may not be in [1]
    nsig = 5
    bounds = [[star.numax[0]-nsig*star.numax[1] , star.numax[0]+nsig*star.numax[1]], # numax
              [star.dnu[0]-nsig*star.dnu[1] , star.dnu[0]+nsig*star.dnu[1]], # Dnu
              [star.epsilon[0]-nsig*star.epsilon[1], star.epsilon[0]+nsig*star.epsilon[1]], # eps
              [-1, 1], # alpha
              [0.01*star.dnu[0] , 0.5*star.dnu[0]], # d02
              [star._env_height*0.5, star._env_height*1.5], #hmax
              [star._env_width*0.9 , star._env_width*1.1], #Ewidth
              [-2, 1.0], # mode width (log10)
              [1e2, 1e4]] # seff
    
    fit = mcmc(star.f[sel], star.s[sel], model, x0, bounds)
    
    star.flatchain = fit() # do the fit with default settings
    
    star.fit_pars = np.median(star.flatchain, axis = 0)
    
    star.asy_model = model.model(*star.fit_pars[:-1])
    
    
    # Get mode ID and frequency list
    #TODO - is there a better/neater way to do this?
    nu0s = np.empty((fit.niter*fit.nwalkers, N))
    for j in range(fit.niter*fit.nwalkers):
        nu0s[j,:] = asymptotic_relation(*star.flatchain[j,:4], N)
   
    nu2s = np.array([nu0s[:,i] - star.flatchain[:,4] for i in range(len(nu0s[0,:]))]).T
        
    nus_mu = np.median(np.array([nu0s, nu2s]), axis = 1)
    nus_std = np.std(np.array([nu0s, nu2s]), axis = 1)

    ells = [0 if i%2 else 2 for i in range(2*len(nus_mu[0,:]))]
    
    nus_mu_out = []
    nus_std_out = []    
    
    for i in range(len(nus_mu[0,:])):
        nus_mu_out  += [nus_mu[1,i], nus_mu[0,i]]
        nus_std_out += [nus_std[1,i], nus_std[0,i]]
        
    star.mode_ID = pd.DataFrame({'ell': ells, 'nu_mu': nus_mu_out, 'nu_std': nus_std_out})
        
    return star.mode_ID




class Prior(pb.epsilon):
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
    f : float, array
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

    def __init__(self, f, s, model, x0, bounds):
        self.f = f
        self.s = s
        self.model = model
        self.x0 = x0
        self.bounds = bounds
        self.ndim = len(x0)
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

        #TODO - change self.model(p[:-1]) to self.model(p), x0 needs to be changed
        logp = self.lp(p)
        if logp == -np.inf:
            return -np.inf
        mod = self.model(p[:-1]) # Last p is seff so ignore.
        like = -1.0 * np.sum(np.log(mod) + self.s / mod)
        return like + logp

    def __call__(self, niter=500, nwalkers=200, spread = 0.01):
        """ Initialize and run the EMCEE afine invariant sampler

        Parameters
        ----------
        niter : int
            Number of steps for the walkers to take (both burn-in and sampling).
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

        import emcee
        
        # Start walkers in a tight random ball
        p0 = np.array([[np.random.uniform(max(self.bounds[i][0], self.x0[i]*(1-spread)), 
                                          min(self.bounds[i][1], self.x0[i]*(1+spread))) for i in range(self.ndim)] for i in range(nwalkers)])
                    
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.likelihood)
        print('Burningham')
        sampler.run_mcmc(p0, self.niter)
        pb = sampler.chain[:,-1,:].copy()
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(pb, self.niter)
        return sampler.flatchain    
    
    