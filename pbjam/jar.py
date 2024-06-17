"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os, jax, json, dynesty, warnings
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
from functools import partial
import scipy.special as sc
import scipy.integrate as si
from dataclasses import dataclass
import pandas as pd
from dynesty import utils as dyfunc
import pbjam.distributions as dist

class generalModelFuncs():

    def __init__(self):
        pass

    



    def chi_sqr(self, mod):
        """ Chi^2 2 dof likelihood

        Evaulates the likelihood of observing the data given the model.

        Parameters
        ----------
        mod : jax device array
            Spectrum model.

        Returns
        -------
        L : float
            Likelihood of the data given the model
        """

        L = -jnp.sum(jnp.log(mod) + self.s / mod)

        return L 
 
    def envelope(self, nu, env_height, numax, env_width, **kwargs):
        """ Power of the seismic p-mode envelope
    
        Computes the power at frequency nu in the oscillation envelope from a 
        Gaussian distribution. Used for computing mode heights.
    
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
 
        return gaussian(nu, 2*env_height, numax, env_width)

    @partial(jax.jit, static_argnums=(0))
    def lnlikelihood(self, theta):
        """
        Calculate the log likelihood of the model given parameters and data.
        
        Parameters
        ----------
        theta : numpy.ndarray
            Parameter values.
        nu : numpy.ndarray
            Array of frequency values.

        Returns
        -------
        float :
            Log-likelihood value.
        """

        thetaU = self.unpackParams(theta)
        
        lnlike = self.addAddObsLike(thetaU)  

        # Constraint from the periodogram 

        mod = self.model(thetaU)
      
        lnlike +=  self.chi_sqr(mod)  
         
        return lnlike 
    
    @partial(jax.jit, static_argnums=(0,))
    def obsOnlylnlikelihood(self, theta):

        thetaU = self.unpackParams(theta)
    
        lnlike = self.addAddObsLike(thetaU)

        return lnlike
    
    def addAddObsLike(self, theta_u):
        """ Add the additional probabilities to likelihood
        
        Adds the additional observational data likelihoods to the PSD likelihood.

        Parameters
        ----------
        p : list
            Sampling parameters.

        Returns
        -------
        lnp : float
            The likelihood of a sample given the parameter PDFs.
        """

        lnp = 0

        for key in self.addObs.keys():       
            lnp += self.addObs[key].logpdf(theta_u[key]) 
 
        return lnp
    
    def setAddObs(self, keys):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 
        """
        
        self.addObs = {}

        for key in keys:
            self.addObs[key] = dist.normal(loc=self.obs[key][0], 
                                           scale=self.obs[key][1])

    def setLabels(self, addPriors, modelParLabels):
        """
        Set parameter labels and categorize them based on priors.

        Parameters
        ----------
        priors : dict
            Dictionary containing prior information for specific parameters.

        Notes
        -----
        - Initializes default PCA and additional parameter lists.
        - Checks if parameters are marked for PCA and not in priors; if so, 
            adds to PCA list.
        - Otherwise, adds parameters to the additional list.
        - Combines PCA and additional lists to create the final labels list.
        - Identifies parameters that use a logarithmic scale and adds them to 
            logpars list.
        """

        with open("pbjam/data/parameters.json", "r") as read_file:
            availableParams = json.load(read_file)
        
        self.variables = {key: availableParams[key] for key in modelParLabels}

        # Default PCA parameters       
        self.pcalabels = []
        
        # Default additional parameters
        self.addlabels = []
        
        # If key appears in priors dict, override default and move it to add. 
        for key in self.variables.keys():
            if self.variables[key]['pca'] and (key not in addPriors.keys()):
                self.pcalabels.append(key)

            else:
                self.addlabels.append(key)

        # Parameters that are in log10
        self.logpars = []
        for key in self.variables.keys():
            if self.variables[key]['log10']:
                self.logpars.append(key)

    def unpackSamples(self, samples=None):
        """
        Unpack a set of parameter samples into a dictionary of arrays.

        Parameters
        ----------
        samples : array-like
            A 2D array of shape (n, m), where n is the number of samples and 
            m is the number of parameters.

        Returns
        -------
        S : dict
            A dictionary containing the parameter values for each parameter 
            label.

        Notes
        -----
        This method takes a 2D numpy array of parameter samples and unpacks each
        sample into a dictionary of parameter values. The keys of the dictionary 
        are the parameter labels and the values are 1D numpy arrays containing 
        the parameter values for each sample.

        Examples
        --------
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.labels = ['a', 'b', 'c']
        ...     def unpackParams(self, theta):
        ...         return {'a': theta[0], 'b': theta[1], 'c': theta[2]}
        ...     def unpackSamples(self, samples):
        ...         S = {key: np.zeros(samples.shape[0]) for key in self.labels}
        ...         for i, theta in enumerate(samples):
        ...             theta_u = self.unpackParams(theta)
        ...             for key in self.labels:
        ...                 S[key][i] = theta_u[key]
        ...         return S
        ...
        >>> model = MyModel()
        >>> samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> S = model.unpackSamples(samples)
        >>> print(S)
        {'a': array([1., 4., 7.]), 'b': array([2., 5., 8.]), 'c': array([3., 6., 9.])}
        """

        if samples is None:
            samples = self.samples

        S = {key: np.zeros(samples.shape[0]) for key in self.pcalabels + self.addlabels}
        
        for i, theta in enumerate(samples):
        
            thetaU = self.unpackParams(theta)
             
            for key in thetaU.keys():
                
                S[key][i] = thetaU[key]
            
        return S

    def testModel(self):
        
        u = np.random.uniform(0, 1, self.ndims)
        
        theta = self.ptform(u)
        
        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u,)
        
        return self.f, m
 
def modeUpdoot(result, sample, key, Nmodes):
    
    result['summary'][key] = np.hstack((result['summary'][key], np.array([smryStats(sample[:, j]) for j in range(Nmodes)]).T))

    result['samples'][key] = np.hstack((result['samples'][key], sample))


class DynestySamplingTools():
    
    def __init__(self):
        """ Generic dynesty sampling methods to be inherited.
        
        The inheriting class must have a callable lnlikelihood function, a
        dictionary of callable prior ppf functions, and an integer ndims 
        attribute.
        
        """
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        """
        Transform a set of random variables from the unit hypercube to a set of 
        random variables distributed according to specified prior distributions.

        Parameters
        ----------
        u : jax device array
            Set of pionts distributed randomly in the unit hypercube.

        Returns
        -------
        theta : jax device array
            Set of random variables distributed according to specified prior 
            distributions.

        Notes
        -----
        This method uses the inverse probability integral transform 
        (also known as the quantile function or percent point function) to 
        transform each element of `u` using the corresponding prior 
        distribution. The resulting transformed variables are returned as a 
        JAX device array.

        Examples
        --------
        >>> from scipy.stats import uniform, norm
        >>> import jax.numpy as jnp
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.priors = {'a': uniform(loc=0.0, scale=1.0), 'b': norm(loc=0.0, scale=1.0)}
        ...     def ptform(self, u):
        ...         theta = jnp.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.priors.keys())])
        ...         return theta
        ...
        >>> model = MyModel()
        >>> u = jnp.array([0.5, 0.8])
        >>> theta = model.ptform(u)
        >>> print(theta)
        [0.5        0.84162123]
        """

        theta = jnp.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.priors.keys())])

        return theta
 
    def getInitLive(self, ndims, nlive, nliveMult=4, logl_kwargs={}, **kwargs):
        
        # TODO put in a check for output dims consistent with nlive.
         
        u = np.random.uniform(0, 1, size=(nliveMult * nlive, ndims))

        v = np.array([self.ptform(u[i, :]) for i in range(u.shape[0])])
         
        L = np.array([self.lnlikelihood(v[i, :], **logl_kwargs) for i in range(u.shape[0])])

        idx = np.isfinite(L)
                
        return [u[idx, :][:nlive, :], v[idx, :][:nlive, :], L[idx][:nlive]]
        
    def testLikelihoood(self, logl_kwargs={}):
        
        u = jnp.zeros(self.ndims) + 0.5
        
        theta = self.ptform(u)
                
        logL = self.lnlikelihood(theta, **logl_kwargs)
        
        assert jnp.isreal(logL)
    
    def runDynesty(self, dynamic=False, progress=True, minSamples=5000, logl_kwargs={}, 
                   sampler_kwargs={}):
         
        #if not hasattr(self, 'ndims'):
        ndims = len(self.priors)
        
        self.testLikelihoood(logl_kwargs=logl_kwargs)

        skwargs = sampler_kwargs.copy()

        # According to the Dynesty docs 50 * ndims is a good estimate of nlive
        if 'nlive' not in skwargs.keys():
            skwargs['nlive'] = 50*self.ndims

        # rwalk seems to perform best out of all the sampling methods...
        if 'sample' not in skwargs.keys():
            skwargs['sample'] = 'rwalk'

        # Set the initial locations of live points based on the prior.
        if 'live_points' not in skwargs.keys() and not dynamic:
            skwargs['live_points'] = self.getInitLive(ndims, logl_kwargs=logl_kwargs, **skwargs)
         
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.lnlikelihood, 
                                                   self.ptform, 
                                                   ndims,  
                                                   **skwargs,
                                                   logl_kwargs=logl_kwargs,
                                                   )
            
            sampler.run_nested(print_progress=progress, 
                               wt_kwargs={'pfrac': 1.0}, 
                               dlogz_init=1e-3 * (skwargs['nlive'] - 1) + 0.01, 
                               nlive_init=skwargs['nlive'])  
            
            _nsamples = sampler.results.niter

            if _nsamples < minSamples:     
                missingSamples = minSamples-_nsamples

                sampler.run_nested(dlogz=1e-9, print_progress=progress, save_bounds=False, maxiter=missingSamples)

        else:           
            sampler = dynesty.NestedSampler(self.lnlikelihood, 
                                            self.ptform, 
                                            ndims,  
                                            **skwargs,
                                            logl_kwargs=logl_kwargs,
                                            )
            
            sampler.run_nested(print_progress=progress, 
                               save_bounds=False,)

            _nsamples = sampler.results.niter + sampler.results.nlive
            
            if _nsamples < minSamples:
                missingSamples = minSamples-_nsamples

                sampler.run_nested(dlogz=1e-9, print_progress=progress, save_bounds=False, maxiter=missingSamples)
 
        result = sampler.results

        self.unweighted_samples, self.weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        self.samples = dyfunc.resample_equal(self.unweighted_samples, self.weights)
 
        self.nsamples = self.samples.shape[0]

        self.logz = result.logz
        
        self.logwt = result.logwt

        sampler.reset()

        del sampler

        return self.samples, self.logz



#@jax.jit
def visell1(emm, inc):
    """ l=1, m=0, 1"""
    y = jax.lax.cond(emm == 0, 
                     lambda : jnp.cos(inc)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1, 
                                           lambda : 0.5*jnp.sin(inc)**2, # m = 1
                                           lambda : jnp.nan # m > 1
                                           ))
                    
    return y

#@jax.jit
def visell2(emm, inc):
    """ l=1, m=0, 1, 2"""
    y = jax.lax.cond(emm == 0, 
                     lambda : 1/4 * (3 * jnp.cos(inc)**2 - 1)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1,
                                           lambda : 3/8 * jnp.sin(2 * inc)**2, # m = 1
                                           lambda : jax.lax.cond(emm == 2, 
                                                                 lambda : 3/8 * jnp.sin(inc)**4, # m = 2
                                                                 lambda : jnp.nan # m > 2
                                                                 )))
    return y

#@jax.jit
def visell3(emm, inc):
    """ l=1, m = 0, 1, 2, 3"""
    y = jax.lax.cond(emm == 0, 
                     lambda : 1/64 * (5 * jnp.cos(3 * inc) + 3 * jnp.cos(inc))**2, # m = 0
                     lambda : jax.lax.cond(emm == 1,
                                           lambda : 3/64 * (5 * jnp.cos(2 * inc) + 3)**2 * jnp.sin(inc)**2, # m =1
                                           lambda : jax.lax.cond(emm == 2,
                                                                 lambda : 15/8 * jnp.cos(inc)**2 * jnp.sin(inc)**4, # m = 2
                                                                 lambda : jax.lax.cond(emm == 3, 
                                                                                       lambda : 5/16 * jnp.sin(inc)**6, # m = 3
                                                                                       lambda : np.nan # m > 3
                                                                                       ))))
    return y

#@jax.jit
def visibility(ell, m, inc):

    emm = abs(m)

    y = jax.lax.cond(ell == 0, 
                     lambda : 1.,
                     lambda : jax.lax.cond(ell == 1,
                                           lambda : visell1(emm, inc),
                                           lambda : jax.lax.cond(ell == 2,
                                                                 lambda : visell2(emm, inc),
                                                                 lambda : jax.lax.cond(ell == 3,
                                                                                       lambda : visell3(emm, inc),
                                                                                       lambda : jnp.nan))))
    return y 

def updatePrior(ID, R, addObs):
    
    prior = pd.read_csv('pbjam/data/prior_data.csv')

    badkeys = ['freq', 'height', 'width', 'teff', 'bp_rp', 'nurot_c', 'inc', 'H3_power', 'H3_nu', 'H3_exp', 'shot']

    r = {key: [R[key][0]] for key in R.keys() if key not in badkeys}

    for key in r.keys():
        if key in ['eps_p', 'eps_g', 'bp_rp', 'H1_exp', 'H2_exp']:
            continue
        else:
            r[key] = np.log10(r[key])

    r['ID'] = ID
    r['teff'] = np.log10(addObs['teff'][0])
    r['bp_rp'] = addObs['bp_rp'][0]

    row = pd.DataFrame.from_dict(r)
     
    prior = prior.append(row, ignore_index=True)

    return prior

@dataclass
class constants:
    
    Teff0: float = 5777 # K
    TeffRed0: float = 8907 # K
    numax0: float = 3090 # muHz
    Delta_Teff: float = 1550 # K
    Henv0: float = 0.1 # ppm^2/muHz
    nu_to_omega: float = 2 * jnp.pi / 1e6 # radians/muHz
    dnu0: float = 135.9 # muHz
    logg0 : float = 4.43775 # log10(2.74e4)

def smryStats(y):

    p = np.array([0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-1, 2)])[::-1]*100
     
    u = np.percentile(y, p)
    
    return np.array([u[1], np.mean(np.diff(u))])

@jax.jit
def attenuation(f, nyq):
    """ The sampling attenuation

    Determine the attenuation of the PSD due to the discrete sampling of the
    variability.

    Parameters
    ----------
    f : np.array
        Frequency axis of the PSD.
    nyq : float
        The Nyquist frequency of the observations.

    Returns
    -------
    eta : np.array
        The attenuation at each frequency.
    """

    eta = jnp.sinc(0.5 * f/nyq)

    return eta

@jax.jit
def lor(nu, nu0, h, w):
    """ Lorentzian to describe an oscillation mode.

    Parameters
    ----------
    nu0 : float
        Frequency of lorentzian (muHz).
    h : float
        Height of the lorentizan (SNR).
    w : float
        Full width of the lorentzian (muHz).

    Returns
    -------
    mode : ndarray
        The SNR as a function frequency for a lorentzian.
    """

    return h / (1.0 + 4.0/w**2*(nu - nu0)**2)

def getCurvePercentiles(x, y, cdf=None, percentiles=None):
    """ Compute percentiles of value along a curve

    Computes the cumulative sum of y, normalized to unit maximum. The returned
    percentiles values are where the cumulative sum exceeds the requested
    percentiles.

    Parameters
    ----------
    x : array
        Support for y.
    y : array
        Array
    percentiles: array

    Returns
    -------
    percs : array
        Values of y at the requested percentiles.
    """
    if percentiles is None:
        percentiles = [0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-2, 3)][::-1]

    y /= np.trapz(y, x)
  
    if cdf is None:
        cdf = si.cumtrapz(y, x, initial=0)
        cdf /= cdf.max()  
         
    percs = np.zeros(len(percentiles))
     
    for i, p in enumerate(percentiles):
        
        q = x[cdf >= p]
          
        percs[i] = q[0]

    return np.sort(percs)

class jaxInterp1D():
 
    def __init__(self, xp, fp, left=None, right=None, period=None):
        """ Replacement for scipy.interpolate.interp1d in jax
    
        Wraps the jax.numpy.interp in a callable class instance.

        Parameters
        ----------
        xp : jax device array 
            The x-coordinates of the data points, must be increasing if argument
             period is not specified. Otherwise, xp is internally sorted after 
             normalizing the periodic boundaries with xp = xp % period.

        fp : jax device array 
            The y-coordinates of the data points, same length as xp.

        left : float 
            Value to return for x < xp[0], default is fp[0].

        right: float 
            Value to return for x > xp[-1], default is fp[-1].

        period : float 
            A period for the x-coordinates. This parameter allows the proper 
            interpolation of angular x-coordinates. Parameters left and right 
            are ignored if period is specified.
        """

        self.xp = xp

        self.fp = fp
        
        self.left = left
        
        self.right = right
        
        self.period = period

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        """ Interpolate onto new axis

        Parameters
        ----------
        x : jax device array 
            The x-coordinates at which to evaluate the interpolated values.

        Returns
        -------
        y : jax device array
            The interpolated values, same shape as x.
        """

        return jnp.interp(x, self.xp, self.fp, self.left, self.right, self.period)

class scalingRelations():
    """ Container for scaling relations

    This is a helper class which contains methods for the various scaling
    relations.

    """

    def __init_(self):
        pass

    @staticmethod
    @jax.jit
    def envHeight(Amax, V, dilution, eta, numax=None, dnu=None, alpha=0.791):
        """ Envelope height

        Scaling relation for the p-mode envelope height. 

        Parameters
        ----------
        Amax : float
            Amplitude of the p-modes for a notinoal radial mode at numax. 
        V : float
            Visibility scaling of the oscillations. This is instrument dependent.
        dilution : float
            Fraction of the flux due to the source, compared to the overall 
            flux in the aperature. Also known as wash-out.
        eta : float
            Attenuation of the continuous signal due to discrete sampling.
        numax : float
            Frequency of maximum power of the p-mode envelope in muHz, by 
            default None.
        dnu : float, optional
            Large separation of the target, by default None
        alpha : float, optional
            Exponent of the dnu/numax scaling relation. Default is 0.791

        Returns
        -------
        Henv : float
            Height of the p-mode envelope at numax.
        """
         
        I = eta * dilution * V

        if numax is not None:
            Henv = constants.Henv0 * I**2 * (numax/constants.numax0)**-alpha * Amax**2

        elif dnu is not None:
            Henv = constants.Henv0 * I**2 * (dnu/constants.dnu0) * Amax**2

        else:
            raise ValueError('Must provide either dnu or numax.')
        
        return Henv

    @staticmethod
    @jax.jit
    def envWidth(numax, Teff=0, Tefflim=5600):
        """ Scaling relation for the envelope width

        Full width at half maximum.

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope in muHz.
        Teff : float
            Stellar effective temperature in K.
        Tefflim : float
            Limit for adding the Teff dependence in K. Default is 5600 K.

        Returns
        -------
        width : float
            Envelope width in muHz

        """
        
        T = jax.lax.lt(Teff, Tefflim)
        
        width = jax.lax.cond(T, lambda numax, teff : 0.66 * numax**0.88,
                                lambda numax, teff : 0.66 * numax**0.88 * (1 + (teff - constants.Teff0) * 6e-4), numax, Teff)
            
        return width

    @staticmethod
    @jax.jit
    def nuHarveyGran(numax):
        """ Harvey frequency for granulation term

        Scaling relation for the characteristic frequency of the granulation
        noise. Based on Kallinger et al. (2014).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.

        Returns
        -------
        nu : float
            Characteristic frequency of Harvey law for granulation.

        """

        nu = 0.317 * numax**0.970

        return nu
    
    @staticmethod
    @jax.jit
    def nuHarveyEnv(numax):
        """ Harvey frequency for envelope term

        Scaling relation for the characteristic frequency of the envelope
        noise. Based on Kallinger et al. (2014).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.

        Returns
        -------
        nu : float
            Characteristic frequency of Harvey law for envelope.

        """

        nu = 0.948 * numax**0.992

        return nu
    
    @staticmethod
    @jax.jit
    def dnuScale(numax=None, rho=None, R=None, Teff=None, gamma=0.0, p=jnp.array([0.79101684, -0.63285292])):
        """ Scaling relation dnu

        Computes an estimate of the large separation from scaling relations.
         
        Default is to use the mean density rho of the star. Otherwise if R
        Teff and numax are given instead, it will use these.

        Finally if only numax is given dnu is evaluated based on the polynomial
        relation between log(dnu) and log(numax). This estimated based on a 
        polynomial fit performed on a set of main-sequence and sub- giant stars 
        in the literature.

        The output may be scaled by a factor gamma, e.g., for setting credible
        intervals.

        Parameters
        ----------
        numax : float
            Value(s) at which to compute dnu in muHz.
        rho : float
            Mean density of the star in solar units.
        Teff : float
            Stellar effective temperature in K.
        R : float
            Stellar radius in solar radii.
        gamma : float, optional
            Scaling factor to apply. Only applies to the polynomial scaling 
            relation. Default is 0.
        p : array-like, optional
            Polynomial coefficients to use for log(dnu), log(numax), starting 
            with the coefficient of the Nth order term and ending at the bias 
            term.

        Returns
        -------
        dnu : float 
            Estimate of the large separation dnu in muHz.
        """
        
        if rho is not None:
            dnu = jnp.sqrt(rho) * constants.dnu0

        elif (R is not None) and (Teff is not None) and (numax is not None):
            dnu = jnp.sqrt(R * (Teff / constants.Teff0)**(-1/2) * (numax/constants.numax0)**(-1))

        else:
            dnu = 10**(jnp.polyval(p, jnp.log10(numax), unroll=128) + gamma)

        return dnu
    
    @staticmethod
    @jax.jit
    def numaxScale(Teff=None, R=None, dnu=None, logg=None, alpha=0.791):
        """ Scaling relation numax

        Compute numax from scaling relations given some combination of 
        parameters.

        The default is to use log(g), and if not available to use dnu and R. If
        only R is given, it uses the mass-less approximation and if only dnu
        is given the power-law relation. 

        Parameters
        ----------
        Teff : float
            Stellar effective temperature in K.
        R : float
            Stellar radius in solar radii.
        dnu : float, optional
            Large frequency separation in muHz, by default None.
        logg : float, optional
            logg for the star in log(cm/s**2), by default None.
        alpha : float, optional
            Exponent to use in the dnu \propto numax**n relation. Default is
            0.791.

        Returns
        -------
        numax : float
            Frequency of maximum power of the p-mode envelope in muHz.

        Raises
        ------
        ValueError
            If none of the combination of parameters match.
        """

        if logg is not None:
            numax = (Teff / constants.Teff0)**(-1/2) * 10**(logg - constants.logg0) * constants.numax0
            
        elif (dnu is not None) and (R is not None):
            numax = (Teff / constants.Teff0)**(-1/2) * R * (dnu/constants.dnu0)**-2 * constants.numax0
            
        elif (dnu is None) and (R is not None):
            A = 0.5/(0.5 - alpha)

            B = -0.25/(0.5 - alpha)

            numax = (Teff / constants.Teff0)**B * R**A * constants.numax0

        elif (dnu is not None) and (R is None) and (Teff is None):
            numax = (dnu/constants.dnu0)**(1/alpha) * constants.numax0
            
        else:
            raise ValueError('Must provide logg, or dnu and/or R to compute numax')

        return numax
    
    @staticmethod
    @jax.jit
    def envBeta(nu, Teff, L=None, a=0.11, b=-0.47, c=-0.093):
        """ Compute beta correction

        Computes the beta correction factor for Amax. This has the effect of
        reducing the amplitude for hotter solar-like stars that are close to
        the red edge of the delta-scuti instability strip, according to the
        observed reduction in the amplitude.

        This method was originally applied by Chaplin et al. 2011, who used a
        Delta_Teff = 1250K, this was later updated (private communcation) to
        Delta_Teff = 1550K.

        Parameters
        ----------
        nu : float
            Value of nu in muHz to compute the beta correction at.
        Teff : float
            Stellar effective temperature in K.
        L : float
            Stellar luminosity in solar units. Default is None.
        a : float
            Scaling relation exponent to compute TeffRed for when L is None. By
            default 0.11.
        b : float
            Scaling relation exponent to compute TeffRed for when L is None. By 
            default -0.47.
        c : float
            Scaling relation exponent to compute TeffRed for when L is not None.
            By default -0.093.

        Returns
        -------
        beta : float
            The correction factor for Amax.
        """
        
        nu = jnp.asarray(nu)
        
        if L is None:
            TeffRed = constants.TeffRed0 * (nu/constants.numax0)**a * (Teff/constants.Teff0)**b

        else:
            TeffRed = constants.TeffRed0 * L**c
            
        _beta = 1.0 - jnp.exp(-(TeffRed-Teff)/constants.Delta_Teff)
         
        beta = jnp.where(_beta <= 0, jnp.exp(-1250), _beta)
        
        return beta
    
    @staticmethod
    @jax.jit
    def Amax(numax, Teff, L=None, M=None):
        """ Compute Amax

        Computes the mode amplitude of a notional radial mode at nu_max, based
        on scaling relations.

        This includes the beta correction factor.

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope in muHz.
        Teff : float
            Stellar effective temperature in K.

        Returns
        -------
        Amax : float
            Amplitude in ppm of a radial order if it were exactly at nu_max.
        """

        beta = scalingRelations.envBeta(numax, Teff)
    
        if (L is None) and (M is None):
            Amax =  beta * (numax/constants.numax0)**-1 * (Teff/constants.Teff0)**1.5
        else:
            Amax =  beta *  L / M * (Teff/constants.Teff0)**-2

        return Amax # solar units

class references():
    """ A class for managing references used when running PBjam.

    This is inherited by session and star. 
    
    Attributes
    ----------
    bibfile : str
        The pathname to the pbjam references list.
    _reflist : list
        List of references that is updated when new functions are used.
    bibdict : dict
        Dictionary of bib items from the PBjam reference list.
    
    """
    
    def __init__(self):
        
        self.bibfile = os.path.join(*[PACKAGEDIR, 'data', 'pbjam_references.bib'])
        
        self._reflist = []
        
        self.bibdict = self._parseBibFile()

    def _findBlockEnd(self, string, idx):
        """ Find block of {}
        
        Go through string starting at idx, and find the index corresponding to 
        the curly bracket that closes the opening curly bracket.
        
        So { will be closed by } even if there are more curly brackets in 
        between.
        
        Note
        ----
        This also works in reverse, so opening with } will be closed by {.
        
        Parameters
        ----------
        string : str
            The string to parse.
        idx : int
            The index in string to start at.         
        """
        
        a = 0
        for i, char in enumerate(string[idx:]):
            if char == '{':
                a -= 1
            elif char == '}':
                a += 1
                
            if (i >= len(string[idx:])-1) and (a != 0):    
                print('Warning: Reached end of bibtex file with no closing curly bracket. Your .bib file may be formatted incorrectly. The reference list may be garbled.')
            if a ==0:
                break  
        
        if string[idx+i] == '{':
            print('Warning: Ended on an opening bracket. Your .bib file may be formatted incorrectly.')
            
        return idx+i
        
    def _parseBibFile(self):
        """ Put contents of a bibtex file into a dictionary.
        
        Takes the contents of the PBjam bib file and stores it as a dictionary
        of bib items.
        
        Article shorthand names (e.g., @Article{shorthand_name) become the
        dictionary key, similar to the way LaTeX handles citations.
        
        Returns
        -------
        bibdict : dict
            Dictionary of bib items from the PBjam reference list.
        """
        
        with open(self.bibfile, 'r') as bib:
            bib = bib.read()
            
            openers = ['@ARTICLE', '@article', '@Article'
                       '@MISC', '@misc',
                       '@BOOK', '@book',
                       '@SOFTWARE', '@software',
                       '@INPROCEEDINGS', '@inproceedings'] #Update this if other types of entries are added to the bib file.
            
            bibitems = []   
            safety = 0
            while any([x in bib for x in openers]):
                for opener in openers:
                    try:
                        start = bib.index(opener)
        
                        end = self._findBlockEnd(bib, start+len(opener))
         
                        bibitems.append([bib[start:end+1]])
        
                        bib = bib[:start] + bib[end+1:]
                            
                    except:
                        pass
                    safety += 1
                    
                    if safety > 1000:
                        break
                    
            bibitems = np.unique(bibitems)
            
            bibdict = {}
            for i, item in enumerate(bibitems):
                key = item[item.index('{')+1:item.index(',')]
                bibdict[key] = item
                
            return bibdict
            
    def _addRef(self, ref):
        """ Add reference from bibdict to active list
        
        The reference must be listed in the PBjam bibfile.
        
        Parameters
        ----------
        ref : str
            Bib entry to add to the list
        """
        if isinstance(ref, list):
            for r in ref:
                self._reflist.append(self.bibdict[r])
        else:
            self._reflist.append(self.bibdict[ref])
        
    def __call__(self, bibfile=None):
        """ Print the list of references used.
        
        Parameters
        ----------
        bibfile : str
            Filepath to print the list of bib items.
        """
        
        out = '\n\n'.join(np.unique(self._reflist))
        print('References used in this run.')
        print(out)
        
        if bibfile is not None:
            with open(bibfile, mode='w') as file_object: #robustify the filepath so it goes to the right place all the time.
                print(out, file=file_object)

def isvalid(number):
    """ Checks if number is finite.
    
    Parameters
    ----------
    number : object
    
    Returns
    -------
    x : bool
        Whether number a real float or not.
    
    """
    if (number is None) or isinstance(number, str) or not np.isfinite(number):
        return False
    else:
        return True
                            
def getNormalPercentiles(X, nsigma=2, **kwargs):
    """ Get percentiles of an distribution
    
    Compute the percentiles corresponding to sigma=1,2,3.. including the 
    median (50th), of an array.
    
    Parameters
    ----------
    X : numpy.array()
        Array to find percentiles of
    sigma : int, optional.
        Sigma values to compute the percentiles of, e.g. 68% 95% are 1 and 2 
        sigma, etc. Default is 2.
    kwargs : dict
        Arguments to be passed to numpy.percentile
    
    returns
    -------
    percentiles : numpy.array()
        Numpy array of percentile values of X.
    
    """

    a = np.array([0.5*(1+erf(z/np.sqrt(2))) for z in range(nsigma+1)])
    
    percs = np.append((1-a[::-1][:-1]),a)*100

    return np.percentile(X, percs, **kwargs)

def to_log10(x, xerr):
    """ Transform to value to log10
    
    Takes a value and related uncertainty and converts them to logscale.

    Approximate.

    Parameters
    ----------
    x : float
        Value to transform to logscale
    xerr : float
        Value uncertainty

    Returns
    -------
    logval : list
        logscaled value and uncertainty
    """
    
    if xerr > 0:
        return [np.log10(x), xerr/x/np.log(10.0)]
    return [x, xerr]

@jax.jit
def normal(x, mu, sigma):
    """ Evaluate logarithm of normal distribution (not normalized!!)

    Evaluates the logarithm of a normal distribution at x. 

    Inputs
    ------
    x : float
        Values to evaluate the normal distribution at.
    mu : float
        Distribution mean.
    sigma : float
        Distribution standard deviation.

    Returns
    -------
    y : float
        Logarithm of the normal distribution at x
    """

    return gaussian(x, 1/jnp.sqrt(2*jnp.pi*sigma**2), mu, sigma)

@jax.jit
def gaussian(x, A, mu, sigma):
    return A*jnp.exp(-(x-mu)**2/(2*sigma**2))

def makeUneven(n):
    if n % 2 == 0:
        n += 1
    return n
