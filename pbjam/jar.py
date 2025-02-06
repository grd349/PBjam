"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os, jax, json
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
from functools import partial
import scipy.special as sc
import scipy.integrate as si
from dataclasses import dataclass
import pandas as pd
import pbjam.distributions as dist

class generalModelFuncs():
    """
    A class containing general model functions for various models in PBjam.

    This class in inherited by the model classes.

    """

    def __init__(self):
        pass

    def getMedianModel(self, samplesU=None, rint=None, N=30):
        """
        Computes the median model from a set of N samples drawn from the posterior.

        Parameters
        ----------
        samplesU : dict, optional
            A dictionary of samples where each key corresponds to a parameter
            and each value is a list of sample values for that parameter.
            If None, it uses `self.samples` unpacked using `self.unpackSamples`.
        rint : array-like, optional
            Indices to select specific samples. If None, `N` indices are randomly
            chosen without replacement.
        N : int, optional
            The number of samples to use for computing the median model. Default is 30.

        Returns
        -------
        ndarray
            The median background model computed from the samples.
        """
                
        if samplesU is None:
            samplesU = self.unpackSamples(self.samples)

        mod = np.zeros((len(self.f), N))
        
        rkey = np.random.choice(list(samplesU.keys()))

        Nsamples = len(samplesU[rkey])

        if rint is None:
            rint = np.random.choice(np.arange(Nsamples), size=N, replace=False)
        
        for i, j in enumerate(rint):
            # Extract background parameters for the selected sample
            theta_u = {k: v[j] for k, v in samplesU.items()}
            
            # Compute the background model for the selected sample
            mod[:, i] = self.model(theta_u)
        
        # Compute the median background model across samples
        return np.median(mod, axis=1)
       
    @partial(jax.jit, static_argnums=(0,))
    def obsOnlylnlikelihood(self, theta):
        """
        Computes the log-likelihood using just the obs parameters.

        This ignores all spectrum information.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Returns
        -------
        lnlike : float
            Log-likelihood value.
        """

        thetaU = self.unpackParams(theta)
    
        lnlike = self.addAddObsLike(thetaU)

        return lnlike

    def setAddObs(self, keys):
        """ 
        Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 

        Parameters
        ----------
        keys : list
            List of keys for additional observational data.
        """
        
        self.addObs = {}

        for key in keys:
            self.addObs[key] = dist.normal(loc=self.obs[key][0], 
                                           scale=self.obs[key][1])
 
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
    
    def addAddObsLike(self, thetaU):
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
            lnp += self.addObs[key].logpdf(thetaU[key]) 
 
        return lnp
    
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
        self.pcaLabels = []
        
        # Default additional parameters
        self.addLabels = []
        
        # If key appears in priors dict, override default and move it to add. 
        for key in self.variables.keys():
            if self.variables[key]['pca'] and (key not in addPriors.keys()):
                self.pcaLabels.append(key)
            else:
                self.addLabels.append(key)
 
        self.logpars = [key for key in self.variables.keys() if self.variables[key]['log10']]

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

        S = {key: np.zeros(samples.shape[0]) for key in self.pcaLabels + self.addLabels}
        
        jUnpack = jax.jit(self.unpackParams)

        for i, theta in enumerate(samples):
        
            thetaU = jUnpack(theta)
             
            for key in thetaU.keys():
                
                S[key][i] = thetaU[key]
            
        return S



 

def envelope(nu, env_height, numax, env_width, **kwargs):
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

def modeUpdoot(result, sample, key, Nmodes):
    """
    Updates the `result` dictionary with summary statistics and samples for a given key.

    Parameters
    ----------
    result : dict
        The result dictionary to be updated. It should contain 'summary' and 'samples' sub-dictionaries.
    sample : ndarray
        The sample data to be added to the result. It is an array of shape (Nsamples, Nmodes).
    key : str
        The key under which the summary statistics and samples are to be stored in the result dictionary.
    Nmodes : int
        The number of modes (columns) in the sample array.    
    """

    result['summary'][key] = np.hstack((result['summary'][key], np.array([smryStats(sample[:, j]) for j in range(Nmodes)]).T))

    result['samples'][key] = np.hstack((result['samples'][key], sample))

def visell1(emm, inc):
    """
    Computes the visibility for l=1 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=1 modes.
    """
    
    y = jax.lax.cond(emm == 0, 
                     lambda : jnp.cos(inc)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1, 
                                           lambda : 0.5*jnp.sin(inc)**2, # m = 1
                                           lambda : jnp.nan # m > 1
                                           ))
                    
    return y

def visell2(emm, inc):
    """
    Computes the visibility for l=2 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=2 modes.
    """

    y = jax.lax.cond(emm == 0, 
                     lambda : 1/4 * (3 * jnp.cos(inc)**2 - 1)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1,
                                           lambda : 3/8 * jnp.sin(2 * inc)**2, # m = 1
                                           lambda : jax.lax.cond(emm == 2, 
                                                                 lambda : 3/8 * jnp.sin(inc)**4, # m = 2
                                                                 lambda : jnp.nan # m > 2
                                                                 )))
    return y

def visell3(emm, inc):
    """
    Computes the visibility for l=3 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=3 modes.
    """

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

def visibility(ell, m, inc):
    """
    Computes the visibility of a mode based on its degree (l), azimuthal order (m), and inclination angle.

    Parameters
    ----------
    ell : int
        The degree of the mode.
    m : int
        The azimuthal order of the mode.
    inc : float
        The inclination angle in radians.

    Returns
    -------
    float
        Visibility for the given mode.
    """

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
    """
    Updates the prior data by adding a new entry based on the provided results and additional observations.

    Parameters
    ----------
    ID : str
        The identifier for the new entry.
    R : dict
        A dictionary containing the results. Keys should correspond to parameter names, and values are typically arrays or lists where the first element is used.
    addObs : dict
        A dictionary containing additional observational data, such as 'teff' and 'bp_rp'.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the updated prior data.

    Notes
    -----
    - The function reads the existing prior data from `pbjam/data/prior_data.csv`.
    - It filters out certain keys from the results that are not meant to be updated.
    - It applies a log10 transformation to certain parameters in the results before updating the prior data.
    - The new entry is then added to the prior DataFrame and returned.
    """

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
    """
    A dataclass for storing astrophysical constants and conversion factors.

    Attributes
    ----------
    nu_to_omega : float
        Conversion factor from frequency (muHz) to angular frequency (radians/muHz). Default is `2 * jnp.pi / 1e6`.
    """

    # Teff0: float = 5777 # K
    # TeffRed0: float = 8907 # K
    # numax0: float = 3090 # muHz
    # Delta_Teff: float = 1550 # K
    # Henv0: float = 0.1 # ppm^2/muHz
    nu_to_omega: float = 2 * jnp.pi / 1e6 # radians/muHz
    c : float = 299792.458 #km/s
    # dnu0: float = 135.9 # muHz
    # logg0 : float = 4.43775 # log10(2.74e4)

def smryStats(y):
    """
    Computes summary statistics (median and uncertainty) for a given array of samples.

    Parameters
    ----------
    y : array-like
        The input array of samples.

    Returns
    -------
    ndarray
        An array containing the median and the average absolute deviation.

    Notes
    -----
    - The function computes percentiles corresponding to the 16th, 50th, and 84th percentiles.
    - The uncertainty is the mean of the differences between the median and the 16th and 84th percentiles.
    """

    p = np.array([0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-1, 2)])[::-1]*100
     
    u = np.percentile(y, p)
    
    return np.array([u[1], np.mean(np.diff(u))])

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
        cdf = si.cumulative_trapezoid(y, x, initial=0)
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

class bibliography():
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

def gaussian(x, A, mu, sigma):
    """
    Computes the Gaussian function.

    Parameters
    ----------
    x : array-like
        Input array of x values.
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean (center) of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.

    Returns
    -------
    array-like
        The computed Gaussian function values.
    """
        
    return A*jnp.exp(-(x-mu)**2/(2*sigma**2))
