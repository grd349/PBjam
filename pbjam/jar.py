"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os, jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
from functools import partial
import scipy.special as sc
import scipy.integrate as si

# Constant factor from cyclic to angular frequency
nu_to_omega = 2 * jnp.pi / 1e6


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
    @partial(jax.jit, static_argnums=(0,))
    def envWidth(numax):
        """ Scaling relation for the envelope width

        Computest he full width at half maximum of the p-mode envelope based
        on numax and Teff (optional).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.
        Teff : float, optional
            Effective surface temperature of the star.
        Teff0 : float, optional
            Solar effective temperature in K. Default is 5777 K.

        Returns
        -------
        width : float
            Envelope width in muHz
        """

        width = 0.66*numax**0.88 # Mosser et al. 201??

        return width

    @partial(jax.jit, static_argnums=(0,))
    def nuHarveyGran(self, numax):
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

    @partial(jax.jit, static_argnums=(0,))
    def nuHarveyEnv(self, numax):
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
                            
def getDistPercentiles(X, nsigma=2, **kwargs):
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
def _normal(x, mu, sigma):
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

    return -0.5 * (x - mu)**2 / sigma**2