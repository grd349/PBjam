"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os
import numpy as np
from scipy.special import erf



class references():
    """ A class for managing references used when running PBjam.

    This is inherited by session and star. 
    
    """
    
    def __init__(self):
        
        self.bibfile = os.path.join(*[PACKAGEDIR, 'data', 'pbjam_references.bib'])
        
        self.reflist = []
        
        entries = np.unique(self._parseBibFile())

        self.bibdict = {}
        for i, block in enumerate(entries):
            key = block[block.index('{')+1:block.index(',')]
            self.bibdict[key] = block
            
    def _findBlockEnd(self, string, idx):
        """ Find block of {}
        
        Go through string starting at idx, and find
        the index corresponding to the curly bracket
        that closes the opening curly bracket.
        
        So { will be closed by } even if there are more
        curly brackets in between.
        
        Note
        ----
        This also works in reverse, so opening with }
        will be closed by {.
        
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
        
        Article shorthand names (e.g., @Article{shorthand_name) becomes the
        dictionary key.
        
        """
        
        with open(self.bibfile, 'r') as bib:
            bib = bib.read()
            
            openers = ['@ARTICLE', '@article', 
                       '@MISC', '@misc',
                       '@BOOK', '@book'] #Update this if other types of entries are added to the bib file.
            
            blocks = []   
            safety = 0
            while any([x in bib for x in openers]):
                for opener in openers:
                    try:
                        start = bib.index(opener)
        
                        end = self._findBlockEnd(bib, start+len(opener))
         
                        blocks.append([bib[start:end+1]])
        
                        bib = bib[:start] + bib[end+1:]
                            
                    except:
                        pass
                    safety += 1
                    
                    if safety > 1000:
                        break
                
            return blocks
            
    def _addRef(self, ref):
        """ Add reference from bibdict to active list
        
        Remember to add the relevant references to the bibfile
        
        """
        
        self.reflist.append(self.bibdict[ref])
        
    def __call__(self, to_file=False):
        """ Print the list of references used.
        
        """
        
        out = '\n\n'.join(np.unique(self.reflist))
        print('References used in this run.')
        print(out)
        
        if to_file:
            with open('pbjam.bib', mode='w') as file_object: #robustify the filepath so it goes to the right place all the time.
                print(out, file=file_object)
            
def get_priorpath():
    """ Get default prior path name
    
    Returns
    -------
    prior_file : str
        Default path to the prior in the package directory structure.
        
    """
    
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])


def get_percentiles(X, nsigma = 2, **kwargs):
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

    a = np.array([0.5*(1+erf(z/np.sqrt(2))) for z in range(nsigma)])
    
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

    if (sigma < 0):
        return 0.0
    return -0.5 * (x - mu)**2 / sigma**2