"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os
import numpy as np
from scipy.special import erf

import functools, logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.debug('Initialised module logger')

def _entering_function(func, logger):
    """ Pre function logging. """
    logger.debug("Entering %s.", func.__qualname__)
    # TODO: stuff to check before entering function

def _exiting_function(func, logger):
    """ Post function logging. """
    # TODO: stuff to check before exiting function
    logger.debug("Exiting  %s.", func.__qualname__)

def log(logger):
    """
    Function logging decorator. 
    
    Parameters
    ----------
    logger: logging.Logger
        Specify the logger in which to submit entering and exiting logs, highly recommended to be the module-level
        logger (see Examples).

    Examples
    --------
    Logging a function called `my_func` defined in a module with name `__name__`,

    ```python
    import logging
    from pbjam.jar import log

    logger = logging.getLogger(__name__)

    @log(logger)
    def my_func(a, b):
        logger.debug('Function in progress.')
        return a + b

    if __name__ == "__main__":
        logging.basicConfig()
        logger.setLevel('DEBUG')
        
        result = my_func(1, 2)
        logger.debug(f'result = {result}')
    ```

    Outputs,

    ```python
    DEBUG:__main__:Entering my_func
    DEBUG:__main__:Function in progress.
    DEBUG:__main__:Exiting  my_func
    DEBUG:__main__:result = 3
    ```

    For use within classes,

    ```python
    import logging
    from pbjam.jar import log

    logger = logging.getLogger(__name__)


    class myClass:

        @log(logger)
        def __init__(self):
            logger.debug('Initializing class.')
            self.a = 1
            self.b = 2

        @log(logger)
        def my_mthd(self):
            logger.debug('Method in progress.')
            return self.a + self.b

    if __name__ == "__main__":
        logging.basicConfig()
        logger.setLevel('DEBUG')
        
        obj = myClass()
        result = obj.my_mthd()
        logger.debug(f'result = {result}')
    ```

    Outputs,

    ```python
    DEBUG:__main__:Entering myClass.__init__.
    DEBUG:__main__:Initializing class.
    DEBUG:__main__:Exiting  myClass.__init__.
    DEBUG:__main__:Entering myClass.my_mthd.
    DEBUG:__main__:Method in progress.
    DEBUG:__main__:Exiting  myClass.my_mthd.
    DEBUG:__main__:result = 3
    ```

    """
    def _log(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            _entering_function(func, logger)
            result = func(*args, **kwargs)
            _exiting_function(func, logger)                
            return result
        return wrap
    
    return _log


class file_logging:
    """
    Context manager for file logging. It logs everything under the `pbjam` parent level in some file at a given `path`.

    Parameters
    ----------
    path : str
        File path to save the log
    
    level : str, optional
        Logging level. Default is 'DEBUG'
    
    **kwargs :
        Keyword arguments passed to `logging.FileHandler`.

    Attributes
    ----------
    handler : logging.FileHandler
        File handler object.

    Examples
    --------
    ```python
    from pbjam.jar import file_logging

    with file_logging('example.log') as flog:
        # Do some stuff here and it will be logged to 'example.log'
        ...

    # Do some stuff here and it won't be logged to 'example.log'

    with flog:
        # Do some stuff here and it will be logged to 'example.log'
        ... 
    ```

    """
    _logger = logging.getLogger('pbjam')
    def __init__(self, path, level='DEBUG', handler_kwargs={}):
        self.path = path
        self.level = level
        self.handler_kwargs = handler_kwargs
        self.file_handler = None
    
    def add_file_handler(self):
        self.file_handler = logging.FileHandler(self.path, **self.handler_kwargs)
        self.file_handler.setFormatter(HANDLER_FMT)
        self.file_handler.setLevel(self.level)

    def __enter__(self):
        self.add_file_handler(self)
        self._logger.addHandler(self.file_handler)
        return self
    
    def __exit__(self, type, value, traceback):
        self._logger.removeHandler(self.file_handler)
        self.file_handler.close()
        self.file_handler = None


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

    if (sigma < 0):
        return 0.0
    return -0.5 * (x - mu)**2 / sigma**2
