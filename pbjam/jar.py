"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os
import numpy as np
import pandas as pd
from scipy.special import erf

import functools, logging, inspect, sys, warnings
from .printer import pretty_printer

HANDLER_FMT = "%(asctime)-23s :: %(levelname)-8s :: %(name)-17s :: %(message)s"
INDENT = 60  # Set to length of logger info before message or just indent by 2?
logger = logging.getLogger(__name__)

_pp_kwargs = {'width': 120}
if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    # 'sort_dicts' kwarg new to Python 3.8
    _pp_kwargs['sort_dicts'] = False

pprinter = pretty_printer(**_pp_kwargs)


class _function_logger:
    """ Handlers the logging upon entering and exiting functions. """

    def __init__(self, func, logger):
        self.func = func
        self.signature = inspect.signature(self.func)
        self.logger = logger

    def _log_bound_args(self, args, kwargs):
        """ Logs bound arguments - ``args`` and ``kwargs`` passed to func. """
        bargs = self.signature.bind(*args, **kwargs)
        bargs_dict = dict(bargs.arguments)
        self.logger.debug(f"Bound arguments:\n{pprinter.pformat(bargs_dict)}")
        
    def _entering_function(self, args, kwargs):
        """ Log before function execution. """
        self.logger.debug(f"Entering {self.func.__qualname__}")
        self.logger.debug(f"Signature:\n{self.func.__name__ + str(self.signature)}")
        self._log_bound_args(args, kwargs)
        # TODO: stuff to check before entering function

    def _exiting_function(self, result):
        """ Log after function execution. """
        # TODO: stuff to check before exiting function
        if result is not None:
            self.logger.debug(f"Returns:\n{pprinter.pformat(result)}")
        self.logger.debug(f"Exiting {self.func.__qualname__}")


def debug(logger):
    """
    Function logging decorator. Logs function metadata upon entering and 
    exiting.
    
    Parameters
    ----------
    logger: logging.Logger
        Specify the logger in which to submit entering and exiting logs, highly 
        recommended to be the module-level logger (see Examples).

    Examples
    --------
    Logging a function called ``my_func`` defined in a module with name ``__name__``,

    .. code-block:: python

        import logging
        from pbjam.jar import debug

        logger = logging.getLogger(__name__)
        debugger = debug(logger)

        @debugger
        def my_func(a, b):
            logger.debug('Function in progress.')
            return a + b

        if __name__ == "__main__":
            logging.basicConfig()
            logger.setLevel('DEBUG')
            
            result = my_func(1, 2)
            logger.debug(f'result = {result}')
    
    Outputs,

    .. code-block:: text

        DEBUG:__main__:Entering my_func
        DEBUG:__main__:Function in progress.
        DEBUG:__main__:Exiting  my_func
        DEBUG:__main__:result = 3
    
    For use within classes,

    .. code-block:: python

        import logging
        from pbjam.jar import debug

        logger = logging.getLogger(__name__)
        debugger = debug(logger)


        class myClass:

            def __init__(self):
                logger.debug('Initializing class.')
                self.a = 1
                self.b = 2

            @debugger
            def my_mthd(self):
                logger.debug('Method in progress.')
                return self.a + self.b

        if __name__ == "__main__":
            logging.basicConfig()
            logger.setLevel('DEBUG')
            
            obj = myClass()
            result = obj.my_mthd()
            logger.debug(f'result = {result}')
    
    Outputs,

    .. code-block:: text

        DEBUG:__main__:Entering myClass.__init__.
        DEBUG:__main__:Initializing class.
        DEBUG:__main__:Exiting  myClass.__init__.
        DEBUG:__main__:Entering myClass.my_mthd.
        DEBUG:__main__:Method in progress.
        DEBUG:__main__:Exiting  myClass.my_mthd.
        DEBUG:__main__:result = 3

    """
    def _log(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            flog = _function_logger(func, logger)
            flog._entering_function(args, kwargs)
            result = func(*args, **kwargs)
            flog._exiting_function(result)
            return result
        return wrap
    
    return _log


class _formatter(logging.Formatter):
    
    def format(self, *args, **kwargs):
        s = super(_formatter, self).format(*args, **kwargs)
        lines = s.split('\n')
        return ('\n' + ' '*INDENT).join(lines)


class _handler(logging.Handler):

    def __init__(self, level='NOTSET', **kwargs):
        super().__init__(**kwargs)
        fmt = _formatter(HANDLER_FMT)
        self.setFormatter(fmt)
        self.setLevel(level)


class _stream_handler(_handler, logging.StreamHandler):
    
    def __init__(self, level='INFO', **kwargs):
        super(_stream_handler, self).__init__(level=level, **kwargs)


class _file_handler(_handler, logging.FileHandler):
    
    def __init__(self, filename, level='DEBUG', **kwargs):
        super(_file_handler, self).__init__(filename=filename, level=level, **kwargs)

  
class log_file:
    """
    Context manager for file logging. It logs everything under the ``loggername`` 
    logger, by default this is the ``'pbjam'`` logger (i.e. logs everything from 
    the pbjam package).

    Parameters
    ----------
    filename : str
        Filename to save the log
    level : str, optional
        Logging level. Default is 'DEBUG'.
    loggername : str, optional
        Name of logger which will send logs to ``filename``. Default is ``'pbjam'``.

    Attributes
    ----------
    handler : pbjam.jar._file_handler
        File handler object.

    Examples
    --------
    .. code-block:: python

        from pbjam.jar import log_file

        with log_file('example.log') as flog:
            # Do some pbjam stuff here and it will be logged to 'example.log'
            ...

        # Do some stuff here and it won't be logged to 'example.log'

        with flog:
            # Do some stuff here and it will be logged to 'example.log'
            ... 
    
    """
    def __init__(self, filename, level='DEBUG', loggername='pbjam'):
        self._filename = filename
        self._level = level
        self._logger = logging.getLogger(loggername)
        self.handler = None
        self._isopen = False

    def open(self):
        """ If log file is not open, creates a file handler at the log level """
        if not self._isopen:
            self.handler = _file_handler(self._filename, level=self._level)
            self._logger.addHandler(self.handler)
            self._isopen = True

    def close(self):
        """ If log file is open, safely closes the file handler """
        if self._isopen:
            self._logger.removeHandler(self.handler)
            self.handler.close()
            self.handler = None
            self._isopen = False

    def get_level(self):
        return self._level

    def set_level(self, level):
        """ 
        Set the level of the file handler.
        
        Parameters
        ----------
        level : str
            Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' or
            'NOTSET'. 
        """
        self._level = level
        if self._isopen:
            self.handler.setLevel(self._level)

    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()


class file_logger:
    """
    Creates a ``log_file`` at ``filename`` to which logs under ``loggername`` at
    a given ``level`` are recorded when the file logger is listening. This
    class is indended to be sub-classed (see Examples). 

    To listen to a method in a sub-class of ``file_logger`` (i.e. record all logs 
    which occur during the method execution) decorate the class method with
    ``@file_logger.listen``.
    
    Parameters
    ----------
    filename : str
        Filename to save the log
    level : str, optional
        Logging level. Default is 'DEBUG'.
    loggername : str, optional
        Name of logger which will send logs to ``filename``. Default is ``'pbjam'``.

    Attributes
    ----------
    log_file : pbjam.jar.log_file

    Examples
    --------
    .. code-block:: python

        # pbjam/example.py
        from .jar import file_logger

        class example_class(file_logger):
            def __init__(self):
                super(example_class, self).__init__('example.log', level='INFO')
                
                with self.log_file:
                    # Records content in context to log_file
                    logger.info('Initializing class.')
                    ...
            
            @file_logger.listen  # records content of example_method to log_file
            def example_method(self):
                logger.info('Performing function tasks.')
                ...
    
    """

    def __init__(self, *args, **kwargs):
        self.log_file = log_file(*args, **kwargs)

    @staticmethod
    def listen(func):
        """
        Decorator for recording logs to ``log_file`` during function operation, 
        closing the log file upon completion.
        """
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            self.log_file.open()
            result = func(self, *args, **kwargs)
            self.log_file.close()
            return result
        return wrap    


class references():
    """ 
    A class for managing references used when running PBjam.

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
                logger.warning('Reached end of bibtex file with no closing curly bracket. Your .bib file may be formatted incorrectly. The reference list may be garbled.')
            if a ==0:
                break  
        
        if string[idx+i] == '{':
            logger.warning('Ended on an opening bracket. Your .bib file may be formatted incorrectly.')
            
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
    """ 
    Get percentiles of an distribution
    
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
    """ 
    Transform to value to log10
    
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
    """ 
    Evaluate logarithm of normal distribution (not normalized!!)

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
