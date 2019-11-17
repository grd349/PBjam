from . import PACKAGEDIR
import os
import numpy as np

def get_priorpath():
    """ Get default prior path name
    
    Returns
    -------
    priorpath : str
        Default path to the prior in the pacakage directory structure
        
    """
    
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])


def get_percentiles(X, sigma = 2, **kwargs):
    """ Get percentiles of an distribution
    
    Compute the percentiles corresponding to sigma=1,2,3.. including the 
    median (50th), of an array.
    
    Parameters
    ----------
    X : numpy.array()
        Array to find percentiles of
    sigma : int
        Sigma values to compute the percentiles of 68% 95% etc.
    kwargs : dict
        Arguments to be passed to numpy.percentile
    
    returns
    -------
    percentiles : numpy.array()
        Numpy array of percentile values of X.
    
    """
    
    percs = np.array([0.682689492137,
                      0.954499736104,
                      0.997300203937,
                      0.999936657516,
                      0.999999426697,
                      0.999999998027])*100/2    
    percs = np.append(0, percs)    
    percs = np.append(-percs[::-1][:-1],percs)
    percs += 50
    
    return np.percentile(X, percs[6-sigma : 6+sigma+1], **kwargs)