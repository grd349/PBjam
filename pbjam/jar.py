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


def to_log10(x, xerr):
    """ Transform to value to log10


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    xerr : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    if xerr > 0:
        return [np.log10(x), xerr/x/np.log(10.0)]
    return [x, xerr]

def normal(x, mu, sigma):
    """ logarithm of normal distribution 

    Evaluates the logarithm of a normal distribution at x.

    Inputs
    ------
    x : real
        observed value
    mu : real
        distribution mean
    sigma : real
        distribution standard deviation

    Returns
    -------
    y : real
        Logarithm of the normal distribution at x
    """

    if (sigma < 0):
        return 0.0
    return -0.5 * (x - mu)**2 / sigma**2 - np.log(sigma * np.sqrt(2.0 * np.pi))
