from . import PACKAGEDIR
import os
import numpy as np

def get_priorpath():
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])


def get_percentiles(X, sigma = 2, **kwargs):
    percs = np.array([0.682689492137,
                      0.954499736104,
                      0.997300203937,
                      0.999936657516,
                      0.999999426697,
                      0.999999998027])*100/2    
    percs = np.append(0, percs)    
    percs = np.append(-percs[::-1][:-1],percs)
    percs += 50
    print(kwargs)
    return np.percentile(X, percs[6-sigma : 6+sigma+1], **kwargs)