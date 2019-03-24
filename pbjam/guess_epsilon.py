import numpy as np
import pandas as pd
import os
from . import PACKAGEDIR

class epsilon():
    ''' A class to predict epsilon.

    Attributes
    ----------
    method : string
        Sets the method used to estimat epsilon
        Possible methods are ['Vrard', ...]
    vrard_dict : dict
        Stores the Vrard coefficients
    data_file : string
        The loc of the prior data file
    '''
    def __init__(self, method='Vrard'):
        self.method = method
        # 0.601 + 0.632 logh∆νi
        self.vrard_dict = {'alpha': 0.601, 'beta': 0.632}
        self.data_file = PACKAGEDIR + os.sep + 'data' + os.sep + 'rg_results.csv'

    def read_prior_data(self):
        ''' Read in the prior data from self.data_file '''
        self.prior_data = pd.read_csv(self.data_file)



    def vrard(self, dnu):
        ''' Calculates epsilon prediction from Vrard 2015
        https://arxiv.org/pdf/1505.07280.pdf

        Uses the equation from Table 1.

        Has no dependence on temperature so not reliable.

        Assumes a fixed uncertainty of 0.1.
        '''
        return [self.vrard_dict['alpha'] +
                self.vrard_dict['alpha'] * np.log10(dnu), 0.1]

    def __call__(self, dnu, numax=-1, teff=-1,
                 dnu_err=-1, numax_err=-1, teff_err=-1):
        ''' Calls the relevant defined method and returns an estimate of
        epsilon.

        Inputs
        ------
        dnu : real
            Large frequency spacing
        numax : real
            Frequency of maximum power
        teff : real
            Stellar effective temperature
        dnu_err : real
            Uncertainty on dnu
        numax_err : real
            uncertainty on numax
        teff_err : real
            uncertainty on teff

        Returns
        -------
        result : array-like
            [estimate of epsilon, unceritainty on estimate]
        '''
        if self.method == 'Vrard':
            if numax > 288.0:
                print('Vrard method really only valid for Giants')
                return [self.vrard(dnu)[0], 1.0]
            return self.vrard(dnu)
