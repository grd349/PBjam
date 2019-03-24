import numpy as np
import pandas as pd
import os
import emcee

from . import PACKAGEDIR

from scipy.stats import gaussian_kde

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
        self.vrard_dict = {'alpha': 0.601, 'beta': 0.632}
        self.data_file = PACKAGEDIR + os.sep + 'data' + os.sep + 'rg_results.csv'
        self.obs = []
        self.seff_offset = 4000.0

    def read_prior_data(self):
        ''' Read in the prior data from self.data_file '''
        self.prior_data = pd.read_csv(self.data_file)
        self.prior_data['Seff'] = self.prior_data.Teff \
                                    - self.seff_offset
        self.prior_data['log_Seff'] = np.log10(self.prior_data.Seff)

    def make_kde(self, bw=0.8):
        ''' Takes the prior data and constructs a KDE function '''
        self.cols = ['log_dnu_', 'log_numax', 'log_Seff', 'eps_mod']
        self.kde = gaussian_kde(self.prior_data[self.cols].values.T, bw)

    def normal(self, y, mu, sigma):
        ''' returns normal log likelihood

        Inputs
        ------
        y : real
            observed value
        mu : real
            distribution mean
        sigma : real
            distribution standard deviation

        returns
        -------
        log likelihood : real
        '''
        if (sigma < 0):
            return 0.0
        return -0.5 * (y - mu)**2 / sigma**2

    def likelihood(self, p):
        ''' Calculates the log likelihood of for the parameters p

        Inputs
        ------
        p : array
            Array of the parameters [log_dnu, log_numax, log_seff, eps]

        Returns
        -------
        like: real
            The log likelihood evaluated at p.

        '''
        log_dnu, log_numax, log_seff, eps = p
        if log_seff < np.log10(4100.0 - self.seff_offset):
            return -np.inf
        # Constraint from prior
        lp = np.log(self.kde(p))
        # Constraint from data
        ld = 0.0
        ld += self.normal(log_dnu, *self.log_obs['dnu'])
        ld += self.normal(log_numax, *self.log_obs['numax'])
        ld += self.normal(log_seff, *self.log_obs['seff'])

        return lp + ld

    def kde_sampler(self):
        ''' Samples from the posterior probability distribution

        p(theta | D) propto p(theta) p(D | theta)

        p(theta) is given by the KDE function of the prior data

        p(D | theta) is given by the observable constraints

        Convergence is far from Guaranteed!

        Returns
        -------

        chains: flatchain
            The emcee chain of samples from the posterior

        '''
        x0 = [self.log_obs['dnu'][0],
              self.log_obs['numax'][0],
              self.log_obs['seff'][0],
              1.0]
        ndim, nwalkers = len(x0), 20
        p0 = [np.array(x0) + np.random.rand(ndim)*1e-3 for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)
        sampler.run_mcmc(p0, 4000)
        sampler.reset()
        sampler.run_mcmc(p0, 2000)
        return sampler.flatchain

    def to_log10(self, x, xerr):
        if xerr > 0:
            return [np.log10(x), xerr/x/np.log(10.0)]
        return [x, xerr]

    def obs_to_log(self, obs):
        self.log_obs = {'dnu': self.to_log10(*self.obs['dnu']),
                        'numax': self.to_log10(*self.obs['numax']),
                        'teff': self.to_log10(*self.obs['teff']),
                        'seff': self.to_log10(*self.obs['seff'])}

    def vrard(self, dnu):
        ''' Calculates epsilon prediction from Vrard 2015
        https://arxiv.org/pdf/1505.07280.pdf

        Uses the equation from Table 1.

        Has no dependence on temperature so not reliable.

        Assumes a fixed uncertainty of 0.1.
        '''
        return [self.vrard_dict['alpha'] +
                self.vrard_dict['alpha'] * np.log10(dnu), 0.1]

    def __call__(self, dnu=[1, -1], numax=[1, -1], teff=[1, -1]):
        ''' Calls the relevant defined method and returns an estimate of
        epsilon.

        Note: The call function creates a seff observable that is just
        the (Teff - self.seff_offset).  The reason for this is to put
        the Teff dynamic range into a more useful range that has
        characteristics that are somewhat consistent with the dex
        uncertainty of the rest of the observables.

        Inputs
        ------
        dnu : [real, real]
            Large frequency spacing and uncertainty
        numax : [real, real]
            Frequency of maximum power and uncertainty
        teff : [real, real]
            Stellar effective temperature and uncertainty

        Returns
        -------
        result : array-like
            [estimate of epsilon, unceritainty on estimate]
        '''
        if self.method == 'Vrard':
            if numax[0] > 288.0:
                print('Vrard method really only valid for Giants')
                return [self.vrard(dnu[0])[0], 1.0]
            return self.vrard(dnu[0])
        self.obs = {'dnu': dnu,
                    'numax': numax,
                    'teff': teff,
                    'seff': [teff[0] - self.seff_offset, teff[1]]}
        if self.method == 'kde':
            if numax[0] > 288.0:
                print('Not yet implemented for SC stars')
            self.read_prior_data()
            self.obs_to_log(self.obs)
            self.make_kde()
            samples = self.kde_sampler()
            return [samples[:,3].mean(), samples[:,3].std()]
