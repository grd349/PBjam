import numpy as np
import pandas as pd
import os, corner
import matplotlib.pyplot as plt
from . import PACKAGEDIR
from .mcmc import mcmc

class epsilon():
    ''' A class to predict epsilon.

    TODO: See the docs for full information. (especially example_advanced)

    '''
    def __init__(self, nthreads=1, verbose=False, bw_fac=1):
        self.data_file = os.path.join(*[PACKAGEDIR,'data', 'prior_data.csv'])
        self.obs = []
        self.samples = []
        self.verbose = verbose
        self.nthreads = nthreads
        self.read_prior_data()
        self.make_kde(bw_fac=bw_fac)

    def read_prior_data(self):
        ''' Read in the prior data from self.data_file '''
        self.prior_data = pd.read_csv(self.data_file)
        self.prior_data = self.prior_data.dropna()

    def make_kde(self, bw_fac=1.0):
        ''' Takes the prior data and constructs a KDE function

        TODO: add details on the band width determination - see example
        advanced for a base explaination.

        I have settled on using the values from a cross validated maximum
        likelihood estimate.

        '''
        self.cols = ['dnu', 'numax', 'eps',
                     'd02', 'alpha', 'env_height',
                     'env_width', 'mode_width', 'teff',
                     'bp_rp']
        # key: [log, log, lin, log, log, log, log, log, log, lin]
        import statsmodels.api as sm
        # bw set using CV ML but times two.
        bw = np.array([0.00774255, 0.01441685, 0.04582654,
                       0.02127414, 0.17830664, 0.54219474,
                       0.04400531, 0.06834085, 0.0054522,
                       0.11864199]) * bw_fac
        self.kde = sm.nonparametric.KDEMultivariate(
                            data=self.prior_data[self.cols].values,
                            var_type='cccccccccc',
                            bw=bw)

    def normal(self, y, mu, sigma):
        ''' Returns normal log likelihood

        Inputs
        ------
        y : real
            observed value
        mu : real
            distribution mean
        sigma : real
            distribution standard deviation

        Returns
        -------
        log likelihood : real
        '''
        if (sigma < 0):
            return 0.0
        return -0.5 * (y - mu)**2 / sigma**2

    def prior(self, p):
        ''' Calculates the log prior from the KDE for the parameters p and
        applies some boundaries.

        Inputs
        ------
        p : array
            Array of the parameters

        Returns
        -------
        like : real
            The log likelihood evaluated at p.

        Note: p = ['dnu', 'numax', 'eps', 'd02', 'alpha', 'env_height',
                   'env_width', 'mode_width', 'teff', 'bp_rp']
        
        key: [log, log, lin, log, log, log, log, log, log, lin]
        '''
        # d02/dnu < 0.2  (np.log10(0.2) ~ -0.7)
        if p[3] - p[0] > -0.7:
            return -np.inf
        # Constraint from prior
        lp = np.log(self.kde.pdf(p))
        return lp

    def likelihood(self, p):
        '''
        Calculates the likelihood of the observed properties given
        the proposed parameters p.
        '''
        log_dnu, log_numax, eps, log_d02, log_alpha, \
            log_env_height, log_env_width, log_mode_width, \
            log_teff, bp_rp = p
        # Constraint from input data
        ld = 0.0
        ld += self.normal(log_dnu, *self.log_obs['dnu'])
        ld += self.normal(log_numax, *self.log_obs['numax'])
        ld += self.normal(log_teff, *self.log_obs['teff'])
        ld += self.normal(bp_rp, *self.log_obs['bp_rp'])
        return ld

    def kde_sampler(self, nwalkers=50):
        ''' Samples from the posterior probability distribution

        p(theta | D) propto p(theta) p(D | theta)

        p(theta) is given by the KDE function of the prior data

        p(D | theta) is given by the observable constraints

        Samples are drawn using `emcee`.

        Returns
        -------

        chains: flatchain
            The emcee chain of samples from the posterior

        Note:['dnu', 'numax', 'eps',
                     'd02', 'alpha', 'env_height',
                     'env_width', 'mode_width', 'teff',
                     'bp_rp']
        key: [log, log, lin, log, log, log, log, log, log, lin]

        '''
        if self.verbose:
            print('Running KDE sampler')
        x0 = [self.log_obs['dnu'][0], #log10 dnu
              self.log_obs['numax'][0], #log10 numax
              1.0, # eps
              np.log10(0.1 * self.obs['dnu'][0]), # log10 d02
              -2.0, # log10 alpha
              1.0, #log10 env height
              1.0, #log10 env width,
              -1.0, #log10 mode width
              self.log_obs['teff'][0],
              self.log_obs['bp_rp'][0]]

        self.fit = mcmc(x0, self.likelihood, self.prior, nwalkers=nwalkers)

        return self.fit()

    def to_log10(self, x, xerr):
        '''
        Transforms observables into log10 space.
        '''
        if xerr > 0:
            return [np.log10(x), xerr/x/np.log(10.0)]
        return [x, xerr]

    def obs_to_log(self, obs):
        '''
        Converted the observed properties into log10 space.
        '''
        self.log_obs = {'dnu': self.to_log10(*obs['dnu']),
                        'numax': self.to_log10(*obs['numax']),
                        'teff': self.to_log10(*obs['teff']),
                        'bp_rp': obs['bp_rp']}

    def plot(self, periodogram):
        '''
        Make a plot of the posterior distribution on the locations
        of the l=0 modes, on top of a
        `lightkurve` periodogram object passed by the user.

        Parameters
        ----------

        periodogram: Periodogram
            A flattened lightkurve Periodogram object for plotting.

        Returns
        -------

        fig: Figure
        '''
        fig, ax = plt.subplots(figsize=[16,9])
        periodogram.plot(ax=ax, alpha=0.5, c='gray', label='Data')
        dnu = 10**(self.samples[:, 0].mean())
        smoo = periodogram.smooth(filter_width=dnu*0.02)
        smoo.plot(ax=ax, c='k', alpha=0.4, lw=3, label='Smoothed Data')
        h = np.max(smoo.power.value)
        f = periodogram.frequency.value
        dnu = 10**(self.samples[:, 0].mean())
        nmin = np.floor(f.min() / dnu)
        nmax = np.floor(f.max() / dnu)
        self.n = np.arange(nmin-1, nmax+1, 1)
        freq, freq_unc = self.kde_predict(self.n)
        y = np.zeros(len(f))
        for i in range(len(self.n)):
            y += h * 0.8 * np.exp(-0.5 * (freq[i] - f)**2 / freq_unc[i]**2)
        ax.fill_between(f, y, alpha=0.3, facecolor='navy', edgecolor='none',
                        label=r'$\propto P(\nu_{\ell=0})$')
        ax.set_xlim([f.min(), f.max()])
        ax.set_ylim([0, h*1.2])
        ax.set_title(periodogram.label)
        ax.legend()
        return fig

    def plot_corner(self):
        '''
        Makes a nice corner plot using corner.corner
        '''
        fig = corner.corner(self.samples, labels=self.cols,  quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
        return fig

    def kde_predict(self, n):
        '''
        Predict the frequencies from the kde method samples.

        The sampler must be run before calling the predict method.

        Parameters
        ----------

        n: numpy_array
            The radial order

        Returns
        -------
        frequencies: numpy-array
            A numpy array of length len(n) containting the frequency estimates.

        frequencies_unc: numpy-array
            A numpy array of length len(n) containting the frequency estimates
            uncertainties.
        '''
        if self.samples == []:
            print('Need to run the sampler first')
            return -1, -1
        dnu = 10**self.samples[:, 0]
        eps = self.samples[:, 2]
        nmax = 10**self.samples[:, 1] / dnu - eps
        alpha = 10**self.samples[:, 4]
        freq = np.array([(nn + eps + alpha/2.0 * (nn - nmax)**2) * dnu for nn in n])
        return freq.mean(axis=1), freq.std(axis=1)

    def __call__(self,
                dnu=[1, -1], numax=[1, -1], teff=[1, -1], bp_rp=[1, -1]):
        ''' Calls the relevant defined method and returns an estimate of
        epsilon.

        Inputs
        ------
        dnu : [real, real]
            Large frequency spacing and uncertainty
        numax : [real, real]
            Frequency of maximum power and uncertainty
        teff : [real, real]
            Stellar effective temperature and uncertainty
        bp_rp : [real, real]
            The Gaia Gbp - Grp color value and uncertainty
            (probably ~< 0.01 dex)
        Returns
        -------
        result : array-like
            [estimate of epsilon, unceritainty on estimate]
        '''
        self.obs = {'dnu': dnu, 'numax': numax, 'teff': teff, 'bp_rp': bp_rp}
        self.obs_to_log(self.obs)

        self.samples = self.kde_sampler()
        result = [self.samples[:,2].mean(), self.samples[:,2].std()]
        return result
