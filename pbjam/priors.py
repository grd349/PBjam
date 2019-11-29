import numpy as np
import pandas as pd
from .mcmc import mcmc
import warnings
from .plotting import plotting
import statsmodels.api as sm
from .jar import get_priorpath, to_log10, normal

class kde(plotting):
    """ A class to predict epsilon.

    TODO: See the docs for full information. (especially example_advanced)

    """

    def __init__(self, starinst=None, prior_file=None):

        if starinst:
            self.f = starinst.f
            self.s = starinst.s
            self.pg = starinst.pg
            starinst.kde = self

            if prior_file is None:
                prior_file = starinst.prior_file
        elif prior_file is None:
            prior_file = get_priorpath()

        self.prior_data = pd.read_csv(prior_file)

    def select_prior_data(self, numax=None, KDEsize = 100):
        """ Selects useful prior data based on proximity to estimated numax.

        Selects a subset of targets around input numax to use for computing the
        KDE. If the number of targets in the range considered for the prior is
        less than 100, the range will be expanded until it ~100. This is to
        ensure that the KDE can be constructed. If the initial range includes
        more than 100 targets (e.g., in dense parts of the RGB) the selection
        of targets will be downsampled randomly.


        Note: does not ensure that the KDE is finite at the location of your
        target

        Parameters
        ----------

        numax : list
            The estimate of numax together with uncertainty in log space.
            If numax==None then no selection will be made - all data will
            be used (you probably don't want to do this).

        KDEsize : int
            Number of targets to include in the KDE estimation.

        """

        if not numax:
            return self.prior_data

        # Select numax range and expand if needed
        idx = self._prior_expand_check(numax, KDEsize)

        # Downsample to KDEsize
        self.prior_data = self._downsample_prior_check(idx, KDEsize)


    def _downsample_prior_check(self, idx, KDEsize):
        """ Reduce prior sample size

        Randomly picks KDEsize targets from the prior sample

        Parameters
        ----------
        idx : boolean array
            Array mask to select targets to be included in the KDE estimation.

        KDEsize : int
            Number of targets to include in the KDE estimation.

        """
        if len(idx) < KDEsize:
            warnings.warn(f'Sample for estimating KDE is less than the request {KDEsize}.')
            KDEsize = len(idx)

        return self.prior_data.sample(KDEsize, weights = idx, replace = False)

    def _prior_expand_check(self, numax, KDEsize):
        """ Expand numax interval to reach sufficient KDE sample size

        If necessary, increases the range around starting numax to use as the
        initial prior, until the sample contains at least N = KDEsize targets.

        Parameters
        ----------
        numax : length 2 list [numax, numax_err]
            The estimate of numax together with uncertainty in log space.

        KDEsize : int
            Number of targets to include in the KDE estimation.

        Returns
        -------
        idx : boolean array
            Array mask to select targets to be included in the KDE estimation.

        """

        nsigma = 1

        idx = np.abs(self.prior_data.numax.values - numax[0]) < nsigma * numax[1]

        flag_warn = False
        while len(self.prior_data[idx]) < KDEsize:
            idx = np.abs(self.prior_data.numax.values - numax[0]) < nsigma * numax[1]

            if not flag_warn:
                warnings.warn(f'Only {len(self.prior_data[idx])} star(s) near provided numax.' +
                'Expanding the range to include ~100 stars.')
                flag_warn = True

            if nsigma >= KDEsize:
                break

            nsigma += 0.1

        return idx



    def make_kde(self, bw_fac=1.0, verbose=False):
        """ Takes the prior data and constructs a KDE function

        TODO: add details on the band width determination - see example
        advanced for a base explaination.

        I have settled on using the values from a cross validated maximum
        likelihood estimate.

        """

        self.par_names = ['dnu', 'numax', 'eps', 'd02', 'alpha', 'env_height',
                          'env_width', 'mode_width', 'teff', 'bp_rp']

        self.select_prior_data(self.log_obs['numax'])

        if self.verbose:
                print(f'Selected data set length {len(self.prior_data)}')

        if bw_fac != 1:
            from statsmodels.nonparametric.bandwidths import select_bandwidth
            bw = select_bandwidth(self.prior_data[self.par_names].values,
                                  bw = 'scott',
                                  kernel=None) * bw_fac
        else:
            if self.verbose:
                print('Selecting sensible stars for kde')
                print(f'Full data set length {len(self.prior_data)}')
            bw = 'cv_ml'

        self.kde = sm.nonparametric.KDEMultivariate(
                            data=self.prior_data[self.par_names].values,
                            var_type='cccccccccc', bw=bw)



    def prior(self, p):
        """ Calculates the log prior

        Evaluates the KDE for the parameters p. Additional hard/soft priors
        can be added here as needed to, e.g., apply boundaries to the fit.

        Hard constraints should be applied at the top so function exits early,
        if necessary.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        lp : real
            The log likelihood evaluated at p.

        """

        # d02/dnu < 0.2  (np.log10(0.2) ~ -0.7)
        if p[3] - p[0] > -0.7:
            return -np.inf

        # Constraint from prior
        lp = np.log(self.kde.pdf(p))

        return lp

    def likelihood(self, p):
        """ Calculate likelihood

        Calculates the likelihood of the observed properties given the proposed
        parameters p.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        lnlike : float
            The log likelihood evaluated at p.

        """

        lnlike = 0.0

        # Constraints from observational data
        lnlike += normal(p[0], *self.log_obs['dnu'])
        lnlike += normal(p[1], *self.log_obs['numax'])
        lnlike += normal(p[8], *self.log_obs['teff'])
        lnlike += normal(p[9], *self.obs['bp_rp'])

        return lnlike

    def kde_sampler(self, nwalkers=50):
        """ Samples from the posterior probability distribution

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

        """

        if self.verbose:
            print('Running KDE sampler')

        x0 = [self.log_obs['dnu'][0],  # log10 dnu
              self.log_obs['numax'][0],  # log10 numax
              1.0,  # eps
              np.log10(0.1 * self.obs['dnu'][0]),  # log10 d02
              -2.0,  # log10 alpha
              1.0,  # log10 env height
              1.0,  # log10 env width,
              -1.0,  # log10 mode width
              self.log_obs['teff'][0],
              self.obs['bp_rp'][0]]

        self.fit = mcmc(x0, self.likelihood, self.prior, nwalkers=nwalkers)

        return self.fit()


    def kde_predict(self, n):
        """
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

        """

        if not hasattr(self, 'samples'):
            print('Need to run the sampler first')
            return -1, -1
        dnu = 10**self.samples[:, 0]
        eps = self.samples[:, 2]
        nmax = 10**self.samples[:, 1] / dnu - eps
        alpha = 10**self.samples[:, 4]
        freq = np.array([(nn + eps + alpha/2.0 * (nn - nmax)**2) * dnu for nn in n])
        return freq.mean(axis=1), freq.std(axis=1)

    def __call__(self, dnu=[1, -1], numax=[1, -1], teff=[1, -1],
                 bp_rp=[1, -1], verbose=False, bw_fac=1):
        """  Compute and sample the KDE.

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

        """
        self.verbose = verbose

        self.obs = {'dnu': dnu, 'numax': numax, 'teff': teff, 'bp_rp': bp_rp}

        self.log_obs = {x: to_log10(*self.obs[x]) for x in self.obs.keys() if x != 'bp_rp'}

        self.make_kde(bw_fac)

        self.samples = self.kde_sampler()

        self.result = [self.samples[:, 2].mean(), self.samples[:, 2].std()]
