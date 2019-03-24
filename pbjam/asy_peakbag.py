import numpy as np
from pbjam import epsilon
import os

from . import PACKAGEDIR

class model():
    def __init__(self, f):
        self.f = f

    def lor(self, freq, h, w):
        '''
        Lorentzian to describe modes.

        Line width in in log_10.
        '''
        w = 10**(w)
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)

    def pair(self, freq0, h, w, d02, hfac=0.7):
        ''' A pair is the sum of two Lorentzians '''
        model = self.lor(freq0, h, w)
        model += self.lor(freq0 - d02, h*hfac, w)
        return model

    def asy(self, numax, dnu, d02, eps, alpha, hmax, Envwidth, w):
        ''' We fit the asymptotic relation:

        nu_nl = (n + epsilon + alpha/2(n - nmax)**2) * log_dnu

        where,

        nmax = numax / dnu - eps.

        We separate the l=0 and l=2 modes by d02*dnu.

        '''
        nmax = numax / dnu - eps
        nn = np.arange(np.floor(nmax-5), np.floor(nmax+6), 1)
        model = np.ones(len(self.f))
        for n in nn:
            f0 = (n + eps + alpha/2*(n - nmax)**2) * dnu
            h = hmax * np.exp(- 0.5 * (f0 - numax)**2 / Envwidth**2)
            model += self.pair(f0, h, w, d02*dnu)
        return model

    def __call__(self, p):
        ''' Return the model '''
        return self.asy(*p)

class Prior(epsilon):
    def __init__(self, _bounds, _gaussian):
        self.bounds = _bounds
        self.gaussian = _gaussian
        self.data_file = PACKAGEDIR + os.sep + 'data' + os.sep + 'rg_results.csv'
        self.seff_offset = 4000.0
        self.read_prior_data()
        self.make_kde()

    def pbound(self, p):
        ''' prior boundaries '''
        for idx, i in enumerate(p):
            if ((self.bounds[idx][0] != 0) & (self.bounds[idx][1] != 0)):
                if ((i < self.bounds[idx][0]) | (i > self.bounds[idx][1])):
                    return -np.inf
        return 0

    def pgaussian(self, p):
        ''' Guassian priors - not yet implemented!!!'''
        lnprior = 0.0
        for idx, i in enumerate(p):
            if self.gaussian[idx][1] != 0:
                lnprior += -0.5 * (i - self.gaussian[idx][0])**2 / \
                           self.gaussian[idx][1]**2
        return lnprior

    def __call__(self, p):
        if self.pbound(p) == -np.inf:
            return -np.inf
        lp = self.kde([np.log10(p[1]), np.log10(p[0]), p[8], p[3]])
        return lp

class mcmc():
    def __init__(self, f, snr, x0):
        self.f = f
        self.snr = snr
        self.sel = np.where(np.abs(self.f - x0[0]) < 3*x0[1])
        self.model = model(f[self.sel])
        # numax, dnu, d02, eps, alpha, hmax, Envwidth, w, seff
        # TODO - tidy this bit up!
        bounds = [[x0[0]*0.9, x0[0]*1.1], [x0[1]*0.9, x0[1]*1.1],
                   [0.04, 0.2], [0.5, 2.5], [0, 1], [x0[5]*0.5, x0[5]*1.5],
                   [x0[6]*0.9, x0[6]*1.1], [-2, 1.0], [2.0, 4.0]]
        self.lp = Prior(bounds, [(0,0) for n in range(len(x0))])

    def likelihood(self, p):
        logp = self.lp(p)
        if logp == -np.inf:
            return -np.inf
        mod = self.model(p[:-1])
        like = -1.0 * np.sum(np.log(mod) + \
                          self.snr[self.sel]/mod)
        return like

    def __call__(self, x0):
        import emcee
        ndim, nwalkers = len(x0), 20
        p0 = [np.array(x0) + np.random.rand(ndim)*1e-3 for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)
        print('Burmingham')
        sampler.run_mcmc(p0, 1000)
        sampler.reset()
        print('Sampling')
        sampler.run_mcmc(p0, 1000)
        return sampler.flatchain
