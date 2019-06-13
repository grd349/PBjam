import matplotlib
matplotlib.use('Agg')

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

class peakbag():
    def __init__(self, f, snr, asy_result):
        self.f = f
        self.snr = snr
        self.asy_result = asy_result
        self.make_ladder()
        self.make_start()
        self.trim_ladder()

    def make_ladder(self):
        dnu = self.asy_result.summary.loc['best'].dnu
        epsilon = self.asy_result.summary.loc['best'].eps
        bin_width = self.f[1] - self.f[0]
        w = int(dnu / bin_width)
        s = int(epsilon * dnu / bin_width * 0.8)
        h = int(np.floor(len(self.snr[s:]) / w))
        self.ladder_p = np.reshape(self.snr[s:h*w+s], [h, w])[:, :int(w/2)]
        self.ladder_f = np.reshape(self.f[s:h*w+s], [h, w])[:, :int(w/2)]

    def make_start(self):
        l0 = self.asy_result.modeID.loc[self.asy_result.modeID.ell == 0].nu_mu.values.flatten()
        l2 = self.asy_result.modeID.loc[self.asy_result.modeID.ell == 2].nu_mu.values.flatten()
        width = 10**(np.ones(len(l0)) * self.asy_result.summary.loc['best'].mode_width).flatten()
        height =  (self.asy_result.summary.loc['best'].env_height * \
                 np.exp(-0.5 * (l0 - self.asy_result.summary.loc['best'].numax)**2 /
                 self.asy_result.summary.loc['best'].env_width**2)).flatten()
        self.start = {'l0': l0,
                      'l2': l2,
                      'width0': width * (1.0 + np.random.randn(len(l0)) * 0.1),
                      'width2': width,
                      'height0': height,
                      'height2': height*0.7,
                      'back': np.ones(len(l0))}
        print(self.start)

    def trim_ladder(self):
        orders = []
        for freq in self.start['l0']:
            for j in range(self.ladder_f.shape[0]):
                if ((freq > np.min(self.ladder_f[j,:])) and
                    (freq < np.max(self.ladder_f[j,:]))):
                    orders.append(j)
        print(orders)
        self.ladder_f = self.ladder_f[orders, :]
        self.ladder_p = self.ladder_p[orders, :]

    def lor(self, freq, w, h):
        diff = (self.ladder_f.T - freq)**2
        norm = 1.0 + 4.0 / w**2 * diff
        return h / norm

    def model(self, l0, l2, width0, width2, height0, height2, back):
        mod = np.ones(self.ladder_f.shape).T * back
        mod += self.lor(l0, width0, height0)
        mod += self.lor(l2, width2, height2)
        return mod.T

    def plot_start_model(self):
        mod = self.model(self.start['l0'],
                         self.start['l2'],
                         self.start['width0'],
                         self.start['width2'],
                         self.start['height0'],
                         self.start['height2'],
                         self.start['back'])
        n = self.ladder_p.shape[0]
        fig, ax = plt.subplots(n, figsize=[16,9])
        for i in range(n):
            ax[i].plot(self.ladder_f[i, :], self.ladder_p[i, :], c='k')
            ax[i].plot(self.ladder_f[i, :], mod[i, :], c='r')

    def simple(self):
        dnu = self.asy_result.summary.loc['best'].dnu
        self.pm_model = pm.Model()
        hfac = 10.0
        wfac = 1.0
        with self.pm_model:
            l0 = pm.Normal('l0', self.start['l0'], dnu*0.1,
                              shape=len(self.start['l0']))
            l2 = pm.Normal('l2', self.start['l2'], dnu*0.1,
                              shape=len(self.start['l2']))
            width0 = pm.HalfNormal('width0', wfac*self.start['width0'],
                                    shape=len(self.start['l2']))
            width2 = pm.HalfNormal('width2', wfac*self.start['width2'],
                                    shape=len(self.start['l2']))
            height0 = pm.HalfNormal('height0', hfac*self.start['height0'],
                                    shape=len(self.start['l2']))
            height2 = pm.HalfNormal('height2', hfac*self.start['height2'],
                                    shape=len(self.start['l2']))
            back = pm.Normal('back', 1.0, 0.1,
                                    shape=len(self.start['l2']))
            limit = self.model(l0, l2, width0, width2, height0, height2, back)
            yobs = pm.Gamma('yobs', alpha=1, beta=1.0/limit, observed=self.ladder_p)

    def width_gp(self):
        dnu = self.asy_result.summary.loc['best'].dnu
        self.pm_model = pm.Model()
        self.n = np.linspace(0.0, 1.0, len(self.start['l0']))[:, None]
        print(self.n)
        hfac = 10.0
        wfac = 1.0
        with self.pm_model:
            l0 = pm.Normal('l0', self.start['l0'], dnu*0.1,
                              shape=len(self.start['l0']))
            l2 = pm.Normal('l2', self.start['l2'], dnu*0.1,
                              shape=len(self.start['l2']))
            # Place a GP over the l=0 mode widths ...
            cov_func = 1.0 * pm.gp.cov.ExpQuad(1, ls=0.3)
            gp = pm.gp.Latent(cov_func=cov_func)
            ln_width0 = gp.prior('ln_width0', X=self.n)
            width0 = pm.Deterministic('width0', pm.math.exp(ln_width0))
            # and on the l=2 mode widths
            ln_width2 = gp.prior('ln_width2', X=self.n)
            width2 = pm.Deterministic('width2', pm.math.exp(ln_width2))
            #Carry on
            height0 = pm.HalfNormal('height0', hfac*self.start['height0'],
                                    shape=len(self.start['l2']))
            height2 = pm.HalfNormal('height2', hfac*self.start['height2'],
                                    shape=len(self.start['l2']))
            back = pm.Normal('back', 1.0, 0.1,
                                    shape=len(self.start['l2']))

            limit = self.model(l0, l2, width0, width2, height0, height2, back)
            yobs = pm.Gamma('yobs', alpha=1, beta=1.0/limit, observed=self.ladder_p)


    def sample(self, model_type='simple',
                     tune=1000,
                     target_accept=0.9,
                     cores=1):
        if model_type == 'simple':
            self.simple()
        elif model_type == 'width_gp':
            self.width_gp()
        with self.pm_model:
            self.samples = pm.sample(tune=tune, start=self.start, cores=cores)
        pm.traceplot(self.samples)

    def plot_linewidth(self, thin=10):
        fig, ax = plt.subplots(figsize=[16,9])
        for i in range(0, len(self.samples), thin):
            ax.scatter(self.samples['l0'][i, :], self.samples['width0'][i, :])
            ax.scatter(self.samples['l2'][i, :], self.samples['width2'][i, :])

    def plot_height(self, thin=10):
        fig, ax = plt.subplots(figsize=[16,9])
        for i in range(0, len(self.samples), thin):
            ax.scatter(self.samples['l0'][i, :], self.samples['height0'][i, :])
            ax.scatter(self.samples['l2'][i, :], self.samples['height2'][i, :])

    def plot_fit(self, thin=100, alpha=0.2):
        n = self.ladder_p.shape[0]
        fig, ax = plt.subplots(n, figsize=[16,9])
        for i in range(n):
            for j in range(0, len(self.samples), thin):
                mod = self.model(self.samples['l0'][j],
                                 self.samples['l2'][j],
                                 self.samples['width0'][j],
                                 self.samples['width2'][j],
                                 self.samples['height0'][j],
                                 self.samples['height2'][j],
                                 self.samples['back'][j])
                ax[i].plot(self.ladder_f[i, :], mod[i, :], c='r', alpha=alpha)
            ax[i].plot(self.ladder_f[i, :], self.ladder_p[i, :], c='k')
