#!/usr/bin/env python 

import sys
sys.path.append('/home/davies/Dropbox/GRDpy')

import GRDdata
import numpy as np
import matplotlib.pyplot as plt
import emcee
import triangle
import os
import sys
from scipy.misc import factorial
from scipy.special import lpmv as legendre

def plot_freqs(f, m=[], samples=[], model=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if len(samples) > 0:
        for i in np.arange(0, len(samples[:,0]), 100):
            mm = modela(samples[i,:])
            xxx = ax1.plot(mm, np.ones(len(mm)), 'ko', alpha=0.2)
    if len(m) > 0:
        m = ax1.plot(m, np.ones(len(m)), 'rs')
    l = ax1.plot(f, np.ones(len(f)), 'bD')
    ax1.set_ylim([0,2])
    ax1.set_xlim([f.min() - 1.0,f[f<1e4].max() + 1.0])

def plot_echelle(f, dnu, m=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l = ax1.plot(f % dnu, f, 'bD')
    if len(m) > 0:
        m = ax1.plot(m % dnu, m, 'rs')
    ax1.set_ylim([f.min(),f[f<1e4].max()])

class RGmodel:
    def __init__(self, _f):
        self.nu = _f
        self.model = np.zeros(len(_f))

    def lorentzian(self, freq, width, height, asym=0.0):
        x = 2.0 * (self.nu * 1e6 - freq) / width
        l = np.absolute(height) * (1.0 + 2.0 * asym/100.0 * x) /\
            (1.0 + x**2)
        return l

    def sphr_lm(self, l, theta):
        ell = int(l)
        amp = np.zeros(ell + 1)
        for mdx, m in enumerate(xrange(0, ell+1)):
            H = (factorial(ell - abs(m))/factorial(ell + abs(m))) \
                * legendre(m, ell, np.cos(theta*np.pi/180))**2
            amp[mdx] = H
        return amp

    def full_model(self, zero, dnu, nominal_pmode, period_spacing, \
                   epsilon_g, coupling, gsplit, rsplit, angle, \
                   linewidth, amplitude, back, L_gamma, A_gamma, \
                   ledoux, led_gam):
        freq = self.freq_model(dnu, nominal_pmode, period_spacing, \
                   epsilon_g, coupling)
        split = self.splitting(freq, dnu, period_spacing, \
                               gsplit, rsplit)
        ledouxs = self.ledouxs(freq*1e6, nominal_pmode, ledoux, led_gam)
        psplit = self.splitting(freq+ledouxs*split/1e6, dnu, period_spacing, \
                               gsplit, rsplit)
        msplit = self.splitting(freq-ledouxs*split/1e6, dnu, period_spacing, \
                               gsplit, rsplit)
        widths = self.widths(freq*1e6, nominal_pmode, linewidth, L_gamma)
        mwidths = self.widths(freq*1e6-msplit*ledouxs, nominal_pmode, linewidth, L_gamma)
        pwidths = self.widths(freq*1e6+psplit*ledouxs, nominal_pmode, linewidth, L_gamma)
        amplitudes = self.amplitudes(freq*1e6, nominal_pmode, amplitude, A_gamma)
        mamplitudes = self.amplitudes(freq*1e6-msplit*ledouxs, nominal_pmode, amplitude, A_gamma)
        pamplitudes = self.amplitudes(freq*1e6+psplit*ledouxs, nominal_pmode, amplitude, A_gamma)
        # amp**2 = np.pi/2 * height * width but here just use a * a = h * w
        heights = amplitudes**2 / widths * 1.5
        mheights = mamplitudes**2 / mwidths * 1.5
        pheights = pamplitudes**2 / pwidths * 1.5
        self.model[:] = back
        eaa = self.sphr_lm(1, angle)
        self.model += self.lorentzian(zero, linewidth, amplitude**2/linewidth) 
        for idx, i in enumerate(freq*1e6):
            self.model += self.lorentzian(i, widths[idx], heights[idx] * eaa[0]) 
            self.model += self.lorentzian(i-msplit[idx], \
                                          mwidths[idx], mheights[idx] * eaa[1]) 
            self.model += self.lorentzian(i+psplit[idx], \
                                          pwidths[idx], pheights[idx] * eaa[1]) 

#        return GRDdata.smooth_power(self.model, 30)
        return self.model

    def freq_model(self, dnu, nominal_pmode, period_spacing, \
                   epsilon_g, coupling):
        lhs = np.pi * (self.nu - nominal_pmode*1e-6) / (dnu * 1e-6)
        rhs = np.arctan(coupling * np.tan(np.pi/(period_spacing * self.nu) \
                                          - epsilon_g))
        mixed = np.ones(1)
        for i in range(len(self.nu)-1):
            y1 = lhs[i] - rhs[i]
            y2 = lhs[i+1] - rhs[i+1]
            if lhs[i] - rhs[i] < 0 and lhs[i+1] - rhs[i+1] > 0:
                m = (y2 - y1) / (self.nu[i+1] - self.nu[i])
                c = y2 - m * self.nu[i+1]
                intp = -c/m
                mixed = np.append(mixed, intp)
        if len(mixed) > 1:
            mixed = mixed[1:]
        return mixed

    def splitting(self, mixed, delta_nu, period_spacing, gsplit, rsplit):
        alpha_0 = (delta_nu * 1e-6)* period_spacing
        chi = 2.0 * mixed / (delta_nu*1e-6) * \
              np.cos(np.pi / (period_spacing * mixed))
        eta = 1.0 / (1.0 + alpha_0 * chi**2)
        splits = gsplit * (eta * (1 - 2.0*rsplit) + 2.0*rsplit)
        return splits

    def widths(self, mixed, nominal_pmode, linewidth, L_gamma):
        x = 2.0 * (mixed - nominal_pmode) / L_gamma
        l = 1.0 / (1.0 + x**2)
        return l * linewidth

    def amplitudes(self, mixed, nominal_pmode, amplitude, A_gamma):
        x = 2.0 * (mixed - nominal_pmode) / A_gamma
        l = 1.0 / (1.0 + x**2)
        return l * amplitude
        
    def ledouxs(self, mixed, nominal_pmode, ledoux, led_gam):
        x = 2.0 * (mixed - nominal_pmode) / led_gam
        l =  ledoux - ((ledoux) / (1.0 + x**2))
        return l

    def __call__(self, params):
        return self.full_model(*params)
        
class Likelihood:
    def __init__(self, _f, _obs, _model):
        self.obs = _obs
        self.f = _f
        self.model = _model

    def __call__(self, params):
        mod = self.model(params)
        L = -1.0 * np.sum(np.log(mod) + self.obs/mod)
        return L

class Prior:
    def __init__(self, _bounds, _gaussian):
        self.bounds = _bounds
        self.gaussian = _gaussian

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        lnprior = 0.0
        for idx, i in enumerate(self.gaussian):
            if i[1] != 0:
                lnprior += -0.5 * (p[idx] - i[0])**2 / i[1]**2
        # Place a prior on i - uniform in cos i
        lnprior += np.log(np.sin(np.radians(p[8])))
        return lnprior
    
def MCMC(params, like, prior, plot=False):
    ntemps, nwalkers, niter, ndims = 1, 100, 800, int(len(params))
#    ntemps, nwalkers, niter, ndims = 6, 1000, 1000, int(len(params))
    sampler = emcee.PTSampler(ntemps, nwalkers, ndims, like, \
                              prior, threads=4)
    param_names = [r'zero', r'dnu', r'nominal pmode', \
                   r'period spacing', r'epsilon g', r'coupling', \
                   r'gsplit', r'rsplit', r'angle', \
                   r'linewidth', r'Amp', \
                   r'Back', \
                   r'L_gamma', r'A_gamma']	
    p0 = np.zeros([ntemps, nwalkers, ndims])
    for i in range(ntemps):
        for j in range(nwalkers):
            p0[i,j,:] = params + 1e-4*np.random.randn(ndims)
#            p0[i,j,2] = params[1] + np.random.randn(1)
#            p0[i,j,3] = np.random.rand(1) * 2.0 + 80
#            p0[i,j,4] = np.random.rand(1) * 1.0 - 0.5
#            p0[i,j,5] = np.random.rand(1) * 0.4
#            p0[i,j,6] = np.random.rand(1) * 0.6
#            p0[i,j,7] = np.random.rand(1) * -0.09
#            p0[i,j,8] = np.degrees(np.arccos(np.random.rand(1)))

    print('... burning in ...')
    for p, lnprob, lnlike in sampler.sample(p0, iterations=niter):
        pass
    sampler.reset()
    print('... running sampler ...')
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                            lnlike0=lnlike,
                                            iterations=niter):
        pass
    samples = sampler.chain[0,:, :, :].reshape((-1, ndims))
#    print("Int time: ", emcee.autocorr.integrated_time(samples))
    results = np.median(samples, axis=0)
    errs = np.std(samples, axis=0)
    print('... plotting ...')
    if plot:
        fig2 = triangle.corner(samples, truth=results)
        fig2.savefig('tmp.png')
    return samples, results, errs

def plot_ps(f, p, mod=[], samples=[]):
    figa = plt.figure()
    axa = figa.add_subplot(111)
    figa.subplots_adjust(right=0.96,left=0.12, bottom=0.12,top=0.96, hspace=0.1)
    axa.plot(f, p, 'k-')
    if len(samples) > 0:
        for i in np.arange(0, len(samples[:,0]), 1000):
            m = mod(samples[i,:])
            axa.plot(f, m, 'b-', alpha=0.5)
    else:
        if len(mod) > 0:
            axa.plot(f, mod, 'r-')
    axa.plot(f, p, 'k-')
    axa.set_xlabel('Frequency ($\mathrm{ \mu Hz}$)', \
                  {'fontsize' : 18})
    axa.set_ylabel('Power spectral density ($\mathrm{ppm^{2} \; \mu Hz^{-1}}$)', \
                  {'fontsize' : 18})
    plt.savefig('ps_fit.png')
    plt.show()

def run(psd_file, low, high, zero, dnu, dnu_err, nom_guess):
    ff, pp, bw = GRDdata.get_psd(psd_file)
    f, p = GRDdata.get_red_psd(ff, pp, bw, low, high)
#    p = GRDdata.smooth_power(p, 30)
    back = np.median(p) 
    param_names = [r'zero', r'dnu', r'nominal pmode', \
                   r'period spacing', r'epsilon g', r'coupling', \
                   r'gsplit', r'rsplit', r'angle', \
                   r'linewidth', r'Amp', \
                   r'Back', \
                   r'L_gamma', r'A_gamma', \
                   r'ledoux', r'led_gam']	
    dnu = 12.97
    params = np.array([zero, dnu, nom_guess, 80.446, \
                       0.0, 0.14, \
                       0.43, 0.0, 89.0, \
                       0.02, np.max(p) * 0.02, \
                       back, \
                       dnu/10, dnu/10, \
                       0.5, 1.3])   
    bounds = [(low, high), (0,dnu*2), (low,high),(49.0,400.0), \
              (-np.pi/2,np.pi/2), (0.01, 0.4), \
              (0.05, 0.8), (-0.5,0.5), (0,90), \
              (bw/2.0,0.8), (0.0,1e6), \
              (0.0, 1e6), \
              (0.0,dnu), (0.0,dnu), \
              (0.01, 1.0), (0.0,dnu)]
    gaussian = [(zero, 1.0), (dnu, dnu_err*10), (0,0), (0,0), \
                (0,0), (0,0), \
                (0,0), (0,0), (0,0), \
                (0,0), (0,0), \
                (0,0), \
                (0,0), (0,0), \
                (0.5,0.2), (0,0)]
    model = RGmodel(f * 1e-6)
    like = Likelihood(f, p, model)
    prior = Prior(bounds, gaussian)
    start = model(params)
    plot_ps(f, p, mod=start)
    samples, results, errs = MCMC(params, like, prior, \
                                 plot=True)
    print(results)
    print(errs)
    plot_ps(f, p, mod=model, samples=samples)
#    plt.show()

if __name__ == "__main__":
    print("PB an RG order")

 



    
