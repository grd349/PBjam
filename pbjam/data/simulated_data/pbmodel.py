#!/usr/bin/env python3
#O. J. Hall 2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import lightkurve
from scipy.special import legendre as P
from scipy.misc import factorial as fct

from omnitool import literature_values as lv
plt.style.use(lightkurve.MPLSTYLE)

import argparse
parser = argparse.ArgumentParser(description='Generate a model of 16 Cyg A')
parser.add_argument('-n', '--noise', action='store_const',
                const=False, default=True, help='Turn off Chi-Sqr 2 d.o.f. noise')
parser.add_argument('-b', '--background', action='store_const', const=False,
                    default=True, help='Turn off Harvey Profile background')
parser.add_argument('-a', '--apodization', action='store_const', const=False,
                    default=True, help='Turn off apodization')
parser.add_argument('years', default = 4., type=float, help='How many years worth of data')
parser.add_argument('-s','--save',action='store_const',const=True,
                    default=False, help='Save output.')
args = parser.parse_args()

class star():
    def __init__(self, freqs, nyquist, numax, dnu, d02, nus, i):
        '''A class model that stores the basic stellar properties'''
        self.freqs = freqs
        self.nyquist = nyquist
        self.numax = numax
        self.dnu = dnu
        self.d02 = d02
        self.epsilon = 0.601 + 0.632*np.log(self.dnu)  #from Vrard et al. 2015 (for RGB)
        self.nmax = self.numax/self.dnu - self.epsilon #from Vrard et al. 2015
        self.lmax = 3     #Don't care about higher order
        self.Gamma = 1.   #Depends on the mode lifetimes (which I don't know)
        self.nus = nus    #Depends on rotation & coriolis force (which I don't understand yet)
        self.i = i        #Determines the mode height
        self.snr  = 10.

    def get_Hn(self, n):
        #The height of the l=0 mode for a given n.
        #These I will draw from a Gaussian with a given FWHM, as they depend on SNR
        nun0 = self.asymodelocs(n, 0, 0)

        hmax=self.snr*1.4

         #I modulate the mode height based on a fudged estimate of the FWHM
        fwhm = 0.25*self.numax  #From LEGACY
        std = fwhm / (2*np.sqrt(2*np.log(2)))
        Hn = hmax * np.exp(-0.5 * (nun0 - self.numax)**2 / std**2)
        return Hn

    def get_Epsilonlm(self, i, l, m):
        #I use the prescriptions from Gizon & Solank 2003 and Handberg & Campante 2012
        if l == 0:
            return 1
        if l == 1:
            if m == 0:
                return np.cos(i)**2
            if np.abs(m) == 1:
                return 0.5 * np.sin(i)**2
        if l == 2:
            if m == 0:
                return 0.25 * (3 * np.cos(i)**2 - 1)**2
            if np.abs(m) ==1:
                return (3/8)*np.sin(2*i)**2
            if np.abs(m) == 2:
                return (3/8) * np.sin(i)**4
        if l == 3:
            if m == 0:
                return (1/64)*(5*np.cos(3*i) + 3*np.cos(i))**2
            if np.abs(m) == 1:
                return (3/64)*(5*np.cos(2*i) + 3)**2 * np.sin(i)**2
            if np.abs(m) == 2:
                return (15/8) * np.cos(i)**2 * np.sin(i)**4
            if np.abs(m) == 3:
                return (5/16)*np.sin(i)**6

    def get_Vl(self, l):
        #Vn depends on the mission, and is usually marginalised over.
        #It is the geometrical visibility of the total power in a multiplet (n, l) as a function of l.
        #Im taking these values from Handberg & Campante 2011 (agree with Chaplin+13)
        if l == 0.:
            return 1.0
        if l == 1.:
            return 1.22
        if l == 2.:
            return 0.71
        if l == 3.:
            return 0.14

    def lorentzian(self, nunlm, n, l, m):
        #We set all mode heights to 1 to start with
        height = self.get_Hn(n) * self.get_Epsilonlm(self.i, l, m) * self.get_Vl(l)**2
        model = height / (1 + (4/self.Gamma**2)*(self.freqs - nunlm)**2)
        return model

    def harvey(self, a, b, c):
        #The harvey profile seems to take different forms depending on who I ask?
        #I'm going to be using the one used in Guy's BackFit code. Why is it different?
        harvey = 0.9*a**2/b/(1.0 + (self.freqs/b)**c);

        return harvey

    def get_background(self):
        #I did a fit to 16CygA using Guy's backfit program. I'm lifting the
        #Harvey components from there
        a = 36.3
        b = 723.52
        c = 31.85
        d = 2002.6
        j = 1.79
        k = 198.31
        white = 0.09

        background = np.zeros(len(self.freqs))
        background += self.harvey(a, b, 4.) +\
                        self.harvey(c, d, 4.) +\
                        self.harvey(j, k, 2.) + white
        return background

    def get_apodization(self):
        return np.sinc((np.pi/2) * self.freqs / self.nyquist)

    def get_noise(self):
        return np.random.chisquare(2, size=len(self.freqs))

    def asymodelocs(self, n, l, m):
        #d00, d01, d02, d03
        dnu0 = [0., 0., self.d02, self.d02]
        return self.dnu * (n + l/2 + self.epsilon) - dnu0[l] + m * self.nus

    def get_model(self):
        nn = np.arange(np.floor(self.nmax-6.), np.floor(self.nmax+6.), 1)
        model = np.ones(len(self.freqs))
        locs = np.ones([len(nn), self.lmax+1])

        for idx, n in enumerate(nn):
            for l in np.arange(self.lmax+1):
                locs[idx, l] = self.asymodelocs(n, l, 0.)
                if l == 0:
                    loc = self.asymodelocs(n, l, 0.)
                    model += self.lorentzian(locs[idx, l], n, l, 0.)
                else:
                    for m in np.arange(-l, l+1):
                        loc = self.asymodelocs(n, l, m)
                        model += self.lorentzian(loc, n, l, m) #change height of multiplet

        #Add the additional components
        if args.background:
            background = self.get_background()
        else:
            background = 0.
        if args.noise:
            noise = self.get_noise()
        else:
            noise = 1.
        if args.apodization:
            apod = self.get_apodization()
        else:
            apod = 1.

        return (model + background) * apod**2 * noise, locs

    def plot_model(self):
        model, locs = self.get_model()
        l0s = np.ones(locs.shape[0])*.82 * np.max(model)
        l1s = np.ones(locs.shape[0])*.82 * np.max(model)
        l2s = np.ones(locs.shape[0])*.81 * np.max(model)
        l3s = np.ones(locs.shape[0])*.81 * np.max(model)

        fig = plt.figure()
        plt.plot(self.freqs, model)
        plt.scatter(locs[:,0],l0s, marker=',',s=10,label='l=0')
        plt.scatter(locs[:,1],l1s, marker='*',s=10,label='l=1')
        plt.scatter(locs[:,2],l2s, marker='^',s=10,label='l=2')
        plt.scatter(locs[:,3],l3s, marker='o',s=10,label='l=3')
        plt.legend(fontsize=20)
        plt.savefig('16CygAmodel.png')
        plt.show()

if __name__ == '__main__':
    nyquist = 0.5 * (1./58.6) * u.hertz
    nyquist = nyquist.to(u.microhertz)
    fs = 1./(args.years*365) * (1/u.day)
    fs = fs.to(u.microhertz)

    #Parameters for 16 Cyg A
    nus = 0.411
    i = np.deg2rad(56.)
    d02 = 6.8
    dnu = 102.
    numax = 2200.

    freqs = np.arange(fs.value, numax*2, fs.value)

    star(freqs, nyquist, numax, dnu, d02, nus, i).plot_model()
    if args.save:
        model, locs = star(freqs, nyquist, numax, dnu, d02, nus, i).get_model()
        np.savetxt('locs.txt',locs)
        np.savetxt('model.txt',model)
        np.savetxt('freqs.txt',freqs)
    # import lightkurve as lk
    # s = star(freqs, nyquist, numax, dnu, d02, nus, i)
    # pg = lk.periodogram.LombScarglePeriodogram(freqs*u.microhertz, s.get_model()[0]*u.hertz)
    #
    # pg.plot()


    # w = s.get_noise()
    # import seaborn as sns
    # sns.distplot(w)
    # plt.show()
