""" 

This module contains a general set of plotting methods for that are inherited by
the different classes of PBjam, so that they can be used to show the status of 
each step that has been performed. 

"""

import matplotlib.pyplot as plt
import astropy.convolution as conv
import pbjam, os, corner, warnings, logging
import numpy as np
import astropy.units as u
import pandas as pd

ellColors = {0: 'C1', 1: 'C4', 2: 'C3', 3: 'C5'}

def smooth_power(freq, power, smooth_filter_width):
    """Smooths the input power array with a Box1DKernel from astropy
    Parameters
    ----------
    power : array-like
        Array of power values
    smooth_filter_width : float
        filter width
    Returns
    -------
    array-like
        Smoothed power
    """
     
    fac = max([1, smooth_filter_width / (freq[1] - freq[0])])

    kernel = conv.Gaussian1DKernel(stddev=np.array(fac))

    smoo = conv.convolve(power, kernel)

    return smoo


def echelle(freq, power, dnu, fmin=0.0, fmax=None, offset=0.0, sampling=0.1):
    """Calculates the echelle diagram. Use this function if you want to do
    some more custom plotting.

    Parameters
    ----------
    freq : array-like
        Frequency values
    power : array-like
        Power values for every frequency
    dnu : float
        Value of deltanu
    fmin : float, optional
        Minimum frequency to calculate the echelle at, by default 0.
    fmax : float, optional
        Maximum frequency to calculate the echelle at. If none is supplied,
        will default to the maximum frequency passed in `freq`, by default None
    offset : float, optional
        An offset to apply to the echelle diagram, by default 0.0

    Returns
    -------
    array-like
        The x, y, and z values of the echelle diagram.
    """
     
    if fmax is None:
        fmax = freq[-1]

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    # trim data
    index = (freq >= fmin) & (freq <= fmax)
    trimx = freq[index]

    samplinginterval = np.median(trimx[1:-1] - trimx[0:-2]) * sampling
    xp = np.arange(fmin, fmax + dnu, samplinginterval)
    yp = np.interp(xp, freq, power)

    n_stack = int((fmax - fmin) / dnu)
    n_element = int(dnu / samplinginterval)

    morerow = 2
    arr = np.arange(1, n_stack) * dnu
    arr2 = np.array([arr, arr])
    yn = np.reshape(arr2, len(arr) * 2, order="F")
    yn = np.insert(yn, 0, 0.0)
    yn = np.append(yn, n_stack * dnu) + fmin + offset

    xn = np.arange(1, n_element + 1) / n_element * dnu
    z = np.zeros([n_stack * morerow, n_element])
    for i in range(n_stack):
        for j in range(i * morerow, (i + 1) * morerow):
            z[j, :] = yp[n_element * (i) : n_element * (i + 1)]
    return xn, yn, z


def plot_echelle(freq, power, dnu, ax=None, cmap="Blues", scale=None,
                 interpolation=None, smooth=False, smooth_filter_width=50, 
                 **kwargs):
    """Plots the echelle diagram.

    Parameters
    ----------
    freq : numpy array
        Frequency values
    power : array-like
        Power values for every frequency
    dnu : float
        Value of deltanu
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A matplotlib axes to plot into. If no axes is provided, a new one will
        be generated, by default None
    cmap : str, optional
        A matplotlib colormap, by default 'BuPu'
    scale : str, optional
        either 'sqrt' or 'log' or None. Scales the echelle to bring out more
        features, by default 'sqrt'
    interpolation : str, optional
        Type of interpolation to perform on the echelle diagram through
        matplotlib.pyplot.imshow, by default 'none'
    smooth_filter_width : float, optional
        Amount by which to smooth the power values, using a Box1DKernel
    **kwargs : dict
        Dictionary of arguments to be passed to `echelle.echelle`

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The plotted echelle diagram on the axes
    """
    if smooth:
        power = smooth_power(freq, power, smooth_filter_width)
    echx, echy, echz = echelle(freq, power, dnu, **kwargs)

    if scale is not None:
        if scale == "log":
            echz = np.log10(echz)
        elif scale == "sqrt":
            echz = np.sqrt(echz)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(echz, aspect="auto", extent=(echx.min(), echx.max(), echy.min(), echy.max()),
              origin="lower", cmap=cmap, interpolation=interpolation, )

    ax.set_xlabel(f"Frequency mod {str(np.round(dnu, 2))} "+r"[$\mu$Hz]")
    ax.set_ylabel(r"Frequency [$\mu$Hz]")

    ax.set_ylim(freq[0], freq[-1])
    return ax





def _echellify_freqs(nu, dnu):
    x = nu%dnu

    y = nu #+ dnu/2

    return x, y

def _baseEchelle(f, s, N_p, numax, dnu, scale, **kwargs):
    """
    Generate a base echelle diagram of the PSD.

    Parameters
    ----------
    numax : float
        Central frequency.
    dnu : float
        Frequency spacing.
    scale : float
        Smoothing scale.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure.
    matplotlib.axes._axes.Axes
        The generated matplotlib Axes.

    Notes
    -----
    - Computes the echelle diagram for the given frequency range.
    - Smoothes the diagram using the specified smoothing scale.
    - Returns the generated Figure and Axes objects.
    """

    n = max([N_p + 1, 10])

    idx = ((numax - n * dnu) < f) & (f < (numax + n * dnu))

    f, s = f[idx], s[idx]

    fig, ax = plt.subplots(figsize=(8,7))    
        
    plot_echelle(f, s, dnu, ax=ax, smooth=True, smooth_filter_width=dnu * scale)

    return fig, ax

def _StarClassEchelle(self, obs, scale, **kwargs):

    dnu = obs['dnu'][0]
    
    numax = obs['numax'][0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)
        
    ax.set_xlim(0, dnu)

    return fig, ax

def _ModeIDClassPriorEchelle(self, scale, colors, dnu=None, numax=None, 
                             DPi1=None, eps_g=None, alpha_g=None, **kwargs):

    if dnu is None:
        dnu = self.obs['dnu'][0]
    if numax is None:
        numax = self.obs['numax'][0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)


     
    if DPi1 is None:
        DPi1 = self.priors['DPi1'].ppf(0.5)
    if eps_g is None:
        eps_g = self.priors['eps_g'].ppf(0.5)
    if alpha_g is None:
        alpha_g = 10**self.priors['alpha_g'].ppf(0.5)
     
    # Overplot gmode frequencies
    nu_g = self.MixFreqModel.asymptotic_nu_g(self.MixFreqModel.n_g, DPi1, eps_g, 
                                             alpha_g, 
                                             )
   
    curlyN = dnu / (DPi1 *1e-6 * numax**2)

    ylims = ax.get_ylim()
    if curlyN > 1:
        nu_g_x, nu_g_y = _echellify_freqs(nu_g, dnu)

        ax.scatter(nu_g_x, nu_g_y, color=colors[1])

    else:
        for i, nu in enumerate(nu_g):
            
            ax.axhline(nu, color='k', ls='dashed')

            if (ylims[0] < nu) & (nu < ylims[1]):
                ax.text(dnu, nu + dnu/2, s=r'$n_g$'+f'={self.MixFreqModel.n_g[i]}', ha='right', fontsize=11)

        ax.axhline(np.nan, color='k', ls='dashed', label='g-modes \n' + r'$\Delta\Pi_1=$'+f'{np.round(DPi1, decimals=0)}s \n' r'$\epsilon_g=$'+f'{np.round(eps_g, decimals=2)}')

    ax.legend(loc=1)

    return fig, ax

def _ModeIDClassPostEchelle(self, Nsamples, colors, dnu=None, numax=None, **kwargs):

    if dnu is None:
        dnu = self.result['summary']['dnu'][0]
    
    if numax is None:
        numax = self.result['summary']['numax'][0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, **kwargs)

    rect_ax = fig.add_axes([0.98, 0.135, 0.2, 0.782])   
    rect_ax.set_xlabel(r'$\sigma_{\nu,\ell=1}$')
    rect_ax.set_yticks([])
    rect_ax.set_ylim(ax.get_ylim())
    rect_ax.fill_betweenx(ax.get_ylim(), 
                  x1=self.priors['freqError0'].mean - self.priors['freqError0'].scale,
                  x2=self.priors['freqError0'].mean + self.priors['freqError0'].scale, color='k', alpha=0.1)
    rect_ax.fill_betweenx(ax.get_ylim(), 
                  x1=self.priors['freqError0'].mean - 2*self.priors['freqError0'].scale,
                  x2=self.priors['freqError0'].mean + 2*self.priors['freqError0'].scale, color='k', alpha=0.1)
    rect_ax.set_xlim(self.priors['freqError0'].mean - 3*self.priors['freqError0'].scale,
                     self.priors['freqError0'].mean + 3*self.priors['freqError0'].scale)
    rect_ax.axvline(0, alpha=0.5, ls='dotted', color='k')

    l1error = np.array([self.result['samples'][key] for key in self.result['samples'].keys() if key.startswith('freqError')]).T

    # Overplot mode frequency samples
    for l in np.unique(self.result['ell']).astype(int):

        idx_ell = self.result['ell'] == l

        freqs = self.result['samples']['freq'][:Nsamples, idx_ell]

        if l==1:
            rect_ax.plot(l1error[:Nsamples, :], self.result['samples']['freq'][:Nsamples, idx_ell], 'o', alpha=0.1, color='C4')

        smp_x, smp_y = _echellify_freqs(freqs, dnu) 

        ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[l], s=100)

        # Add to legend
        ax.scatter(np.nan, np.nan, alpha=1, color=colors[l], s=100, label=r'$\ell=$'+str(l))

    # Overplot gmode frequencies
    nu_g = self.MixFreqModel.asymptotic_nu_g(self.MixFreqModel.n_g, 
                                          self.result['summary']['DPi1'][0], 
                                          self.result['summary']['eps_g'][0], 
                                          self.result['summary']['alpha_g'][0], )
    
    ylims = ax.get_ylim()

    for i, nu in enumerate(nu_g):

        if (ylims[0] < nu) & (nu < ylims[1]):
            ax.text(dnu, nu + dnu/2, s=r'$n_g$'+f'={self.MixFreqModel.n_g[i]}', ha='right', fontsize=11)

        ax.axhline(nu, color='k', ls='dashed')

    ax.axhline(np.nan, color='k', ls='dashed', label='g-modes')

    #Overplot l=1 p-modes
    nu0_p, _ = self.AsyFreqModel.asymptotic_nu_p(numax, 
                                                dnu,  
                                                self.result['summary']['eps_p'][0], 
                                                self.result['summary']['alpha_p'][0],)
    
    nu1_p = nu0_p + self.result['summary']['d01'][0]

    nu1_p_x, nu1_p_y = _echellify_freqs(nu1_p, dnu) 

    ax.scatter(nu1_p_x, nu1_p_y, edgecolors='k', fc='None', s=100, label='p-like $\ell=1$')
    
    ax.set_xlim(0, dnu)

    ax.legend(ncols=2)
    
    return fig, ax

def _PeakbagClassPriorEchelle(self, scale, colors, **kwargs):

    dnu = self.dnu[0]
    
    numax = self.numax[0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)

    freqPriors = {key:val for key,val in self.priors.items() if 'freq' in key}

    for l in np.unique(self.ell).astype(int):

        idx_ell = self.ell == l

        nu = np.array([freqPriors[key].ppf(0.5) for key in np.array(list(freqPriors.keys()))[idx_ell]])
        
        nu_err = np.array([[freqPriors[key].ppf(0.5)-freqPriors[key].ppf(0.16), freqPriors[key].ppf(0.84)-freqPriors[key].ppf(0.5)] for key in np.array(list(freqPriors.keys()))[idx_ell]])

        nu_x, nu_y = _echellify_freqs(nu, dnu)

        ax.errorbar(nu_x, nu_y, xerr=nu_err.T, color=colors[l], fmt='o')

        # Add to legend
        ax.errorbar(-100, -100, xerr=1, color=colors[l], fmt='o', label=r'$\ell=$'+str(l))
    
    ax.set_xlim(0, dnu)

    ax.legend(loc=1)

    return fig, ax

def _PeakbagClassPostEchelle(self, Nsamples, scale, colors, **kwargs):
    
    dnu = self.dnu[0]
    
    numax = self.numax[0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)
    
    for l in np.unique(self.ell).astype(int):

        idx_ell = self.ell == l

        freqs = self.result['samples']['freq'][idx_ell, :Nsamples]

        smp_x, smp_y = _echellify_freqs(freqs, dnu) 

        ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[l], s=100)

        # Add to legend
        ax.scatter(np.nan, np.nan, alpha=1, color=colors[l], s=100, label=r'$\ell=$'+str(l))
    
    ax.legend(loc=1)

    return fig, ax 


def _baseSpectrum(ax, f, s, smoothness=0.1, xlim=[None, None], ylim=[None, None], **kwargs):
 
    ax.plot(f, s, 'k-', label='Data', alpha=0.2)
    
    smoo = smooth_power(f, s, smoothness)
    
    ax.plot(f, smoo, 'k-', label='Smoothed', lw=3, alpha=0.6)
    
    _ylim = list(ax.get_ylim())
    
    if ylim[0] is None:
        _ylim[0] = smoo.min()*0.1
    else:
        _ylim[0] = ylim[0]

    if ylim[1] is None:
        _ylim[1] = smoo.max()*1.3
    else:
        _ylim[1] = ylim[1]
    
    ax.set_ylim(_ylim)
    
    _xlim = list(ax.get_xlim())
    if xlim[0] is None:
        _xlim[0] = f.min()
    else:
        _xlim[0] = xlim[0]
    
    if xlim[1] is None:
        _xlim[1] = f.max()
    else:
        _xlim[1] = xlim[1]
    
    ax.set_xlim(_xlim)

def _StarClassSpectrum(self):
    
    fig, ax = plt.subplots(2, 1, figsize=(16,9))
    
    dnu = self.obs['dnu'][0]
    
    numax = self.obs['numax'][0]
    

    # Full frame
    _baseSpectrum(ax[0], self.f, self.s)
    
    ax[0].set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

    ax[0].set_yscale('log')
    
    ax[0].set_xscale('log')
    
    # Zoom on envelope

    xlim = [max([min(self.f), numax - 5 * dnu]), 
            min([max(self.f), numax + 5 * dnu])]

    sel = (xlim[0] <= self.f) & (self.f <= xlim[1])

    _baseSpectrum(ax[1], self.f[sel], self.s[sel])
    
    ax[1].set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')
    
    ax[1].set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax[1].legend()

    return fig, ax

def _ModeIDClassPriorSpectrum(self, N):
    
    fig, ax = plt.subplots(figsize=(16,9))

    _baseSpectrum(ax, self.f, self.s, smoothness=0.5)
 
    for i in range(N):
 
        u = np.random.uniform(0, 1, size=self.ndims)

        theta = self.ptform(u)
        
        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u, self.f)
        
        ax.plot(self.f, m, alpha=0.2, color='C3')

    ax.plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)

    ax.set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

    ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax.set_yscale('log')
    
    ax.set_xscale('log')

    ax.legend(loc=3)

    return fig, ax

def _ModeIDClassPostSpectrum(self, N):

    rint = np.random.randint(0, self.samples.shape[0], size=N)

    numax = self.obs['numax'][0]

    dnu = self.obs['dnu'][0]


    fig, ax = plt.subplots(2, 1, figsize=(16,18))
    
    _baseSpectrum(ax[0], self.f, self.s)

    xlim = [max([min(self.f), numax - (self.N_p + 2) * dnu]), 
            min([max(self.f), numax + ((self.N_p+2)/2) * dnu])]
     
    sel = (xlim[0] <= self.f) & (self.f <= xlim[1])

    _baseSpectrum(ax[1], self.f[sel], self.s[sel], ylim=[0, None])

    for k in rint:
    
        theta = self.samples[k, :]

        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u, self.f)
        
        ax[0].plot(self.f, m, color='C3', alpha=0.2)

        ax[1].plot(self.f[sel], m[sel], color='C3', alpha=0.2)

    ax[0].set_yscale('log')

    ax[0].set_xscale('log')
 
    ax[0].plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)
    
    ax[0].legend(loc=3)
    
    ax[0].set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

    for nu in self.result['summary']['freq'][0]:
        ax[1].axvline(nu, c='k', linestyle='--')
    
    ax[1].plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)

    ax[1].axvline(-100, c='k', linestyle='--', label='Median frequencies')

    ax[1].set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')
    
    ax[1].set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax[1].legend(loc=2)

    return fig, ax

def _PeakbagClassPriorSpectrum(self, N):
    
    fig, ax = plt.subplots(figsize=(16,9))

    _baseSpectrum(ax, self.f, self.snr, smoothness=0.5)

    for i in range(N):
 
        u = np.random.uniform(0, 1, size=self.ndims)

        theta = self.ptform(u)
        
        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u)
        
        ax.plot(self.f[self.sel], m, alpha=0.2, color='C3')

    ax.plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)

    ax.set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

    ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax.set_xlim(self.f[self.sel].min(), self.f[self.sel].max())

    ax.legend(loc=1)

    return fig, ax

def _PeakbagClassPostSpectrum(self, N):

    rint = np.random.randint(0, self.samples.shape[0], size=N)

    fig, ax = plt.subplots(figsize=(16,9))

    _baseSpectrum(ax, self.f, self.snr, smoothness=0.5)

    for k in rint:
    
        theta = self.samples[k, :]

        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u)
        
        ax.plot(self.f[self.sel], m, color='C3', alpha=0.2)

    ax.plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)

    ax.set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')
    
    ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax.set_xlim(self.f[self.sel].min(), self.f[self.sel].max())

    ax.legend(loc=1)

    return fig, ax


def _baseCorner(samples, labels):
     
    fig = corner.corner(np.array(samples), hist_kwargs={'density': True}, labels=labels)
    
    return fig

def _setSampleToPlot(self, samples, unpacked, labels=None):

    if unpacked:
        _samples = self.unpackSamples(samples)

    else:
        _samples = {label: samples[:, i] for i, label in enumerate(self.priors.keys())}

    return _samples

def _ModeIDClassPriorCorner(self, samples, labels, unpacked, **kwargs):

    _samples = _setSampleToPlot(self, samples, unpacked)

    if labels == None:
        labels = list(_samples.keys())
        
    fig = _baseCorner(_samples, labels)    

    axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

    if not unpacked:
        for i, key in enumerate(labels):
            if key in self.priors.keys():
            
                x = np.linspace(self.priors[key].ppf(1e-6), 
                                self.priors[key].ppf(1-1e-6), 100)

                pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])

                axes[i, i].plot(x, pdf, color='C3', alpha=0.5, lw =5)
        
    return fig, axes

def _ModeIDClassPostCorner(self, samples, labels, unpacked, **kwargs):

    _samples = _setSampleToPlot(self, samples, unpacked)

    if labels == None:
        labels = list(_samples.keys())

    fig = _baseCorner(_samples, labels)  

    axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

    if not unpacked:
        for i, key in enumerate(labels):
        
            if key in self.priors.keys():
            
                x = np.linspace(self.priors[key].ppf(1e-6), self.priors[key].ppf(1-1e-6), 100)

                pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])

                axes[i, i].plot(x, pdf, color='C3', alpha=0.5, lw =5) 

    return fig, axes

def _PeakbagClassPriorCorner(self, samples, labelType, **kwargs):
     
    _samples = self.unpackSamples(samples)

    subSamples = {key: v for key, v in _samples.items() if any([l in key for l in labelType])}

    plotLabels = list(subSamples.keys())

    fig = _baseCorner(subSamples, plotLabels)    

    axes = np.array(fig.get_axes()).reshape((len(plotLabels), len(plotLabels)))
     
    for i, key in enumerate(subSamples.keys()):
         
        x = np.linspace(self.priors[key].ppf(1e-6), 
                        self.priors[key].ppf(1-1e-6), 100)

        pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])
        
        isLog10 = [self.variables[varKey]['log10'] for varKey in self.variables if key.startswith(varKey)][0]

        if isLog10:
            axes[i, i].plot(10**x, pdf/10**x/np.log(10.0), color='C2', alpha=0.5, lw=5)
        else:
            axes[i, i].plot(x, pdf, color='C2', alpha=0.5, lw=5)

        if  any([key.startswith(l) for l in ['height', 'freq', 'width']]):
            axes[i,i].patch.set_facecolor(self.ell[i])
            axes[i,i].patch.set_alpha(0.25)

    return fig, axes

def _PeakbagClassPostCorner(self, samples, labelType, colors, **kwargs):

    _samples = self.unpackSamples(samples)

    subSamples = {key: v for key, v in _samples.items() if any([l in key for l in labelType])}
    
    plotLabels = list(subSamples.keys())

    fig = _baseCorner(subSamples, plotLabels)    

    axes = np.array(fig.get_axes()).reshape((len(plotLabels), len(plotLabels)))
     
    for i, key in enumerate(subSamples.keys()):
         
        x = np.linspace(self.priors[key].ppf(1e-6), 
                        self.priors[key].ppf(1-1e-6), 100)

        pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])
        
        isLog10 = [self.variables[varKey]['log10'] for varKey in self.variables if key.startswith(varKey)][0]
        
        if isLog10:
            axes[i, i].plot(10**x, pdf/10**x/np.log(10.0), color='C2', alpha=0.5, lw=5)
        else:
            axes[i, i].plot(x, pdf, color='C2', alpha=0.5, lw=5)

        if any([key.startswith(l) for l in ['height', 'freq', 'width']]):
            axes[i,i].patch.set_facecolor(colors[int(self.ell[i])])
            axes[i,i].patch.set_alpha(0.25)

    return fig, axes


class plotting():
    """ Class inherited by PBjam modules to plot results
    
    This is used to standardize the plots produced at various steps of the 
    peakbagging process. 
    
    As PBjam class is initialized, these plotting methods will be inherited.
    The methods will plot the relevant result based on the class they are being
    called from. 
    
    """
    
    def __init__(self):
        pass

    def _save_my_fig(self, fig, figtype, path, ID):
        """ Save the figure object
        
        Saves the figure object with a predefined path name pattern.

        Parameters
        ----------
        fig : Matplotlib figure object
            Figure object to be saved.
        figtype : str
            The type of figure in question. This is used to set the filename.
        path : str, optional
            Used along with savefig, sets the output directory to store the
            figure. Default is to save the figure to the star directory.
        ID : str, optional
            ID of the target to be included in the filename of the figure.

        """
        
        # TODO there should be a check if path is full filepath or just dir

        if path and ID:
            outpath = os.path.join(*[path,  type(self).__name__+f'_{figtype}_{str(ID)}.png'])
            fig.savefig(outpath)

    def echelle(self, stage='posterior', ID=None, savepath=None, kwargs={}, save_kwargs={}):

        if not 'colors' in kwargs:
            kwargs['colors'] = ellColors

        if not 'scale' in kwargs:
            kwargs['scale'] = 1/350

        if not 'Nsamples' in kwargs:
            kwargs['Nsamples'] = 200
        
        # TODO the self.__class__.__name__ check is instead of isinstance
        # since it plays better with auto reload
        if self.__class__.__name__ == 'star': 

            fig, ax = _StarClassEchelle(self, self.obs, **kwargs)
            
        elif self.__class__.__name__ == 'modeIDsampler':

            if stage=='prior':
                fig, ax = _ModeIDClassPriorEchelle(self, **kwargs)

            elif stage=='posterior': 
                
                fig, ax = _ModeIDClassPostEchelle(self, **kwargs)
                
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')

        elif self.__class__.__name__ == 'peakbag':  
            
            if stage=='prior':
                fig, ax = _PeakbagClassPriorEchelle(self, **kwargs)
                
            elif stage=='posterior':
                fig, ax = _PeakbagClassPostEchelle(self, **kwargs)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')

        else:
            raise ValueError('Unrecognized class type')

        if ID is not None:
            ax.set_title(ID)

                       
        fig.tight_layout()

        if (savepath is not None):
            fig.savefig(savepath, **save_kwargs)

        return fig, ax

    def spectrum(self, stage='posterior', ID=None, savepath=None, kwargs={}, save_kwargs={}, N=20):

        if self.__class__.__name__ == 'star': 
            
            _StarClassSpectrum(self, **kwargs)

        elif self.__class__.__name__ == 'modeIDsampler':
            if stage=='prior':

                fig, ax = _ModeIDClassPriorSpectrum(self, N, **kwargs)

            elif stage=='posterior': 

                assert hasattr(self, 'result')

                fig, ax = _ModeIDClassPostSpectrum(self, N, **kwargs)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')

        elif self.__class__.__name__ == 'peakbag': 

            if stage=='prior':
                fig, ax = _PeakbagClassPriorSpectrum(self, N, **kwargs)
                
            elif stage=='posterior':
                fig, ax = _PeakbagClassPostSpectrum(self, N, **kwargs)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')
        
        else:
            raise ValueError('Unrecognized class type')
        
        if ID is not None:
            ax.set_title(ID)
                        
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, **save_kwargs)

        return fig, ax

    def corner(self, stage='posterior', ID=None, labels=None, savepath=None, unpacked=False, kwargs={}, save_kwargs={}, N=5000):
         
        if not 'colors' in kwargs:
            kwargs['colors'] = ellColors

        if self.__class__.__name__ == 'star': 
            
            _StarClassCorner(self, **kwargs)

        elif self.__class__.__name__ == 'modeIDsampler':
             
            if stage=='prior':
                print('bla')
                samples = np.array([self.ptform(np.random.uniform(0, 1, size=self.ndims)) for i in range(N)])

                fig, ax = _ModeIDClassPriorCorner(self, samples, labels, unpacked, **kwargs)
            
            elif stage=='posterior': 
                 
                samples = self.samples

                fig, ax = _ModeIDClassPostCorner(self, samples, labels, unpacked, **kwargs)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')
            
        elif self.__class__.__name__ == 'peakbag': 
             
            if stage=='prior':

                samples = np.array([self.ptform(np.random.uniform(0, 1, size=self.ndims)) for i in range(N)])

                fig, ax = _PeakbagClassPriorCorner(self, samples, labels, **kwargs)
                
            elif stage=='posterior':

                samples = self.samples

                fig, ax = _PeakbagClassPostCorner(self, samples, labels, **kwargs)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')
            
        else:
            raise ValueError('Unrecognized class type')
        
        return fig, ax

    def plotLatentCorner(self, samples, labels=None):
    
        if labels == None:
            labels = list(self.priors.keys()) 
        
        fig = corner.corner(samples, hist_kwargs = {'density': True}, labels=labels)

        axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

        for i, key in enumerate(labels):
        
            if key in self.priors.keys():
            
                x = np.linspace(self.priors[key].ppf(1e-6), self.priors[key].ppf(1-1e-6), 100)

                pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])
    
                axes[i, i].plot(x, pdf, color='C3', alpha=0.5, lw =5)    

    def plot_corner(self, path=None, ID=None, savefig=False):
        """ Make corner plot of result.
        
        Makes a nice corner plot of the fit parameters.

        Parameters
        ----------
        path : str, optional
            Used along with savefig, sets the output directory to store the
            figure. Default is to save the figure to the star directory.
        ID : str, optional
            ID of the target to be included in the filename of the figure.
        savefig : bool
            Whether or not to save the figure to disk. Default is False.

        Returns
        -------
        fig : Matplotlib figure object
            Figure object with the corner plot.

        """

        if not hasattr(self, 'samples'):
            warnings.warn(f"'{self.__class__.__name__}' has no attribute 'samples'. Can't plot a corner plot.")
            return None

        fig = corner.corner(self.samples, labels=self.par_names,
                            show_titles=True, quantiles=[0.16, 0.5, 0.84],
                            title_kwargs={"fontsize": 12})

        if savefig:
            self._save_my_fig(fig, 'corner', path, ID)

        return fig




    def _fill_diag(self, axes, vals, vals_err, idxs):
        """ Overplot diagnoal values along a corner plot diagonal.
        
        Plots a set of specified values over the 1D histograms in the diagonal 
        frames of a corner plot.

        Parameters
        ----------
        axes : Matplotlib axis object
            The particular axis element to be plotted in.
        vals : float
            Mean values to plot.
        vals_err : float
            Error estimates for the value to be plotted.
        idxs : list
            List of 2D indices that represent the diagonal. 

        """

        N = int(np.sqrt(len(axes)))
        axs = np.array(axes).reshape((N,N)).T
        
        for i,j in enumerate(idxs):
            yrng = axs[j,j].get_ylim()
            
            v, ve = vals[i], vals_err[i]
            
            axs[j,j].fill_betweenx(y=yrng, x1= v-ve[0], x2 = v+ve[-1], color = 'C3', alpha = 0.5)
    
    def _plot_offdiag(self, axes, vals, vals_err, idxs):
        """ Overplot offdiagonal values in a corner plot.
        
        Plots a set of specified values over the 2D histograms or scatter in the
        off-diagonal frames of a corner plot.

        Parameters
        ----------
        axes : Matplotlib axis object
            The particular axis element to be plotted in.
        vals : float
            Mean values to plot.
        vals_err : float
            Error estimates for the value to be plotted.
        idxs : list
            List of 2D indices that represent the diagonal. 

        """
        
        N = int(np.sqrt(len(axes)))

        axs = np.array(axes).reshape((N,N)).T
        
        for i, j in enumerate(idxs):
            
            for m, k in enumerate(idxs):
                if j >= k:
                    continue
                    
                v, ve = vals[i], vals_err[i]

                w, we = vals[m], vals_err[m]
    
                axs[j,k].errorbar(v, w, xerr=ve.reshape((2,1)), yerr=we.reshape((2,1)), fmt = 'o', ms = 10, color = 'C3')

    def _make_prior_corner(self, df, numax_rng = 100):
        """ Show dataframe contents in a corner plot.
        
        This is meant to be used to show the contents of the prior_data that is
        used by KDE and Asy_peakbag. 

        Parameters
        ----------
        df : pandas.Dataframe object
            Dataframe of the data to be shown in the corner plot.
        numax_rng : float, optional
            Range in muHz around the input numax to be shown in the corner plot.
            The default is 100. 

        Returns
        -------
        crnr : matplotlib figure object
            Corner plot figure object containing NxN axis objects.
        crnr_axs : list
            List of axis objects from the crnr object.

        """
        
        idx = abs(10**df['numax'] - self._obs['numax'][0]) <= numax_rng
        
        # This is a temporary fix for disabling the 'Too few samples' warning 
        # from corner.hist2d. The github bleeding edge version has 
        # hist2d_kwargs = {'quiet': True}, but this isn't in the pip 
        # installable version yet. March 2020.
        logging.disable(logging.WARNING)
        crnr = corner.corner(df.to_numpy()[idx,:-1], data_kwargs = {'alpha': 0.5}, labels = df.keys());
        logging.getLogger().setLevel(logging.WARNING)
    
        return crnr,  crnr.get_axes()
        


    def plot_prior(self, path=None, ID=None, savefig=False):
        """ Corner of result in relation to prior sample.
        
        Create a corner plot showing the location of the star in relation to
        the rest of the prior.

        Parameters
        ----------
        path : str, optional
            Used along with savefig, sets the output directory to store the
            figure. Default is to save the figure to the star directory.
        ID : str, optional
            ID of the target to be included in the filename of the figure.
        savefig : bool
            Whether or not to save the figure to disk. Default is False.

        Returns
        -------
        crnr : matplotlib figure object
            Corner plot figure object containing NxN axis objects.
            
        """
        
        df = pd.read_csv(self.prior_file)
        crnr, axes = self._make_prior_corner(df)
       
        if type(self) == pbjam.star:
            vals, vals_err = np.array([self._log_obs['dnu'],
                                       self._log_obs['numax'], 
                                       self._log_obs['teff'], 
                                       self._obs['bp_rp']]).T
            
            vals_err = np.vstack((vals_err, vals_err)).T
            self._fill_diag(axes, vals, vals_err, [0, 1, 8, 9])
            self._plot_offdiag(axes, vals, vals_err, [0, 1, 8, 9])
            
            # break this if statement if asy_fit should plot something else
        elif (type(self) == pbjam.priors.kde) or (type(self) == pbjam.asy_peakbag.asymptotic_fit): 
            percs = np.percentile(self.samples, [16, 50, 84], axis=0)
            vals = percs[1,:]
            vals_err = np.diff(percs, axis = 0).T
            
            self._fill_diag(axes, vals, vals_err, range(len(vals)))
            self._plot_offdiag(axes, vals, vals_err, range(len(vals)))

        elif type(self) == pbjam.peakbag:
            raise AttributeError('The result of the peakbag run cannot be plotted in relation to the prior, since it does not know what the prior is anymore. plot_corner is only available for star, kde and asy_peakbag.')


        return crnr
    
    # def plot_start(self):
    #     """ Plot starting point for peakbag
        
    #     Plots the starting model to be used in peakbag as a diagnotstic.
        
    #     """

    #     dnu = 10**np.median(self.start_samples, axis=0)[0]
    #     xlim = [min(self.f[self.sel])-dnu, max(self.f[self.sel])+dnu]
    #     fig, ax = plt.subplots(figsize=[16,9])
    #     ax.plot(self.f, self.s, 'k-', label='Data', alpha=0.2)
    #     smoo = dnu * 0.005 / (self.f[1] - self.f[0])
    #     kernel = conv.Gaussian1DKernel(stddev=smoo)
    #     smoothed = conv.convolve(self.s, kernel)
    #     ax.plot(self.f, smoothed, 'k-', label='Smoothed', lw=3, alpha=0.6)
    #     ax.plot(self.f[self.sel], self.model(self.start_samples.mean(axis=0)),
    #             'r-', label='Start model', alpha=0.7)
    #     ax.set_ylim([0, smoothed.max()*1.5])
    #     ax.set_xlim(xlim)
    #     ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
    #     ax.set_ylabel(r'SNR')
    #     ax.legend()
    #     return fig

