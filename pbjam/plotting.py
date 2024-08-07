""" 

This module contains a general set of plotting methods for that are inherited by
the different classes of PBjam, so that they can be used to show the status of 
each step that has been performed. 

"""

import matplotlib.pyplot as plt
import astropy.convolution as conv
import os, corner, warnings, logging
import numpy as np
import jax

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

    smoo = conv.convolve_fft(power, kernel)

    return smoo


def echelle(freq, power, dnu, fmin=0.0, fmax=None, offset=0.0, sampling=1):
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

    for x in np.arange(echy.min(), echy.max(), dnu):
        ax.axhline(x, color='k', alpha=0.1)
        
    return ax


def _scatterFrame(model, samples, key1, key2, ax,):
     
    df = model.DR.priorData
    
    # Relevant bit of prior_data.csv
    prior1 = df[key1]
    
    prior2 = df[key2]

    # Sample used to construct prior
    select1 = model.DR.selectedSubset[key1]
    
    select2 = model.DR.selectedSubset[key2]
    
    # Samples from the sampling
    samplesU = model.unpackSamples(samples)
    
    # Compute some limits
    dx = (select1.max()-select1.min())*0.25
    minx = select1.min()-dx
    
    maxx = select1.max()+dx
    
    dy = (select2.max()-select2.min())*0.25
     
    miny = select2.min()-dy
    
    maxy = select2.max()+dy
    
    # Plot the things
    pidx = (minx < prior1) & (prior1 < maxx) & (miny < prior2) & (prior2 < maxy)
 
    ax.scatter(prior1[pidx], prior2[pidx], ec='k', alpha=0.25, s=8, fc='None')

    ax.scatter(select1, select2, c='C3', alpha=0.55, s=15)
  
    ax.set_xlim(minx, maxx)
    
    ax.set_ylim(miny, maxy)

    if not np.isnan(samples).all():
        # Only plot the summary stats
        result1 = np.percentile(samplesU[key1], [15, 50, 85])
        
        result2 = np.percentile(samplesU[key2], [15, 50, 85])
        
        if key1 in model.logpars:
            result1 = np.log10(result1)
        if key2 in model.logpars:
            result2 = np.log10(result2)
        
        ax.errorbar(result1[1], 
                    result2[1], 
                    xerr=np.array([[result1[1]-result1[0]], [result1[2]-result1[1]]]), 
                    yerr=np.array([[result2[1]-result2[0]], [result2[2]-result2[1]]]), 
                    fmt='.-', lw=5, ms=25, markeredgecolor='k', color='C0')
    
def _baseReference(model, samples, fac=3):

    labels = model.pcaLabels + model.addLabels

    labelsInfile = [label for label in labels if label in model.DR.priorData.keys()]

    Nf = len(labelsInfile)-1
    
    fig, axes = plt.subplots(Nf, Nf, figsize=(fac*Nf, fac*Nf))

    for i in range(Nf):
        for j in range(Nf):
            ax = axes[j, i]

            key1 = labelsInfile[i]

            key2 = labelsInfile[j+1]

            if j >= i:
                _scatterFrame(model, samples, key1, key2, ax)
            else:
                ax.set_visible(False)
                
            if i != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(key2)

            if j != Nf-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(key1)
      
    axes[0, 0].scatter(np.nan, np.nan, ec='k', alpha=1, s=8, fc='None', label=f'Viable prior sample')
    
    axes[0, 0].scatter(np.nan, np.nan, c='C3', alpha=0.55, s=15, label='Selected prior sample')
    
    if not np.isnan(samples).all():
        axes[0, 0].errorbar(np.nan, np.nan, xerr=np.array([[np.nan], [np.nan]]), yerr=np.array([[np.nan], [np.nan]]), 
                            fmt='.-', lw=5, ms=25, markeredgecolor='k', color='C0', label='Result summary statistics')

    axes[0, 0].legend(bbox_to_anchor=(4, 1), fontsize=24, markerscale=2.)
 
    return fig, axes

def _ModeIDPriorReference(model, N=1000):

    samples = np.zeros_like(model.samples[:N, :]) * np.nan

    fig, axes = _baseReference(model, samples)

    return fig, axes

def _ModeIDPosteriorReference(model, N=1000):
    
    samples = model.samples[:N, :].copy()

    fig, axes = _baseReference(model, samples)

    return fig, axes


@jax.jit
def _echellify_freqs(nu, dnu, offset=0):
    x = (nu - offset*dnu)  % dnu  

    y =  nu  

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
        
    plot_echelle(f, s, dnu, ax=ax, smooth=True, smooth_filter_width=dnu * scale, **kwargs)

    return fig, ax

def _ModeIDClassPriorEchelle(self, Nsamples, scale, colors, dnu=None, numax=None, 
                             DPi1=None, eps_g=None, **kwargs):

    if dnu is None:
        dnu = self.obs['dnu'][0]

    if numax is None:
        numax = self.obs['numax'][0]

    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)

    if hasattr(self, 'l20model'):
 
        junpackl20 = jax.jit(self.l20model.unpackParams)
 
        jptforml20 = jax.jit(self.l20model.ptform)

        jasy_nu_p = jax.jit(self.l20model.asymptotic_nu_p)

        for _ in range(Nsamples):

            u = np.random.uniform(0, 1, size=self.l20model.ndims)
        
            theta = jptforml20(u)

            thetaU = junpackl20(theta)
            
            nu0p, _ = jasy_nu_p(**thetaU)

            nu2p  = nu0p + thetaU['d02']

            for freqs, ell in zip([nu0p, nu2p], [0, 2]):
                smp_x, smp_y = _echellify_freqs(freqs, dnu) 

                ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[ell], s=100)
 
        for ell in [0, 2]:
            ax.scatter(np.nan, np.nan, alpha=1, color=colors[ell], s=100, label=r'$\ell=$'+str(ell))

    if hasattr(self, 'l1model'):

        junpackl1 = jax.jit(self.l1model.unpackParams)
 
        jptforml1 = jax.jit(self.l1model.ptform)

        jnu = jax.jit(self.l1model.nu1_frequencies)

        for _ in range(Nsamples):

            u = np.random.uniform(0, 1, size=self.l1model.ndims)
        
            theta = jptforml1(u)

            thetaU = junpackl1(theta)
            
            nu1 = jnu(thetaU)
            
            smp_x, smp_y = _echellify_freqs(nu1, dnu) 

            ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[1], s=100)

        ax.scatter(np.nan, np.nan, alpha=1, color=colors[1], s=100, label=r'$\ell=$'+str(1))
 
        # if DPi1 is None:
        #     DPi1 = self.l1model.priors['DPi1'].ppf(0.5)
        # if eps_g is None:
        #     eps_g = self.l1model.priors['eps_g'].ppf(0.5)

        # # Overplot gmode frequencies
        # nu_g = self.l1model.asymptotic_nu_g(self.l1model.n_g, DPi1, eps_g,) 

        # curlyN = dnu / (DPi1 *1e-6 * numax**2)

        # ylims = ax.get_ylim()

        # if curlyN > 1:
        #     nu_g_x, nu_g_y = _echellify_freqs(nu_g, dnu)

        #     ax.scatter(nu_g_x, nu_g_y, color=colors[1])

        # else:
        #     for i, nu in enumerate(nu_g):
            
        #         ax.axhline(nu, color='k', ls='dashed')

        #         if (ylims[0] < nu) & (nu < ylims[1]):
        #             ax.text(dnu, nu + dnu/2, s=r'$n_g$'+f'={self.l1model.n_g[i]}', ha='right', fontsize=11)

        #     ax.axhline(np.nan, color='k', ls='dashed', label='g-modes \n' + r'$\Delta\Pi_1=$'+f'{np.round(DPi1, decimals=0)}s \n' r'$\epsilon_g=$'+f'{np.round(eps_g, decimals=2)}')
 
    ax.set_xlim(0, dnu)
    
    ax.legend(loc=1)

    return fig, ax

def _ModeIDClassPostEchelle(self, Nsamples, colors, dnu=None, numax=None, **kwargs):

    if (dnu is None) and hasattr(self, 'result'):
        dnu = self.result['summary']['dnu'][0]
    else:
        dnu = self.obs['dnu'][0]
    
    
    if (numax is None) and hasattr(self, 'result'):
        numax = self.result['summary']['numax'][0]
    else:
        numax = self.obs['numax'][0]

    offset = (self.result['summary']['eps_p'][0])  - 0.25
     
    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, offset=offset * dnu, **kwargs)

    axes = np.array([ax])
    if hasattr(self, 'result'):
        for l in np.unique(self.result['ell']).astype(int):

            idx_ell = (self.result['ell'] == l ) & (self.result['emm'] == 0)
            
            freqs = self.result['samples']['freq'][:Nsamples, idx_ell]

            smp_x, smp_y = _echellify_freqs(freqs, dnu, offset) 

            ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[l], s=100)

            med_freqs = self.result['summary']['freq'][0, self.result['emm'] == 0]

            med_x, med_y = _echellify_freqs(med_freqs, dnu, offset) 

            #ax.scatter(med_x, med_y, alpha=1, s=100, facecolors='none', edgecolors='k', linestyle='--')

            # Add to legend
            ax.scatter(np.nan, np.nan, alpha=1, color=colors[l], s=100, label=r'$\ell=$'+str(l))

    ylims = ax.get_ylim()

    # If fudge frequencies are used plot those
    if hasattr(self, 'l1model') and 'freqError0' in self.result['summary'].keys():

        rect_ax = fig.add_axes([0.92, 0.107, 0.2, 0.775])   
        rect_ax.set_xlabel(r'$\sigma_{\nu,\ell=1}$')
        rect_ax.set_yticks([])
        rect_ax.set_ylim(ax.get_ylim())
    
        rect_ax.fill_betweenx(ax.get_ylim(), 
                              x1=self.l1model.priors['freqError0'].mean - self.l1model.priors['freqError0'].scale,
                              x2=self.l1model.priors['freqError0'].mean + self.l1model.priors['freqError0'].scale, color='k', alpha=0.1)
    
        rect_ax.fill_betweenx(ax.get_ylim(), 
                              x1=self.l1model.priors['freqError0'].mean - 2*self.l1model.priors['freqError0'].scale,
                              x2=self.l1model.priors['freqError0'].mean + 2*self.l1model.priors['freqError0'].scale, color='k', alpha=0.1)
    
        rect_ax.set_xlim(self.l1model.priors['freqError0'].mean - 3*self.l1model.priors['freqError0'].scale,
                         self.l1model.priors['freqError0'].mean + 3*self.l1model.priors['freqError0'].scale)
    
        rect_ax.axvline(0, alpha=0.5, ls='dotted', color='k')

        l1error = np.array([self.result['samples'][key] for key in self.result['samples'].keys() if key.startswith('freqError')]).T
         
        rect_ax.plot(l1error[:Nsamples, :], self.result['samples']['freq'][:Nsamples, (self.result['ell']==1) & (self.result['emm']==0)], 'o', alpha=0.1, color='C4')
        axes = np.append(axes, ax)

    # Overplot gmode frequencies
    if hasattr(self, 'l1model'):
        if self.l1model.N_g > 0:

            curlyN = dnu / (self.result['summary']['DPi1'][0] *1e-6 * numax**2)
            
            if curlyN < 1:
                nu_g = self.l1model.asymptotic_nu_g(self.l1model.n_g, 
                                                    self.result['summary']['DPi1'][0], 
                                                    self.result['summary']['eps_g'][0], 
                                                    )
                                                    
                for i, nu in enumerate(nu_g):

                    if (ylims[0] < nu) & (nu < ylims[1]):
                        ax.text(dnu, nu + dnu/2, s=r'$n_g$'+f'={self.l1model.n_g[i]}', ha='right', fontsize=11)

                    ax.axhline(nu, color='k', ls='dashed')

                ax.axhline(np.nan, color='k', ls='dashed', label='g-modes')

        #Overplot l=1 p-modes
        nu0_p, _ = self.l20model.asymptotic_nu_p(self.result['summary']['numax'][0], 
                                                self.result['summary']['dnu'][0],  
                                                self.result['summary']['eps_p'][0], 
                                                self.result['summary']['alpha_p'][0],)

        nu1_p = nu0_p + self.result['summary']['d01'][0]

        nu1_p_x, nu1_p_y = _echellify_freqs(nu1_p, dnu) 

        #ax.scatter(nu1_p_x, nu1_p_y, edgecolors='k', fc='None', s=100, label='p-like $\ell=1$')
                        
    
 
    ax.set_xlim(0, dnu)

    ax.legend(ncols=len(np.unique(self.result['ell'])), loc=1)
    
    return fig, axes

def _PeakbagClassPriorEchelle(self, scale, colors, dnu=None, numax=None, **kwargs):

    if dnu is None:
        dnu = self.dnu[0]
    
    if numax is None:
        numax = np.median(self.freq)
   
    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)

    maxL = 0

    for inst in self.pbInstances:
        freqPriors = {key:val for key,val in inst.priors.items() if 'freq' in key}
        
        for l in np.unique(inst.ell).astype(int):

            idx_ell = inst.ell == l

            nu = np.array([freqPriors[key].loc for key in np.array(list(freqPriors.keys()))[idx_ell]])
            
            nu_err = np.array([freqPriors[key].scale for key in np.array(list(freqPriors.keys()))[idx_ell]])
        
            nu_x, nu_y = _echellify_freqs(nu, dnu)

            ax.errorbar(nu_x, nu_y, xerr=nu_err, color=colors[l], fmt='o')

            maxL = max([maxL, l])

    # Add to legend
    for l in range(maxL+1):
        ax.errorbar(-100, -100, xerr=1, color=colors[l], fmt='o', label=r'$\ell=$'+str(l))
        
    ax.set_xlim(0, dnu)

    ax.legend(loc=1)

    return fig, ax

def _PeakbagClassPostEchelle(self, Nsamples, scale, colors, dnu=None, numax=None, **kwargs):
    
    
    if dnu is None:
        dnu = self.dnu[0]
    
    if numax is None:
        numax = np.median(self.freq[0, :])
 
    fig, ax = _baseEchelle(self.f, self.s, self.N_p, numax, dnu, scale)
    
    maxL = 0

    for inst in self.pbInstances:
        for l in np.unique(inst.ell).astype(int):

            idx_ell = inst.ell == l

            freqs = inst.result['samples']['freq'][:Nsamples, idx_ell]

            smp_x, smp_y = _echellify_freqs(freqs, dnu) 

            ax.scatter(smp_x, smp_y, alpha=0.05, color=colors[l], s=100)

            maxL = max([maxL, l])

    # Add to legend
    for l in range(maxL+1):
        ax.scatter(np.nan, np.nan, alpha=1, color=colors[l], s=100, label=r'$\ell=$'+str(l))
    
    ax.legend(loc=1)

    return fig, ax 


def _baseSpectrum(ax, f, s, smoothness=0.1, xlim=[None, None], ylim=[None, None], **kwargs):
 
    #ax.plot(f, s, 'k-', label='Data', alpha=0.1)
    
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

def _makeBaseFrames(self):

    if not hasattr(self, 'l20model'):
        fig, ax = plt.subplots(2, 1, figsize=(16,18))

        _baseSpectrum(ax[0], self.f, self.s)

        _baseSpectrum(ax[1], self.f[self.sel], self.s[self.sel])


    elif hasattr(self, 'l20model') and not hasattr(self, 'l1model'): # only l20 has been run

        fig, ax = plt.subplots(3, 1, figsize=(16,18))

        _baseSpectrum(ax[0], self.f, self.s)

        _baseSpectrum(ax[1], self.f[self.sel], self.s[self.sel])

        _baseSpectrum(ax[2], self.f[self.sel], self.s[self.sel] / self.l20model.getMedianModel())
    
    elif hasattr(self, 'l20model') and hasattr(self, 'l1model'):

        fig, ax = plt.subplots(4, 1, figsize=(16,18))

        _baseSpectrum(ax[0], self.f, self.s)

        _baseSpectrum(ax[1], self.f[self.sel], self.s[self.sel])

        _baseSpectrum(ax[2], self.f[self.sel], self.s[self.sel] / self.l20model.getMedianModel())

        _baseSpectrum(ax[3], self.f[self.sel], self.l20residual / self.l1model.getMedianModel())
    
    else:
        raise ValueError('Unable to make plots')
  
    ax[0].set_xlim(self.f.min(), self.f.max())
    
    ax[0].set_yscale('log')

    ax[0].set_xscale('log')
         
    for i in range(ax.shape[0]):
        ax[i].set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

        if i > 0:
            ax[i].set_xlim(self.f[self.sel].min(), 
                        self.f[self.sel].max())
            
        if i > 1:
            ax[i].set_ylabel(r'Residual')

    ax[-1].set_xlabel(r'Frequency ($\mu \rm Hz$)')
   
    return fig, ax

def _ModeIDClassPriorSpectrum(self, N):
     
    fig, ax = _makeBaseFrames(self)
 
    if hasattr(self, 'l20model'):

        rint = np.random.randint(0, len(self.result['samples']['dnu']), size=N)

        junpackl20 = jax.jit(self.l20model.unpackParams)

        jmodell20 = jax.jit(self.l20model.model)

        jptforml20 = jax.jit(self.l20model.ptform)

        for k in rint:
            
            u = np.random.uniform(0, 1, size=self.l20model.ndims)
        
            theta = jptforml20(u)

            thetaU = junpackl20(theta)
            
            mod = jmodell20(thetaU)

            ax[0].plot(self.f[self.sel], mod, color='C3', alpha=0.2)

            ax[1].plot(self.f[self.sel], mod, color='C3', alpha=0.2)

        ax[0].plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)
        ax[1].plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)
    
    if hasattr(self, 'l1model'):

        rint = np.random.randint(0, len(self.result['samples']['d01']), size=N)

        junpackl1 = jax.jit(self.l1model.unpackParams)

        jmodell1 = jax.jit(self.l1model.model)

        jptforml1 = jax.jit(self.l1model.ptform)

        for k in rint:
            u = np.random.uniform(0, 1, size=self.l1model.ndims)
        
            theta = jptforml1(u)
        
            thetaU = junpackl1(theta)
            
            mod = jmodell1(thetaU,)

            ax[2].plot(self.f[self.sel], mod, color='C3', alpha=0.2)

        ax[2].plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)
  
    ax[0].legend(loc=3)
     
    return fig, ax
 
def _ModeIDClassPostSpectrum(self, N):
 
    rint = np.random.randint(0, len(self.result['samples']['dnu']), size=N)

    fig, ax = _makeBaseFrames(self)
 
    if hasattr(self, 'l20model'):

        junpackl20 = jax.jit(self.l20model.unpackParams)

        jmodell20 = jax.jit(self.l20model.model)

        for k in rint:
        
            thetaU = junpackl20(self.l20Samples[k, :])
            
            mod = jmodell20(thetaU)

            ax[0].plot(self.f[self.sel], mod, color='C3', alpha=0.2)

            ax[1].plot(self.f[self.sel], mod, color='C3', alpha=0.2)
    
    if hasattr(self, 'l1model'):

        junpackl1 = jax.jit(self.l1model.unpackParams)

        jmodell1 = jax.jit(self.l1model.model)

        for k in rint:
            thetaU = junpackl1(self.l1Samples[k, :])
            
            mod = jmodell1(thetaU,)

            ax[2].plot(self.f[self.sel], mod, color='C3', alpha=0.2)

            llim, ulim = ax[2].get_ylim()
             
            line_bottom = ulim - 0.1*(ulim-llim)

            nu_l0 = self.result['summary']['freq'][0, self.result['ell']==0]

            for _, nu_l0 in enumerate(nu_l0):
                ax[2].plot([nu_l0, nu_l0],[line_bottom, ulim], lw=5, color=ellColors[0])

            nu_l2 = self.result['summary']['freq'][0, self.result['ell']==2]

            for _, nu_l2 in enumerate(nu_l2):
                ax[2].plot([nu_l2, nu_l2],[line_bottom, ulim], lw=5, color=ellColors[2])
  
    ax[0].plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)
    
    ax[0].legend(loc=3)
    
    # for i in range(1, ax.shape[0]):
    #     for j, nu in enumerate(self.result['summary']['freq'][0]):
            
    #         if (i==1 and self.result['ell'][j]==1) or (i==2 and self.result['ell'][j]!=1):
    #             _alpha=0.35

    #         else:
    #             _alpha=1.0

    #         ax[i].axvline(nu, c='k', linestyle='--', alpha=_alpha, lw=3)
        
    #     ax[i].plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)

    #     ax[i].axvline(-100, c='k', linestyle='--', label='Median frequencies')
 
    #     ax[i].legend(loc=2)
 
    return fig, ax

def _PeakbagClassPriorSpectrum(self, N):
    
    fig, ax = plt.subplots(figsize=(16,9))

    _baseSpectrum(ax, self.f, self.snr, smoothness=0.5)

    for inst in self.pbInstances:

        junpack = jax.jit(inst.unpackParams)

        jmodel = jax.jit(inst.model)

        jptform = jax.jit(inst.ptform)

        for _ in range(N):
    
            u = np.random.uniform(0, 1, size=inst.ndims)

            theta = jptform(u)
            
            theta_u = junpack(theta)
             
            m = jmodel(theta_u)
             
            ax.plot(inst.f[inst.sel], m, alpha=0.2, color='C3')

    xlims = [float(min([min(inst.f[inst.sel]) for inst in self.pbInstances])),
             float(max([max(inst.f[inst.sel]) for inst in self.pbInstances]))]

    ax.plot([-100, -100], [-100, -100], color='C3', label='Prior samples', alpha=1)

    ax.set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')

    ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax.set_xlim(xlims)

    ax.legend(loc=1)

    return fig, ax

def _PeakbagClassPostSpectrum(self, N):

    fig, ax = plt.subplots(figsize=(16,9))

    _baseSpectrum(ax, self.f, self.snr, smoothness=0.1)

    for inst in self.pbInstances:

        randInt = np.random.randint(0, inst.samples.shape[0], size=N)
        
        junpack = jax.jit(inst.unpackParams)

        jmodel = jax.jit(inst.model)

        for k in randInt:
        
            theta = inst.samples[k, :]

            theta_u = junpack(theta)
            
            m = jmodel(theta_u)
            
            ax.plot(inst.f[inst.sel], m, color='C3', alpha=0.2)

    xlims = [float(min([min(inst.f[inst.sel]) for inst in self.pbInstances])),
             float(max([max(inst.f[inst.sel]) for inst in self.pbInstances]))]
                     
    ax.plot([-100, -100], [-100, -100], color='C3', label='Posterior samples', alpha=1)

    ax.set_ylabel(r'PSD [$\mathrm{ppm}^2/\mu \rm Hz$]')
    
    ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')

    ax.set_xlim(xlims)

    #ax.legend(loc=1)

    return fig, ax


def _baseCorner(samples, labels):

    logging.disable(logging.WARNING)
    fig = corner.corner(np.array([samples[key] for key in labels]).T, hist_kwargs={'density': True}, labels=labels)
    logging.getLogger().setLevel(logging.WARNING)

    return fig

def _setSampleToPlot(self, N, unpacked=True, stage='prior'):

    if stage == 'prior':
        samples = np.array([self.ptform(np.random.uniform(0, 1, size=self.ndims)) for i in range(N)])
    else:
        samples = self.samples

    if unpacked:
        _samples = self.unpackSamples(samples)

    else:
        _samples = {label: samples[:, i] for i, label in enumerate(self.priors.keys())}

    return _samples

def _ModeIDClassPriorCorner(self, modObj, unpacked, N, **kwargs):

    _samples = _setSampleToPlot(modObj, N, unpacked=unpacked, stage='prior')
 
    labels = list(_samples.keys())
        
    fig = _baseCorner(_samples, labels)    

    axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

    if not unpacked:
        for i, key in enumerate(labels):
            if key in modObj.priors.keys():
            
                x = np.linspace(modObj.priors[key].ppf(1e-6), 
                                modObj.priors[key].ppf(1-1e-6), 100)

                pdf = np.array([modObj.priors[key].pdf(x[j]) for j in range(len(x))])

                axes[i, i].plot(x, pdf, color='C3', alpha=0.5, lw =5)
        
    return fig, axes

def _ModeIDClassPostCorner(self, modObj, unpacked, N, **kwargs):

    _samples = _setSampleToPlot(modObj, N, unpacked=unpacked, stage='posterior')
    
    labels = list(_samples.keys())

    fig = _baseCorner(_samples, labels)  

    axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

    if not unpacked:
        for i, key in enumerate(labels):
        
            if key in modObj.priors.keys():
                 
                x = np.linspace(modObj.priors[key].ppf(1e-9), modObj.priors[key].ppf(1-1e-9), 100)

                pdf = np.array([modObj.priors[key].pdf(x[j]) for j in range(len(x))])

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

    def echelle(self, stage='posterior', ID=None, savepath=None, save_kwargs={}, kwargs={}):

        if not 'colors' in kwargs:
            kwargs['colors'] = ellColors

        if not 'scale' in kwargs:
            kwargs['scale'] = 1/350

        if not 'Nsamples' in kwargs:
            kwargs['Nsamples'] = 200
            
        if self.__class__.__name__ == 'modeID':

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
            raise ValueError('Unrecognized class type. Only modeID and peakbag have this plotting function built in.')

        if ID is not None:
            ax.set_title(ID)
 
        if (savepath is not None):
            fig.savefig(savepath, **save_kwargs)

        return fig, ax

    def spectrum(self, stage='posterior', ID=None, savepath=None, kwargs={}, save_kwargs={}, N=30):
         
        if self.__class__.__name__ == 'modeID':
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
            raise ValueError('Unrecognized class type. Only modeID and peakbag have this plotting function built in.')        
        
        if ID is not None:
            ax.set_title(ID)
                        
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, **save_kwargs)

        return fig, ax

    def corner(self, stage='posterior', ID=None, labels=None, savepath=None, unpacked=False, kwargs={}, save_kwargs={}, N=5000):
         
        if not 'colors' in kwargs:
            kwargs['colors'] = ellColors

        
        if self.__class__.__name__ == 'modeID':
             
            if stage=='prior':
                 
                fig, ax = [], []
                
                if hasattr(self, 'l20model'):
                    figl20, axl20 = _ModeIDClassPriorCorner(self, self.l20model, unpacked, N, **kwargs)
                    
                    fig.append(figl20)
                    
                    ax.append(axl20)

                else:
                    warnings.warn('modeID does not currently have and l20model attribute, use runl20model first.')

                if hasattr(self, 'l1model'):
                    figl1, axl1 = _ModeIDClassPriorCorner(self, self.l1model, unpacked, N, **kwargs)

                    fig.append(figl1)
                
                    ax.append(axl1)
                    
                else:
                    warnings.warn('modeID does not currently have and l1model attribute, use runl1model first.')
            
            elif stage=='posterior': 
                
                fig, ax = [], []
                
                if hasattr(self, 'l20model'):
                    figl20, axl20 = _ModeIDClassPostCorner(self, self.l20model, unpacked, N, **kwargs)
                    
                    fig.append(figl20)
                    
                    ax.append(axl20)

                else:
                    warnings.warn('modeID does not currently have and l20model attribute, use runl20model first.')

                if hasattr(self, 'l1model'):
                    figl1, axl1 = _ModeIDClassPostCorner(self, self.l1model, unpacked, N, **kwargs)

                    fig.append(figl1)
                
                    ax.append(axl1)

                else:
                    warnings.warn('modeID does not currently have and l1model attribute, use runl1model first.')

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
            raise ValueError('Unrecognized class type. Only modeID and peakbag have this plotting function built in.')
        
        return fig, ax

    def reference(self, stage='posterior', ID=None):
        """Make a corner plot of the prior sample with relevant overplotted values."""

        if self.__class__.__name__ == 'modeID':
            
            fig, axes = [], []

            if stage=='prior':
 
                if hasattr(self, 'l20model'):
                    figl20, axl20 = _ModeIDPriorReference(self.l20model)
                    
                    fig.append(figl20)
                    
                    axes.append(axl20)

                if hasattr(self, 'l1model'):
                    figl1, axl1 = _ModeIDPriorReference(self.l1model)

                    fig.append(figl1)
                    
                    axes.append(axl1)

            elif stage=='posterior':
                if hasattr(self, 'l20model'):
                    figl20, axl20 = _ModeIDPosteriorReference(self.l20model)

                    fig.append(figl20)
                    
                    axes.append(axl20)

                if hasattr(self, 'l1model'):
                    figl1, axl1 = _ModeIDPosteriorReference(self.l1model)

                    fig.append(figl1)
                    
                    axes.append(axl1)
            else:
                raise ValueError('Set stage optional argument to either prior or posterior')
            
            return fig, axes
        else:
            raise ValueError('This kind of plot is only available for the modeIDsampler module.')













    # def plotLatentCorner(self, samples, labels=None):
    
    #     if labels == None:
    #         labels = list(self.priors.keys()) 
        
    #     fig = corner.corner(samples, hist_kwargs = {'density': True}, labels=labels)

    #     axes = np.array(fig.get_axes()).reshape((len(labels), len(labels)))

    #     for i, key in enumerate(labels):
        
    #         if key in self.priors.keys():
            
    #             x = np.linspace(self.priors[key].ppf(1e-6), self.priors[key].ppf(1-1e-6), 100)

    #             pdf = np.array([self.priors[key].pdf(x[j]) for j in range(len(x))])
    
    #             axes[i, i].plot(x, pdf, color='C3', alpha=0.5, lw =5)    

    # def plot_corner(self, path=None, ID=None, savefig=False):
    #     """ Make corner plot of result.
        
    #     Makes a nice corner plot of the fit parameters.

    #     Parameters
    #     ----------
    #     path : str, optional
    #         Used along with savefig, sets the output directory to store the
    #         figure. Default is to save the figure to the star directory.
    #     ID : str, optional
    #         ID of the target to be included in the filename of the figure.
    #     savefig : bool
    #         Whether or not to save the figure to disk. Default is False.

    #     Returns
    #     -------
    #     fig : Matplotlib figure object
    #         Figure object with the corner plot.

    #     """

    #     if not hasattr(self, 'samples'):
    #         warnings.warn(f"'{self.__class__.__name__}' has no attribute 'samples'. Can't plot a corner plot.")
    #         return None

    #     fig = corner.corner(self.samples, labels=self.par_names,
    #                         show_titles=True, quantiles=[0.16, 0.5, 0.84],
    #                         title_kwargs={"fontsize": 12})

    #     if savefig:
    #         self._save_my_fig(fig, 'corner', path, ID)

    #     return fig




    # def _fill_diag(self, axes, vals, vals_err, idxs):
    #     """ Overplot diagnoal values along a corner plot diagonal.
        
    #     Plots a set of specified values over the 1D histograms in the diagonal 
    #     frames of a corner plot.

    #     Parameters
    #     ----------
    #     axes : Matplotlib axis object
    #         The particular axis element to be plotted in.
    #     vals : float
    #         Mean values to plot.
    #     vals_err : float
    #         Error estimates for the value to be plotted.
    #     idxs : list
    #         List of 2D indices that represent the diagonal. 

    #     """

    #     N = int(np.sqrt(len(axes)))
    #     axs = np.array(axes).reshape((N,N)).T
        
    #     for i,j in enumerate(idxs):
    #         yrng = axs[j,j].get_ylim()
            
    #         v, ve = vals[i], vals_err[i]
            
    #         axs[j,j].fill_betweenx(y=yrng, x1= v-ve[0], x2 = v+ve[-1], color = 'C3', alpha = 0.5)
    
    # def _plot_offdiag(self, axes, vals, vals_err, idxs):
    #     """ Overplot offdiagonal values in a corner plot.
        
    #     Plots a set of specified values over the 2D histograms or scatter in the
    #     off-diagonal frames of a corner plot.

    #     Parameters
    #     ----------
    #     axes : Matplotlib axis object
    #         The particular axis element to be plotted in.
    #     vals : float
    #         Mean values to plot.
    #     vals_err : float
    #         Error estimates for the value to be plotted.
    #     idxs : list
    #         List of 2D indices that represent the diagonal. 

    #     """
        
    #     N = int(np.sqrt(len(axes)))

    #     axs = np.array(axes).reshape((N,N)).T
        
    #     for i, j in enumerate(idxs):
            
    #         for m, k in enumerate(idxs):
    #             if j >= k:
    #                 continue
                    
    #             v, ve = vals[i], vals_err[i]

    #             w, we = vals[m], vals_err[m]
    
    #             axs[j,k].errorbar(v, w, xerr=ve.reshape((2,1)), yerr=we.reshape((2,1)), fmt = 'o', ms = 10, color = 'C3')

    # def _make_prior_corner(self, df, numax_rng = 100):
    #     """ Show dataframe contents in a corner plot.
        
    #     This is meant to be used to show the contents of the prior_data that is
    #     used by KDE and Asy_peakbag. 

    #     Parameters
    #     ----------
    #     df : pandas.Dataframe object
    #         Dataframe of the data to be shown in the corner plot.
    #     numax_rng : float, optional
    #         Range in muHz around the input numax to be shown in the corner plot.
    #         The default is 100. 

    #     Returns
    #     -------
    #     crnr : matplotlib figure object
    #         Corner plot figure object containing NxN axis objects.
    #     crnr_axs : list
    #         List of axis objects from the crnr object.

    #     """
        
    #     idx = abs(10**df['numax'] - self._obs['numax'][0]) <= numax_rng
        
    #     # This is a temporary fix for disabling the 'Too few samples' warning 
    #     # from corner.hist2d. The github bleeding edge version has 
    #     # hist2d_kwargs = {'quiet': True}, but this isn't in the pip 
    #     # installable version yet. March 2020.
    #     logging.disable(logging.WARNING)
    #     crnr = corner.corner(df.to_numpy()[idx,:-1], data_kwargs = {'alpha': 0.5}, labels = df.keys());
    #     logging.getLogger().setLevel(logging.WARNING)
    
    #     return crnr,  crnr.get_axes()
        


    # def plot_prior(self, path=None, ID=None, savefig=False):
    #     """ Corner of result in relation to prior sample.
        
    #     Create a corner plot showing the location of the star in relation to
    #     the rest of the prior.

    #     Parameters
    #     ----------
    #     path : str, optional
    #         Used along with savefig, sets the output directory to store the
    #         figure. Default is to save the figure to the star directory.
    #     ID : str, optional
    #         ID of the target to be included in the filename of the figure.
    #     savefig : bool
    #         Whether or not to save the figure to disk. Default is False.

    #     Returns
    #     -------
    #     crnr : matplotlib figure object
    #         Corner plot figure object containing NxN axis objects.
            
    #     """
        
    #     df = pd.read_csv(self.prior_file)
    #     crnr, axes = self._make_prior_corner(df)
       
    #     if type(self) == pbjam.star:
    #         vals, vals_err = np.array([self._log_obs['dnu'],
    #                                    self._log_obs['numax'], 
    #                                    self._log_obs['teff'], 
    #                                    self._obs['bp_rp']]).T
            
    #         vals_err = np.vstack((vals_err, vals_err)).T
    #         self._fill_diag(axes, vals, vals_err, [0, 1, 8, 9])
    #         self._plot_offdiag(axes, vals, vals_err, [0, 1, 8, 9])
            
    #         # break this if statement if asy_fit should plot something else
    #     elif (type(self) == pbjam.priors.kde) or (type(self) == pbjam.asy_peakbag.asymptotic_fit): 
    #         percs = np.percentile(self.samples, [16, 50, 84], axis=0)
    #         vals = percs[1,:]
    #         vals_err = np.diff(percs, axis = 0).T
            
    #         self._fill_diag(axes, vals, vals_err, range(len(vals)))
    #         self._plot_offdiag(axes, vals, vals_err, range(len(vals)))

    #     elif type(self) == pbjam.peakbag:
    #         raise AttributeError('The result of the peakbag run cannot be plotted in relation to the prior, since it does not know what the prior is anymore. plot_corner is only available for star, kde and asy_peakbag.')


    #     return crnr
     