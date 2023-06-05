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

    def echelle(self, ID=None, path=None, savefig=False, N=200):
            """ Make echelle plot

            Plots an echelle diagram with mode frequencies if available.

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
                Figure object with the echelle diagram.

            """

            freqs = {'l'+str(i): {'nu': [], 'err': [], 'zeta': [], 'samples': None} for i in range(4)}
 

            if isinstance(self, pbjam.core.star): 
                dnu = self.obs['dnu'][0]
                
                numax = self.obs['numax'][0]

            elif isinstance(self, pbjam.modeID.modeIDsampler):  
                dnu = self.result['summary']['dnu'][0]

                numax = self.result['summary']['numax'][0]

                for l in np.arange(4):
                    idx = self.result['ell'] == l

                    freqs[f'l{str(l)}']['nu'] = self.result['summary']['freq'][0, idx]

                    freqs[f'l{str(l)}']['err'] = self.result['summary']['freq'][1, idx]

                    freqs[f'l{str(l)}']['samples'] = self.result['samples']['freq'][:N, idx]

            elif isinstance(self, pbjam.peakbagging.peakbag):  
                numax = 10**self.asy_fit.summary.loc['numax', '50th']

                for l in np.arange(4):
                    ell = 'l'+str(l)

                    freqs[ell]['nu'] = self.summary.filter(like=ell, axis=0).loc[:, 'mean']

                    freqs[ell]['err'] = self.summary.filter(like=ell, axis=0).loc[:, 'sd']

                dnu = np.median(np.diff(freqs['l0']['nu']))

            else:
                raise ValueError('Unrecognized class type')

            n = max([self.N_p + 1, 10])

            idx = ((numax - n * dnu) < self.f) & (self.f < (numax + n * dnu))

            fig, ax = plt.subplots(figsize=(8,7))    
            
            plot_echelle(self.f[idx], self.s[idx], dnu, ax=ax, smooth=True, smooth_filter_width=0.1)
        
            # Overplot modes
            cols = ['C1', 'C2', 'C3', 'C4']

            for l in np.arange(4):
                ell = 'l'+str(l)
                
                if len(freqs[ell]['nu']) > 0:
 
                    err = freqs[ell]['err']

                    smry_x, smry_y = self.echelle_freqs(freqs[ell]['nu'], dnu) 

                    ax.errorbar(smry_x, smry_y, xerr=err, fmt='o', color=cols[l], label=r'$\ell=$%i' % (l), ms=5)
 
                    smp_x, smp_y = self.echelle_freqs(freqs[ell]['samples'], dnu) 

                    ax.scatter(smp_x, smp_y, alpha=0.2, color=cols[l])
                        
            ax.legend()

            fig.tight_layout()

            if savefig:
                self._save_my_fig(fig, 'echelle', path, ID)

            return fig, ax

    def echelle_freqs(self, nu, dnu):
        x = nu%dnu

        y = (nu//dnu) * dnu + dnu/2

        return x, y
        
        

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

    def plot_spectrum(self, f=None, s=None, path=None, ID=None, savefig=False):
        """ Plot the power spectrum

        Plot the power spectrum around the p-mode envelope. Calling this
        method from the different classes such as KDE or peakbag, will plot
        the relevant result from those classes if available.

        Parameters
        ----------
        pg : Lightkurve.periodogram object, optional
            A lightkurve periodogram to plot
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
            Figure object with the spectrum.

        """

        if not pg and hasattr(self, 'pg'):
            pg = self.pg

        if pg:
            f = pg.frequency.value
            s = pg.power.value

        elif hasattr(self, 'f') & hasattr(self, 's'):
            f = self.f
            s = self.s
        else:
            raise ValueError('Unable to plot spectrum.')

        # The raw and smoothed spectrum will always be plotted
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(f, s, 'k-', label='Data', alpha=0.2)
        fac = max([1, 0.1 / (f[1] - f[0])])
        kernel = conv.Gaussian1DKernel(stddev=fac)
        smoo = conv.convolve(s, kernel)
        ax.plot(f, smoo, 'k-', label='Smoothed', lw=3, alpha=0.6)
              
        if isinstance(self, pbjam.star): #type(self) == pbjam.star:
             xlim = [self.numax[0]-5*self.dnu[0], self.numax[0]+5*self.dnu[0]]

        elif isinstance(self, pbjam.priors.kde): #type(self) == pbjam.priors.kde:
            h = max(smoo)
            numax = 10**(np.median(self.samples[:, 1]))
            dnu = 10**(np.median(self.samples[:, 0]))
            nmin = np.floor(min(f) / dnu)
            nmax = np.floor(max(f) / dnu)
            enns = np.arange(nmin-1, nmax+1, 1)
            freq, freq_sigma = self.kde_predict(enns)
            y = np.zeros(len(f))
            for i in range(len(enns)):
                y += 0.8 * h * np.exp(-0.5 * (freq[i] - f)**2 / freq_sigma[i]**2)
            ax.fill_between(f, y, alpha=0.3, facecolor='navy', edgecolor='none',
                            label=r'$\propto P(\nu_{\ell=0})$')
           
            xlim = [numax-5*dnu, numax+5*dnu]

        elif isinstance(self, pbjam.asy_peakbag.asymptotic_fit): #type(self) == pbjam.asy_peakbag.asymptotic_fit:
            for j in np.arange(-50,0):
                if j==-1:
                    label='Model'
                else:
                    label=None
                ax.plot(f[self.sel], self.model(self.samples[j, :]), 'r-',
                        alpha=0.1)
            for nu in self.modeID['nu_med']:
                ax.axvline(nu, c='k', linestyle='--')
            dnu = 10**self.summary.loc['dnu', '50th']
            xlim = [min(f[self.sel])-dnu,
                    max(f[self.sel])+dnu]

        elif isinstance(self, pbjam.peakbag): #type(self) == pbjam.peakbag:
            n = self.ladder_s.shape[0]
            par_names = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2',
                         'back']
            for i in range(n):
                for j in range(-50, 0):
                    if (i == 0) and (j==-1):
                        label='Model'
                    else:
                        label=None
                    mod = self.model(*[self.traces[x][j] for x in par_names])
                    ax.plot(self.ladder_f[i, :], mod[i, :], c='r', alpha=0.1,
                            label=label)

            dnu = 10**self.asy_fit.summary.loc['dnu', '50th']
            xlim = [min(f[self.asy_fit.sel])-dnu,
                    max(f[self.asy_fit.sel])+dnu]

        else:
            raise ValueError('Unrecognized class type')

        ax.set_ylim([0, smoo.max()*1.5])
        ax.set_xlim([max([min(f), xlim[0]]), min([max(f), xlim[1]])])
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend(loc=1)
        
        fig.tight_layout()
        if savefig:
            self._save_my_fig(fig, 'spectrum', path, ID)

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
    
    def plot_start(self):
        """ Plot starting point for peakbag
        
        Plots the starting model to be used in peakbag as a diagnotstic.
        
        """

        dnu = 10**np.median(self.start_samples, axis=0)[0]
        xlim = [min(self.f[self.sel])-dnu, max(self.f[self.sel])+dnu]
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(self.f, self.s, 'k-', label='Data', alpha=0.2)
        smoo = dnu * 0.005 / (self.f[1] - self.f[0])
        kernel = conv.Gaussian1DKernel(stddev=smoo)
        smoothed = conv.convolve(self.s, kernel)
        ax.plot(self.f, smoothed, 'k-', label='Smoothed', lw=3, alpha=0.6)
        ax.plot(self.f[self.sel], self.model(self.start_samples.mean(axis=0)),
                'r-', label='Start model', alpha=0.7)
        ax.set_ylim([0, smoothed.max()*1.5])
        ax.set_xlim(xlim)
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend()
        return fig

