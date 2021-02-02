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

from .jar import debug

logger = logging.getLogger(__name__)  # For module-level logging


class plotting:
    """ Class inherited by PBjam modules to plot results
    
    This is used to standardize the plots produced at various steps of the 
    peakbagging process. 
    
    As PBjam class is initialized, these plotting methods will be inherited.
    The methods will plot the relevant result based on the class they are being
    called from. 
    
    """

    def __init__(self, *args, **kwargs):
        super(plotting, self).__init__(*args, **kwargs)

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
    
    @debug(logger)
    def plot_echelle(self, pg=None, path=None, ID=None, savefig=False):
        """ Make echelle plot

        Plots an echelle diagram with mode frequencies if available.

        Parameters
        ----------
        pg : Lightkurve.periodogram object, optional
            A lightkurve periodogram to plot the echelle diagram of
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

        freqs = {'l'+str(i): {'nu': [], 'err': []} for i in range(4)}

        if isinstance(self, pbjam.star):#type(self) == pbjam.star:
            dnu = self.dnu[0]
            numax = self.numax[0]

        elif isinstance(self, pbjam.priors.kde): #type(self) == pbjam.priors.kde:
            dnu = 10**np.median(self.samples[:,0])
            numax = 10**np.median(self.samples[:,1])

        elif isinstance(self, pbjam.asy_peakbag.asymptotic_fit): #type(self) == pbjam.asy_peakbag.asymptotic_fit:
            dnu = 10**self.summary.loc['dnu', '50th']
            numax = 10**self.summary.loc['numax', '50th']
            for l in np.arange(4):
                idx = self.modeID.ell == l
                freqs['l'+str(l)]['nu'] = self.modeID.loc[idx, 'nu_med']
                freqs['l'+str(l)]['err'] = self.modeID.loc[idx, 'nu_mad']

        elif isinstance(self, pbjam.peakbag): #type(self) == pbjam.peakbag:
            numax = 10**self.asy_fit.summary.loc['numax', '50th']
            for l in np.arange(4):
                ell = 'l'+str(l)
                freqs[ell]['nu'] = self.summary.filter(like=ell, axis=0).loc[:, 'mean']
                freqs[ell]['err'] = self.summary.filter(like=ell, axis=0).loc[:, 'sd']
            dnu = np.median(np.diff(freqs['l0']['nu']))

        elif isinstance(self, pbjam.ellone): #type(self) == pbjam.ellone:
            numax = 10**self.pbinst.asy_fit.summary.loc['numax', '50th']
            for l in [0, 2]:
                ell = 'l'+str(l)
                freqs[ell]['nu'] = self.pbinst.summary.filter(like=ell, axis=0).loc[:, 'mean']
                freqs[ell]['err'] = self.pbinst.summary.filter(like=ell, axis=0).loc[:, 'sd']
            freqs['l1']['nu'] = self.nu_l1
            freqs['l1']['err'] = self.nu_l1_std

            dnu = np.median(np.diff(freqs['l0']['nu']))

        else:
            raise ValueError('Unrecognized class type')

        # make dnu an intger multiple of bw
        dnu -= dnu % (self.f[1] - self.f[0])
        #nmin = np.floor(self.f.min() / dnu) + 1

        if pg:
            peri = pg
        elif hasattr(self, 'pg'):
            peri = self.pg
        else:
            raise ValueError('Need spectrum to plot echelle diagram')

        seismology = peri.flatten().to_seismology()

        fig, ax = plt.subplots(figsize=[16,9])
        ax = seismology.plot_echelle(deltanu=dnu * u.uHz,
                                     numax=numax * u.uHz,
                                     ax=ax)

        # Overplot modes
        cols = ['C1', 'C2', 'C3', 'C4']

        for l in np.arange(4):
            ell = 'l'+str(l)
            if len(freqs[ell]['nu']) > 0:
                nu = freqs[ell]['nu']
                err = freqs[ell]['err']
                ax.errorbar(nu%dnu, (nu//dnu) * dnu, xerr=err, fmt='o', color = cols[l], label = r'$\ell=$%i' % (l))
        ax.legend(fontsize = 'x-small')
        
        fig.tight_layout()

        if savefig:
            self._save_my_fig(fig, 'echelle', path, ID)
            
        return fig

    @debug(logger)
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

    @debug(logger)
    def plot_spectrum(self, pg=None, path=None, ID=None, savefig=False):
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

        elif isinstance(self, pbjam.ellone): #type(self) == pbjam.ellone:
            n = self.pbinst.ladder_s.shape[0]
            par_names = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2',
                         'back']
            for i in range(n):
                for j in range(-50, 0):
                    if (i == 0) and (j==-1):
                        mlabel='Model'
                        l1label = r'$\nu_{l=1}$'
                    else:
                        mlabel=None
                    mod = self.pbinst.model(*[self.pbinst.samples[x][j] for x in par_names])
                    ax.plot(self.pbinst.ladder_f[i, :], mod[i, :], c='r', alpha=0.1,
                            label=mlabel)

                ax.axvline(self.nu_l1, color = 'k', alpha = 0.5, label = l1label)

            dnu = 10**self.pbinst.asy_fit.summary.loc['dnu', '50th']

            xlim = [min(f[self.pbinst.asy_fit.sel])-dnu,
                    max(f[self.pbinst.asy_fit.sel])+dnu]

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
        
    @debug(logger)
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
    
    @debug(logger)
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


# =============================================================================
# To be implemented later
# =============================================================================
#def plot_trace(stage):
#    """ Make a trace plot of the MCMC chains
#    """
#
#    import pymc3 as pm
#
#    if type(stage) == pbjam.priors.kde:
#        # TODO - make this work for kde
#        print('Traceplot for kde not yet implimented')
#
#    if type(stage) == pbjam.asy_peakbag.asymptotic_fit:
#        # TODO - make this work for asy_peakbag
#        print('Traceplot for asy_peakbag not yet implimented')
#
#    if type(stage) == pbjam.peakbag:
#        pm.traceplot(stage.samples)
#
#
## Peakbag
#def plot_linewidth(self, thin=10):
#    """ Plot estimated line width as a function of scaled n.
#    """
#
#    fig, ax = plt.subplots(1, 2, figsize=[16,9])
#
#    if self.gp0 != []:
#
#        n_new = np.linspace(-0.2, 1.2, 100)[:,None]
#        with self.pm_model:
#            f_pred0 = self.gp0.conditional("f_pred0", n_new)
#            f_pred2 = self.gp2.conditional("f_pred2", n_new)
#            self.pred_samples = pm.sample_posterior_predictive(self.samples,
#                           vars=[f_pred0, f_pred2], samples=1000)
#        plot_gp_dist(ax[0], self.pred_samples["f_pred0"], n_new)
#        plot_gp_dist(ax[1], self.pred_samples["f_pred2"], n_new)
#
#        for i in range(0, len(self.samples), thin):
#            ax[0].scatter(self.n,
#                          self.samples['ln_width0'][i, :], c='k', alpha=0.3)
#            ax[1].scatter(self.n,
#                          self.samples['ln_width2'][i, :], c='k', alpha=0.3)
#
#
#    else:
#        for i in range(0, len(self.samples), thin):
#            ax[0].scatter(self.n,
#                          np.log(self.samples['width0'][i, :]), c='k', alpha=0.3)
#            ax[1].scatter(self.n,
#                          np.log(self.samples['width2'][i, :]), c='k', alpha=0.3)
#
#    ax[0].set_xlabel('normalised order')
#    ax[1].set_xlabel('normalised order')
#    ax[0].set_ylabel('ln line width')
#    ax[1].set_ylabel('ln line width')
#    ax[0].set_title('Radial modes')
#    ax[1].set_title('Quadrupole modes')
#    return fig
#
#def plot_height(self, thin=10):
#    """ Plots the estimated mode height.
#    """
#
#    fig, ax = plt.subplots(figsize=[16,9])
#    for i in range(0, len(self.samples), thin):
#        ax.scatter(self.samples['l0'][i, :], self.samples['height0'][i, :])
#        ax.scatter(self.samples['l2'][i, :], self.samples['height2'][i, :])
#    return fig
#
#def plot_ladder(self, thin=10, alpha=0.2):
#    """
#    Plots the ladder data and models from the samples
#
#    Parameters
#    ----------
#    thin: int
#        Uses every other thin'th value from the samkles, i.e. [::thin].
#    alpha: float64
#        The alpha to use for plotting the models from samples.
#
#    """
#
#    n = self.ladder_s.shape[0]
#    fig, ax = plt.subplots(n, figsize=[16,9])
#    for i in range(n):
#        for j in range(0, len(self.samples), thin):
#            mod = self.model(self.samples['l0'][j],
#                             self.samples['l2'][j],
#                             self.samples['width0'][j],
#                             self.samples['width2'][j],
#                             self.samples['height0'][j],
#                             self.samples['height2'][j],
#                             self.samples['back'][j])
#            ax[i].plot(self.ladder_f[i, :], mod[i, :], c='r', alpha=alpha)
#        ax[i].plot(self.ladder_f[i, :], self.ladder_s[i, :], c='k')
#        ax[i].set_xlim([self.ladder_f[i, 0], self.ladder_f[i, -1]])
#    ax[n-1].set_xlabel(r'Frequency ($\mu \rm Hz$)')
#    fig.tight_layout()
#    return fig
