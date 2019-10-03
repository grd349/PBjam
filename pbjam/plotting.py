import matplotlib.pyplot as plt
import astropy.convolution as conv
import pbjam, os, corner
import numpy as np
from pymc3.gp.util import plot_gp_dist
import astropy.units as u
import pymc3 as pm

class plotting():
    
    def __init__(self):
        pass
    
    
    def plot_corner(self, path=None, ID=None, savefig=False):
       
        '''
        Makes a nice corner plot using corner.corner
        '''
        
        fig = corner.corner(self.samples, labels=self.par_names,
                            show_titles=True, quantiles=[0.16, 0.5, 0.84],
                            title_kwargs={"fontsize": 12})
        
        # TODO there should be a check if path is full filepath or just dir
        if path and ID:
            outpath = os.path.join(*[path,  type(self).__name__+ '_corner_' + str(ID) + '.png'])
            if savefig:
                fig.savefig(outpath)
                
        return fig


    def plot_spectrum(self, pg=None, path=None, ID=None, savefig=False):
    
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
        fac = 0.1 / (f[1] - f[0]) #0.005 * self.dnu[0]  / (f[1] - f[0])
        kernel = conv.Gaussian1DKernel(stddev=fac)
        smoo = conv.convolve(s, kernel)
        ax.plot(f, smoo, 'k-', label='Smoothed', lw=3, alpha=0.6)
    
        # Overplot kde diagnostic  
    
        if type(self) == pbjam.star:
             xlim = [self.numax[0]-5*self.dnu[0], self.numax[0]+5*self.dnu[0]]
         
        elif type(self) == pbjam.priors.kde:
            h = max(smoo)
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
            xlim = [min(freq)-dnu, max(freq)+dnu]
    
        # Overplot asy_peakbag diagnostic        
        elif type(self) == pbjam.asy_peakbag.asymptotic_fit:
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
            
        # Overplot peakbag diagnostic    
        elif type(self) == pbjam.peakbag:  
            n = self.ladder_s.shape[0]                
            par_names = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2', 
                         'back']
            for i in range(n):
                for j in range(-50, 0):
                    if (i == 0) and (j==-1):
                        label='Model'
                    else:
                        label=None
                    mod = self.model(*[self.samples[x][j] for x in par_names])
                    ax.plot(self.ladder_f[i, :], mod[i, :], c='r', alpha=0.1, 
                            label=label)
                    
            dnu = 10**self.asy_result.summary.loc['dnu', '50th']        
            xlim = [min(f[self.asy_result.sel])-dnu, 
                    max(f[self.asy_result.sel])+dnu]
        else:
            raise ValueError('Unrecognized class type')
        
        ax.set_ylim([0, smoo.max()*1.5])
        ax.set_xlim([max([min(f), xlim[0]]), min([max(f), xlim[1]])])
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend(loc=1)
    
        if savefig:
            outpath = os.path.join(*[path, f'{type(self).__name__}_{str(ID)}.png'])
            fig.savefig(outpath)


    def plot_trace(stage):
        '''
        Will make a pymc3 traceplot.
        '''
        import pymc3 as pm
        
        if type(stage) == pbjam.priors.kde:
            # TODO - make this work for kde
            print('Traceplot for kde not yet implimented')
        
        if type(stage) == pbjam.asy_peakbag.asymptotic_fit:
            # TODO - make this work for asy_peakbag
            print('Traceplot for asy_peakbag not yet implimented')
        
        if type(stage) == pbjam.peakbag:
            pm.traceplot(stage.samples)


# Asy_peakbag  
    def plot_start(self):
        '''
        Plots the starting model as a diagnotstic.
        '''
        fig, ax = plt.subplots(figsize=[16,9])
        ax.plot(self.f, self.s, 'k-', label='Data', alpha=0.2)
        smoo = self.start[0] * 0.005 / (self.f[1] - self.f[0])
        kernel = conv.Gaussian1DKernel(stddev=smoo)
        smoothed = conv.convolve(self.s, kernel)
        ax.plot(self.f, smoothed, 'k-', label='Smoothed', lw=3, alpha=0.6)
        ax.plot(self.f[self.sel], self.model(self.start_samples.mean(axis=0)), 
                'r-', label='Start model', alpha=0.7)
        ax.set_ylim([0, smoothed.max()*1.5])
        ax.set_xlabel(r'Frequency ($\mu \rm Hz$)')
        ax.set_ylabel(r'SNR')
        ax.legend()
        return fig


 
      
# Peakbag
        


def plot_linewidth(self, thin=10):
    """
    Plots the estimated line width as a function of scaled n.
    """
    fig, ax = plt.subplots(1, 2, figsize=[16,9])

    if self.gp0 != []:

        n_new = np.linspace(-0.2, 1.2, 100)[:,None]
        with self.pm_model:
            f_pred0 = self.gp0.conditional("f_pred0", n_new)
            f_pred2 = self.gp2.conditional("f_pred2", n_new)
            self.pred_samples = pm.sample_posterior_predictive(self.samples,
                           vars=[f_pred0, f_pred2], samples=1000)
        plot_gp_dist(ax[0], self.pred_samples["f_pred0"], n_new)
        plot_gp_dist(ax[1], self.pred_samples["f_pred2"], n_new)

        for i in range(0, len(self.samples), thin):
            ax[0].scatter(self.n,
                          self.samples['ln_width0'][i, :], c='k', alpha=0.3)
            ax[1].scatter(self.n,
                          self.samples['ln_width2'][i, :], c='k', alpha=0.3)


    else:
        for i in range(0, len(self.samples), thin):
            ax[0].scatter(self.n,
                          np.log(self.samples['width0'][i, :]), c='k', alpha=0.3)
            ax[1].scatter(self.n,
                          np.log(self.samples['width2'][i, :]), c='k', alpha=0.3)

    ax[0].set_xlabel('normalised order')
    ax[1].set_xlabel('normalised order')
    ax[0].set_ylabel('ln line width')
    ax[1].set_ylabel('ln line width')
    ax[0].set_title('Radial modes')
    ax[1].set_title('Quadrupole modes')
    return fig

def plot_height(self, thin=10):
    """
    Plots the estimated mode height.
    """
    fig, ax = plt.subplots(figsize=[16,9])
    for i in range(0, len(self.samples), thin):
        ax.scatter(self.samples['l0'][i, :], self.samples['height0'][i, :])
        ax.scatter(self.samples['l2'][i, :], self.samples['height2'][i, :])
    return fig

def plot_ladder(self, thin=10, alpha=0.2):
    """
    Plots the ladder data and models from the samples

    Parameters
    ----------
    thin: int
        Uses every other thin'th value from the samkles, i.e. [::thin].
    alpha: float64
        The alpha to use for plotting the models from samples.
    """
    n = self.ladder_s.shape[0]
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
        ax[i].plot(self.ladder_f[i, :], self.ladder_s[i, :], c='k')
        ax[i].set_xlim([self.ladder_f[i, 0], self.ladder_f[i, -1]])
    ax[n-1].set_xlabel(r'Frequency ($\mu \rm Hz$)')
    fig.tight_layout()
    return fig



def plot_echelle(self, pg):
    '''
    Plots an echelle diagram with mode frequencies over plotted.

    Parameters
    ----------
    pg : periodogram
        A lightkurve periodogram (we use the plot echelle method
        of the periodogram (well actually seismology) class)

    Returns
    -------
    fig : figure
        The figure containing the plot.
    '''
    
    dnu = np.median(np.diff(np.sort(self.samples['l0'].mean(axis=0))))
    # make dnu an intger multiple of bw
    bw = self.f[1] - self.f[0]
    dnu -= dnu % bw
    numax = 10**self.asy_result['summary'].loc['numax', 'mean']
    seismology = pg.flatten().to_seismology()
    nmin = np.floor(self.f.min() / dnu) + 1
    ax = seismology.plot_echelle(deltanu=dnu * u.uHz,
                                 numax=numax * u.uHz,
                                 minimum_frequency=dnu*nmin)
    pbjam_mean_l0 = self.samples['l0'].mean(axis=0)
    pbjam_std_l0 = self.samples['l0'].std(axis=0)
    pbjam_mean_l2 = self.samples['l2'].mean(axis=0)
    pbjam_std_l2 = self.samples['l2'].std(axis=0)
    ax.errorbar(pbjam_mean_l0 % dnu, (pbjam_mean_l0 // dnu) * dnu,
                xerr=pbjam_std_l0, fmt='ro', alpha=0.5, label=r'$\ell=0$')
    ax.errorbar(pbjam_mean_l2 % dnu, (pbjam_mean_l2 // dnu) * dnu,
                xerr=pbjam_std_l2, fmt='gs', alpha=0.5, label=r'$\ell=2$')
    ax.legend(fontsize = 'x-small')