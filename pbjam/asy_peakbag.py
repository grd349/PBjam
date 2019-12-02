"""Fitting the asymptotic relation to an SNR spectrum

This module fits the asymptotic relation to the p-modes in a frequency range
around nu_max, the central frequency of the seismic mode envelope,
in a solar-like oscillator. Only l=0 and l=2 are fit, l=1 modes are ignored.

"""

import numpy as np
import pbjam as pb
import pandas as pd
import scipy.stats as scist
from .priors import kde
from .plotting import plotting
from .jar import to_log10, normal
from collections import OrderedDict

class asymp_spec_model():
    """Class for spectrum model using asymptotic relation.

    Parameters
    ---------_
    f : float, ndarray
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.
    norders : int
        Number of radial order to fit

    Attributes
    ----------
    f : float, ndarray
        Array of frequency bins of the spectrum (muHz). Truncated to the range
        around numax.

    norders : int
        Number of radial order to fit
        
    """

    def __init__(self, f, norders):
        self.f = np.array([f]).flatten()
        self.norders = int(norders)
    

    def _get_nmax(self, dnu, numax, eps):
        """Compute radial order at numax.
    
        Compute the radial order at numax, which in this implimentation of the
        asymptotic relation is not necessarily integer.
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (muHz).
    
        Returns
        -------
            nmax : float
                non-integer radial order of maximum power of the p-mode envelope
                
        """
    
        return numax / dnu - eps

    def _get_enns(self, nmax, norders):
        """Compute radial order numbers.
    
        Get the enns that will be included in the asymptotic relation fit. 
        These are all integer.
    
        Parameters
        ----------
        nmax : float
            Frequency of maximum power of the p-mode envelope
        norders : int
            Total number of radial orders to consider
    
        Returns
        -------
        enns : ndarray
                Numpy array of norders radial orders (integers) around numax (nmax).
                
        """
    
        below = np.floor(nmax - np.floor(norders/2)).astype(int)
        above = np.floor(nmax + np.ceil(norders/2)).astype(int)
    
        # Handling of single input (during fitting), or array input when evaluating
        # the fit result
        if type(below) != np.ndarray:
            return np.arange(below, above)
        else:
            out = np.concatenate([np.arange(x, y) for x, y in zip(below, above)])
            return out.reshape(-1, norders)

    def _asymptotic_relation(self, numax, dnu, eps, alpha, norders):
        """ Compute the l=0 mode frequencies from the asymptotic relation for
        p-modes
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (unitless).
        alpha : float
            Curvature factor of l=0 ridge (second order term, unitless).
        norders : int
            Number of desired radial orders to calculate frequncies for, centered
            around numax.
    
        Returns
        -------
        nu0s : ndarray
            Array of l=0 mode frequencies from the asymptotic relation (muHz).
    
        """
        
        nmax = self._get_nmax(dnu, numax, eps)
        enns = self._get_enns(nmax, norders)
        return (enns.T + eps + alpha/2*(enns.T - nmax)**2) * dnu


    def _P_envelope(self, nu, hmax, numax, width):
        """ Power of the seismic p-mode envelope
    
        Computes the power at frequency nu in the p-mode envelope from a Gaussian
        distribution. Used for computing mode heights.
    
        Parameters
        ----------
        nu : float
            Frequency (in muHz).
    
        hmax : float
            Height of p-mode envelope (in SNR).
    
        numax : float
            Frequency of maximum power of the p-mode envelope (in muHz).
    
        width : float
            Width of the p-mode envelope (in muHz).
    
        Returns
        -------
        h : float
            Power at frequency nu (in SNR)
            
        """
    
        return hmax * np.exp(- 0.5 * (nu - numax)**2 / width**2)

    def _lor(self, freq, h, w):
        """ Lorentzian to describe a mode.

        Parameters
        ----------
        freq : float
            Frequency of lorentzian (muHz).
        h : float
            Height of the lorentizan (SNR).
        w : float
            Full width of the lorentzian (log10(muHz)).

        Returns
        -------
        mode : ndarray
            The SNR as a function frequency for a lorentzian.
            
        """
       
        return h / (1.0 + 4.0/w**2*(self.f - freq)**2)


    def _pair(self, freq0, h, w, d02, hfac=0.7):
        """Define a pair as the sum of two Lorentzians.

        A pair is assumed to consist of an l=0 and an l=2 mode. The widths are
        assumed to be identical, and the height of the l=2 mode is scaled
        relative to that of the l=0 mode. The frequency of the l=2 mode is the
        l=0 frequency minus the small separation.

        Parameters
        ----------
        freq0 : float
            Frequency of the l=0 (muHz).
        h : float
            Height of the l=0 (SNR).
        w : float
            The mode width (identical for l=2 and l=0) (log10(muHz)).
        d02 : float
            The small separation (muHz).
        hfac : float, optional
            Ratio of the l=2 height to that of l=0 (unitless).

        Returns
        -------
        pair_model : array
            The SNR as a function of frequency of a mode pair.
            
        """
        
        pair_model = self._lor(freq0, h, w)
        pair_model += self._lor(freq0 - d02, h*hfac, w)
        return pair_model

    
    def model(self, dnu, numax, eps, d02, alpha, hmax, envwidth, modewidth,
              *args):
        """ Constructs a spectrum model from the asymptotic relation.

        The asymptotic relation for p-modes with angular degree, l=0, is
        defined as:

        nu_nl = (n + epsilon + alpha/2(n - nmax)**2) * log(dnu) ,

        where nmax = numax / dnu - epsilon.

        We separate the l=0 and l=2 modes by the small separation d02.

        Parameters
        ----------
        dnu : float
            Large separation log10(muHz)
        lognumax : float
            Frequency of maximum power of the p-mode envelope log10(muHz)

        eps : float
            Phase term of the asymptotic relation (unitless)
        alpha : float
            Curvature of the asymptotic relation log10(unitless)
        d02 : float
            Small separation log10(muHz)
        loghmax : float
            Gaussian height of p-mode envelope log10(SNR)
        logenvwidth : float
            Gaussian width of the p-mode envelope log10(muHz)
        logmodewidth : float
            Width of the modes (log10(muHz))
        *args : array-like
            List of additional parameters (Teff, bp_rp) that aren't actually
            used to construct the spectrum model, but just for evaluating the
            prior.

        Returns
        -------
        model : ndarray
            spectrum model around the p-mode envelope
            
        """

        f0s = self._asymptotic_relation(10**numax, 10**dnu, eps, 10**alpha, self.norders)
        Hs = self._P_envelope(f0s, 10**hmax, 10**numax, 10**envwidth)
        
        modewidth = 10**modewidth # widths are the same for all modes
        d02 = 10**d02
        
        mod = np.ones(len(self.f))
        for n in range(len(f0s)):
            mod += self._pair(f0s[n], Hs[n], modewidth, d02)
        return mod

    def __call__(self, p):
        """ Produce model of the asymptotic relation

        Parameters
        ----------
        p : list
            list of model parameters

        Returns
        -------
        model : array
            spectrum model around the p-mode envelope
            
        """

        return self.model(*p)


class asymptotic_fit(kde, plotting, asymp_spec_model):
    """ Class for fitting a spectrum based on the asymptotic relation.

    Parameters
    ----------
    f : ndarray
        Numpy array of frequency bins of the spectrum (muHz).
    s : ndarray
        Numpy array of power in each frequency bin (SNR).
    start_samples: ndarray
        Samples representing the starting guess for the asymptotic
        peakbagging.
    teff : [real, real]
        Stellar effective temperature and uncertainty
    bp_rp : [real, real]
        The Gaia Gbp - Grp color value and uncertainty (probably ~< 0.01 dex)
    store_chains : bool, optional
        Flag for storing all the full set of samples from the MCMC run.
        Warning, if running multiple targets, make sure you have enough memory.
    norders : int, optional
        Number of radial orders to fit

    Attributes
    ----------
    f : ndarray
        Numpy array of frequency bins of the spectrum (muHz).
    s : ndarray
        Numpy array of power in each frequency bin (SNR).
    sel : ndarray, bool
        Numpy array of boolean values specifying the frequency range to be
        considered in the asymptotic relation fit.
    model : asy_peakbag.model.model instance
        Function for computing a spectrum model given a set of parameters.
    bounds : ndarray
        Numpy array of upper and lower boundaries for the asymptotic relation
        fit. These limits truncate the likelihood function.
    gaussian : ndarray
        Numpy array of tuples of mean and sigma for Gaussian
        priors on each of the fit parameters (To be removed when full
        KDE is implimented).
        
    """

    def __init__(self, starinst, kdeinst=None, norders=None):

        self.pg = starinst.pg
        self.f = starinst.f
        self.s = starinst.s
        self.norders = norders

        self.par_names = ['dnu', 'numax', 'eps', 'd02', 'alpha', 'env_height',
                          'env_width', 'mode_width', 'teff', 'bp_rp']

        self._set_kde(starinst, kdeinst)

        self.start = self._get_asy_start()

        lfreq, ufreq = self._get_freq_range()

        self.sel = (lfreq < self.f) & (self.f < ufreq)

        self.model = asymp_spec_model(self.f[self.sel], self.norders)

        starinst.asy_fit = self
            
    def __call__(self, dnu=[1, -1], numax=[1, -1], teff=[1, -1], bp_rp=[1, -1],
                 store_chains=False):
        """ Setup, run and parse the asymptotic relation fit using EMCEE.

        Parameters
        ----------
        dnu : [real, real]
            Large frequency spacing and uncertainty
        numax : [real, real]
            Frequency of maximum power and uncertainty
        teff : [real, real]
            Stellar effective temperature and uncertainty
        bp_rp : [real, real]
            The Gaia Gbp - Grp color value and uncertainty
            (probably ~< 0.01 dex).

        Returns
        -------
        asy_result : Dict
            A dictionary of the modeID DataFrame and the summary DataFrame.
            
        """
        self.store_chains = store_chains
        
        self.obs = {'dnu': dnu, 'numax': numax, 'teff': teff, 'bp_rp': bp_rp}
        
        self.log_obs = {x: to_log10(*self.obs[x]) for x in self.obs.keys() if x != 'bp_rp'}

        #self._start_init() # TODO - finish up this function

        self.fit = pb.mcmc(np.median(self.start_samples, axis=0), 
                           self.likelihood, self.prior)

        self.fit(start_samples=self.start_samples)

        self.modeID = self.get_modeIDs(self.fit, self.norders)

        self.summary, self.mle_model = self._get_summary_stats()

        self.samples = self.fit.flatchain

        self.acceptance = self.fit.acceptance

        return {'modeID': self.modeID, 'summary': self.summary}

    def _get_summary_stats(self):
        """ Make dataframe with fit summary statistics
    
        Creates a dataframe that contains various quantities that summarize the
        fit. Note, these are predominantly derived from the marginalized posteriors.
       
        Returns
        -------
        summary : pandas.DataFrame
            Dataframe with the summary statistics.
        mle_model : ndarray
            Numpy array with the model spectrum corresponding to the maximum
            likelihood solution.
            
        """
    
        idx = np.argmax(self.fit.flatlnlike)
        
        # Append here to add other statistics
        stats = OrderedDict({'mle' : self.fit.flatchain[idx,:], 
                             'mean' : np.mean(self.fit.flatchain, axis = 0),
                             'std' : np.std(self.fit.flatchain, axis = 0),
                             'skew' : scist.skew(self.fit.flatchain, axis = 0),
                             '2nd' : np.percentile(self.fit.flatchain, 50-95.4499736104/2, axis=0),
                             '16th' : np.percentile(self.fit.flatchain, 50-68.2689492137/2, axis=0),
                             '50th' : np.percentile(self.fit.flatchain, 50, axis=0),
                             '84th' : np.percentile(self.fit.flatchain, 50+68.2689492137/2, axis=0),
                             '97th' : np.percentile(self.fit.flatchain, 50+95.4499736104/2, axis=0),
                             'MAD' : scist.median_absolute_deviation(self.fit.flatchain, axis=0)})
    
        summary = pd.DataFrame(stats, index = self.par_names)
        
        mle_model = self.model(stats['mle'])
        
        return summary, mle_model

    def _get_asy_start(self):
        """ Get start averages for sampling
        """
 
        mu = np.median(self.start_samples, axis=0)
        start = [10**mu[0], 10**mu[1], mu[2], 10**mu[3], 10**mu[4], mu[5], 
                 mu[6], mu[7], 10**mu[8], mu[9]]
        return start

    def _get_freq_range(self):
        """ Get frequency range for model
        """
        
        dnu, numax, eps = self.start[:3]
        
        nmax = self._get_nmax(dnu, numax, eps)
        enns = self._get_enns(nmax, self.norders)

        lfreq = (min(enns) - 1.25 + eps) * dnu
        ufreq = (max(enns) + 1.25 + eps) * dnu
        return lfreq, ufreq

    def _start_init(self, verbose=False):
        """ Bodge a better starting point
        """
        
        like_start = np.ones(len(self.start_samples[:, 0]))
        for idx, samp in enumerate(self.start_samples):
            like_start[idx] = self.likelihood(samp)
        if verbose:
            print(f'Likelihood at the start : {np.max(like_start)}')
            print(f'Start params from init : {self.start_samples[np.argmax(like_start), :]}')

    def get_modeIDs(self, fit, norders):
        """ Set mode ID in a dataframe

        Evaluates the asymptotic relation for each walker position from the
        MCMC fit. The median values of the resulting set of frequencies are
        then returned in a pandas.DataFrame

        Parameters
        ----------
        fit : asy_peakbag.mcmc class instance
            mcmc class instances used in the fit
        norders : int
            Number of radial orders to output. Note that doesn't have to be
            the same as that used int he fit itself.

        Returns
        -------
        modeID : pandas.DataFrame
            Dataframe of radial order, n (best guess), angular degree, l,
            frequency and frequency error.
            
        """

        flatchain = fit.flatchain

        nsamps = np.shape(flatchain)[0]

        nu0_samps = np.empty((nsamps, norders))
        nu2_samps = np.empty((nsamps, norders))

        nu0_samps = self._asymptotic_relation(10**flatchain[:, 1], 
                                              10**flatchain[:, 0],
                                              flatchain[:, 2], 
                                              10**flatchain[:, 4], norders)
        nu2_samps = nu0_samps - 10**flatchain[:, 3]

        nus_med = np.median(np.array([nu0_samps, nu2_samps]), axis=2)
        nus_mad = scist.median_absolute_deviation(np.array([nu0_samps, 
                                                            nu2_samps]), axis=2)

        ells = [0 if i % 2 else 2 for i in range(2*norders)]

        nus_med_out = []
        nus_mad_out = []

        for i in range(norders):
            nus_med_out += [nus_med[1, i], nus_med[0, i]]
            nus_mad_out += [nus_mad[1, i], nus_mad[0, i]]

        modeID = pd.DataFrame({'ell': ells, 'nu_med': nus_med_out,
                               'nu_mad': nus_mad_out})
        return modeID


    def _set_kde(self, starinst, kdeinst):
        """ Set the kde attribute    
        """
        
        if kdeinst:
            self.start_samples = kdeinst.samples
            self.kde = kdeinst.kde
            
        elif hasattr(starinst, 'kde'):
            self.start_samples = starinst.kde.samples
            self.kde = starinst.kde.kde
        else:
            raise ValueError("Asy_peakbag won't run without samples of the prior")

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
        
        lp = 0
        
        # Added linewidth constraints
        if (p[7] > self.start[7] + np.log10(1.5)):
            lp += normal(10**p[7], 10**self.start[7]*1.5, 10**self.start[7]*0.1)
        
        # Constraints from KDE
        lp += np.log(self.kde.pdf(p))
        
        return lp
        
    def likelihood(self, p):
        """ Likelihood function for set of model parameters

        Evaluates the likelihood function for a set of
        model parameters.  This includes the constraint from
        the observed variables.

        The code now includes a penalty to limit very large linewidth
        examples.  The penalty is very basic and in some cases where the
        true linewidth is much larger than the linewidth in the
        prior it will become informative.  For the most part this
        should not be a problem because if you  care about linewidths
        you should be using the outpout from the peakbag model.

        Parameters
        ----------
        p : array
            Array of model parameters

        Returns
        -------
        lnlike : float
            The log likelihood evaluated at p.

        """
        
        lnlike = 0
        
        # Constraint from input obs
        lnlike += normal(p[-2], *self.log_obs['teff'])
        lnlike += normal(p[-1], *self.obs['bp_rp'])
        
        # Constraint from the periodogram
        mod = self.model(p)
        lnlike += -np.sum(np.log(mod) + self.s[self.sel] / mod)
        return lnlike