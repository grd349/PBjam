"""
The peakbagging module contains the classes used for the peakbag stage of PBjam. The 
main class to interacte with is the peakbag class, which will handle potential slicing
of the spectrum or not and combine results for each case.
"""

from functools import partial
import jax, emcee, warnings, time
import jax.numpy as jnp
from pbjam.plotting import plotting 
import pbjam.distributions as dist
from pbjam import jar, samplers
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import scipy.stats as st

class peakbag(plotting):
    """
    A class for handling the peakbagging run(s). The spectrum can be divided into a 
    desired number of slices which can be peakbagged independently, which tends to be
    faster than fitting the whole spectrum at once. 

    The default setting is to divide the spectrum into roughly the number of radial orders
    provided in the input frequency list. However, checks are performed to make sure closely
    spaced modes aren't split so any potential correlation is correctly accounted for. The 
    number of slices may therefore be less than the requested if it's not possible to separate
    the modes.

    The slicing is done by a K-means clustering algorithm, where clusters are then merged if they
    split up for example an l=2,0 mode pair.  

    Parameters
    ----------
    f : array-like
        The frequency array of the spectrum.
    s : array-like
        The values of the power density spectrum.
    ell : array-like
        Angular degree of the modes.
    freq : array-like
        Frequencies of the modes corresponding to the angular degrees.
    height : array-like, optional
        Heights of the modes corresponding to the angular degrees. Default is None, 
        in which case an SNR of 1 is assumed. Providing betters estimates from 
        modeID is however strongly recommended.
    width : array-like, optional
        Widths of the modes corresponding to the angular degrees. Default is None, 
        in which case an width of 0.1 is assumed. Providing betters estimates from 
        modeID is however strongly recommended.
    zeta : array-like, optional
        Mixed-mode coupling factors of the modes corresponding to the angular degrees. 
        Default is None, in which case 0 (no mixing) is assumed for all modes.
    dnu : float, optional
        Large frequency separation. Default is None, in which case it's estimated from 
        the provided `ell` and `freq` arrays.
    d02 : float, optional
        Small frequency separation between l=0 and l=2 modes. Default is None, in which 
        case it's estimated from the provided `ell` and `freq` arrays.
    freqLimits : list, optional
        Frequency limits for the peakbagging. Default is an empty list, in which case the 
        lowest and heighest radial mode frequencies -/+ 1 radial order are used to set the limits.
    rotAsym : array-like, optional
        Rotational asymmetry parameters. Default is None.
    RV : array-like, optional
        Radial velocity and associated error of the star in km/s. Default is None.
    slices : int, optional
        Number of slices for mode fitting. Default is -1, in which case the number of radial orders 
        determined from `freq` is used. 
    snrInput : bool, optional
        Flag indicating if the input is signal-to-noise ratio (SNR) spectrum. Default is False.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    N_p : int
        Number of radial (l=0) modes.
    Nmodes : int
        Total number of modes.
    """

    def __init__(self, f, s, ell, freq, height=None, width=None, zeta=None, dnu=None, d02=None, freqLimits=[], rotAsym=None, RV=None, slices=0, snrInput=False, samplerType='emcee', **kwargs):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        # Assign some default parameters if none are given.
        self._checkDefaults()

        self.width[0, :] = self.width[0, :] / (1-self.zeta)

        # Select modes to include based on freqLimits        
        self._pickModes()
        
        self.N_p = len(self.ell[self.ell==0])

        self.Nmodes = len(self.freq[0, :])
 
        # If is not the SNR spectrum, remove the packground first and turn the input heights into SNR.
        if not self.snrInput:
            self.bkg = self.getBkg()

            self.snr = self.s / self.bkg(self.f)

            self.height[0, :] = self.height[0, :] / self.bkg(self.freq[0, :])

        else:
            self.snr = self.s
 
        # Create list of peakbag instances.
        self.createPeakbagInstances()

    def _checkDefaults(self):
        """
        Checks and sets default values for attributes if they are None. Just to tidy up the init function.
        """

        self.dnu = self._setDnu()
                        
        self.d02 = self._setd02()

        if self.height is None:
            self.height = np.ones_like(self.freq)
        
        if self.width is None:
            self.width = 0.1 * np.ones_like(self.freq)

        if self.zeta is None:
            self.zeta = np.zeros(self.freq.shape[-1])
        
        if self.rotAsym is None:
            self.rotAsym = np.zeros_like(self.freq)

        if self.RV is None:
            self.RV = (0, 0)

    def _pickModes(self, fac=1):
        """
        Selects the modes to include in the model based on frequency limits.

        Checks if `freqLimits` is empty. If so, it sets default frequency limits based on the minimum and maximum frequencies of the l=0 modes +/- fac*dnu.

        The method updates the attributes `ell`, `freq`, `height`, `width`, `rotAsym`, and `zeta` to include only the selected modes.

        Parameters
        ----------
        fac : float, optional
            A factor used to adjust the frequency limits. Default is 1.

        Returns
        -------
        idx : array-like
            A boolean array indicating which modes are selected based on the frequency limits.
        """

        if len(self.freqLimits) == 0:
            self.freqLimits = [min(self.freq[0, self.ell==0]) - fac * self.dnu[0],
                                max(self.freq[0, self.ell==0]) + fac * self.dnu[0]]

        idx = (min(self.freqLimits) < self.freq[0, :]) & (self.freq[0,:] < max(self.freqLimits))

        self.ell = self.ell[idx]

        self.freq = self.freq[:, idx]
        
        self.height = self.height[:, idx]
        
        self.width = self.width[:, idx]
        
        self.rotAsym = self.rotAsym[:, idx]

        self.zeta = self.zeta[idx]
 
        return idx
    
    def _setd02(self):
        """
        Set d02 if not provided as input. This is used to define a wide PDF which is 
        used to penalize l=0 and l=2 frequencies that drift away from each other too 
        much, and also ensure they are unlikely to be negative.

        A precise values is not strictly necessary.

        Returns
        -------
        d02 : array-like
            Estimate of the l=2,0 small separation.
        """
           
        if self.d02 is None and (2 in self.ell and 0 in self.ell):
            try:
                d02 = np.array([np.median(self.freq[0, self.ell==0]-self.freq[0, self.ell==2]), jnp.nan])
            except:
                warnings.warn("Estimating d02 as 0.1*dnu")
                d02 = np.array([0.1 * self.dnu[0], jnp.nan])
            
        elif isinstance(self.d02, (float, int)):
            d02 = np.array([self.d02, jnp.nan])
        
        else:
            d02 = np.array(self.d02)
            
            assert (d02.dtype==float) or (d02.dtype==int)
        
        return d02
    
    def _setDnu(self):
        """
        Set dnu if not provided as input. This is used in defining the slicing so a 
        precise values is not critical. 

        Returns
        -------
        dnu : array-like
            Estimated large separation of the modes.
        """

        if self.dnu is None:
            if 0 in self.ell:
                ref_l = 0
            elif 2 in self.ell:
                ref_l = 2
            elif 1 in self.ell:
                ref_l = 1
            else:
                raise ValueError('ells must contain either l=0,1, or 2.')

            dnu = np.array([jnp.median(jnp.diff(self.freq[0, self.ell==ref_l])), jnp.nan])
        
        elif isinstance(self.dnu, (float, int)):
            dnu = np.array([self.dnu, jnp.nan])
        
        else:
            dnu = np.array(self.dnu)
            
            assert (dnu.dtype==float) or (dnu.dtype==int)

        return dnu
    
    def _checkSmallDiffs(self, cuts, nu, Gamma):
        """
        Check for any slices that split closely spaced modes defined in termes 
        of a provided width `Gamma`.

        The frequency differences simply have to be larger than the given width.

        Parameters
        ----------
        cuts : array-like
            Frequencies at which slices are made.
        nu : array-like
            Mode frequencies.
        Gamma : float
            The width criterion.

        Returns
        -------
        goodCutIdx : bool
            Array of boolean values corresponding to the valid cuts.
        """

        distances = np.abs(nu[:, np.newaxis] - cuts)
    
        diffs = np.array([distances[m, i] for i, m in enumerate(np.argmin(distances, axis=0))])
    
        goodCutIdx = diffs > Gamma

        return goodCutIdx

    def _checkNoSmallSep(self, C, nu, ells):
        """
        Check for any slices that split l=2,0 mode pairs, if they do the slice is rejected. 

        Parameters
        ----------
        C : array-like
            Frequencies at which slices are made.
        nu : array-like
            Mode frequencies.
        ells : float
            Angular degrees of the modes in `nu`.

        Returns
        -------
        goodCutIdx : bool
            Array of boolean values corresponding to the valid cuts.
        """

        goodCutIdx = np.ones(len(C),dtype=bool)
    
        for i, c in enumerate(C):
        
            if (i==0) or (i==len(C)-1):
                continue
        
            bidx = nu < c
            nub_idx = np.argmax(nu[bidx] - c)
            lb = ells[bidx][nub_idx]
    
            aidx = nu > c
            nua_idx = np.argmin(nu[aidx] - c)
            la = ells[aidx][nua_idx]
        
            if (int(lb)==2) and (int(la)==0):
                goodCut = False

            elif (int(lb)==3) and (int(la)==1):
                goodCut = False

            else:
                goodCut = True
    
            goodCutIdx[i] *= goodCut

        return goodCutIdx

    def _kmeans(self, nu, ells, centroids=None, max_iters=100):
        """
        Assign a mode to a cluster of modes using K-means clustering. 

        The initial list of cluster centroids are taken as l=0 modes by default.

        Parameters
        ----------
        nu : array-like
            Mode frequencies.
        ells : float
            Angular degrees of the modes in `nu`.
        centroids : array-like, optional
            Array of centroids to start at, by default None in which case the l=0 modes frequencies are used.
        max_iters : int, optional
            Maximum number of iterations to use to establish the clusters, by default 100

        Returns
        -------
        centroids : array-like
            Locations of the new centroids after clustering.
        labels : list
            Labels of the clusters.
        k : int
            Number of clusters.
        """

        nu = np.sort(nu)
         
        if not centroids:
            _k = int(np.ceil(len(ells[ells==0]) / self.Nslices))

            centroids = nu[ells==0][::_k]
         
        k = len(centroids) # should be the same as _k
         
        for _ in range(max_iters):
        
            # Assign each data point to the closest centroid
            distances = np.abs(nu[:, np.newaxis] - centroids) # distance of each point to all the centroids

            labels = np.argmin(distances, axis=1) # assign to closest centroid
             
            # Update centroids based on the mean of data points assigned to each cluster
            new_centroids = np.array([nu[labels == i].mean() for i in range(k)])
        
            # Merge clusters with only one point into the nearest cluster
            for i in range(k):
            
                if len(labels[labels == i]) == 1:
                    closest_centroid_idx = np.argmin(np.abs(centroids - centroids[i]))
            
                    labels[labels == i] = closest_centroid_idx
        
            # If centroids have not changed significantly, stop
            if np.allclose(centroids, new_centroids):
                break
        
            centroids = new_centroids
    
        return centroids, labels, k

    def _determineCuts(self, dnu, nu, labels):
        """
        Determine initial location of the slice locations based on the K-means clustering,
        where the slice locations are initially placed half-way between two adjacent clusters.

        Appends an upper and lower limits are also appended to the list. 

        Parameters
        ----------
        dnu : float
            Large separation, used to define the upper and lower limits that are appended.
        nu : array-like
            Mode frequencies.
        labels : list
            Labels of the clusters.

        Returns
        -------
        cuts : array-like
            Array of frequencies at which to place the slices.
        """

        x = np.where(np.diff(labels) == 1)[0]
         
        cuts = np.append(nu.min() - dnu / 3, # append lower limit 
                         np.append(nu[x] + (nu[x+1] - nu[x])/2,  # append list of cuts based on clustering
                         nu.max() + dnu / 3)) # append upper limit

        return cuts

    def getBkg(self, a=0.66, b=0.88, skips=100):
        """ Estimate the background

        Takes an average of the power at linearly spaced points along the
        log(frequency) axis, where the width of the averaging window increases
        as a power law.

        The mean power values are interpolated back onto the full linear
        frequency axis to estimate the background noise level at all
        frequencies.

        Returns
        -------
        b : array
            Array of psd values approximating the background.
        """

        freq_skips = np.exp(np.linspace(np.log(self.f[0]), np.log(self.f[-1]), skips))

        m = np.array([np.median(self.s[np.abs(self.f-fi) < a*fi**b]) for fi in freq_skips])

        bkgModel = jar.jaxInterp1D(freq_skips, m/np.log(2))

        return bkgModel
  
    def _slc(self, x, low, high):
        """ Select indices of x based on provided low and high limits.

        Parameters
        ----------
        x : array-like
            Array to base selection on.
        low : float
            Low limit to base the selection on.
        high : float
            Low limit to base the selection on.
        
        Returns
        -------
        idx : bool
            Boolean array of shape equal to x with the high/low 
            selection applied.
        """

        return (low <= x) & (x <= high)
    
    def sliceSpectrum(self):
        """ 
        Slice up the envelope. Sets a series of frequency limits which divide the detected 
        modes into roughly equal number of modes per slice. 

        Parameters
        ----------
        result : dict
            Dictionary of results from the mode ID stage.
        fac : int, optional
            Factor scale dnu to include modes outside envelope, usually very mixed l=1 
            modes, by default 1.

        Returns
        -------
        limits : np.array
            Frequencies delimiting the slices. 
        """
   
        sortidx = np.argsort(self.freq[0, :])
        
        freqs = self.freq[0, sortidx]
        ells = self.ell[sortidx]
         
        _, labels, nclusters = self._kmeans(freqs, ells)
         
        # find cuts
        cuts = self._determineCuts(self.dnu[0], freqs, labels)
 
        assert len(cuts) == nclusters + 1

        # weed out small distances
        goodCutIdx0 = self._checkSmallDiffs(cuts, freqs, 3*np.median(self.width[0, sortidx]))

        # weed out 2, 0 and 3, 1 splitting cuts
        goodCutIdx1 = self._checkNoSmallSep(cuts, freqs, ells)

        goodCutIdx = goodCutIdx0 * goodCutIdx1

        limits = cuts[goodCutIdx]

        return limits
    
    def createPeakbagInstances(self, ):
        """
        Create a list of class instances which do the modeling in each spectrum slice. 
        Only the relevant part of the spectrum and mode parameters are passed to each 
        instance.

        Unless `self.slices` is set to 0, attempts to find the best number of slices to use. 
        Starts at the number of l=0 modes, but may adjust down if there are closely 
        spaced sets of modes. Similarly if a choice of self.slices > 0 means a slice has 
        to be placed between two closely spaced modes, the number of slices is also adjusted 
        downward to avoid slicing between closely spaced peaks.
  
        Notes
        -----
        - If `self.slices` is 0, no slicing occurs.
        - The method stores the created instances in the `pbInstances` attribute.
        """

        self.pbInstances = []

        if self.samplerType.lower()=='emcee':
            print('Using emcee to sample.')
            sampler = EmceePeakbag
        elif self.samplerType.lower()=='dynesty':
            print('Using Dynesty to sample.')
            sampler = DynestyPeakbag
        else:
            raise ValueError('Sampler choice must be emcee or dynesty')
 
        if self.slices < 0 or self.slices > len(self.ell):
            self.Nslices = len(self.ell[self.ell==0])
        
        elif self.slices == 0:
            self.Nslices = 0
        
        else:
            self.Nslices = self.slices
    
        if self.Nslices > 0:
                
            sliceLimits = self.sliceSpectrum()
            
            print('Creating envelope slices')
            for i in tqdm(range(len(sliceLimits) - 1)):
                 
                slcIdx = self._slc(self.freq[0, :], sliceLimits[i], sliceLimits[i + 1])

                _ell = self.ell[slcIdx]
                
                _zeta = self.zeta[slcIdx]
                
                _freq = self.freq[:, slcIdx]
                
                _height = self.height[:, slcIdx]
                
                _width = self.width[:, slcIdx]

                _rotAsym = self.rotAsym[:, slcIdx]
                 
                self.pbInstances.append(sampler(self.f, self.snr, _ell, _freq, _height, _width, _zeta, self.dnu, self.d02, sliceLimits[i: i+2], _rotAsym, self.RV))
                                                       
        else:
            self.pbInstances.append(sampler(self.f, self.snr, self.ell, self.freq, self.height, self.width, self.zeta, self.dnu, self.d02, self.freqLimits, self.rotAsym, self.RV))
    
    def __call__(self, sampler_kwargs={}, Nsamples=10000, **kwargs):
        """
        Run the sampler for the slice(s) and parse the results. 

        Parameters
        ----------
        dynamic : bool, optional
            Whether or not to use dynamic sampling in Dynesty, by default False. 
        progress : bool, optional
            Whether or not to show the progress, by default False. Setting this to True doesn't play well with tqdm for the slice loop.
        sampler_kwargs : dict, optional
            Additional kwargs to be passed to the sampler.
        Nsamples : int, optional
            Samples to retain in the parsed sample, by default 10000

        Returns
        -------
        result : dict
            Dictionary containing the results of the sampling.
        """

        # if self.Nslices > 0:
        #     print('Peakbagging envelope slices')
        # else:
        #     print('Peakbagging the whole envelope')
        t0 = time.time()

        for i, inst in enumerate(self.pbInstances):
            print(f'Peakbagging slice {i+1}/{len(self.pbInstances)}')
            inst(sampler_kwargs);
        
        print(f'Time taken {np.round((time.time() - t0)/60, 1)} minutes')

        self.result = self.parseSamples(Nsamples)
        
        return self.result
    
    def parseSamples(self, Nsamples):
        """
        Parses the samples to extract and organize the model parameters.

        Attempts to include at most N samples from the model, but will default
        to the actual number of samples of the model parameters if it's less than N.

        The resulting dictionary contains some global parameters, ell, enn, emm etc. and 
        two dictionaries, one containing the samples drawn and one with their summary 
        statistics.
  
        Parameters
        ----------
        smp : dict
            A dictionary of sampled parameters.
        Nmax : int, optional
            Maximum number of samples to include. Default is 5000.

        Returns
        -------
        result : dict
            A dictionary containing parsed and organized model parameters.
        """

        N = min([Nsamples, min([inst.nsamples for inst in self.pbInstances])])

        result = {'ell': np.array([]),
                  'enn': np.array([]),
                  'emm': np.array([]),
                  'zeta': np.array([]),
                  'summary': {'freq'  : np.empty(shape=(2, 0), dtype=float),
                              'height': np.empty(shape=(2, 0), dtype=float),
                              'width' : np.empty(shape=(2, 0), dtype=float),},     
                  'samples': {'freq'  : np.empty(shape=(N, 0), dtype=float),
                              'height': np.empty(shape=(N, 0), dtype=float),
                              'width' : np.empty(shape=(N, 0), dtype=float),},
                  'kstest': {'significant': np.array([], dtype=bool),
                             'pvalue': np.array([]),
                             'statistic': np.array([]),}
                  }
        
        for inst in self.pbInstances:

            # Merge top level keys.
            for key in ['ell', 'enn', 'emm', 'zeta']:

                result[key] = np.append(result[key], inst.result[key])

            # Merge lower level keys into summary and samples.
            n = result['samples']['freq'].shape[0]
            
            m = inst.result['samples']['freq'].shape[0]
                
            randInt = np.random.choice(np.arange(m), size=n, replace=False)

            for key in ['freq', 'height', 'width']:
        
                result['summary'][key] = np.append(result['summary'][key], 
                                                   inst.result['summary'][key], 
                                                   axis=1)
                
                smpl = inst.result['samples'][key][randInt, :]
                
                result['samples'][key] = np.append(result['samples'][key], 
                                                   smpl, 
                                                   axis=1)
                
            # Compute KS-statistic for mode frequencies
            for key in ['significant', 'pvalue', 'statistic']:
                result['kstest'][key] = np.append(result['kstest'][key], 
                                                  inst.result['kstest'][key])
          
        return result

    def getRotationInclination(self,):
        """
        Computes the joint posterior distribution for rotation and inclination 
        given the samples from the peakbag instances.
 
        Kernel density estimates are computed for all the marginalized posterior distributions.
        These are then sampled using an MCMC sampler, where the resulting posterior is given
        by the joint probability of getting the individual posteriors.

        Summary statistic and samples are added to the pb.result dictionary.
        """

        R = jointRotInc(self)

        samples = R()

        samplesU = R.unpackSamples(samples)

        self.result['samples'].update(samplesU)

        for key in samplesU.keys():
            self.result['summary'][key] = jar.smryStats(samplesU[key])

class jointRotInc(samplers.DynestySampling):
    """
    A class to perform joint posterior sampling for rotation and inclination parameters using emcee.

    Takes the posterior distributions from the slices in the peakbag object and samples the joint 
    probability of all the slices. 

    Notes
    -----
    This is a coarse estimate of the rotation rate and inclination. A better estimate can be achieved
    by modeling the entire envelope at once instead of using the slices. To do this set slices=0 when
    initializing peakbag.

    Parameters
    ----------
    pb : object
        Peakbagging object containing the instances for analysis.
    NKDE : int, optional
        Number of samples for KDE estimation. Default is 2500.
    bw : float, optional
        Bandwidth for KDE. Default is 0.03.
    """

    def __init__(self, pb, NKDE=2500, bw=0.03):
        
        self.insts = pb.pbInstances
        
        self.labels = ['nurot_e', 'nurot_c', 'inc']
        
        self.priors = {lbl: self.insts[0].priors[lbl] for lbl in self.labels}

        self.ndims = len(self.priors)
        
        self.Nslices = len(pb.pbInstances)
        
        self.makeKDEs(NKDE, bw)
    
    def makeKDEs(self, N, bw):
        """
        Creates KDEs of the posterior distributions from the spectrum slices.

        Parameters
        ----------
        N : int
            Number of samples for KDE estimation.
        bw : float
            Bandwidth for KDE.
        """

        self.kdes = []
        
        sampleSizes = np.array([self.insts[i].samples.shape[0] for i in range(self.Nslices)])
        
        Nc = np.append(N, sampleSizes).min()
        
        for i in range(self.Nslices):
            
            indx = np.random.choice(np.arange(self.insts[i].samples.shape[0], dtype=int), Nc)
 
            samplesU = self.insts[i].unpackSamples(self.insts[i].samples[indx, :])

            kde = sm.nonparametric.KDEMultivariate(data=[samplesU[lbl] for lbl in self.labels], 
                                                   var_type='c'*self.ndims, 
                                                   bw=[bw]*self.ndims)

            self.kdes.append(kde)
 
    @partial(jax.jit, static_argnums=(0,))
    def unpackParams(self, theta):
        """ Cast the parameters in a dictionary

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        thetaU : dict
            The unpacked parameters.

        """

        thetaU = {lbl: theta[i] for i, lbl in enumerate(self.labels)}

        return thetaU
    
    def unpackSamples(self, samples):
        """
        Unpacks the sample matrix into a dictionary.

        Parameters
        ----------
        samples : array-like
            Sample matrix.

        Returns
        -------
        S : dict
            Dictionary of samples.
        """

        S = {}
    
        for i, lbl in enumerate(self.labels):
            S[f'{lbl}'] = samples[:, i]
        
        return S
    
    @partial(jax.jit, static_argnums=(0,))
    def priorLnProb(self, thetaU):
        """
        Computes the log-prior probability for the given parameters.

        Parameters
        ----------
        thetaU : dict
            Dictionary of parameters.

        Returns
        -------
        lnP : float
            Log-prior probability.
        """
        
        lnP = jnp.sum(jnp.array([self.priors[lbl].logpdf(thetaU[lbl]) for lbl in self.labels]))
        
        return lnP
   
    def lnJointPost(self, theta):
        """
        Computes the joint log-posterior probability for the given parameters.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Returns
        -------
        lnP float
            Joint log-posterior probability.
        """

        thetaU = self.unpackParams(theta)
        
        lnPrior = self.priorLnProb(thetaU)
    
        lnL = 0  
        
        for kde in self.kdes:
            lnL += jnp.log(kde.pdf(theta)) - lnPrior 
    
        lnP = lnL + lnPrior  
 
        # Some values return nan since the KDE is only defined on a linear scale.
        if np.isnan(lnP) or np.isinf(lnP):
            lnP = -jnp.inf
        
        return lnP
 
    def __call__(self,  nwalkers=500, nsteps=500, burnFraction=0.1, accept=0.1, progress=True):
        """
        Runs the MCMC sampling to obtain the posterior samples.

        Parameters
        ----------
        nwalkers : int, optional
            Number of walkers for MCMC. Default is 500.
        nsteps : int, optional
            Number of steps for MCMC. Default is 500.
        burnFraction : float, optional
            Fraction of steps to discard as burn-in. Default is 0.1.
        accept : float, optional
            Acceptance threshold for MCMC samples. Default is 0.1.
        progress : bool, optional
            Whether to display progress during MCMC sampling. Default is True.

        Returns
        -------
        ndarray
            Posterior samples.
        """
        
        itr = 0

        p0 = np.empty(shape=(0, self.ndims), dtype=float)

        while p0.shape[0] < nwalkers:
            itr += 1

            u = np.random.uniform(0, 1, size=self.ndims)

            _p0 = self.ptform(u)

            lnP = self.lnJointPost(_p0)

            if np.isfinite(lnP):
                p0 = np.vstack((p0, _p0))

            if itr>10000:
                p0 = np.vstack((p0, _p0))

        sampler = emcee.EnsembleSampler(nwalkers, self.ndims, self.lnJointPost)

        sampler.run_mcmc(p0, nsteps=nsteps, progress=True);
        
        idx = sampler.acceptance_fraction > accept
        
        samples = sampler.get_chain()[int(burnFraction*nsteps):, idx, :].reshape((-1, self.ndims))
        
        return samples
 
class basePeakbag(plotting):
    """
    Base model class for peakbagging a section of the power spectrum.
 
    Parameters
    ----------
    f : array-like
        The frequency array of the spectrum.
    s : array-like
        The values of the power density spectrum.
    ell : array-like
        Angular degree of the modes.
    freq : array-like
        Frequencies of the modes corresponding to the angular degrees.
    height : array-like
        Heights of the modes corresponding to the angular degrees. Default is None, 
        in which case an SNR of 1 is assumed. Providing betters estimates from 
        modeID is however strongly recommended.
    width : array-like
        Widths of the modes corresponding to the angular degrees. Default is None, 
        in which case an width of 0.1 is assumed. Providing betters estimates from 
        modeID is however strongly recommended.
    zeta : array-like
        Mixed-mode coupling factors of the modes corresponding to the angular degrees.
    dnu : float
        Large frequency separation.
    d02 : float
        Small frequency separation between l=0 and l=2 modes.
    freqLimits : list
        Frequency limits for the peakbagging. 
    rotAsym : array-like
        Rotational asymmetry parameters. 
    RV : array-like
        Radial velocity of the star in km/s.
    addPriors : dict, optional
        Additional priors to be added. Default is an empty dictionary.
    **kwargs : dict
        Additional keyword arguments.
    """
 
    def __init__(self, f, s, ell, freq, height, width, zeta, dnu, d02, freqLimits, rotAsym, RV, addPriors={}, sampling='emcee', **kwargs):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
         
        # Convert everything to jax array
        for key, val in self.__dict__.items():
            if type(val) == np.ndarray:
                self.__dict__.update({key: jnp.array(val)})

        # Except ell. For some reason it's no longer jit-able?
        self.ell = list(np.array(self.ell))
         
        self.Nmodes = len(self.ell)
        
        self.setLabels()
        
        self.setPriors()
              
        self.sel = self.setFreqRange()
        
        self.ndims = len(self.priors.keys())

        self.setAddLikeTerms()

        self.zeros = jnp.zeros_like(self.f[self.sel])

    def setPriors(self, freq_err=0.03):
        """
        Sets the prior distributions for model parameters.

        Parameters
        ----------
        freq_err : float, optional
            The error in frequency, used to set the scale in percent of dnu of the normal 
            distribution for frequency priors. Default is 0.03.

        Notes
        -----
        - Initializes the `priors` dictionary and updates it with additional priors specified in `self.addPriors`.
        - For each mode, sets normal priors for 'freq', 'height', and 'width' if not already specified in `self.priors`.
        - Sets uniform priors for 'nurot_e' (envelope rotation) and 'nurot_c' (core rotation) if not already specified.
        - Sets a truncated sine prior for 'inc' (inclination) if not already specified.
        - Sets a normal prior for 'shot' noise if not already specified.

        Raises 
        ------
        ValueError if the length of labels does not match the length of priors.
        """

        self.priors = {}

        self.priors.update((k, v) for k, v in self.addPriors.items())
        
        for i in range(self.Nmodes):
            _key = f'freq{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=self.freq[0, i],  scale=freq_err * self.dnu[0])
                 
        for i in range(self.Nmodes):
            _key = f'height{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=jnp.log10(self.height[0, i]), scale=0.5)

        for i in range(self.Nmodes):
            _key = f'width{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=jnp.log10(self.width[0, i]), scale=0.1)
        
        # Envelope rotation prior
        if 'nurot_e' not in self.priors.keys():
            self.priors['nurot_e'] = dist.uniform(loc=1e-9, scale=2.)

        # Core rotation prior
        if 'nurot_c' not in self.priors.keys():
            self.priors['nurot_c'] = dist.uniform(loc=1e-9, scale=2.)

        # The inclination prior is a cos(i)~U(0,1)
        if 'inc' not in self.priors.keys():
            self.priors['inc'] = dist.truncsine()
            #self.priors['inc'] = dist.uniform(loc=0., scale=1)

        if 'shot' not in self.priors.keys():
            self.priors['shot'] = dist.normal(loc=0., scale=0.01)

        if not all([key in self.labels for key in self.priors.keys()]):
            raise ValueError('Length of labels doesnt match lenght of priors.')
     
    def setLabels(self):
        """
        Sets labels for the model parameters.

        For each key in `self.variables`, if the key is 'freq', 'height', or 'width', 
        it appends numbered labels for each mode.
        """

        # Default additional parameters
        self.labels = []
        
        # If key appears in priors dict, override default and move it to add.
        for key in self.variables.keys():

            if key in ['freq', 'height', 'width']:
                self.labels += [f'{key}{i}' for i in range(self.Nmodes)]
            else:
                self.labels.append(key)
        
        # Parameters that are in log10
        self.logpars = [key for key in self.variables.keys() if self.variables[key]['log10']]
        
    def setFreqRange(self,):
        """ Get frequency range around numax for model 

        Returns
        -------
        idx : jax device array
            Array of boolean values defining the interval of the frequency axis
            where the oscillation modes present.
        """
        
        if len(self.freqLimits) != 2:
            raise ValueError('freqLimits should be an iterable of length 2.')

        return (self.freqLimits[0] < self.f) & (self.f < self.freqLimits[1])  
    
    def chi_sqr(self, mod):
        """ Chi^2 2 dof likelihood

        Evaulates the likelihood of observing the data given the model.

        Parameters
        ----------
        mod : jax device array
            Spectrum model.

        Returns
        -------
        L : float
            Likelihood of the data given the model
        """

        lnp = -jnp.sum(jnp.log(mod) + self.s[self.sel] / mod)
 
        return jax.lax.cond(jnp.isfinite(lnp), lambda : lnp, lambda : -jnp.inf)
        
    variables = {'freq'   : {'info': 'mode frequency list'      , 'log10': False},
                 'height' : {'info': 'mode height list'         , 'log10': True},
                 'width'  : {'info': 'mode width list'          , 'log10': True},
                 'nurot_e': {'info': 'envelope rotation rate'   , 'log10': False},
                 'nurot_c': {'info': 'core otation rate'        , 'log10': False}, 
                 'inc'    : {'info': 'stellar inclination axis' , 'log10': False},
                 'shot'   : {'info': 'Shot noise level'         , 'log10': True }}
   
    @partial(jax.jit, static_argnums=(0,))
    def lnlikelihood(self, theta):
        """ Likelihood function for set of model parameters

        Evaluates the likelihood function for a set of model parameters given
        the data. This includes the constraint from the observed variables.

        The samples l are drawn from the latent parameter priors and are first
        projected into the model space before the model is constructed and the
        likelihood is constructed.

        Parameters
        ----------
        l : list
            Array of latent parameters

        Returns
        -------
        lnlike : float
            The log likelihood evaluated at the model parameters p.
        """

        thetaU = self.unpackParams(theta)
  
        # Constraint from the periodogram 
        mod = self.model(thetaU)

        lnlike = self.chi_sqr(mod)

        lnlike += self.AddLikeTerms(theta, thetaU)
        
        return lnlike
 
    def unpackParams(self, theta): 
        """ Cast the parameters in a dictionary

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        thetaU : dict
            The unpacked parameters.

        """
        
        thetaU = {'freq'    : theta[0: self.Nmodes],
                   'height' : theta[self.Nmodes: 2 * self.Nmodes],
                   'width'  : theta[2 * self.Nmodes: 3 * self.Nmodes],
                   'nurot_e': theta[self.labels.index('nurot_e')],
                   'nurot_c': theta[self.labels.index('nurot_c')],
                   'inc'    : theta[self.labels.index('inc')],
                   'shot'   : theta[self.labels.index('shot')],
                   }

        thetaU['nurot_e'] = thetaU['nurot_e'] / jnp.sin(thetaU['inc'])
        thetaU['nurot_c'] = thetaU['nurot_c'] / jnp.sin(thetaU['inc'])

        for key in self.logpars:
            thetaU[key] = 10**thetaU[key]
  
        return thetaU
 
    def model(self, thetaU):
        """
        Computes the model spectrum using a sum of Lorentzian profiles with a 
        multiplet of 2l+1 modes per mode nl. 

        The rotational splitting is a weighted sum of the core and envelope 
        rotation rates, where the weighting is the degree of mixing zeta. 
        The splitting is symmetric.

        Parameters
        ----------
        thetaU : dict
            A dictionary of model parameters.

        Returns
        -------
        modes : array-like
            The computed model spectrum.
        """ 

        modes = self.zeros + thetaU['shot']
        
        omega = self.zeta * thetaU['nurot_c'] + (1 - self.zeta) * thetaU['nurot_e']
 
        for i, l in enumerate(self.ell):
            
            for m in jnp.arange(-l, l+1):
                
                E = jar.visibility(l, m, thetaU['inc'])
                
                nu = thetaU['freq'][i] + omega[i] * m 

                H = E * thetaU['height'][i]

                modes += jar.lor(self.f[self.sel], nu, H, thetaU['width'][i])
        
        return modes
    
    def __call__(self, sampler_kwargs):
        """
        Run the sampler for the slice(s) and parse the results. 

        Parameters
        ----------
        dynamic : bool, optional
            Whether or not to use dynamic sampling in Dynesty, by default False. 
        progress : bool, optional
            Whether or not to show the progress, by default False. Setting this to True doesn't play well with tqdm for the slice loop.
        sampler_kwargs : dict, optional
            Additional kwargs to be passed to the sampler.
         
        Returns
        -------
        result : dict
            Dictionary containing the results of the sampling.
        """

        self.runSampler(**sampler_kwargs)

        self.result = self.parseSamples(self.samples)
  
        return self.result 

    def dopplerRVCorrection(self, N):
    
        if self.RV[1] > 0:
            RVs = np.random.normal(loc=self.RV[0], scale=self.RV[1], size=N)
        else:
            RVs = np.zeros(N)
        
        beta = RVs / jar.constants.c

        doppler = np.sqrt((1 + beta) / (1 - beta))

        return doppler

    def testFreqs(self, thetaU, pvalueLim=0.05):
       
        
        testResult = {'pvalue': np.zeros(self.Nmodes),
                      'statistic': np.zeros(self.Nmodes),
                      'significant': np.zeros(self.Nmodes, dtype=bool)}
        
        for k in range(self.Nmodes):
                        
            kstestResult = st.kstest(thetaU[f'freq{k}'], 
                                     self.priors[f'freq{k}'].cdf)
            
            testResult['pvalue'][k] = kstestResult.pvalue
            
            testResult['significant'][k] = kstestResult.pvalue <= pvalueLim
            
            testResult['statistic'][k] = kstestResult.statistic
            
        return testResult
    
    def parseSamples(self, samples, N=10000):
        """
        Parses the samples to extract and organize the model parameters.

        Attempts to include at most N samples from the model, but will default
        to the actual number of samples of the model parameters if it's less than N.

        The resulting dictionary contains some global parameters, ell, enn, emm etc. and 
        two dictionaries, one containing the samples drawn and one with their summary 
        statistics.
  
        Parameters
        ----------
        smp : dict
            A dictionary of sampled parameters.
        Nmax : int, optional
            Maximum number of samples to include. Default is 5000.

        Returns
        -------
        result : dict
            A dictionary containing parsed and organized model parameters.
        """

        thetaU = self.unpackSamples(samples)

        kstestResult = self.testFreqs(thetaU)

        D = self.dopplerRVCorrection(samples.shape[0])

        for key in thetaU:
            if 'freq' in key:
                thetaU[key] = D*thetaU[key]

        result = {'ell': np.array([self.ell]),
                  'enn': np.array([]),
                  'emm': np.array([]),
                  'zeta': np.array([self.zeta]),
                  'summary': {},
                  'samples': {},
                  'kstest': kstestResult
                 }
        
        for key in self.variables:

            arr = np.array([thetaU[_key] for _key in thetaU.keys() if key in _key])
            
            result['summary'][key] = np.array([np.mean(arr, axis=1), np.std(arr, axis=1)]) 
            
            result['samples'][key] = arr[:, :N].T

        return result
 
    def unpackSamples(self, samples):
        """
        Unpack a set of parameter samples into a dictionary of arrays.

        Parameters
        ----------
        samples : array-like
            A 2D array of shape (n, m), where n is the number of samples and 
            m is the number of parameters.

        Returns
        -------
        S : dict
            A dictionary containing the parameter values for each parameter 
            label.
        """
         
        S = {key: np.zeros(samples.shape[0]) for key in self.labels}
    
        for i, key in enumerate(self.labels):
            S[f'{key}'] = samples[:, i]
        
        for key in self.labels:
            if any([key.startswith(logkey) for logkey in self.logpars]):
                S[key] = 10**S[key]
    
        return S

    def setAddLikeTerms(self):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 
        """

        self.addObs = {}

        self.addObs['freq diff'] = dist.beta(a=1.00, b=1.00, loc=0., scale=100.)

    # def setAddLikeTerms(self):
    #     """ Set attribute containing additional observational data

    #     Additional observational data other than the power spectrum goes here. 

    #     Can be Teff or bp_rp color, but may also be additional constraints on
    #     e.g., numax, dnu. 
    #     """

    #     self.addObs = {}

    #     # Frequency diff prior so that they have to be positive.
    #     freq_diff = jnp.diff(self.freq)

    #     df = 0.08*self.dnu

    #     loc = max([0.01, min(freq_diff)-df])

    #     scale = min([max(freq_diff)+df, 1.25*self.dnu])

    #     self.addObs['freq diff'] = dist.beta(a=1.00, b=1.00, loc=loc, scale=scale)

        # Correlated Noise Regularisation for width
        # wGPtheta={'amp': 1, 'scale': self.dnu[0]}

        # wGPmuFunc = jar.jaxInterp1D(self.freq[0, :], jnp.log10(self.width[0, :]/(1-self.zeta)))
    
        # wGP = self.build_gp(wGPtheta, self.freq[0, :], wGPmuFunc)

        # self.addObs['widthGP'] = wGP.log_probability


        # # Correlated Noise Regularisation for amplitude
        # hGPtheta={'amp': 1, 'scale': self.dnu[0]}

        # hGPmuFunc = jar.jaxInterp1D(self.freq[0, :], jnp.log10(self.height[0, :]))
    
        # hGP = self.build_gp(hGPtheta, self.freq[0, :], hGPmuFunc)

        # self.addObs['heightGP'] = hGP.log_probability
 
    def AddLikeTerms(self, theta, thetaU):
        """ Add the additional probabilities to likelihood
        
        Adds the additional observational data likelihoods to the PSD likelihood.

        Parameters
        ----------
        p : list
            Sampling parameters.

        Returns
        -------
        lnp : float
            The likelihood of a sample given the parameter PDFs.
        """
        
        delta = jnp.diff(jnp.sort(thetaU['freq']))

        lnp = 0
        
        for d in delta:
            lnp += self.addObs['freq diff'].logpdf(d)
                
        #lnp += self.addObs['heightGP'](theta[self.Nmodes: 2 * self.Nmodes])

        #lnp += self.addObs['widthGP'](theta[2 * self.Nmodes: 3 * self.Nmodes])

        return lnp

    # def build_gp(self, theta, X, muFunc, muKwargs={}):

    #     kernel = theta["amp"] * kernels.ExpSquared(theta["scale"])

    #     GP = GaussianProcess(kernel, X, diag=1e-6, mean=partial(muFunc, **muKwargs))

    #     return GP

class DynestyPeakbag(basePeakbag, samplers.DynestySampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EmceePeakbag(basePeakbag, samplers.EmceeSampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

  
