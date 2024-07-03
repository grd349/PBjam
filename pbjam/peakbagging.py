from functools import partial
import jax, emcee, warnings
import jax.numpy as jnp
from pbjam.plotting import plotting 
import pbjam.distributions as dist
from pbjam import jar
import numpy as np
from tinygp import GaussianProcess, kernels
import statsmodels.api as sm
from tqdm import tqdm

class peakbag(plotting):

    def __init__(self, f, s, ell, freq, height, width, zeta=None, dnu=None, d02=None, freqLimits=[], rotAsym=None, slice=True, Nslices=0, snrInput=False, **kwargs):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        self.pbInstances = []
         
        self.pickModes()
        
        self.N_p = len(self.ell[self.ell==0])

        self.Nmodes = len(self.freq[0, :])

        if self.zeta is None:
            self.zeta == jnp.zeros_like(self.freq)

        self.dnu = self.setDnu()
                        
        self.d02 = self.setd02()
 
        if not self.snrInput:
            self.bkg = self.getBkg()

            self.snr = self.s / self.bkg(self.f)

            self.height[0, :] = self.height[0, :] / self.bkg(self.freq[0, :])

        else:
            self.snr = self.s

        self.width[0, :] = self.width[0, :] / (1-self.zeta)
        
        if self.rotAsym is None:
            self.rotAsym = jnp.zeros((2, len(self.ell)))

        self.createPeakbagInstances()

    def setd02(self):
            
        if self.d02 is None and (2 in self.ell and 0 in self.ell):
            try:
                self.d02 = np.median(self.freq[self.ell==0]-self.freq[self.ell==2])
            except:
                warnings.warn("Estimating d02 as 0.1*dnu")
                self.d02 = 0.1 * self.dnu[0]

        elif isinstance(self.d02, (float, int)):
            d02 = np.array([self.d02, jnp.nan])
        
        else:
            d02 = np.array(self.d02)
            
            assert (d02.dtype==float) or (d02.type==int)

        return d02
    
    def setDnu(self):
            
        if self.dnu is None:
            if 0 in self.ell:
                ref_l = 0
            elif 2 in self.ell:
                ref_l = 2
            elif 1 in self.ell:
                ref_l = 1

            dnu = np.array([jnp.median(jnp.diff(self.freq[self.ell==ref_l])), jnp.nan])
        
        elif isinstance(self.dnu, (float, int)):
            dnu = np.array([self.dnu, jnp.nan])
        
        else:
            dnu = np.array(self.dnu)
            
            assert (dnu.dtype==float) or (dnu.type==int)

        return dnu
    
    def checkSmallDiffs(self, cuts, nu, Gamma):
    
        distances = np.abs(nu[:, np.newaxis] - cuts)
    
        diffs = np.array([distances[m, i] for i, m in enumerate(np.argmin(distances, axis=0))])
    
        goodCutIdx = diffs > Gamma

        return goodCutIdx

    def checkNoSmallSep(self, C, nu, ells):

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

    def kmeans(self, nu, ells, centroids=None, max_iters=100):
 
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

    def determineCuts(self, dnu, nu, labels):
    
        x = np.where(np.diff(labels) == 1)[0]
         
        cuts = np.append(nu.min() - dnu / 3, # append lower limit 
                         np.append(nu[x] + (nu[x+1] - nu[x])/2,  # append list of cuts
                         nu.max() + dnu / 3)) # append upper limit

        return cuts

    def createPeakbagInstances(self):

        if self.slice:
    
            if self.Nslices < 1:
                self.Nslices = len(self.ell[self.ell==0])

            sliceLimits = self.sliceSpectrum()
            
            print('Creating envelope slices')
            for i in tqdm(range(len(sliceLimits) - 1)):
                 
                slcIdx = self.slc(self.freq[0, :], sliceLimits[i], sliceLimits[i + 1])

                _ell = self.ell[slcIdx]
                
                _zeta = self.zeta[slcIdx]
                
                _freq = self.freq[:, slcIdx]
                
                _height = self.height[:, slcIdx]
                
                _width = self.width[:, slcIdx]

                _rotAsym = self.rotAsym[:, slcIdx]
                 
                self.pbInstances.append(DynestyPeakbag(self.f, self.snr, _ell, _freq, _height, _width, _zeta, self.dnu, self.d02, sliceLimits[i: i+2], _rotAsym))
                                                       
        else:
            self.pbInstances.append(DynestyPeakbag(self.f, self.snr, self.ell, self.freq, self.height, self.width, self.zeta, self.dnu, self.d02, self.freqLimits, _rotAsym))

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
    
    def pickModes(self, fac=1):
       
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

    def slc(self, x, low, high):
            return (low <= x) & (x <= high)
    
    def sliceSpectrum(self):
        """ Slicing up the envelope 

        Sets a series of frequency limits which divide the detected modes into roughly 
        equal number of modes per slice. 

        Parameters
        ----------
        result : dict
            Dictionary of results from the mode ID stage.
        fac : int, optional
            Factor scale dnu to include modes outside envelope, usually very mixed l=1 modes, by default 1

        Returns
        -------
        limits : np.array
            Frequencies delimiting the slices. 
        """
   
        sortidx = np.argsort(self.freq[0, :])
        
        freqs = self.freq[0, sortidx]
        ells = self.ell[sortidx]
         
        _, labels, nclusters = self.kmeans(freqs, ells)
         
        # find cuts
        cuts = self.determineCuts(self.dnu[0], freqs, labels)
 
        assert len(cuts) == nclusters + 1

        # weed out small distances
        goodCutIdx0 = self.checkSmallDiffs(cuts, freqs, 3*np.median(self.width[0, sortidx]))

        # weed out 2, 0 and 3, 1 splitting cuts
        goodCutIdx1 = self.checkNoSmallSep(cuts, freqs, ells)

        goodCutIdx = goodCutIdx0 * goodCutIdx1

        limits = cuts[goodCutIdx]

        return limits
    
    def __call__(self, dynamic=False, progress=False, sampler_kwargs={}, Nsamples=10000):
  
        if self.slice:
            print('Peakbagging envelope slices')
        else:
            print('Peakbagging the whole envelope')

        for inst in tqdm(self.pbInstances):
             
            inst(dynamic, progress, sampler_kwargs)
 
        samplesSaved = min([Nsamples, min([inst.nsamples for inst in self.pbInstances])])

        mainResult = {'ell': np.array([]),
                      'enn': np.array([]),
                      'emm': np.array([]),
                      'zeta': np.array([]),
                      'summary': {'freq'  : np.empty(shape=(2, 0), dtype=float),
                                'height': np.empty(shape=(2, 0), dtype=float),
                                'width' : np.empty(shape=(2, 0), dtype=float),},     
                      'samples': {'freq'  : np.empty(shape=(samplesSaved, 0), dtype=float),
                                'height': np.empty(shape=(samplesSaved, 0), dtype=float),
                                'width' : np.empty(shape=(samplesSaved, 0), dtype=float),},
                      }
        
        for inst in self.pbInstances:

            mainResult = self.compileResults(mainResult, inst.result)

        self.result = mainResult
 
        return self.result
    
    def compileResults(self, mainResult, instResult):
    
        for key in ['ell', 'enn', 'ell', 'zeta']:
            mainResult[key] = np.append(mainResult[key], instResult[key])
            
        n = mainResult['samples']['freq'].shape[0]
        
        m = instResult['samples']['freq'].shape[0]
            
        randInt = np.random.choice(np.arange(m), size=n, replace=False)
        
        for key in ['freq', 'height', 'width']:
        
            mainResult['summary'][key] = np.append(mainResult['summary'][key], instResult['summary'][key], axis=1)
            
            smpl = instResult['samples'][key][randInt, :]
            
            mainResult['samples'][key] = np.append(mainResult['samples'][key], smpl, axis=1)
            
        return mainResult

    def getRotationInclination(self,):

        R = jointRotInc(self)

        samples = R()

        samplesU = R.unpackSamples(samples)

        self.result['samples'].update(samplesU)

        for key in samplesU.keys():
            self.result['summary'][key] = jar.smryStats(samplesU[key])

class jointRotInc(jar.DynestySamplingTools):
    
    def __init__(self, pb, NKDE=2500, bw=0.03):
        
        self.insts = pb.pbInstances
        
        self.labels = ['nurot_e', 'nurot_c', 'inc']
        
        self.priors = {lbl: self.insts[0].priors[lbl] for lbl in self.labels}

        self.ndims = len(self.priors)
        
        self.Nslices = len(pb.pbInstances)
        
        self.makeKDEs(NKDE, bw)
    
    def makeKDEs(self, N, bw):
        
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
        thetaU = {lbl: theta[i] for i, lbl in enumerate(self.labels)}
        return thetaU
    
    def unpackSamples(self, samples):
        S = {}
    
        for i, lbl in enumerate(self.labels):
            S[f'{lbl}'] = samples[:, i]
        
        return S
    
    @partial(jax.jit, static_argnums=(0,))
    def priorLnProb(self, thetaU):
        return jnp.sum(jnp.array([self.priors[lbl].logpdf(thetaU[lbl]) for lbl in self.labels]))
   
    def lnJointPost(self, theta):
         
        thetaU = self.unpackParams(theta)
        
        lnPrior = self.priorLnProb(thetaU)
    
        lnL = 0  
        
        for kde in self.kdes:
            lnL += jnp.log(kde.pdf(theta)) - lnPrior 
    
        lnP = lnL + lnPrior  
 
        if np.isnan(lnP) or np.isinf(lnP):
            lnP = -jnp.inf
        
        return lnP
 
    def __call__(self,  nwalkers=500, nsteps=500, burnFraction=0.1, accept=0.1, progress=True):
        
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

class DynestyPeakbag(jar.DynestySamplingTools, plotting):
    
    def __init__(self, f, s, ell, freq, height, width, zeta, dnu, d02, freqLimits, rotAsym, addPriors={}, **kwargs):
        
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

        L = -jnp.sum(jnp.log(mod) + self.s[self.sel] / mod)
 
        return L 
        
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

        theta_u = self.unpackParams(theta)
  
        # Constraint from the periodogram 
        mod = self.model(theta_u)

        lnlike = self.chi_sqr(mod)

        lnlike += self.AddLikeTerms(theta, theta_u)
        
        return lnlike

    def unpackParams(self, theta): 
        """ Cast the parameters in a dictionary

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        theta_u : dict
            The unpacked parameters.

        """
        
        theta_u = {'freq'   : theta[0: self.Nmodes],
                   'height' : theta[self.Nmodes: 2 * self.Nmodes],
                   'width'  : theta[2 * self.Nmodes: 3 * self.Nmodes],
                   'nurot_e': theta[self.labels.index('nurot_e')],
                   'nurot_c': theta[self.labels.index('nurot_c')],
                   'inc'    : theta[self.labels.index('inc')],
                   'shot'   : theta[self.labels.index('shot')],
                   }

        theta_u['nurot_e'] = theta_u['nurot_e'] / jnp.sin(theta_u['inc'])
        theta_u['nurot_c'] = theta_u['nurot_c'] / jnp.sin(theta_u['inc'])

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
  
        return theta_u
 
    def model(self, theta_u):
         
        modes = self.zeros + theta_u['shot']
        
        omega = self.zeta * theta_u['nurot_c'] + (1 - self.zeta) * theta_u['nurot_e']
 
        for i, l in enumerate(self.ell):
            
            for m in jnp.arange(-l, l+1):
                
                E = jar.visibility(l, m, theta_u['inc'])
                
                nu = theta_u['freq'][i] + omega[i] * m 

                H = E * theta_u['height'][i]

                modes += jar.lor(self.f[self.sel], nu, H, theta_u['width'][i])
        
        return modes
    
    def __call__(self, dynamic=False, progress=False, sampler_kwargs={}):

        self.runDynesty(dynamic, progress, sampler_kwargs=sampler_kwargs)

        self.result = self.parseSamples(self.samples)
  
        return self.samples, self.result 

    def parseSamples(self, samples, N=10000):
    
        theta_u = self.unpackSamples(samples)
        
        result = {'ell': np.array([self.ell]),
                  'enn': np.array([]),
                  'emm': np.array([]),
                  'zeta': np.array([self.zeta]),
                  'summary': {},
                  'samples': {}
                 }
        
        for key in self.variables:
            arr = np.array([theta_u[_key] for _key in theta_u.keys() if key in _key])
            
            result['summary'][key] = np.array([np.mean(arr, axis=1), np.std(arr, axis=1)]) 
            
            result['samples'][key] = arr[:, :N].T

        return result
 
    def unpackSamples(self, samples):
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

        # l=0 and l=2 can't swap places.
        # TODO this should be changed to something that can't go below 0
        self.addObs['d02'] = dist.normal(loc=self.d02[0], 
                                         scale=10 * self.d02[1])

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
        delta = thetaU['freq'][self.ell==0] - thetaU['freq'][self.ell==2]

        lnp = jnp.sum(self.addObs['d02'].logpdf(delta))
        
        #lnp += self.addObs['heightGP'](theta[self.Nmodes: 2 * self.Nmodes])

        #lnp += self.addObs['widthGP'](theta[2 * self.Nmodes: 3 * self.Nmodes])

        return lnp

    def build_gp(self, theta, X, muFunc, muKwargs={}):

        kernel = theta["amp"] * kernels.ExpSquared(theta["scale"])

        GP = GaussianProcess(kernel, X, diag=1e-6, mean=partial(muFunc, **muKwargs))

        return GP


    