import warnings, os
import jax.numpy as jnp
import numpy as np
from pbjam.l1models import Asyl1model, Mixl1model, RGBl1model
from pbjam.pairmodel import Asyl20model
from pbjam.plotting import plotting
import pandas as pd

class modeIDsampler(plotting, ):

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freqLimits=None, 
                 vis={'V20': 0.71, 'V10': 1.22}, Npca=50, PCAdims=8, 
                 priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        if self.freqLimits is None:
            self.freqLimits = [self.obs['numax'][0] - self.obs['dnu'][0]*(self.N_p//2+1), 
                               self.obs['numax'][0] + self.obs['dnu'][0]*(self.N_p//2+1),]

        if np.isscalar(self.Npca):
            self.Npca_asy = self.Npca
            self.Npca_mix = self.Npca
        elif np.size(self.Npca) == 2:
            self.Npca_asy = self.Npca[0]
            self.Npca_mix = self.Npca[1]
        else:
            raise ValueError('Npca is wrong')
        
        if np.isscalar(self.PCAdims):
            self.PCAdims_asy = self.PCAdims
            self.PCAdims_mix = self.PCAdims
        elif np.size(self.PCAdims) == 2:
            self.PCAdims_asy = self.PCAdims[0]
            self.PCAdims_mix = self.PCAdims[1]
        else:
            raise ValueError('PCAdims is wrong')

    def runl20model(self, progress, sampler_kwargs, logl_kwargs, freqLimits=None):

        if freqLimits is None:
            freqLimits = self.freqLimits
        else:
            self.freqLimits = freqLimits
        
        self.l20sel = (np.array(freqLimits).min() < self.f) & (self.f < np.array(freqLimits).max())    

        f = self.f[self.l20sel]
        s = self.s[self.l20sel]
        self.l20model = Asyl20model(f, s, 
                                    self.obs, 
                                    self.addPriors, 
                                    self.N_p, 
                                    self.Npca_asy, 
                                    self.PCAdims_asy,
                                    priorpath=self.priorpath)
        
        self.l20Samples, self.l20logz = self.l20model.runDynesty(progress=progress, logl_kwargs=logl_kwargs, 
                                                                            sampler_kwargs=sampler_kwargs)

        l20samples_u = self.l20model.unpackSamples(self.l20Samples)

        self.l20result = self.l20model.parseSamples(l20samples_u)

        self.summary = {}

        self.summary['n_p'] = self.l20result['enn'][self.l20result['ell']==0]

        self.summary['nu0_p'] = self.l20result['summary']['freq'][0, self.l20result['ell']==0]

        for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
            self.summary[key] = self.l20result['summary'][key]

        l20model = self.l20model.getMedianModel(l20samples_u)

        self.l20residual = self.s[self.l20sel]/l20model

        return self.l20result

    def runl1model(self, progress, sampler_kwargs, logl_kwargs, model='asy', freqLimits=None, BayesFactorLimit=1/2):

        if freqLimits is None:
            freqLimits = self.freqLimits
        else:
            if freqLimits.min() < self.freqLimits.min():
                warnings.warn('Provided lower frequency limit is less than that of the l20 model. Results may be wonky.')
            elif freqLimits.max() > self.freqLimits.max():
                warnings.warn('Provided upper frequency limit is greater than that of the l20 model. Results may be wonky.')
            
        self.l1sel = (np.array(freqLimits).min() < self.f[self.l20sel]) & (self.f[self.l20sel] < np.array(freqLimits).max())
        
        f = self.f[self.l20sel][self.l1sel]

        s = self.l20residual[self.l1sel]

        if model.lower() == 'asy':
            self.l1model = Asyl1model(f, s, 
                                        self.summary, 
                                        self.addPriors,
                                        self.Npca_asy, 
                                        priorpath=self.priorpath)

        elif model.lower() == 'mix':    
            self.l1model = Mixl1model(f, s, 
                                        self.summary, 
                                        self.addPriors,
                                        self.Npca_mix, 
                                        self.PCAdims_mix,
                                        priorpath=self.priorpath)

        elif model.lower() == 'rgb':
            self.l1model = RGBl1model(f, s,  
                                        self.summary, 
                                        self.addPriors, 
                                        NPriorSamples=50,
                                        rootiter=15,
                                        priorpath=self.priorpath,
                                        modelChoice='complicated')
        else:
            raise ValueError(f'Model {model} is invalid. Please use either Asy, Mix or RGB.')

        self.l1samples, self.l1logz  = self.l1model.runDynesty(progress=progress, 
                                                                logl_kwargs=logl_kwargs, 
                                                                sampler_kwargs=sampler_kwargs)
        
        l1samples_u = self.l1model.unpackSamples(self.l1samples)

        self.l1result = self.l1model.parseSamples(l1samples_u)

        return self.l1result

    def __call__(self, progress=True, sampler_kwargs={}, logl_kwargs={}):
        """
        Run the Dynesty sampler.

        Parameters
        ----------
        dynesty_kwargs : dict, optional
            Additional keyword arguments for the Dynesty sampler. 

        Returns
        -------
        samples : jax device array
            The generated samples
        result : dict
            Dictionary of parsed result. Contains summary statistics and samples
            in human-readable form.

        Notes
        -----
        - Calls the `runDynesty` method with provided keyword arguments.
        - Unpacks and parses the generated samples.
        - Stores the parsed result and returns both the samples and the result.
        """
        
        self.runl20model(progress, sampler_kwargs, logl_kwargs)
        
        self.runl1model(progress, sampler_kwargs, logl_kwargs)

        self.result = self.mergeResults(self.l20result, self.l1result)


    def mergeResults(self, l20result, l1result):
    
        R = {'summary': {},
             'samples': {}}

        N = min([l20result['samples']['freq'].shape[0], l1result['samples']['freq'].shape[0]])

        for key in ['ell', 'enn', 'zeta']:
            R[key] = np.append(l20result[key], l1result[key])
        
        for rootkey in ['summary', 'samples']:
            for D in [l1result, l20result]: # Note this overrides the numax, dnu and teff estimates from l1result since they are included in l20result.
                for subkey in list(D[rootkey].keys()):
                    if subkey not in ['freq', 'height', 'width']:
                        R[rootkey][subkey] = D[rootkey][subkey]

            for subkey in ['freq', 'height', 'width']:
 
                R[rootkey][subkey] = np.hstack((l20result[rootkey][subkey][:N, :], l1result[rootkey][subkey][:N, :]))
        
        return R

    def storeResult(self, resultDict, ID=None):

        # TODO ID should come from star?
        if ID is not None:
            _ID = ID
        elif (ID is None) and hasattr(self, 'ID'):
            _ID = self.ID
        else:   
            _ID = f'unknown_tgt_{np.random.randint(0, 1e10)}'

            warnings.warn(f'Output stored under {_ID}. You should probably specify a target ID.')
        
        path = _ID
        
        basefilename = os.path.join(*[path, f'asymptotic_fit_summary_{_ID}'])
        
        if not os.path.exists(path):
            os.makedirs(path)

        # store everything
        np.savez(basefilename+'.npz', resultDict)

        # grab just model parameters and save
        _tmp = {key: self.result['summary'][key] for key in self.result['summary'].keys() if key not in ['freq', 'height', 'width']}

        df_data = [{'name': key, 'mean': value[0], 'error': value[1]} for key, value in _tmp.items()]

        # Create a DataFrame
        df = pd.DataFrame(df_data)
        
        df.to_csv(basefilename+'.csv', index=False)
 
    
# if (self.Mixl1model.DR.viableFraction <= 0.1) or self.Mixl1model.badPrior: 
            #     print('Not enough prior samples for mixed model. Using the asymptotic model.')
                
            #     self.Asyl1Samples, self.Asyl1logz  = self.Asyl1model.runDynesty(progress=progress, 
            #                                                                    logl_kwargs=logl_kwargs, 
            #                                                                    sampler_kwargs=sampler_kwargs)
            #     self.useMixResult = False
                
            # elif ((0.1 < self.Mixl1model.DR.viableFraction) and (self.Mixl1model.DR.viableFraction <= 0.90)) and not self.Mixl1model.badPrior:
            #     print('Testing both asymptotic and mixed mode models.')
                
            #     if not 'nlive' in sampler_kwargs.keys():
            #         sampler_kwargs['nlive'] = 200*self.Mixl1model.ndims

            #     self.Asyl1Samples, self.Asyl1logz  = self.Asyl1model.runDynesty(progress=progress, 
            #                                                                    logl_kwargs=logl_kwargs, 
            #                                                                    sampler_kwargs=sampler_kwargs)
                
            #     self.Mixl1Samples, self.Mixl1logz  = self.Mixl1model.runDynesty(progress=progress, 
            #                                                                    logl_kwargs=logl_kwargs, 
            #                                                                    sampler_kwargs=sampler_kwargs)
                
            #     # evidence check
            #     self.AsyMixBayesFactor = self.Asyl1logz.max() - self.Mixl1logz.max()

            #     self.useMixResult = self.AsyMixBayesFactor < BayesFactorLimit              
                 
            # elif (0.9 <= self.Mixl1model.DR.viableFraction) and not self.Mixl1model.badPrior:
            #     print('Using the mixed mode model.')

            #     if not 'nlive' in sampler_kwargs.keys():
            #         sampler_kwargs['nlive'] = 100*self.Mixl1model.ndims
                
            #     self.Mixl1Samples, self.Mixl1logz  = self.Mixl1model.runDynesty(progress=progress, 
            #                                                                    logl_kwargs=logl_kwargs, 
            #                                                                    sampler_kwargs=sampler_kwargs)
                
            #     self.useMixResult = True

            # else:
            #     print('Mixed model nan-fraction:', self.Mixl1model.DR.nanFraction)
            #     print('Asy model nan-fraction:', self.Asyl1model.DR.nanFraction)
            #     raise ValueError('Somethings gone wrong when picking the asy/mix model')

            # if self.useMixResult:
            #     l1samples_u = self.Mixl1model.unpackSamples(self.Mixl1Samples)

            #     self.l1result = self.Mixl1model.parseSamples(l1samples_u)
            # else:
