import warnings, os
import jax.numpy as jnp
import numpy as np
from pbjam import jar
from pbjam.l1models import Asyl1Model
from pbjam.l1models import Mixl1Model
from pbjam.pairmodel import Asyl20Model
from pbjam.plotting import plotting
import pandas as pd

class modeIDsampler(plotting, ):

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freqLimits=[1, 5000], 
                 vis={'V20': 0.71, 'V10': 1.22}, Npca=50, PCAdims=8, 
                 priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        if np.isscalar(self.Npca):
            self.Npca_pair = self.Npca
            self.Npca_mix = self.Npca
        elif np.size(self.Npca) == 2:
            self.Npca_pair = self.Npca[0]
            self.Npca_mix = self.Npca[1]
        else:
            raise ValueError('Npca is wrong')
        
        if np.isscalar(self.PCAdims):
            self.PCAdims_pair = self.PCAdims
            self.PCAdims_mix = self.PCAdims
        elif np.size(self.PCAdims) == 2:
            self.PCAdims_pair = self.PCAdims[0]
            self.PCAdims_mix = self.PCAdims[1]
        else:
            raise ValueError('PCAdims is wrong')

    def runl20Model(self, progress, logl_kwargs, sampler_kwargs):

            self.l20sel = (np.array(self.freqLimits).min() < self.f) & (self.f < np.array(self.freqLimits).max())   # self.setFreqRange(*self.freqLimits)

            self.Asyl20Model = Asyl20Model(self.f[self.l20sel], 
                                             self.s[self.l20sel], 
                                             self.obs, 
                                             self.addPriors, 
                                             self.N_p, 
                                             self.Npca_pair, 
                                             self.PCAdims_pair,
                                             priorpath=self.priorpath)
            
            self.Asyl20Samples, self.Asyl20logz = self.Asyl20Model.runDynesty(progress=progress, logl_kwargs=logl_kwargs, 
                                                                             sampler_kwargs=sampler_kwargs)
            
            l20samples_u = self.Asyl20Model.unpackSamples(self.Asyl20Samples)

            self.l20res = self.Asyl20Model.parseSamples(l20samples_u)
        

            asymptotic_samps = np.array([self.Asyl20Model.asymptotic_nu_p(l20samples_u['numax'][i], 
                                                                           l20samples_u['dnu'][i], 
                                                                           l20samples_u['eps_p'][i], 
                                                                           l20samples_u['alpha_p'][i]) for i in range(50)])
            self.summary = {}

            self.summary['n_p'] = np.median(asymptotic_samps[:, 1, :], axis=0).astype(int)

            self.summary['nu0_p'] = np.median(asymptotic_samps[:, 0, :], axis=0)

            for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
                self.summary[key] = jar.smryStats(l20samples_u[key])

            l20model = self.Asyl20Model.getMedianModel(l20samples_u)

            self.l20residual = self.s[self.l20sel]/l20model

            return self.l20res

    def runl1Model(self, progress, logl_kwargs, sampler_kwargs, freqLimits=None):

            if freqLimits is None:
                                
                 freqLimits = [self.obs['numax'][0] - self.obs['dnu'][0]*(self.Asyl20Model.N_p//2+1), 
                               self.obs['numax'][0] + self.obs['dnu'][0]*(self.Asyl20Model.N_p//2+1),]

            self.l1sel = (np.array(freqLimits).min() < self.f[self.l20sel]) & (self.f[self.l20sel] < np.array(freqLimits).max())
            
            print('Initializing asymptotic l=1 model')
            self.Asyl1Model = Asyl1Model(self.f[self.l20sel][self.l1sel], 
                                         self.l20residual[self.l1sel], 
                                         self.summary, 
                                         {},
                                         self.N_p, 
                                         self.Npca_pair, 
                                         priorpath=self.priorpath)
            print('Viable prior sample fraction of requested:', np.round(1-self.Asyl1Model.DR.nanFraction, 2))
            print('Initializing mixed mode l=1 model')
            self.Mixl1Model = Mixl1Model(self.f[self.l20sel][self.l1sel], 
                                         self.l20residual[self.l1sel], 
                                         self.summary, 
                                         self.addPriors,
                                         self.N_p, 
                                         self.Npca_mix, 
                                         self.PCAdims_mix,
                                         priorpath=self.priorpath)
            print('Viable prior sample fraction of requested:', np.round(1-self.Mixl1Model.DR.nanFraction, 2))

            
            if (self.Asyl1Model.DR.nanFraction < 0.05) & (0.95 < self.Mixl1Model.DR.nanFraction):#( >= 0.05 ) & (self.Asyl1Model.DR.nanFraction < 0.05):
                print('Not enough prior samples for mixed model. Using asymptotic.')
                
                self.Asyl1Samples, self.Asyl1logz  = self.Asyl1Model.runDynesty(progress=progress, 
                                                                               logl_kwargs=logl_kwargs, 
                                                                               sampler_kwargs=sampler_kwargs)
                self.useMixResult = False
                
            elif 0.05 < self.Mixl1Model.DR.nanFraction <= 0.95:
                print('Testing both models.')
                
                self.Asyl1Samples, self.Asyl1logz  = self.Asyl1Model.runDynesty(progress=progress, 
                                                                               logl_kwargs=logl_kwargs, 
                                                                               sampler_kwargs=sampler_kwargs)
                
                self.Mixl1Samples, self.Mixl1logz  = self.Mixl1Model.runDynesty(progress=progress, 
                                                                               logl_kwargs=logl_kwargs, 
                                                                               sampler_kwargs=sampler_kwargs)
                
                # evidence check
                self.AsyMixBayesFactor = self.Asyl1logz.max() - self.Mixl1logz.max()

                self.useMixResult = self.AsyMixBayesFactor  < 1/2                
                 
            elif self.Mixl1Model.DR.nanFraction <= 0.05:
                print('Using the mixed model.')
                
                self.Mixl1Samples, self.Mixl1logz  = self.Mixl1Model.runDynesty(progress=progress, 
                                                                               logl_kwargs=logl_kwargs, 
                                                                               sampler_kwargs=sampler_kwargs)
                
                self.useMixResult == True

            else:
                print('Mixed model nan-fraction:', self.Mixl1Model.DR.nanFraction)
                print('Asy model nan-fraction:', self.Asyl1Model.DR.nanFraction)
                raise ValueError('Somethings gone wrong when picking the asy/mix model')

            if self.useMixResult:
                l1samples_u = self.Mixl1Model.unpackSamples(self.Mixl1Samples)

                self.l1res = self.Mixl1Model.parseSamples(l1samples_u)
            else:
                l1samples_u = self.Asyl1Model.unpackSamples(self.Asyl1Samples)

                self.l1res = self.Asyl1Model.parseSamples(l1samples_u)

            return self.l1res

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
        
        self.runl20Model(progress, sampler_kwargs, logl_kwargs)
        
        self.runl1Model(progress, sampler_kwargs, logl_kwargs)

        self.result = self.mergeResults(self.l20res, self.l1res)


    def mergeResults(self, l20res, l1res):
    
        R = {'summary': {},
             'samples': {}}

        N = min([l20res['samples']['freq'].shape[0], l1res['samples']['freq'].shape[0]])

        for key in ['ell', 'enn', 'zeta']:
            R[key] = np.append(l20res[key], l1res[key])
        
        for rootkey in ['summary', 'samples']:
            for D in [l20res, l1res]:
                for subkey in list(D[rootkey].keys()):
                    if subkey not in ['freq', 'height', 'width']:
                        R[rootkey][subkey] = D[rootkey][subkey]

            for subkey in ['freq', 'height', 'width']:

                

                R[rootkey][subkey] = np.hstack((l20res[rootkey][subkey][:N, :], l1res[rootkey][subkey][:N, :]))
        
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
 
    
