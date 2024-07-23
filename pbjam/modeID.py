import warnings, os
import jax.numpy as jnp
import numpy as np
from pbjam.l1models import Asyl1model, Mixl1model, RGBl1model
from pbjam.l20models import Asyl20model
from pbjam.plotting import plotting
from pbjam import jar
import pandas as pd

class modeID(plotting, ):
    
    """
    Class for identifying modes in solar-like oscillators.

    Parameters
    ----------
    f : array-like
        The frequency array of the spectrum.
    s : array-like
        The values of the power density spectrum.
    obs : dict
        Dictionary of observational inputs.
    addPriors : dict, optional
        Additional priors to be added. Default is an empty dictionary.
    N_p : int, optional
        Number of radial orders to use for mode identification. Default is 7.
    freqLimits : list, optional
        Frequency limits for mode identification. If None, it is calculated based on 'numax' and 'dnu'.
    priorpath : str, optional
        Path to prior sample csv. If None, a default path is used.

    """

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freqLimits=None, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        # Set frequency range to compute the model on. Default is one radial order above/below the requested number.
        if self.freqLimits is None:
            self.freqLimits = [self.obs['numax'][0] - self.obs['dnu'][0]*(self.N_p//2+1), 
                               self.obs['numax'][0] + self.obs['dnu'][0]*(self.N_p//2+1),]
            
        self.sel = (np.array(self.freqLimits).min() < self.f) & (self.f < np.array(self.freqLimits).max())   

        if priorpath is None:
            self.priorpath = jar.getPriorpath()
 

    def runl20model(self, progress=True, sampler_kwargs={}, logl_kwargs={}, PCAsamples=50, PCAdims=6):
        """
        Runs the l20 model on the selected spectrum.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during the model run. Default is True.
        sampler_kwargs : dict, optional
            Additional keyword arguments for the sampler. Default is an empty dictionary.
        logl_kwargs : dict, optional
            Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        PCAsamples : int, optional
            Number of samples for PCA. Default is 50.
        PCAdims : int, optional
            Number of dimensions for PCA. Default is 6.

        Returns
        -------
        result : dict
            Parsed results from the l20 model.
        """

        f = self.f[self.sel]

        s = self.s[self.sel]

        self.l20model = Asyl20model(f, s, 
                                    self.obs, 
                                    self.addPriors, 
                                    self.N_p, 
                                    PCAsamples, 
                                    PCAdims,
                                    priorpath=self.priorpath)
        
        self.l20Samples, self.l20logz = self.l20model.runDynesty(progress=progress, 
                                                                 logl_kwargs=logl_kwargs, 
                                                                 sampler_kwargs=sampler_kwargs)

        l20samples_u = self.l20model.unpackSamples(self.l20Samples)

        self.l20result = self.l20model.parseSamples(l20samples_u)

        self.result = self.mergeResults(l20result=self.l20result)
 
        return self.l20result

    def runl1model(self, progress=True, sampler_kwargs={}, logl_kwargs={}, model='MS', PCAsamples=100, PCAdims=5):
        """
        Runs the l1 model on the selected spectrum.

        Should follow the l20 model run.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during the model run. Default is True.
        sampler_kwargs : dict, optional
            Additional keyword arguments for the sampler. Default is an empty dictionary.
        logl_kwargs : dict, optional
            Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        model : str
            Choice of which model to use for estimating the l=1 mode locations. Choices are MS, SG, RGB models.
        PCAsamples : int, optional
            Number of samples for PCA. Default is 100.
        PCAdims : int, optional
            Number of dimensions for PCA. Default is 5.

        Returns
        -------
        result : dict
            Parsed results from the l1 model.
        """

        # Compute the l=2,0 model residual. 
        self.l20residual = self.s[self.sel] / self.l20model.getMedianModel()
      
        f = self.f[self.sel]

        s = self.l20residual

        summary = {'n_p': self.l20result['enn'][self.l20result['ell']==0],
                   'nu0_p': self.l20result['summary']['freq'][0, self.l20result['ell']==0]}
 
        for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
            summary[key] = self.l20result['summary'][key]

        if model.lower() == 'ms':
            self.l1model = Asyl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      priorpath=self.priorpath)

        elif model.lower() == 'sg':    
            self.l1model = Mixl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      PCAdims,
                                      priorpath=self.priorpath)

        elif model.lower() == 'rgb':
            self.l1model = RGBl1model(f, s,  
                                      summary, 
                                      self.addPriors, 
                                      PCAsamples,
                                      rootiter=15,
                                      priorpath=self.priorpath,
                                      modelChoice='simple')
        else:
            raise ValueError(f'Model {model} is invalid. Please use either MS, SG or RGB.')

        self.l1Samples, self.l1logz  = self.l1model.runDynesty(progress=progress, 
                                                               logl_kwargs=logl_kwargs, 
                                                               sampler_kwargs=sampler_kwargs)
        
        l1SamplesU = self.l1model.unpackSamples(self.l1Samples)

        self.l1result = self.l1model.parseSamples(l1SamplesU)

        self.result = self.mergeResults(l20result=self.l20result, l1result=self.l1result)

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
 
    def mergeResults(self, l20result=None, l1result=None, N=5000):
    
        R = {'ell': np.array([]),
             'enn': np.array([]),
             'emm': np.array([]),
             'zeta': np.array([]),
             'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                         'height': np.array([]).reshape((2, 0)), 
                         'width' : np.array([]).reshape((2, 0)),
                         'rotAsym': np.array([]).reshape((2, 0)),
                        },
             'samples': {'freq'  : np.array([]).reshape((N, 0)),
                         'height': np.array([]).reshape((N, 0)), 
                         'width' : np.array([]).reshape((N, 0)),
                         'rotAsym' : np.array([]).reshape((N, 0))
                        },
            }
        
        if l20result is None and hasattr(self, 'l20result'):
            l20result = self.l20result

        _N = np.append(N, l20result['samples']['freq'].shape[0])
         
        if l1result is None and hasattr(self, 'l1model'):
            l1result = self.l1result
        
        if l1result is not None:
            _N = np.append(_N, l1result['samples']['freq'].shape[0])
 
        N = np.min(_N)
         
        resList = [l1result, l20result] # This order overrides the numax, dnu and teff from l1result in the output since the l20result is more reliable. 

        for rootkey in ['ell', 'enn', 'emm', 'zeta']:
            for D in resList:
                if D is not None:
                    R[rootkey] = np.append(R[rootkey], D[rootkey])
        
        for rootkey in ['summary', 'samples']:
            for D in resList: 
                if D is not None:
                    for subkey in list(D[rootkey].keys()):
                        if subkey not in ['freq', 'height', 'width', 'rotAsym']:
                            R[rootkey][subkey] = D[rootkey][subkey]

                    for subkey in ['freq', 'height', 'width', 'rotAsym']:
                        R[rootkey][subkey] = np.hstack((R[rootkey][subkey][:N, :], 
                                                        D[rootkey][subkey][:N, :]))

                else:
                    continue

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
 
  
