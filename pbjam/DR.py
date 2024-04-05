import numpy as np
import pandas as pd
import jax, warnings
import jax.numpy as jnp
from functools import partial
jax.config.update('jax_enable_x64', True)

class PCA():
    def __init__(self, obs, varLabels, fName, nSamples, selectLabels, weights=None, weightArgs={}, dropNansIn='all'):
        """ Class for handling PCA-based dimensionality reduction

        Parameters
        ----------
        obs : list
            List of observational parameters, e.g., numax, dnu, teff, bp_rp. To 
            be used to find local covariance. Must be in the same units as in 
            the prior sample file.
        varLabels : list
            List of labels to be used for the PCA. Should probably correspond to
            columns in csv file or you won't get very far.
        fName : str
            Full pathname of the csv file containing the prior sample
        nsamples : int
            Number of neightbors to use.
        weights : object, optional
            Array corresponding to N or callable function to get a list of 
            weights to apply to the data sample, by default None
        weightArgs : dict, optional
            Dictionary of arguments if weights is a callable function, empty by 
            default.
        """
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
         
        if self.nSamples > 5000:
            warnings.warn('The requested PCA sample is very large, you may run in to memory issues.')
        
        self.dataF, self.dimsF, self.nSamples = self.getSample(self.fName, self.nSamples)
         
        self.setWeights(self.weights, self.weightArgs)

        self.mu  = jnp.average(self.dataF, axis=0, weights=self.weights)

        self.var = jnp.average((self.dataF - self.mu)**2, axis=0, weights=self.weights)

        self.std = jnp.sqrt(self.var)
 

    def setWeights(self, w, kwargs):
        """
        Set the PCA weights. If None is given then the weights are uniform.

        Parameters
        ----------
        w : np.array
            Array of weights of length equal to number of samples.
        kwargs : dict
            Dictionary of kwargs for the weight function.
        """
         
        if w is None:
            self.weights = jnp.ones(self.nSamples)

        elif callable(w):
            self.weights = w(self, **kwargs)

        else:
            self.weights = w


    def readPriorData(self, fName, labels):
        """
        Read and preprocess prior data from a CSV file.
        
        Parameters
        ----------
        fName : str 
            The file name or path of the CSV file to be read.
        labels : list: 
            A list of column labels to be extracted from the CSV file.
        
        Returns
        -------
        pdata : pandas.DataFrame
            A pandas DataFrame containing the extracted data with preprocessing 
            applied.
        
        Raises
        ------
            FileNotFoundError: If the specified file `fName` does not exist.
            ValueError: If the `labels` parameter is empty or contains invalid column labels.
        
        Notes
        -----
            - The function reads the CSV file specified by `fName` and extracts the columns
            specified by the `labels` parameter.
            - It performs preprocessing on the extracted data, including replacing infinite
            values with NaN, dropping rows with any NaN values, and resetting the DataFrame
            index.
            - The resulting preprocessed DataFrame is returned.
        """

        self.priorData = pd.read_csv(fName, usecols=labels)
         
        self.priorData.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if self.dropNansIn == 'all':
            self.dropNanLabels = self.varLabels + self.selectLabels
        else:
            self.dropNanLabels = self.selectLabels

        self.priorData.dropna(axis=0, how="any", inplace=True, subset=self.dropNanLabels)
         
        self.priorData.reset_index(drop=True, inplace=True)
         
        return self.priorData

    def getSample(self, fName, nSamples):
        """_summary_

        Parameters
        ----------
        fName : str
            File name where prior samples are stored.
        nSamples : int
            Number of samples from the prior to draw around the target in terms 
            of numax.

        Returns
        -------
        pdata : jax device array
            The nSamples drawn from the nearest region around the required 
            point in the prior sample file.
        _ndim : int
            Number of dimensions of the output. Should be the same length as
            varLabels.
        _nSamples : int
            Number of samples in the output. Might be less than the requested
            if the prior sample file is small in comparison.
        """
         
        readlabels = self.varLabels + [key for key in self.selectLabels if key not in self.varLabels]
         
        priorData = self.readPriorData(fName, readlabels)
        
        self.selectedSubset = self.findNearest(priorData, nSamples)
        
        _nSamples, _ndim = self.selectedSubset[self.varLabels].shape
        
        return jnp.array(self.selectedSubset[self.varLabels].to_numpy()), _ndim, _nSamples

    def findNearest(self, fullPriorData, N):
        """ Find nearest neighbours.

        Uses Euclidean distance to find the N nearest neighbors to a set of 
        observational parameters.

        Parameters
        ----------
        pdata : pandas dataframe
            Dataframe of the set of observations to search.
        N : int
            Number of neighbors to find.

        Returns
        -------
        pandas dataframe
            Subset of pdata that contains only the N nearest neightbors.
        """

        limits = {'numax': 0.2,
                  'dnu': 0.2,
                  'teff': 9999,
                  'bp_rp': 9999}
 
        idx = np.prod(np.array([abs(fullPriorData[key].values - self.obs[key][0]) < limits[key] for key in self.selectLabels], dtype=bool), axis=0).astype(bool)
         
        priorData = fullPriorData.loc[idx, :].reset_index(drop=True)
         
        priorSampleMean = np.mean(priorData[self.selectLabels].values, axis=0)

        priorSampleStd = np.std(priorData[self.selectLabels].values, axis=0)
 
        normedObs = {key: (self.obs[key][0] - priorSampleMean[i]) / priorSampleStd[i] for i, key in enumerate(self.selectLabels)}
 
        normedPrior = {key: (priorData[key].values - priorSampleMean[i]) / priorSampleStd[i] for i, key in enumerate(self.selectLabels)}

        delta_i = np.array([normedPrior[key] - normedObs[key] for i, key in enumerate(self.selectLabels)])
            
        euclDist = np.sqrt(np.average(delta_i**2, axis=0))
         
        sortidx = np.argsort(euclDist)
        
        selectedSubset = priorData.loc[sortidx, :][:N]

        self.nanFraction = np.max(np.sum(np.isnan(selectedSubset).values, axis=0)/len(selectedSubset))

        self.viableFraction = 1 - self.nanFraction

        selectedSubset.dropna(axis=0, how="any", inplace=True)
        
        self.badPrior = False
        for i, key in enumerate(self.selectLabels):
            
            S = selectedSubset[key].values
 
            if (min(S) - self.obs[key][0] > 0.1) or (self.obs[key][0]- max(S) > 0.1):
                self.badPrior = True
                warnings.warn(f'Target {key} more than 10 percent beyond limits of the viable prior sample. Prior may not be reliable.', stacklevel=2)
        
        return selectedSubset.reset_index(drop=True)

    @partial(jax.jit, static_argnums=(0,))
    def scale(self, data):
        """
        Scale a sample of data such that it has zero mean and unit standard
        deviation.

        Parameters
        ----------
        data : jax.DeviceArray
            Sample of data

        Returns
        -------
        scaledData : jax.DeviceArray
            The sample of data scaled.
        """

        scaledData = (data - self.mu) / self.std

        return scaledData

    @partial(jax.jit, static_argnums=(0,))
    def inverse_scale(self, scaledData):
        """
        Invert the scaling of the data.

        Parameters
        ----------
        scaledData : jax.DeviceArray
            Scaled sample of data.

        Returns
        -------
        unscaled : jax.DeviceArray
            The sample of data unscaled.

        """

        unscaled = scaledData * self.std + self.mu

        return unscaled

    @partial(jax.jit, static_argnums=(0,))
    def transform(self, X):
        """ Project model space parameters into latent space

        Parameters
        ----------
        X : jax device array
            Sample of model space parameters.

        Returns
        -------
        Y : jax device array
            The coordinates of the model space samples in the latent space.
        """
        _X = self.scale(X)
         
        Y = self.eigvectors[:, self.sortidx].T.dot(_X.T)

        return Y.T.real

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, Y):
        """ Project from latent space into model space

        Parameters
        ----------
        Y : jax device array
            Sample of latent space parameters.

        Returns
        -------
        X : jax device array
            The coordinates of the latent space samples in the model space.
        """

        _X = jnp.dot(Y, self.eigvectors[:, self.sortidx].T)

        return self.inverse_scale(_X).real

    def fit_weightedPCA(self, dim):
        """ Compute PCAs and transform
        
        Wrapper for various functions to compute the covariance matrix of a 
        sample, a few metrics for the PCAs and then project the sample of model
        parameters into the the latent space.

        Parameters
        ----------
        dim : int
            Set the number of dimensions to use. All PCAs are computed but
            only dim are used in the projection into the latent space.
        """

        
        self.dimsR = min([dim, len(self.varLabels)])

        _X = self.scale(self.dataF)
         
        self.covariance = self.covarianceMatrix(_X)
        
        self.eigvals, self.eigvectors = jnp.linalg.eig(self.covariance)

        self.sortidx = sorted(range(len(self.eigvals)), key=lambda i: self.eigvals[i], reverse=True)[:self.dimsR]

        self.explained_variance_ratio = sorted(self.eigvals / jnp.sum(self.eigvals), reverse=True)

        self.erank = jnp.exp(-jnp.sum(self.explained_variance_ratio * np.log(self.explained_variance_ratio))).real

        self.dataR = self.transform(self.dataF)

    def covarianceMatrix(self, _X):
        """ Compute the weighted covariance matrix

        Parameters
        ----------
        _X : jax device array
            Sample of parameters.

        Returns
        -------
        jax device array
            Covariane matrix of the sample.
        """

        W = jnp.diag(self.weights)
        
        C = _X.T@W@_X * jnp.sum(self.weights) / (jnp.sum(self.weights)**2 - jnp.sum(self.weights**2))

        return C
