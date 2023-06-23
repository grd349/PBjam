import numpy as np
import pandas as pd
import jax, warnings
import jax.numpy as jnp
from functools import partial
jax.config.update('jax_enable_x64', True)

class PCA():
    def __init__(self, obs, pcalabels, fname, nsamples, selectlabels, 
                 weights=None, weight_args={}):
        """ Class for handling PCA-based dimensionality reduction

        Parameters
        ----------
        obs : list
            List of observational parameters, e.g., numax, dnu, teff, bp_rp. To 
            be used to find local covariance. Must be in the same units as in 
            the prior sample file.
        pcalabels : list
            List of labels to be used for the PCA. Should probably correspond to
            columns in csv file or you won't get very far.
        fname : str
            Full pathname of the csv file containing the prior sample
        nsamples : int
            Number of neightbors to use.
        weights : object, optional
            Array corresponding to N or callable function to get a list of 
            weights to apply to the data sample, by default None
        weight_args : dict, optional
            Dictionary of arguments if weights is a callable function, empty by 
            default.
        """

        self.pcalabels = pcalabels

        self.selectlabels = selectlabels

        self.obs = obs

        if nsamples > 5000:
            warnings.warn('The requested PCA sample is very large, you may run in to memory issues.')

        self.data_F, self.dims_F, self.nsamples = self.getPCAsample(fname, 
                                                                    nsamples)
         
        self.setWeights(weights, weight_args)

        self.mu  = jnp.average(self.data_F, axis=0, weights=self.weights)

        self.var = jnp.average((self.data_F - self.mu)**2, axis=0, 
                               weights=self.weights)

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
            self.weights = jnp.ones(self.nsamples)

        elif callable(w):
            self.weights = w(self, **kwargs)

        else:
            self.weights = w


    def readPriorData(self, fname, labels):
        """
        Read and preprocess prior data from a CSV file.
        
        Parameters
        ----------
        fname : str 
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
            FileNotFoundError: If the specified file `fname` does not exist.
            ValueError: If the `labels` parameter is empty or contains invalid column labels.
        
        Notes
        -----
            - The function reads the CSV file specified by `fname` and extracts the columns
            specified by the `labels` parameter.
            - It performs preprocessing on the extracted data, including replacing infinite
            values with NaN, dropping rows with any NaN values, and resetting the DataFrame
            index.
            - The resulting preprocessed DataFrame is returned.
        """

        pdata = pd.read_csv(fname, usecols=labels)

        pdata.replace([np.inf, -np.inf], np.nan, inplace=True)
 
        pdata.dropna(axis=0, how="any", inplace=True)

        pdata.reset_index(inplace=True)

        return pdata

    def getPCAsample(self, fname, nsamples):
        """_summary_

        Parameters
        ----------
        fname : str
            File name where prior samples are stored.
        nsamples : int
            Number of samples from the prior to draw around the target in terms 
            of numax.

        Returns
        -------
        pdata : jax device array
            The nsamples drawn from the nearest region around the required 
            point in the prior sample file.
        _ndim : int
            Number of dimensions of the output. Should be the same length as
            pcalabels.
        _nsamples : int
            Number of samples in the output. Might be less than the requested
            if the prior sample file is small in comparison.
        """
         
        readlabels = self.pcalabels + [key for key in self.selectlabels if key not in self.pcalabels]
         
        priorData = self.readPriorData(fname, readlabels)
         
        nearestSubset = self.findNearest(priorData, nsamples)
         
        selectedSubset = nearestSubset[self.pcalabels]
         
        _nsamples, _ndim = selectedSubset.shape
        
        return jnp.array(selectedSubset.to_numpy()), _ndim, _nsamples

    def findNearest(self, pdata, N):
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

        mu = np.mean(pdata[self.selectlabels].values, axis=0)
    
        std = np.std(pdata[self.selectlabels].values, axis=0)
        
        deltas = np.array([(pdata[key].values-mu[i]) / std[i] - (self.obs[key][0] - mu[i]) / std[i] for i, key in enumerate(self.selectlabels)])

        sortidx = np.argsort(np.sqrt(np.sum(deltas**2, axis=0)))
         
        out = pdata.loc[sortidx, :][:N]
        
        return out

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
    def invert_scale(self, scaledData):
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

        return self.invert_scale(_X).real

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

        self.dims_R = dim

        _X = self.scale(self.data_F)
         
        self.covariance = self.covarianceMatrix(_X)
        
        self.eigvals, self.eigvectors = jnp.linalg.eig(self.covariance)

        self.sortidx = sorted(range(len(self.eigvals)), key=lambda i: self.eigvals[i], reverse=True)[:self.dims_R]

        self.explained_variance_ratio = sorted(self.eigvals / jnp.sum(self.eigvals), reverse=True)

        self.erank = jnp.exp(-jnp.sum(self.explained_variance_ratio * np.log(self.explained_variance_ratio))).real

        self.data_R = self.transform(self.data_F)

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
