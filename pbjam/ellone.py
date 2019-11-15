""" Detect l=1 ridges

This is a basic module for detecting the l=1 modes in a spectrum. 

It attempts to find the l=1 ridge by using a clustering function on frequency
bins that are inconsistent with noise at a ~10% probability (H0 test).

It uses the output from the pbjam.peakbag module, where the best-fit models for
the l=2,0 modes is divided out, to form a residual spectrum. This is used in 
the following to detect the l=1 ridge under the assumption that the remaining
significant peaks are most likely caused by l=1 modes (not l=3).

Several levels of binning are applied and the H0 test is performed at each
level. Peaks that persist at many levels of binning will be picked up by 
Hdbscan (unsupervised clustering algorithm), and the average frequencies of the 
clusters are taken to be an estimate of the l=1 frequencies. 

The frequency range between l=0 and l=2, for a given n, is searched for the
cluster with the largest number of samples, and it is assumed that this is
the l=1 ridge. Only 1 l=1 mode is picked up for each radial order, i.e., 
mixed modes are not considered. 

Note
----
It is recommended that pbjam.peakbag be run prior to using this module.
"""

import numpy as np
from scipy.special import gammaincc
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle as skshuffle
import hdbscan as Hdbscan
import warnings
from .plotting import plotting


class ellone(plotting):
    """ Basic l=1 detection 

    Uses either an instance of the pbjam.peakbag module where the l=2,0 modes 
    have been fit, or separate spectrum as inputs to attempt to find the l=1
    modes.
    
    Parameters
    ----------
    pbinst : pbjam.peakbag instance (optional)
        An instance of peakbag where the l=2,0 modes have been fit
    f : ndarray (optional)
        Numpy array of frequency bins of the spectrum (muHz).
    s : ndarray (optional)
        Numpy array of power in each frequency bin (SNR).
    
    Attributes
    ----------
    norders : int
        Number of orders to consider, taken from the peakbag instance
    res : ndarray
        Array of residual power after dividing out the l=2,0 modes
    hdblabels : ndarray
        Array of cluster labelss assigned by HDBscan
    hdbX : ndarray
        Array of input samples to HDBscan
    hdb_clustern : ndarray
        Number of samples retained in each cluster
    
    Note
    ----
    If not pbjam.peakbag instance is supplied, a spectrum must be supplied 
    instead, in which case the l=2,0 modes may be picked up instead of the l=1.
    """
    
    def __init__(self, pbinst=None, f=None, s=None):
        
        if pbinst:
            
            if (f is not None) and (s is not None):
                self.f = f#
                self.s = s#
            else:
                self.f = pbinst.f
                self.s = pbinst.s
                
            self.norders = pbinst.norders
            self.pbinst = pbinst
            self.res = self.residual()
        
        elif (not f is None) and (not s is None):
            self.f = f
            self.s = s
            self.res = s
        else:
            raise AssertionError('Must provide frequency and spectrum input')
        
        self.hdblabels = None
        self.hdbX = None
        self.hdb_clusterN = None
        
    def residual(self,):
        """ Compute the residual after dividing out l=2,0
        
        Uses the best-fit model from a pbjam.peakbag run and divides out the 
        l=2,0 mode power from the supplied spectrum.
        
        Returns
        -------
        res : ndarray
            Array of residual power after dividing out the l=2,0 modes
        """
        
        res = self.s.copy()
        keys = ['l0','l2', 'width0', 'width2', 'height0', 'height2', 'back']
        orders = range(self.norders)
        smry = self.pbinst.summary
        varbs = np.array([[smry.loc[[f'{string}__{i}'], 'mean'].values[0] for i in orders] for string in keys])
        mod = self.pbinst.model(*varbs)
        for i in orders:
            flad = self.pbinst.ladder_f[i,:]
            idx = (flad[0] <= self.f) & (self.f <= flad[-1])
            res[idx] /= mod[i,:]
        return res
     
    def binning(self, nbin):
        """ Simply mean-binning
        
        Bins the spectrum by a factor of nbin
        
        Parameters
        ----------
        nbin : int
            Number of bins to average
        
        Note
        ----
        Assumes the frequency bins are equidistant 
        """
        
        nmax = int(len(self.res)/nbin)*nbin
        sbin = self.res[:nmax].reshape((-1, nbin)).mean(axis=1)
        fbin = self.f[:nmax].reshape((-1, nbin)).mean(axis=1)
        return fbin, sbin
    
    def H0test(self, fbin, sbin, nbin, dnu, reject=0.1): 
        """ Perform H0 test on SNR spectrum
        
        Parameters
        ----------
        fbin : ndarray
            Numpy array of frequency bins of the spectrum (muHz), binned by a
            factor of nbin.
        sbin : ndarray
            Numpy array of power in each frequency bin (SNR), binned by a
            factor of nbin. 
        nbin : int
            Binning factor
        dnu : float
            Large separation of the p-modes (muHz)
        reject: float
            Rejection level to use for the H0 test. (~0.1)
        """
        
        Nind = int(dnu/(2*nbin*np.median(np.diff(fbin))))
        g = gammaincc(nbin, nbin*sbin)
        k = 1-(1-g)**Nind
        idx = k < reject
        return idx, k
    
    
    def H0_inconsistent(self, dnu, Nmax, rejection_level):
        """ Find bins inconsistent with noise
        
        Perform the H0 test at several degrees of binning in freqeuency. 
        Peaks that are inconsistent with noise are retained. 
        
        Parameters
        ----------
        dnu : float
            Large separation of the p-modes (muHz)
        Nmax : int
            Maximum level of binning to try. Default level might require 
            adjusting for lower resolution spectra.
        rejection_level: float
            Probability level to perform H0 test at. Peaks in the spectrum 
            with a probability of being consistent with noise, that is less
            than this are retained.
        
        Returns
        -------
        nu : ndarray
            Frequency of peaks that satisfy the H0 test.
        N : ndarray
            Bin factors at which the significant peaks were selected.
        pH0s : ndarray
            Probabilities of the peaks that are retained
        """
        N = np.array([])
        nu = np.array([])
        pH0s = np.array([])
        
        for nbin in range(1, Nmax+1):
            fbin, sbin = self.binning(nbin) 
            idx, ps = self.H0test(fbin, sbin, nbin, dnu, rejection_level) 
            N = np.append(N, np.zeros(len(fbin[idx]))+nbin)
            nu = np.append(nu, fbin[idx])
            pH0s = np.append(pH0s, ps[idx])
            
        return nu, N, pH0s
    
    def clustering_preprocess(self, nu, N, limits = (0, 100000)):
        """ Preprocess the samples before clustering
        
        Preprocesses the list of frequencies at which significant peaks in the 
        power spectrum were found. The binning factors are shuffled to prevent
        clustering along that axis (axis=1). 
        
        The binning factors are scaled to range between 0 and 1. 
        
        Parameters
        ----------
        nu : ndarray
            Frequency of peaks that satisfy the H0 test.
        N : ndarray
            Bin factors at which the significant peaks were selected.
        limits : list
            Lower and upper limits in nu to use for clustering. Samples
            beyond these limits are rejected.
        
        Returns
        -------
        X : ndarray
            Array of samples to be used by HDBscan        
        """
        
        nuidx = (limits[0] < nu) & (nu < limits[1])
        Nscaler = MinMaxScaler().fit(N.reshape(-1,1))
        Ns = skshuffle(Nscaler.transform(N[nuidx].reshape(-1,1))).flatten()
        return np.vstack((nu[nuidx], Ns)).T
    
    def span(self, x):
        """ Compute span of array
        
        Parameters
        ----------
        x : ndarray
            List of floats/integers
        
        Returns
        -------
        span : float
            Range spanned by the minimum and maximum values in x
        """
        return max(x)-min(x)
    
    def clustering(self, nu, N, Nmax, outlier_limit=0.5, cluster_prob=0.9):
        """ Perform HDBscan clustering
        
        Uses HDBscan to perform an unsupervised clustering analysis of the 
        frequencies of the H0 significance test. 
        
        Samples in the clusters are retained based on their probability of 
        being part of an individual cluster, and the odds of them being an 
        outlier of all the samples in the list.
        
        Parameters
        ----------
        nu : ndarray
            Frequency of peaks that satisfy the H0 test.
        N : ndarray
            Bin factors at which the significant peaks were selected.
        Nmax: int
            Maximum binning factor used in the H0 test
        Wcut : float
            Unused. Level at which to cut broad clusters (those unlikely to 
            be due to a single mode.)
        outlier_limiter : float
            Probability that a sample is an outlier of a cluster. Samples with
            a probability above this are rejected.
        cluster_prob : float
            Probability that samples belong to a particular cluster. Samples
            with a probability below this are rejected.
            
        Returns
        -------
        nus : ndarray
            Array of mean frequencies of the clusters identified by HDBscan
        Wratios : ndarray
            Ratio of the span of the cluster and the average mode width of the
            p-modes. (unused)
        """
        
        X = self.clustering_preprocess(nu, N)
        hdbscan = Hdbscan.HDBSCAN(min_cluster_size = 2*Nmax).fit(X)
        
        labels = hdbscan.labels_
        ulabels = np.unique(labels)
        
        nus = np.zeros(len(ulabels))
        
        for i, u in enumerate(ulabels):
            hdbidx = (labels==u) & (hdbscan.outlier_scores_<outlier_limit) & (hdbscan.probabilities_>cluster_prob) 
            if (len(X[hdbidx,0])==0) or (u == -1):
                continue
            nus[i] = np.mean(X[hdbidx,0])
            #Wratios[i] = self.span(X[hdbidx,0])/W
    
        self.hdblabels = labels
        self.hdbX = X
        self.hdb_clusterN = np.array([len(labels[labels==i]) for i in ulabels])
        
        return nus[1:]
    

    
    def get_ell1(self, dnu):
        """ Estimate frequency of l=1 modes (p-modes)
        
        Takes the best-fit l=2,0 mode frequencies from pbjam.peakbag and 
        searches the frequency range in between subsequent radial orders for
        clusters of significant peaks. The cluster with the largest number of 
        peaks is assumed to be the most likely l=1 location. 
        
        Parameters
        ----------
        dnu : float
            Large separation of the p-modes (muHz)
            
        Returns
        -------
        nul1s : ndarray
            Array of estimated frequencies of the l=1 modes. 
            
        Note
        ----
        Raises a warning if the detected frequency is far from the expected 
        frequency based on the Universal Pattern. 
        
        """
        pbsmry = self.pbinst.summary
        N = self.norders
        ell02 = ['l0', 'l2']
        nul0s, nul2s = np.array([[pbsmry.loc[f'{x}__{i}', 'mean'] for i in range(N)] for x in ell02])

        d01 = (1./2 -0.0056 -0.002*np.log10(dnu))*dnu
        
        nul1s = np.zeros(len(nul0s))
        for i in range(len(nul0s)):
            nuidx = (nul0s[i] < self.cluster_means) & (self.cluster_means < nul2s[i]+dnu)
            
            if len(self.hdb_clusterN[1:][nuidx]) == 0:
                continue
            
            maxidx = np.argmax(self.hdb_clusterN[1:][nuidx])
            nul1s[i] = self.cluster_means[nuidx][maxidx]
        
            if (nul0s[i] - nul1s[i])/d01 > 0.2:
                warnings.warn('Cluster nu_l1 exceeds UP estimate by more than 20%')
                
        return nul1s

    def __call__(self, dnu, Nmax = 30, rejection_level = 0.1):
        """ Perform all the steps to estimate l=1 frequencies
        
        Check which peaks in the provided spectrum are inconsistent with the
        noise level, at various degrees of binning.
        
        Peaks that are persistent over many binning levels are clustered and 
        the mean frequencies are assumed to be likely locations of the l=1.
        modes.
        
        Parameters
        ----------
        dnu : float
            Large separation of the p-modes (muHz)
        Nmax : int (optional)
            Maximum level of binning to try. Default level might require 
            adjusting for lower resolution spectra.
        rejection_level: float (optional)
            Probability level to perform H0 test at. Peaks in the spectrum 
            with a probability of being consistent with noise, that is less
            than this are retained.
        
        Returns
        -------
        nul1s : ndarray
            Array of estimated frequencies of the l=1 modes. 
        
        """
        nus, counts, pH0s = self.H0_inconsistent(dnu, Nmax, rejection_level)
        self.cluster_means, _ = self.clustering(nus, counts, Nmax)
        nul1s = self.get_ell1(dnu)
        return nul1s
    
         
# d02 = 10**st.asy_fit.summary.loc['d02', '50th']