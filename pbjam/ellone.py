import numpy as np
from scipy.special import gammaincc
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle as skshuffle
import hdbscan as Hdbscan
import warnings
from .plotting import plotting


class ellone(plotting):
    
    def __init__(self, pbinst=None, f=None, s=None):
        
        if pbinst:
            
            if (not f is None) and (not s is None):
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
            
        
    def residual(self,):
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
        nmax = int(len(self.res)/nbin)*nbin
        sbin = self.res[:nmax].reshape((-1, nbin)).mean(axis=1)
        fbin = self.f[:nmax].reshape((-1, nbin)).mean(axis=1)
        return fbin, sbin
    
    def H0test(self, fbin, sbin, dnu, nbin, reject=0.02):       
        Nind = int(dnu/(2*nbin*np.median(np.diff(fbin))))
        g = gammaincc(nbin, nbin*sbin)
        idx = 1-(1-g)**Nind < reject
        self.H0passed = idx
        return idx, g
    
    def clustering_preprocess(self, nu, N, limits = (0, 100000)):
        nuidx = (limits[0] < nu) & (nu < limits[1])
        Nscaler = MinMaxScaler().fit(N.reshape(-1,1))
        Ns = skshuffle(Nscaler.transform(N[nuidx].reshape(-1,1))).flatten()
        return np.vstack((nu[nuidx], Ns)).T
    
    def span(self, x):
        return max(x)-min(x)
    
    def clustering(self, nu, N, Nmax, W, Wcut = 2, outlier_limit=0.5, cluster_prob = 0.9):
        X = self.clustering_preprocess(nu, N)
        hdbscan = Hdbscan.HDBSCAN(min_cluster_size = 2*Nmax).fit(X)
        
        labels = hdbscan.labels_
        ulabels = np.unique(labels)
        
        nus = np.zeros(len(ulabels))
        Wratios = np.zeros(len(ulabels))
        for i, u in enumerate(ulabels):
            hdbidx = (labels==u) & (hdbscan.outlier_scores_<outlier_limit) & (hdbscan.probabilities_>cluster_prob) 
            if (len(X[hdbidx,0])==0) or (u == -1):
                continue
            nus[i] = np.mean(X[hdbidx,0])
            Wratios[i] = self.span(X[hdbidx,0])/W
    
        self.hdblabels = labels
        self.hdbX = X
        self.hdb_clusterN = np.array([len(labels[labels==i]) for i in ulabels])
        
        return nus[1:], Wratios[1:]
    
    def H0_inconsistent(self, dnu, Nmax, rejection_level):
        N = np.array([])
        nu = np.array([])
        pH0s = np.array([])
        
        for nbin in range(1, Nmax):
            fbin, sbin = self.binning(nbin) 
            idx, ps = self.H0test(fbin, sbin, dnu, nbin, rejection_level) 
            N = np.append(N, np.zeros(len(fbin[idx]))+nbin)
            nu = np.append(nu, fbin[idx])
            pH0s = np.append(pH0s, ps[idx])
            
        return nu, N, pH0s
    
    def get_ell1(self, dnu):
        pbsmry = self.pbinst.summary
        N = self.norders
        ell02 = ['l0', 'l2']
        nul0s, nul2s = np.array([[pbsmry.loc[f'{x}__{i}', 'mean'] for i in range(N)] for x in ell02])

        d01 = (1./2 -0.0056 -0.002*np.log10(dnu))*dnu
        
        nul1s = np.zeros(len(nul0s))
        for i in range(len(nul0s)):
            nuidx = (nul0s[i] < self.cluster_means) & (self.cluster_means < nul2s[i]+dnu)
            maxidx = np.argmax(self.hdb_clusterN[1:][nuidx])
            nul1s[i] = self.cluster_means[nuidx][maxidx]
        
            if (nul0s[i] - nul1s[i])/d01 > 0.2:
                warnings.warn('Cluster nu_l1 exceeds UP estimate by more than 10%')
                
        return nul1s

    def __call__(self, W, dnu, Nmax = 30, rejection_level = 0.09):
        nus, counts = self.H0_inconsistent(dnu, Nmax, rejection_level)
        self.cluster_means, _ = self.clustering(nus, counts, Nmax, W)
        nul1s = self.get_ell1(dnu)
        return nul1s
    
         
# d02 = 10**st.asy_fit.summary.loc['d02', '50th']