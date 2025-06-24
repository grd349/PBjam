import scipy.stats as st
import pbjam.distributions as dist
import numpy as np
import scipy.spatial.distance as ssd
import statsmodels.api as sm


class validate():

    def __init__(self, priors, postSamples):
        """
        Parameters
        ----------
        priors : dict
            Dictionary of prior class instances. Must have the pdf, logpdf, cdf and ppf methods.
            Keywords must correspond to the variables to be compared.
        posteriorSamples : dict
            Dictionary of posterior samples. Keywords must correspond to the variables to be compared.
        """

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.ndim = len(self.priors.keys())

    def KStest(self, threshold=0.05):
        """ Method for running KS test on posteriorSamples vs priors

        Returns
        -------
        testResult : dict
        """

        testResult = {'statistic': np.zeros(self.ndim), 'pvalue': np.zeros(self.ndim), 'significant': np.zeros(self.ndim, dtype=bool)}

        # Loop through list of priors can compute the KS statistic for each sample
        for i, key in enumerate(self.priors.keys()):
                        
            kstestResult = st.kstest(self.postSamples[key], 
                                     self.priors[key].cdf)
            
            testResult['pvalue'][i] = kstestResult.pvalue
            
            testResult['significant'][i] = kstestResult.pvalue <= threshold
            
            testResult['statistic'][i] = kstestResult.statistic
         
        return testResult
    
    def JStest(self, threshold=0.05, N=10000):
        """ Method for running JS test on posteriorSamples vs priors

        Computes the Jensen-Shannon (JS) distance between the probability distribution
        P(x) and Q(x), where x is the posterior sample, P is a KDE of the posterior sample
        and Q is the prior probability density. 

        Computes the p-value for the JS distance by generating a null-distribution of JS values.
        This is produced by drawing N sets of samples of size equal to the set of posterior samples
        and computing the JS distane for random combinations of these sets. The p-value is then the 
        fraction of N where the null-JS values are above the JS value of the posterior sample.

        The significance of a result is determined by comparing the p-value to a threshold.

        Parameters
        ----------
        threshold : float
            p-value threshold to use to define significant vs. not
        N : int
            Number of realizations of the prior to define the null-distribution of JS values.
        
        Returns
        -------
        testResult : dict
        """
 
        # Limit on poster sample size for memory?
        M = min([*[self.postSamples[key].shape[0] for key in self.priors.keys()], self.testSizeLim])
         
        testResult = {'statistic': np.zeros(self.ndim), 'pvalue': np.zeros(self.ndim), 'significant': np.zeros(self.ndim, dtype=bool)}
 
        # Loop through each item in the priors list and corresponding posterior sample and compute JS value and p-value for each.
        for i, key in enumerate(self.priors.keys()):

            x = np.sort(np.array(self.postSamples[key]).real).squeeze()

            # Get PDF for posterior sample
            kde = sm.nonparametric.KDEUnivariate(x)

            kde.fit(cut=5)

            # Get PDF for prior
            sprior = self._getScipyDistVersion(self.priors[key])

            # Make null sample of JS values
             
            null_JS = self._generateJSNullSample(sprior, N, M)
 
            # Compute JS and corresponding p-value for posterior sample            
            JS = ssd.jensenshannon(sprior.pdf(x), 
                               kde.evaluate(x.T), base=2)
 
            pvalue = len(null_JS[null_JS >= JS]) / N

            # Put things in an output dict.
            testResult['pvalue'][i] = pvalue
            testResult['statistic'][i] = JS
            testResult['significant'][i] = pvalue <= threshold

        return testResult
    
    def widthRatio(self, threshold=0.5):
        """ Method for running prior/posterior width ratio test

        Returns
        -------
        testResult : dict
        """
        
        testResult = {'statistic': np.zeros(self.ndim), 'significant': np.zeros(self.ndim, dtype=bool)}
 
        for i, key in enumerate(self.priors.keys()):

            priorType = self.priors[key].__class__.__name__
            
            if priorType == 'normal':
                 
                w0 = self.priors[key].scale
                 

            elif priorType == 'beta':
                 
                a = self.priors[key].a
                b = self.priors[key].b

                w0 = np.sqrt(a*b / ((a+b)**2 * (a + b + 1)))
            else:
                raise TypeError('Width ratio test currently only supports normal and beta distributions')
            
            w1 = np.std(self.postSamples[key])

            ratio = w1/w0
            
            testResult['statistic'][i] = ratio

            testResult['significant'][i] = ratio <= threshold

        return testResult

    def _generateJSNullSample(self, prior, N, M, maxArr=1e6):
         
        n = int(maxArr//M)

        k = int(N//(maxArr//M))
        
        null_JS = np.zeros((k, n))
        
        idx0 = np.random.choice(range(n), size=n, replace=False)
        idx1 = np.random.choice(range(n), size=n, replace=False)
    
        for i in range(k):

            nullProbs = prior.pdf(np.sort(prior.rvs((n, M)), axis=1))
    
            null_JS[i, :] = ssd.jensenshannon(nullProbs[idx0, :],  
                                              nullProbs[idx1, :], base=2, axis=1)
        
        return null_JS.flatten()
    
    def _getScipyDistVersion(self, prior):
        """ This is a hack for getting the Scipy version of the distribution.
        """
        priorType = prior.__class__.__name__
        
        if priorType =='normal':
            priorType = 'norm'
        
        assert priorType in ['beta', 'norm'], 'JS test currently only supports the beta and norm distributions.'

        priorCls = getattr(st, priorType)

        possibleKwargs = ['loc', 'scale', 'a', 'b', 's']
        
        distKwargs = {k: v for k,v in vars(prior).items() if k in possibleKwargs}
        
        return priorCls(**distKwargs)

    def _runValidationMethod(self, method_name):
        method = getattr(self, method_name, None)  # Get method if it exists

        if callable(method):  # Check if it's actually a method
            return method()  # Call the method
        
        else:
            raise AttributeError(f"Method '{method_name}' not found")
        
    def __call__(self, tests='all'):

        availableTests = ['kstest', 'jstest', 'widthratio']
        
        if tests=='all':
            testsToRun = availableTests

        elif isinstance(tests, list):
            testsToRun = [test.lower() for test in tests if test.lower() in availableTests]
        
        elif isinstance(tests, str):
             
            assert tests.lower() in availableTests, 'Please select one or more valid tests. See PBjam docs for a list.'
            testsToRun = [tests]

        funcNameMap = {'kstest': 'KStest',
                       'jstest': 'JStest',
                       'widthratio': 'widthRatio'}

        valid = {test: self._runValidationMethod(funcNameMap[test]) for test in testsToRun}

        return valid