 ## Function
 # does it run
 # does it return something if it should? Or set an attribute
 # is the output the type of thing you want
 # is the output the shape you want

 ## Reasonable/unreasonable?
 # is the output something reasonable if I give it something reasonable?
 # is the output something unreasonable if I give it something unreasonable


import numpy as np
from ..jar import to_log10
import lightkurve as lk
import astropy.units as units
from ..star import star
import statsmodels.api as sm
from ..asy_peakbag import asymptotic_fit

class case():
    
    def __init__(self, ID):
        
        self.ID = ID
        
        self.pars = self.load_example_pars(ID)

        self.st = self.init_dummy_star(ID)
        
        self.load_kde()
        
        self.load_asy_fit()


    def init_dummy_star(self, ID):
        """ Initialize a dummy star class instance
        
        Parameters
        ----------
        ID : str
            ID of the dummy star to use
        
        Returns
        -------
        st : star class instance
            A star class instance with a set of dummy parameters
        
        """

        st = star(self.ID, self.pars['pg'], *[self.pars['obs'][x] for x in self.pars['obs'].keys()])
               
        return st

            
    def load_example_pars(self, ID):
          
        # A silly example
        if self.ID == 'silly':
            pars = {'asypars': [10, 10, 1, 10, 10, 1, 1, 1, 10, 1], 
                    'obs': {'dnu': (1, 1), 'numax': (1, 1), 'teff': (1, 1), 'bp_rp': (1, 1)},
                    'nmax': 1, 
                    'freqs' : np.array([1,10]),
                    }
            
            pars['norders'] = len(pars['freqs'])            
            pars['log_obs'] = {x: to_log10(*pars['obs'][x]) for x in pars['obs'].keys() if x != 'bp_rp'}       
            pars['SC_TS'] = np.vstack((np.linspace(0,1000,100),
                                       np.linspace(0,1000,100)))
            pars['LC_TS'] = np.vstack((np.linspace(0,1000,10),
                                       np.linspace(0,1000,10)))
            
            pars['pg'] = lk.periodogram.Periodogram(np.array([1,1])*units.microhertz, units.Quantity(np.array([1,1]), None))
            
            pars['nsamples'] = 10
            
        if self.ID == 'boeing':
            pars = {'pars': [1.23072917362692,
                             2.34640449860884,
                             1.33476843739203,
                             0.347104562462547,
                            -2.17266900058679,
                             1.47003574648704,
                             1.31305213708824,
                             -0.668935969601942,
                             3.69363755249007,
                             1.22605501887223],
                             'obs' : {'dnu': (16.97, 0.05), 'numax': (220.0, 3.0), 'teff': (4750, 100), 'bp_rp': (1.34, 0.1)}, 
                             'nmax': 11,
                             'freqs': np.array([159.583, 176.226, 192.983, 209.855, 226.841, 243.942, 261.157])}
            
        return pars
    
    def load_kde(self,):
        
        if self.ID == 'silly':
            self.st.kde = type('kde', (object,), {})()
            self.st.kde.samples = np.ones((2,10))
            
            data = np.array(self.pars['asypars']).repeat(11).reshape((10,-1))
            self.st.kde.kde = sm.nonparametric.KDEMultivariate(data=data, var_type='cccccccccc', bw='scott')
        
    def load_asy_fit(self, ):

        if self.ID == 'silly':

            asymptotic_fit(self.st, norders=self.pars['norders'])
            
            self.st.asy_fit.fit = type('fit', (object,), {})()
            self.st.asy_fit.fit.flatchain = np.ones((100, 10))
            self.st.asy_fit.fit.flatchain[:,1] = 2
            self.st.asy_fit.fit.flatchain[:,3] = 0
            self.st.asy_fit.fit.flatchain[:,4] = -2
            self.st.asy_fit.fit.flatlnlike = np.ones(100)
            self.st.asy_fit.fit.flatlnlike[0] = 2
            self.st.asy_fit.log_obs = self.pars['log_obs']

            
    
def does_it_run(func, args):
    """Test to see if function runs 
    
    Given standard inputs test that function doesn't crash. 
    
    Parameters
    ----------
    func : function
        Function to be tested
    args : list
        List of arguments to the provided function
        
    """
    
    if args is None:
        func()
    else:
        func(*args)

def does_it_return(func, args):
    """ Test if function returns something
    
    Given a set of arguments, does the function actually return something?
    
    Parameters
    ----------
    func : function
        Function to be tested
    args : list
        List of arguments to the provided function
        
    """
    
    if args is None:
        assert(func() is not None)
    else:
        assert(func(*args) is not None)

def right_type(func, args, expected):
    """ Check the type of the returned object
    
    Calls func given args, and compares it to the expected output type.
    
    Parameters
    ----------
    func : function
        Function to be tested
    args : list
        List of arguments to the provided function
    expected : type
        The expected type of the output.
        
    """
    
    if args is None:
        assert(isinstance(func(), expected))
    else:
        assert(isinstance(func(*args), expected))

def right_shape(func, args, expected):
    """ Check that output has the correct shape
    
    Calls func given args, and compares the output shape to the expected output
    shape.
    
    Parameters
    ----------
    func : function
        Function to be tested
    args : list
        List of arguments to the provided function
    expected : tuple
        The expected shape tuple of the output.
        
    """
    
    if args is None:
        assert(np.shape(func())==expected)
    else:
        assert(np.shape(func(*args))==expected)

def assert_positive(x):
    """ Check that all elements of x are positive
    
    Parameters
    ----------
    x : ndarray
        Array of values to check 
        
    """
    
    assert(all(x) >= 0)

def assert_hasattributes(obj, attributes):
    """ Check that object has attributes
    
    Loops through a list of attribute names and checks that the object has them.
    
    obj : object
        Object to test for attributes
    attributes : list
        List of attributes to check 
    
    """
    
    for attr in attributes:
        # print(attr)
        assert(hasattr(obj, attr))


#        # A more realistic example
#        self.solar_extra = {'pars': [np.log10(135.0), # dnu
#                                  np.log10(3050.0), # numax 
#                                  1.25, #eps
#                                  0.5, # d02
#                                  -2.5, # alpha
#                                  1.5, # envheight
#                                  2.2, # envwidth
#                                  0.0, # modwidth
#                                  3.77, #teff
#                                  0.8], # bp_rp
#                            'obs': {'dnu': (135, 1.35), 'numax': (3050, 30), 'teff': (5777, 60), 'bp_rp': (0.8, 0.01)},
#                            'nmax': 21, 
#                            'freqs' : np.array([2601.134, 2734.921, 2869.134, 3003.775, 3138.842, 3274.336, 3410.257])}
#        
#        # Another more realistic example
#        self.boing_extra = {'pars': [1.23072917362692,
#                                    2.34640449860884,
#                                    1.33476843739203,
#                                    0.347104562462547,
#                                   -2.17266900058679,
#                                    1.47003574648704,
#                                    1.31305213708824,
#                                    -0.668935969601942,
#                                    3.69363755249007,
#                                    1.22605501887223],
#                            'obs' : {'dnu': (16.97, 0.05), 'numax': (220.0, 3.0), 'teff': (4750, 100), 'bp_rp': (1.34, 0.1)}, 
#                            'nmax': 11,
#                            'freqs': np.array([159.583, 176.226, 192.983, 209.855, 226.841, 243.942, 261.157])}
#        
#        for K in [self.silly_extra, self.solar_extra, self.beoing_extra]:
#            K['norders'] = len(K['freqs'])
#            
#            K['log_obs'] = {x: to_log10(*K['obs'][x]) for x in K['obs'].keys() if x != 'bp_rp'}
#        
#            K['SC_TS'] = np.vstack((np.linspace(1e-20,8500,100),
#                                    np.linspace(1e-20,8500,100)))
#            K['LC_TS'] = np.vstack((np.linspace(1e-20,850,10),
#                                    np.linspace(1e-20,850,10)))
#            K['nsamples'] = 10