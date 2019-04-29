

class star():
    
    def __init__(self, f, s, numax, dnu, teff):
        self.f = f
        self.s = s
        self.numax = numax
        self.dnu = dnu
        self.teff = teff
        self.epsilon = None
        self.mode_ID = {}
        self.asy_model = None
        
        
        

    def parse_asy_pars(self, verbose = False):
        
        self._d02 = None
        self._alpha = None
        self._seff = None
        self._mode_width = None
        self._env_width = None
        self._env_height = None
        
        
        if not self.epsilon:
            ge_vrard = pb.epsilon()
            self.epsilon = ge_vrard(self.dnu, self.numax, self.teff)
        
        if not self._d02:
            self._d02 = 0.1*self.dnu[0]
            
        if not self._alpha:
            self._alpha = 1e-3
        
        if not self._seff:
            # TODO this needs to be done properly
            self._seff = 4000 
        
        if not self._mode_width:
            self._mode_width = 1e-20 # must be non-zero for walkers' start pos
        
        if not self._env_width:
            self._env_width = 0.66 * self.numax[0]**0.88      
        
        if not self._env_height:
            df = np.median(np.diff(self.f))
            a = int(np.floor(self.dnu[0]/df)) 
            b = int(len(self.s) / a)
            smoo = self.s[:a*b].reshape((b,a)).mean(1)
            self._env_height = max(smoo)
        
        pars = [self.numax[0], self.dnu[0], self.epsilon[0], self._alpha, 
                self._d02, self._env_height, self._env_width, self._mode_width,
                self._seff]
        
        parsnames = ['numax', 'large separation', 'epsilon', 'alpha', 'd02', 
                     'p-mode envelope height', 'p-mode envelope width',
                     'mode width (log10)', 'Seff (adjusted Teff)']
        if verbose:
            for i in range(len(pars)):
                print('%s: %f' % (parsnames[i], pars[i]))
                
        return pars