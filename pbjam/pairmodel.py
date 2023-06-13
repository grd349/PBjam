import jax
import jax.numpy as jnp
from functools import partial
jax.config.update('jax_enable_x64', True)

class AsyFreqModel():

    def __init__(self, N_p, V20=0.5):
        
        self.N_p = N_p

        self.V20 = V20

    @partial(jax.jit, static_argnums=(0,))
    def _get_n_p_max(self, dnu, numax, eps):
        """Compute radial order at numax.
    
        Compute the radial order at numax, which in this implimentation of the
        asymptotic relation is not necessarily integer.
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (muHz).
    
        Returns
        -------
        nmax : float
            non-integer radial order of maximum power of the p-mode envelope      
        """
    
        return numax / dnu - eps

    @partial(jax.jit, static_argnums=(0,))
    def _get_n_p(self, nmax):
        """Compute radial order numbers.

        Get the enns that will be included in the asymptotic relation fit.
        These are all integer.

        Parameters
        ----------
        nmax : float
            Frequency of maximum power of the oscillation envelope.

        Returns
        -------
        enns : jax device array
            Array of norders radial orders (integers) around nu_max (nmax).
        """

        below = jnp.floor(nmax - jnp.floor(self.N_p/2)).astype(int)
         
        enns = jnp.arange(self.N_p) + below

        return enns 

    @partial(jax.jit, static_argnums=(0,))
    def asymptotic_nu_p(self, numax, dnu, eps_p, alpha_p, **kwargs):
        """ Compute the l=0 mode frequencies from the asymptotic relation for
        p-modes
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (unitless).
        alpha : float
            Curvature factor of l=0 ridge (second order term, unitless).
    
        Returns
        -------
        nu0s : ndarray
            Array of l=0 mode frequencies from the asymptotic relation (muHz).
        """
        
        n_p_max = self._get_n_p_max(dnu, numax, eps_p)

        n_p = self._get_n_p(n_p_max)

        return (n_p + eps_p + alpha_p/2*(n_p - n_p_max)**2) * dnu, n_p
 
    
 


 
    

