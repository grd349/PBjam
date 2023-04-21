from pbjam import jar
from functools import partial
import jax

class bkgModel():
    def __init__(self, Nyquist):

        self.Nyquist = Nyquist

    @partial(jax.jit, static_argnums=(0,))
    def harvey(self, nu, a, b, c):
        """ Harvey-profile

        Parameters
        ----------
        f : np.array
            Frequency axis of the PSD.
        a : float
            The amplitude (divided by 2 pi) of the Harvey-like profile.
        b : float
            The characeteristic frequency of the Harvey-like profile.
        c : float
            The exponent parameter of the Harvey-like profile.

        Returns
        -------
        H : np.array
            The Harvey-like profile given the relevant parameters.
        """
         
        H = a / b * 1 / (1 + (nu / b)**c)

        return H

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, theta_u, nu):
            # Background
        H1 = self.harvey(nu, theta_u['H_power'], theta_u['H1_nu'], theta_u['H1_exp'],)

        H2 = self.harvey(nu, theta_u['H_power'], theta_u['H2_nu'], theta_u['H2_exp'],)

        H3 = self.harvey(nu, theta_u['H3_power'], theta_u['H3_nu'], theta_u['H3_exp'],)
            
        eta = jar.attenuation(nu, self.Nyquist)**2

        bkg = (H1 + H2 + H3) * eta + theta_u['shot']

        return bkg

