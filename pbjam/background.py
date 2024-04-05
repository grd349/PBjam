from pbjam import jar
from functools import partial
import jax

class bkgModel():
    def __init__(self, nu, Nyquist):
        """
        Initialize the background model calculator.

        Parameters
        ----------
        Nyquist : float
            Nyquist frequency.
        """
        self.nu = nu

        self.Nyquist = Nyquist

        self.eta = jar.attenuation(self.nu, self.Nyquist)**2


    #@partial(jax.jit, static_argnums=(0,))
    def harvey(self, nu, a, b, c):
        """ Harvey-profile

        Parameters
        ----------
        nu : np.array
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

    #@partial(jax.jit, static_argnums=(0,))
    def __call__(self, theta_u,):
        """
        Calculate the background model.

        Parameters
        ----------
        theta_u : dict
            A dictionary of background model parameters.
        nu : numpy.ndarray
            Array of frequency values.

        Returns
        -------
        array
            The calculated background model.

        Notes
        -----
        - Computes the Harvey components H1, H2, and H3 for the given frequency values.
        - Calculates the attenuation factor eta.
        - Combines the Harvey components with the attenuation factor and shot noise to
          compute the background model.
        """

        H1 = self.harvey(self.nu, theta_u['H_power'], theta_u['H1_nu'], theta_u['H1_exp'],)

        H2 = self.harvey(self.nu, theta_u['H_power'], theta_u['H2_nu'], theta_u['H2_exp'],)

        H3 = self.harvey(self.nu, theta_u['H3_power'], theta_u['H3_nu'], theta_u['H3_exp'],)
            

        bkg = (H1 + H2 + H3) * self.eta + theta_u['shot']

        return bkg

