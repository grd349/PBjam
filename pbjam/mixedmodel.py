import jax.numpy as jnp
import jax, warnings
from pbjam import jar
from functools import partial
from pbjam.jar import constants as c
jax.config.update('jax_enable_x64', True)
 
class MixFreqModel():

    def __init__(self,  N_p, obs, n_g_ppf):
   
        self.N_p = N_p

        self.obs = obs
 
        self.n_g = self.select_n_g(n_g_ppf)

        self.N_g = len(self.n_g)

        self.ones_block = jnp.ones((self.N_p, self.N_g))

        self.zeros_block = jnp.zeros((self.N_p, self.N_g))

        self.eye_N_p = jnp.eye(self.N_p)

        self.eye_N_g = jnp.eye(self.N_g)
 
        self.D_gamma = jnp.vstack((jnp.zeros((self.N_p, self.N_p + self.N_g)), 
                                   jnp.hstack((self.zeros_block.T, self.eye_N_g))))

    def select_n_g(self, n_g_ppf, fac=2):
        """ Select and initial range for n_g

        Computes the number of g-modes that are relevant near the oscillation
        envelope. This is based on the expected range for DPi1 and eps_g and 
        numax.

        This is used to set the number of g-modes at the start of the run, and
        sets the number of g-modes at or near the p-mode envelope. The range is
        significantly wider than the actual power distribution of the envelope
        so there is room for DPi1 and eps_g to change.

        Returns
        -------
        n_g_ppf : list
            The quauntile functions for DPi1 and eps_g. 
        fac : float
            g-modes are considered if they fall within +/- fac * envelope_width
            of numax. A larger may(??) increase precision at the cost of time
            to perform eigendecomposition.
        """
  
        n = self.N_p // 2 + 1
 
        width = max((n + 1) * self.obs['dnu'][0], fac * jar.scalingRelations.envWidth(self.obs['numax'][0]))
         
        freq_lims = (self.obs['numax'][0] - width, 
                     self.obs['numax'][0] + width)
         
        # Start with an exagerated number of g-modes.
        init_n_g = jnp.arange(10000)[::-1] + 1

        min_n_g = init_n_g.max()

        max_n_g = init_n_g.min()

        # Loop over combinations of DPi1 and eps_g as drawn from the respective PDFs.       
        for DPi1 in jnp.linspace(n_g_ppf[0](1e-3), n_g_ppf[0](1-1e-3), 3):
            
            for eps_g in jnp.linspace(n_g_ppf[1](1e-3), n_g_ppf[1](1-1e-3), 3):
                
                nu_g = self.asymptotic_nu_g(init_n_g, DPi1, eps_g, 1e-4)
                
                idx = (freq_lims[0] < nu_g) & (nu_g < freq_lims[1])
                 
                t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                 
                min_n_g = jnp.minimum(min_n_g, t)
                 
                t = jnp.where(idx, init_n_g, 0 * init_n_g - 1).max()
                 
                max_n_g = jnp.maximum(max_n_g, t)

        n_g = jnp.arange(min_n_g, max_n_g, dtype=int)[::-1]
        
        if len(n_g) > 100:
            warnings.warn(f'{len(n_g)} g-modes in the p-mode envelope area.')

        return n_g
 
    @partial(jax.jit, static_argnums=(0,))
    def asymptotic_nu_g(self, n_g, DPi1, eps_g, alpha_g, max_N2=jnp.inf):
        """Asymptotic relation for g-modes

        Asymptotic relation for the g-mode frequencies in terms of a fundamental
        period offset (defined by the maximum Brunt-Vaisala frequency), the 
        asymptotic g-mode period spacing, the g-mode phase offset, and an 
        optional curvature term.

        Parameters
        ----------
        n_g : jax device array
            Array of radial orders for the g-modes.
        DPi1 : float
            Period spacing for l=1 in seconds).
        eps_g : float
            Phase offset of the g-modes.
        alpha_g : float
            Curvature scale of the g-modes.
        max_N2 : float
            Maximum of the Brunt-Vaisala frequency.
        Returns
        -------
        jax device array
            Frequencies of the notionally pure g-modes of degree l.
        """
         
        P0 = 1 / (jnp.sqrt(max_N2) / c.nu_to_omega)

        #DPi0 = DPi1 / jnp.sqrt(2)  
        DPi1 *= 1e-6 # DPi1 in s to Ms.  

        P_max = 1/self.obs['numax'][0]

        n_gmax = (P_max - P0) / DPi1 - eps_g

        P = P0 + DPi1 * (n_g + eps_g + alpha_g * (n_g - n_gmax)**2)

        return 1/P

    @partial(jax.jit, static_argnums=(0,))
    def mixed_nu1(self, nu0_p, n_p, d01, DPi1, p_L, p_D, eps_g, alpha_g, **kwargs):
        """
        Calculate mixed nu1 values and associated zeta values.
        
        Parameters
        ----------
        nu0_p : float
            Initial nu0 value.
        n_p : int
            Number of n values.
        d01 : float
            The d01 frequency separation
        DPi1 : float
            Period spacing for l=1.
        p_L : jax device array
            Polynomial coefficients for the L coupling strength matrix.
        p_D : jax device array
            Polynomial coefficients for the D coupling strength matrix.
        eps_g : float
            Phase offset of the g-modes.
        alpha_g : float
            Curvature scale of the g-modes.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        nu : jax device array
            Array of frequencies of the mixed l=1 modes. 
        zeta : jax device array
            Array of mixing degrees for the modes.
        """

        nu1_p = nu0_p + d01  
    
        nu_g = self.asymptotic_nu_g(self.n_g, DPi1, eps_g, alpha_g)
        
        L, D = self.generate_matrices(n_p, self.n_g, nu1_p, nu_g, p_L, p_D)
         
        nu, zeta = self.new_modes(L, D)

        return nu, zeta 

    @partial(jax.jit, static_argnums=(0,))
    def _wrap_polyval2d(self, x, y, p):
        """Evaluate 2D polynomial

        Evaluates 2D polynomial for the the top right or bottom left blocks of 
        the L and D matrices. 

        Parameters
        ----------
        x : jax device array
            Array of radial orders for the pi modes.
        y : jax device array
            Array of radial orders for the gamma modes.
        p : jax device array
            Polynomial coeffecients.

        Returns
        -------
        jax device array
            Block of shape (len(x), len(y) to be inserted in L or D coupling 
            strength matrices.
        
        Notes
        -----
        This is legacy code which is not currently in use. May be used in the 
        future if it's necessary to compute more complicated 2D poly's.

        """
      
        xx = (x + 0 * y)

        yy = (y + 0 * x)

        shape = xx.shape

        P = self.poly_params_2d(p)

        block = self.polyval2d(xx.flatten(), yy.flatten(), P)

        return block.reshape(shape)

    @partial(jax.jit, static_argnums=(0,))
    def poly_params_2d(self, p):
        """Reshape polynomial coefficients

        Turn a parameter vector into an upper triangular coefficient matrix.

        For a list of polynomial coefficients [1,2,3] the result is of the
        form 
        [1 3 0]
        [2 0 0]
        [0 0 0]

        Parameters
        ----------
        p : jax device array
            Array of polynomial coefficients

        Returns
        -------
        jax device array
            Square matrix of reshaped polynomial coefficients.
        
        Notes
        -----
        This is legacy code which is not currently in use. May be used in the 
        future if it's necessary to compute more complicated 2D poly's.

        """
 
        # Check that len(p) a triangular number
        if len(p) == 1:
            n = 1
        elif len(p) == 3:
            n = 2
        elif len(p) == 6:
            n = 3
        else:
            raise ValueError('The length of p must be 1, 3, or 6.')

        P = jnp.zeros((n, n))

        # TODO: I think this part is slow in jax, can be speeded up?
        for i in range(n):
            # ith triangular number as offset
            n_i = i * (i + 1) // 2

            for j in range(i + 1):

                P = P.at[i - j, j].set(p[n_i + j])

        return P

    @partial(jax.jit, static_argnums=(0,))
    def polyval2d(self, x, y, P, increasing=True):
        """Evaluate a 2D polynomial

        Jaxxable replacement for Numpy's polyval2d. Doesn't seem to exist in 
        jax.numpy yet.

        Parameters
        ----------
        x : jax device array
            x-coordinates to evaluate the polynomial at. In this case n_g or 
            n_p
        y : jax device array
            y-coordinates to evaluate the polynomial at. In this case n_g or 
            n_p
        P : jax device array
            Upper triangular matrix of polynomial coefficients.
        increasing : bool, optional
            Ordering of the powers of the columns of the Vandermonde matrices. 
            If True, the powers increase from left to right, if False 
            (the default) they are reversed.

        Returns
        -------
        jax device array
            Matrix of the polymial values.

        Notes
        -----
        This is legacy code which is not currently in use. May be used in the 
        future if it's necessary to compute more complicated 2D poly's.
        """

        X = jnp.vander(x, P.shape[0], increasing=increasing)

        Y = jnp.vander(y, P.shape[1], increasing=increasing)

        return jnp.sum(X[:, :, None] * Y[:, None, :]*P, axis=(2, 1))

    @partial(jax.jit, static_argnums=(0,))
    def generate_matrices(self, n_p, n_g, nu_p, nu_g, p_L, p_D):
        """Generate coupling strength matrices

        Computes the coupling strength matrices based on the asymptotic p- and
        g-mode frequencies and the polynomial representation of the coupling
        strengths.

        Parameters
        ----------
        n_p : jax device array
            Array containing p-mode radial orders.
        n_g : jax device array
            Array containing g-mode radial orders.
        nu_p : jax device array
            Array containing asymptotic l=1 p-mode frequencies.
        nu_g : jax device array
            Array containing asymptotic l=1 g-mode frequencies.
        p_L : jax device array
            Parameter vector describing 2D polynomial coefficients for coupling 
            strengths.
        p_D : jax device array
            Parameter vector describing 2D polynomial coefficients for overlap 
            integrals.

        Returns
        -------
        L : jax device array
            Matrix of coupling strengths.
        D : jax device array
            Matrix of overlap integrals.
        """
 
        #L_cross = self._wrap_polyval2d(n_p[:, jnp.newaxis], n_g[jnp.newaxis, :], p_L) * (nu_g * c.nu_to_omega)**2
        L_cross = self.ones_block * p_L * (nu_g * c.nu_to_omega)**2

        #D_cross = self._wrap_polyval2d(n_p[:, jnp.newaxis], n_g[jnp.newaxis, :], p_D) * (nu_g[jnp.newaxis, :]) / (nu_p[:, jnp.newaxis])
        D_cross = self.ones_block * p_D * (nu_g[jnp.newaxis, :]) / (nu_p[:, jnp.newaxis])

        L = jnp.hstack((jnp.vstack((jnp.diag(-(nu_p * c.nu_to_omega)**2), L_cross.T)),
                        jnp.vstack((L_cross, jnp.diag( -(nu_g * c.nu_to_omega)**2 )))
                        ))

        D = jnp.hstack((jnp.vstack((self.eye_N_p       , D_cross[::-1, ::-1].T)),
                        jnp.vstack((D_cross[::-1, ::-1], self.eye_N_g))
                        ))

        return L, D
    
    @partial(jax.jit, static_argnums=(0,))
    def new_modes(self, L, D):
        """ Solve for mixed mode frequencies

        Given the matrices L and D such that we have eigenvectors

        L cᵢ = -ωᵢ² D cᵢ,

        with ω in Hz, we solve for the frequencies ν (μHz), mode mixing 
        coefficient zeta.

        Parameters
        ----------
        L : jax device array
            The coupling strength matrix.
        D : jax device array
            The overlap integral.

        Returns
        -------
        nu_mixed : jax device array
            Array of mixed mode frequencies.
        zeta : jax device array
            The mixing degree for each of the mixed modes.
        """

        Lambda, U = self.eigh(L, D)
    
        new_omega2 = -Lambda
        
        zeta = jnp.diag(U.T @ self.D_gamma @ U)

        sidx = jnp.argsort(new_omega2)

        return jnp.sqrt(new_omega2)[sidx] / c.nu_to_omega, zeta[sidx]  

    # @partial(jax.jit, static_argnums=(0,))
    # def symmetrize(self, x):
    #     """ Symmetrize matrix.

    #     Parameters
    #     ----------
    #     x : jax device array
    #         A square matrix.

    #     Returns
    #     -------
    #     w : jax device array
    #         Symmetrized matrix.
    #     """
    #     return (x + jnp.conj(jnp.swapaxes(x, -1, -2))) / 2

    @partial(jax.jit, static_argnums=(0,))
    def standardize_angle(self, w, b):
        """ Helper for eigh solver

        Parameters
        ----------
        w : jax device array
            Matrix
        b : jax device array
            Matrix

        Returns
        -------
        w : jax device array
            Modified w.
        """

        return w * jnp.sign(w[0, :])


    @partial(jax.jit, static_argnums=(0,))
    def eigh(self, a, b):
        """ Jaxxable replacement for numpy.eigh.

        From https://jackd.github.io/posts/generalized-eig-jvp/

        Compute the solution to the symmetrized generalized eigenvalue problem.

        a_s @ w = b_s @ w @ np.diag(v)

        where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized 
        versions of the inputs and H is the Hermitian (conjugate transpose) 
        operator.

        For self-adjoint inputs the solution should be consistent with 
        `scipy.linalg.eigh` i.e.

        v, w = eigh(a, b)
        v_sp, w_sp = scipy.linalg.eigh(a, b)

        np.testing.assert_allclose(v, v_sp)
        np.testing.assert_allclose(w, standardize_angle(w_sp))

        Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which
        will be slow because there is no GPU implementation of `eig` and it's 
        just a generally inefficient way of doing it. Future implementations 
        should wrap cuda primitives. This implementation is provided primarily 
        as a means to test `eigh_jvp_rule`.

        Parameters
        ----------
        a : jax device array
            [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        b : jax device array
            [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

        Returns
        -------
        v : jax device array
            Eigenvalues of the generalized problem in ascending order.
        w : jax device array
            Eigenvectors of the generalized problem, normalized such that
            w.H @ b @ w = I.
        """

        #a = self.symmetrize(a)

        #b = self.symmetrize(b)
        
        b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)

        v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)

        v = v.real
 
        w = w.real

        # reorder as ascending in w
        order = jnp.argsort(v)

        v = v.take(order, axis=0)

        w = w.take(order, axis=1)

        # renormalize so v.H @ b @ H == 1
        norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)

        w = w / jnp.sqrt(norm2)

        w = self.standardize_angle(w, b)

        return v, w

    