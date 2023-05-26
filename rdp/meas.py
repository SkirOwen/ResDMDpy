from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.sparse import issparse

from utils.kernels import unitary_kernel
from utils.filters import phi_inft, phi_opt4, phi_fejer, phi_cosine, phi_sharp_cosine
from rdp.utils.linalg_op import guarantee_hermitian


# IsomMeas
def isom_meas(
		G: np.ndarray,
		A: np.ndarray,
		L,
		f: np.ndarray,
		theta: np.ndarray,
		epsilon: float,
		order: int = 2
) -> np.ndarray:

	G = guarantee_hermitian(G)

	# compute the poles and residues
	delta = (2 * np.arange(1, order+1) / (order+1)-1) * 1j + 1
	c, d = unitary_kernel(delta, epsilon)

	if issparse(A) and issparse(G):
		v1 = G * f
		v2 = v1.copy()
		v3 = A.conj().T * f
	else:
		# TODO: what is this test for?
		if np.linalg.norm(G - np.eye(G.shape[0])) < (1e-14 * np.linalg.norm(G)):
			q, s, _ = scipy.linalg.schur(A, output="complex")   # TODO: look into this schur not in __init__
			z = q.copy()
			t = scipy.sparse.eye(A.shape[0], format="csr")  # TODO: ditto
		else:
			s, t, q, z = scipy.linalg.qz(A, G)
			q = q.conj().T  # TODO: what????????????????????????? why

		v1 = t @ z.conj().T * f
		v2 = t.conj().T @ q.conj().T * f
		v3 = s.conj().T @ q.conj().T * f

	nu = 0 * theta
	# TODO: look at parallel here
	for k in range(len(theta)):
		for j in range(order):
			lamb = np.exp(1j * theta[k]) * (1 + epsilon * delta[j])
			Ij = np.linalg.solve((A - lamb @ G), v1)
			nu[k] -= np.real(
				1 / (2 * np.pi) * (c[j] * np.conj(lamb) * (Ij.conj().T @ v2) + d[j] * (v3.conj().T @ Ij))
			)

	return nu


def moment_meas(mu: np.ndarray, filt: str = 'inft'):
	"""
	This code computes smoothed spectral measures of an isometry using the
	computed moments (Fourier coefficients). Requires chebfun.

	Parameters:
	mu (ndarray): Vector of Fourier coefficients (-N to N)
	filt (str): Type of filter (default: 'inft')

	Returns:
	nu (chebfun): Smoothed measure as a chebfun
	"""
	N = (len(mu) - 1) // 2

	if filt == "fejer":
		FILTER = phi_fejer(abs(np.arange(-N, N + 1) / N))
	elif filt == "cosine":
		FILTER = phi_cosine(abs(np.arange(-N, N + 1) / N))
	elif filt == "vand":
		FILTER = phi_opt4(abs(np.arange(-N, N + 1) / N))
	elif filt == "sharp_cosine":
		FILTER = phi_sharp_cosine(abs(np.arange(-N, N + 1) / N))
	else:
		FILTER = phi_inft(abs(np.arange(-N, N + 1) / N))
	FILTER[0] = 0
	FILTER[-1] = 0

	nu = 0
	# TODO: look into chebfun
	# nu = chebfun.Chebfun(FILTER * mu, domain=[-np.pi, np.pi], trig=True)
	# or
	# from numpy.polynomial.chebyshev import Chebyshev
	# from scipy.integrate import fixed_quad
	# def integrand(t):
	# 	return np.array([np.cos(k * t) for k in range(-N, N + 1)]) * FILTER
	#
	# integrals = np.array([fixed_quad(integrand, 0, np.pi, n=50)[0] for i in range(MU.shape[0])])
	# nu = Chebyshev.fromroots(integrals)
	return nu
