from __future__ import annotations

import numpy as np
import scipy
from scipy.sparse.linalg import eigs, lobpcg, eigsh
from tqdm import tqdm
from functools import partial
import multiprocessing


from rdp.utils.linalg_op import guarantee_hermitian

# KoopPseudoSpec


def compute_RES(jj, SQ, L, A, G, z_pts):
	return np.sqrt(
		np.real(
			eigs(
				SQ @ (L - z_pts[jj] * A.T - np.conj(z_pts[jj]) * A + (abs(z_pts[jj])**2) * G) @ SQ,
				1,
				which='SM'
			)[0]
		)
	)


def koop_pseudo_spec(
		G: np.ndarray,
		A: np.ndarray,
		L: np.ndarray,
		z_pts: np.ndarray,
		parallel: bool = False,
		z_pts2: None | np.ndarray = None,
		reg_param: float = 1e-14
) -> tuple:
	"""

	Parameters
	----------
	G : 2d-ndarray
	A : 2d-ndarray
	L : 2d-ndarray
	z_pts : 1d-ndarray
	parallel : bool, optional
		The default is False
	z_pts2 : ndarray, optional
		The default is None.
	reg_param : float, Optional
		The default is 1e-14

	Returns
	-------
	tuple

	"""
	# safeguards
	G = guarantee_hermitian(G)
	L = guarantee_hermitian(L)

	G = (G + G.conj().T) / 2
	L = (L + L.conj().T) / 2
	# [VG, DG] = eig(G + norm(G) * (p.Results.reg_param) * eye(size(G)));
	# MATLAB norm is by default 2, whereas for numpy is frobenius for matrices
	DG, VG = np.linalg.eig(G + np.linalg.norm(G, 2) * reg_param * np.eye(*G.shape))

	# DG(abs(DG) > 0) = sqrt(1. / abs(DG(abs(DG) > 0)));
	# SQ = VG * DG * (VG'); % needed to compute pseudospectra according to Gram matrix G
	# TODO: abs(DG[abs(DG)> 0]) sounds redundant
	DG[abs(DG) > 0] = np.sqrt(1 / DG[abs(DG) > 0])  # TODO: this doesnt give the same results
	SQ = VG @ (DG * np.identity(len(DG))) @ VG.conj().T
	# TODO: DG * np.eye to make it diag, there must be a better way!

	#
	z_pts = z_pts.reshape(-1, 1)
	LL = len(z_pts)
	RES = np.zeros((LL, 1))
	print(LL)

	if LL > 0:
		if parallel:
			with multiprocessing.Pool() as pool:
				func = partial(compute_RES, SQ=SQ, L=L, A=A, G=G, z_pts=z_pts)
				for idx, res in enumerate(tqdm(pool.imap(func, range(LL)), total=LL)):
					RES[idx] = res

		else:
			for jj in tqdm(range(LL), desc="Koopman pseudospectra"):

				# "smallestabs in matlab is SM in scipy and sm in Octave and Matlab<R2017a
				RES[jj] = np.sqrt(
					np.real(
						eigs(   # TODO: look eigsh after fixing bottleneck
							SQ @ (L - z_pts[jj] * A.conj().T - np.conj(z_pts[jj]) * A + (abs(z_pts[jj])**2) * G) @ SQ,
							1,
							which='SM')[0]
					)
				)

	RES2 = []
	V2 = []

	# if len(z_pts2):
	# 	# TODO: do I really need this?
	# 	RES2 = np.zeros((len(z_pts2), 1))
	# 	V2 = np.zeros(G.shape[0], len(z_pts2))
	#
	# 	if parallel:
	# 		for jj in range(len(z_pts2)):
	# 			D, V = eigs(
	# 				SQ @ (L - z_pts2[jj] * A.conj().T - np.conj(z_pts2[jj]) * A + (abs(z_pts2[jj])**2) * G) @ SQ,
	# 				1,
	# 				which="sn"
	# 			)
	#
	# 			RES2[jj] = np.sqrt(np.real(D[0, 0]))
	# 			V2[:, jj] = V.copy()
	# 	else:
	# 		for jj in range(len(z_pts2)):
	# 			[V, D] = eigs(
	# 				SQ @ (L - z_pts2[jj] * A.conj().T - np.conj(z_pts2[jj]) * A + (abs(z_pts2[jj])**2) * G) @ SQ,
	# 				1,
	# 				which="sm"
	# 			)
	# 			RES2[jj] = np.sqrt(np.real(D[0, 0]))
	# 			V2[:, jj] = V.copy()
	#
	# 	V2 = SQ @ V2

	return RES, RES2, V2
#
# # ErgodicMoments


def ergodic_moments(x: np.ndarray, n: int) -> np.ndarray:
	"""Compute the moments -n to n of an array x using the ergodic formula.

	Given an array x of a signal and a non-negative integer N, this function computes the auto-correlations of X,
	or moments of the Koopman operator, from -N to N.
	This code uses the FFT for rapid computation.

	Parameters
	----------
	x : array-like
	n : int

	Returns
	-------
	mu : ndarray
	"""
	x = x.flatten()
	m = len(x)
	mu = np.zeros(n + 1)
	mu[0] = np.dot(x.conj().T, x) / (m * 2 * np.pi)

	w = np.convolve(x, np.conj(np.flip(x)))

	mu[1: n+1] = w[m-2: n-1: -1].T / (2 * np.pi * np.arange(m-1, n, -1))
	mu = np.concatenate((np.conj(np.flip(mu[1:])), mu))

	return mu


