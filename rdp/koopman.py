import numpy as np
import scipy
from scipy.sparse.linalg import eigs, lobpcg, eigsh
from tqdm import tqdm
from functools import partial
import multiprocessing


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


def koop_pseudo_spec(G, A, L, z_pts, parallel=False, z_pts2=None, reg_param=1e-14):
	# %Collect the optional inputs
	# p = inputParser;

	# %addRequired(p, 'G', @ isnumeric);
	# %addRequired(p, 'A', @ isnumeric);
	# %addRequired(p, 'L', @ isnumeric);
	# %addRequired(p, 'z_pts', @ isnumeric);

	# validPar = {'on', 'off'};
	# checkPar = @(x) any(validatestring(x, validPar));

	# addParameter(p, 'Parallel', 'off', checkPar)
	# addParameter(p, 'z_pts2', [], @ isnumeric)
	# addParameter(p, 'reg_param', 10 ^ (-14), @ isnumeric)

	# p.CaseSensitive = false;
	# parse(p, varargin{:})

	# %%compute the pseudospectrum
	# G = (G + G')/2; L=(L+L') / 2; %safeguards
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
	#
# 	if ~isempty(p.Results.z_pts2)
# 		RES2=zeros(length(p.Results.z_pts2), 1);
# 		V2=zeros(size(G, 1), length(p.Results.z_pts2));
# 		pf = parfor_progress(length(p.Results.z_pts2));
# 		pfcleanup = onCleanup( @ () delete(pf));
# 		if p.Results.Parallel == "on"
# 			parfor jj=1:length(p.Results.z_pts2)
# 				warning('off', 'all')
# 				[V, D] = eigs(
# 					SQ * ((L) - p.Results.z_pts2(jj) * A'-conj(p.Results.z_pts2(jj))*A+(abs(p.Results.z_pts2(jj))^2)*G)*SQ,
# 					1,
# 					'smallestabs'
# 				);
# 				V2(:, jj) = V;
# 				RES2(jj) = sqrt(real(D(1, 1)));
# 				parfor_progress(pf);
# 		end
# 		else
# 			for jj=1:length(p.Results.z_pts2)
# 				[V, D] = eigs(
# 					SQ * ((L) - p.Results.z_pts2(jj) * A'-conj(p.Results.z_pts2(jj))*A+(abs(p.Results.z_pts2(jj))^2)*G)*SQ,
# 					1,
# 					'smallestabs'
# 				);
# 				V2(:, jj) = V;
# 				RES2(jj) = sqrt(real(D(1, 1)));
# 				parfor_progress(pf);
# 			end
# 		end
# 		V2 = SQ * V2;
# 	end
# 	#
# 	warning('on', 'all')
# 	#
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
	w = np.convolve(x, np.conj(np.flipud(x)))

	mu = np.zeros((1, n + 1))
	mu[0] = np.dot(x) / (m * 2 * np.pi)
	# TODO: WIP


