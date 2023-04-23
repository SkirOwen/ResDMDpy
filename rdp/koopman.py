import numpy as np
from scipy.sparse.linalg import eigs
from tqdm import tqdm

# KoopPseudoSpec


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
	G = (G + G.T) / 2
	L = (L + L.T) / 2
	# [VG, DG] = eig(G + norm(G) * (p.Results.reg_param) * eye(size(G)));
	# MATLAB norm is by default 2, whereas for numpy is frobenius for matrices
	DG, VG = np.linalg.eig(G + np.linalg.norm(G, 2) * reg_param * np.eye(*G.shape))

	# DG(abs(DG) > 0) = sqrt(1. / abs(DG(abs(DG) > 0)));
	# SQ = VG * DG * (VG'); % needed to compute pseudospectra according to Gram matrix G
	# TODO: abs(DG[abs(DG)> 0]) sounds redundant
	DG[abs(DG) > 0] = np.sqrt(1 / abs(DG[abs(DG) > 0]))  # TODO: this doesnt give the same results
	SQ = VG @ (DG * np.eye(len(DG))) @ VG.T
	# TODO: DG * np.eye to make it diag, there must be a better way!

	#
	z_pts = z_pts.reshape(-1, 1)
	LL = len(z_pts)
	RES = np.zeros((LL, 1))
	print(LL)

	if LL > 0:
		if parallel:
			pass
		else:
			for jj in tqdm(range(LL)):

				# "smallestabs in matlab is SM in scipy and sm in Octave and Matlab<R2017a
				RES[jj] = np.sqrt(
					np.real(
						eigs(
							SQ @ (L - z_pts[jj] * A.T - np.conj(z_pts[jj]) * A + (abs(z_pts[jj])**2) * G) @ SQ,
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