from __future__ import annotations

import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import Literal

import cmocean as cm

from tqdm import tqdm

from rdp import logger
from rdp.koopman import koop_pseudo_spec
from rdp.utils.mat_loader import loadmat
from rdp.utils.plotting import plot_pseudospectra, plot_eig_res, plot_error

plt.rcParams['text.usetex'] = True


def get_dict(dmd: Literal["linear", "combined", "pre-computed", "non-linear"]):
	if dmd == "linear":
		data = scipy.io.loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_DMD.mat")
		G_matrix = data["G_matrix"]
		A_matrix = data["A_matrix"]
		L_matrix = data["L_matrix"]
		N = data["N"][0, 0]
		PSI_x = data["PSI_x"]

	elif dmd == "combined":
		data_dmd = scipy.io.loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_DMD.mat")
		PSI_x0 = data_dmd["PSI_x"]
		PSI_y0 = data_dmd["PSI_y"]
		data_edmd = scipy.io.loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
		N = 2 * data_edmd["N"][0, 0]
		PSI_x = np.hstack([data_edmd["PSI_x"], PSI_x0])
		PSI_y = np.hstack([data_edmd["PSI_y"], PSI_y0])

		G_matrix = (PSI_x.conj().T @ PSI_x) / data_edmd["M2"]
		A_matrix = (PSI_x.conj().T @ PSI_y) / data_edmd["M2"]
		L_matrix = (PSI_y.conj().T @ PSI_y) / data_edmd["M2"]

	elif dmd == "non-linear":
		data_edmd = scipy.io.loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
		N = data_edmd["N"][0, 0]
		PSI_x = data_edmd["PSI_x"]
		PSI_y = data_edmd["PSI_y"]

		G_matrix = (PSI_x.conj().T @ PSI_x) / data_edmd["M2"]
		A_matrix = (PSI_x.conj().T @ PSI_y) / data_edmd["M2"]
		L_matrix = (PSI_y.conj().T @ PSI_y) / data_edmd["M2"]

	else:   # pre-computed
		data = scipy.io.loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
		G_matrix = data["G_matrix"]
		A_matrix = data["A_matrix"]
		L_matrix = data["L_matrix"]
		N = data["N"][0, 0]
		PSI_x = data["PSI_x"]

	return G_matrix, A_matrix, L_matrix, N, PSI_x


def main():
	G_matrix, A_matrix, L_matrix, N, PSI_x = get_dict(dmd="non-linear")

	x_pts = np.arange(-1.5, 1.55, 0.05)
	y_pts = np.arange(-1.5, 1.55, 0.05)
	X, Y = np.meshgrid(x_pts, y_pts)
	z_pts = X + 1j * Y
	z_pts = z_pts.flatten()

	RES = koop_pseudo_spec(G_matrix, A_matrix, L_matrix, z_pts, parallel=False)[0]
	RES = RES.reshape(len(y_pts), len(x_pts))

	D, V = np.linalg.eig(np.linalg.inv(G_matrix) @ A_matrix)
	# E = np.diag(D)

	plot_pseudospectra(D, RES, X, Y, x_pts, y_pts)

	# N = data["N"][0, 0]
	RES2 = np.zeros(N)
	for j in range(N):
		M = L_matrix - D[j] * A_matrix.conj().T - D[j].conj() * A_matrix + np.abs(D[j]) ** 2 * G_matrix
		num = np.sqrt(V[:, j].conj().T @ M @ V[:, j])
		den = np.sqrt(V[:, j].conj().T @ G_matrix @ V[:, j])
		RES2[j] = abs(num / den)

	# I = np.argsort(RES2, axis=0)
	I = np.argsort(RES2)
	RES_p = RES2[I]
	# RES_p = np.take_along_axis(RES2, I, axis=0)

	plot_eig_res(D, RES2)

	evec_x = PSI_x @ V[:, I]
	lam = D[I]
	t1 = 0.967585567481353 + 0.252543401421919j  # TODO: where does this come form?
	t1 = lam[abs(lam - t1) == min(abs(lam - t1))]  # TODO: re centering so min(abs(lam-t1)) = 0
	# TODO: what append if the equality has more than one output, t1 becomes longer than 1
	# TODO: see np.argmin => this gives a number with more precison
	# >>> lam[abs(lam - t1) == min(abs(lam - t1))]
	# array([0.96758562+0.25254341j])
	# >>> lam[np.argmin(abs(lam - t1))]
	# (0.9675856210485554+0.2525434147325198j)   # type is numpy.complex128

	lam1 = np.zeros(100)
	ang1 = np.zeros(100)
	res1 = np.zeros(100)

	for j in range(100):
		# find the indices of eigenvalues close to t1^j    (GPT???)
		I2 = np.where(np.abs(lam - t1 ** (j + 1)) < 0.001)[0]
		# TODO: max(0.001, 0) really useful, why is max(lam1) multiplied by 0 ?????
		# TODO: look at np.nonzero

		# check if only one eigenvalue was found
		if len(I2) == 1:
			# compute the error between the eigenspaces
			b1 = evec_x[:, np.abs(lam - t1) < 0.001] ** (j + 1)
			b2 = evec_x[:, I2]
			# TODO: fix invalid value incountered in arccos
			ang1[j] = np.arccos(np.abs(b1.conj().T @ b2 / (np.linalg.norm(b1, 2) * np.linalg.norm(b2, 2))))
			# compute the error between the eigenvalues
			lam1[j] = np.abs(lam[I2] - t1 ** (j + 1))
			res1[j] = RES2[I[I2]]
		else:
			break  # TODO: why break the loop, why not continue?

	ang1[0], lam1[0], res1[0] = 0, 0, 0

	plot_error(lam1, ang1, res1)

	# for ind2 it is the same as the one to perform the computation on the data file
	# TODO make this either read from file or be defined before since I'll be using the raw data
	m1 = 500
	m2 = 1000
	ind1 = np.arange(0, m1) + 6000    # slicing in matlab include the last item
	ind2 = np.arange(0, m2) + (m1 + 6000) + 500

	logger.info("Loading raw data ..")
	raw_file = loadmat("D:\\PythonProjects\\ai4er\\ResDMDpy\\rdp\\examples\\Cylinder_data.mat")
	logger.info("Done!")
	raw_data = raw_file["DATA"]
	obst_r = raw_file["obst_r"][0, 0]
	obst_x = raw_file["obst_x"][0, 0]
	obst_y = raw_file["obst_y"][0, 0]
	x = raw_file["x"]
	y = raw_file["y"]

	# the issue here is, I create an array which going to become 2D later, i.e change of type
	# and numpy as not happy.
	# I can either preset the size of the array or use a list
	# TODO: also maybe preparing the array to the correct size of the power we are going for
	# TODO: as it unnecessary to create more never used dimenesion.
	# TODO: using enumerate would be a solution.

	# TODO: maybe use dict here, so that the key is the power
	contour_1 = np.zeros((21, 21))  # this works
	contour_2 = [0] * 21            # This works

	xi = np.linalg.pinv(V) @ np.linalg.pinv(PSI_x) @ raw_data[:(raw_data.shape[0] // 2), ind2].T

	for power in tqdm([1, 2, 20]):
		lambda_ = t1 ** power
		idd = np.argmin(np.abs(D - lambda_))    # TODO: check this, seem good but return Number, and np.where an array
		# TODO: matlab returns a Number
		tt = np.linalg.norm(PSI_x @ V[:, idd]) / np.sqrt(m2)
		xi = np.linalg.pinv(V) @ np.linalg.pinv(PSI_x) @ raw_data[:(raw_data.shape[0] // 2), ind2].T
		xi = xi[idd, :]
		xi = (-1j * xi.reshape(100, 400) * tt).T

		h = plt.figure()
		plt.subplot(2, 1, 1)
		d = 2 * obst_r
		contour_1[power] = np.linspace(np.min(np.real(xi)), np.max(np.real(xi)), 21)
		plt.contourf(
			(x - obst_x) / d,
			(y - obst_y) / d,
			np.real(xi),
			contour_1[power],
		)
		plt.colorbar()    # TODO: fix this
		# plt.axis('equal')
		plt.fill(
			obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) / d,
			obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) / d,
			"r"
			# [200, 200, 200] / 255, edgecolor='none'
		)
		plt.xlim([-2, np.max((x - obst_x) / d)])        # TODO: check
		T = f"Mode {power} (real part)"
		plt.title(T, fontsize=16)       # TODO: check
		plt.box(True)
		plt.data_aspect_ratio = [1, 1, 1]
		plt.plot_box_aspect_ratio = [8.25, 2.25, 1]
		plt.clim([np.min(contour_1[power]), np.max(contour_1[power])])   # TODO: fix
		plt.tight_layout()      # TODO: check

		plt.subplot(2, 1, 2)
		d = 2 * obst_r
		contour_2[power] = np.linspace(0, np.max(np.abs(xi)), 21)
		plt.contourf(
			(x - obst_x) / d,
			(y - obst_y) / d,
			np.abs(xi),
			contour_2[power],
		)
		plt.colorbar()
		plt.axis('equal')
		plt.tight_layout()
		plt.fill(
			obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) / d,
			obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) / d,
			"r"
			# [200, 200, 200] / 255, edgecolor='none'
		)
		plt.xlim([-2, np.max((x - obst_x) / d)])
		T = f"Mode {power} (absolute value)"
		plt.title(T, fontsize=16)
		plt.box(True)
		plt.data_aspect_ratio = [1, 1, 1]
		plt.plot_box_aspect_ratio = [8.25, 2.25, 1]
		plt.clim([np.min(contour_2[power]), np.max(contour_2[power])])
		# h.set_position([360.0000, 262.3333, 560.0000, 355.6667])

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
