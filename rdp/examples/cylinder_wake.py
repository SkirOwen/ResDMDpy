from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from typing import Literal, Sequence
from tqdm import tqdm

from rdp import logger
from rdp.koopman import koop_pseudo_spec
from rdp.utils.plotting import plot_pseudospectra, plot_eig_res, plot_error, plot_koop_mode
from rdp.utils.file_ops import save_data
from rdp.utils.directories import get_koopmode_dir

from rdp.examples import load_cylinder_data, load_cylinder_dmd, load_cylinder_edmd

from rdp.utils.directories import get_example_dir
from rdp.data.basis_func import gen_from_file

plt.rcParams['text.usetex'] = True


def get_dict(
		dmd: Literal["linear", "combined", "pre-computed", "non-linear", "generate"],
		n: int = 200,
		m1: int = 500,
		m2: int = 1000,
		linear_dict: bool = True,
):
	"""
	Get the dictionary (or generate) for the koopman mode from the cylinder data.

	Parameters
	----------
	dmd : {'linear', 'combined', 'pre-computed', 'generate'}
	n : int, optional
		Size of the computed dictionary. The default is 200.
	m1 : int, optional
		Number of snapshots to compute the basis. The default is 500.
	m2 : int, optional
		Number of snapshots used for ResDMD matrices. The default is 1000.
	linear_dict : bool, optional
		If True, compute the linear dictionary. Otherwise (default), compute the non-linear.

	Returns
	-------
	G_matrix : ndarray
	A_matrix : ndarray
	L_matrix : ndarray
	n : int
		Size of the computed dictionary
	psi_x : ndarray
	"""
	if dmd == "linear":
		data = load_cylinder_dmd()
		G_matrix = data["G_matrix"]
		A_matrix = data["A_matrix"]
		L_matrix = data["L_matrix"]
		n = data["N"][0, 0]
		psi_x = data["PSI_x"]

	elif dmd == "combined":
		data_dmd = load_cylinder_dmd()
		psi_x0 = data_dmd["PSI_x"]
		psi_y0 = data_dmd["PSI_y"]
		data_edmd = load_cylinder_edmd()
		n = 2 * data_edmd["N"][0, 0]
		psi_x = np.hstack([data_edmd["PSI_x"], psi_x0])
		psi_y = np.hstack([data_edmd["PSI_y"], psi_y0])

		G_matrix = (psi_x.conj().T @ psi_x) / data_edmd["M2"]
		A_matrix = (psi_x.conj().T @ psi_y) / data_edmd["M2"]
		L_matrix = (psi_y.conj().T @ psi_y) / data_edmd["M2"]

	elif dmd == "non-linear":
		data_edmd = load_cylinder_edmd()
		n = data_edmd["N"][0, 0]
		psi_x = data_edmd["PSI_x"]
		psi_y = data_edmd["PSI_y"]

		G_matrix = (psi_x.conj().T @ psi_x) / data_edmd["M2"]
		A_matrix = (psi_x.conj().T @ psi_y) / data_edmd["M2"]
		L_matrix = (psi_y.conj().T @ psi_y) / data_edmd["M2"]

	elif dmd == "generate":
		# data_raw = load_cylinder_data()
		filepath = os.path.join(get_example_dir(), "Cylinder_data.mat")
		psi_x, psi_y = gen_from_file(filepath, n=n, m1=m1, m2=m2, linear_dict=linear_dict)

		G_matrix = (psi_x.conj().T @ psi_x) / m2
		A_matrix = (psi_x.conj().T @ psi_y) / m2
		L_matrix = (psi_y.conj().T @ psi_y) / m2

	else:   # pre-computed
		data = load_cylinder_edmd()
		G_matrix = data["G_matrix"]
		A_matrix = data["A_matrix"]
		L_matrix = data["L_matrix"]
		n = data["N"][0, 0]
		psi_x = data["PSI_x"]

	return G_matrix, A_matrix, L_matrix, n, psi_x


def gen_koop_modes(
		V: np.ndarray,
		PSI_x: np.ndarray,
		t1: int,
		D: np.ndarray,
		powers: Sequence,
		plot: bool = True,
		filename: str = "cylinder_xi_v3_p.h5",
) -> tuple:
	# for ind2 it is the same as the one to perform the computation on the data file
	# TODO make this either read from file or be defined before since I'll be using the raw data
	# TODO: m1, m2, ind1, ind2 are stored in the mat file
	m1 = 500
	m2 = 1000
	ind1 = np.arange(0, m1) + 6000    # slicing in matlab include the last item
	ind2 = np.arange(0, m2) + (m1 + 6000) + 500

	logger.info("Loading raw data ...")
	raw_file = load_cylinder_data()
	logger.info("Done!")
	raw_data = raw_file["DATA"]
	obst_r = raw_file["obst_r"][0, 0]
	obst_x = raw_file["obst_x"][0, 0]
	obst_y = raw_file["obst_y"][0, 0]
	x = raw_file["x"]
	y = raw_file["y"]

	metadata = {
		"powers": powers,
		"obst_x": obst_x,
		"obst_y": obst_y,
		"obst_r": obst_r,
		"xy": x.shape,
	}

	# TODO: what does it mean to have a non-int power?

	# TODO: what is XI?
	xi = np.linalg.pinv(V) @ np.linalg.pinv(PSI_x) @ raw_data[:(raw_data.shape[0] // 2), ind2].T
	all_xi = []

	for i, power in enumerate(tqdm(powers, desc="Calculating koopman modes")):
		# TODO: make this a function
		lambda_ = t1 ** power
		idd = np.argmin(np.abs(D - lambda_))    # TODO: check this, seem good but return Number, and np.where an array
		# TODO: matlab returns a Number
		# eigval = D[idd]
		tt = np.linalg.norm(PSI_x @ V[:, idd]) / np.sqrt(m2)

		xi_ = xi[idd, :]
		xi_ = (-1j * xi_.reshape(100, 400) * tt).T

		if plot:
			plot_koop_mode(xi_, power, obst_x, obst_y, obst_r, x, y)
		all_xi.append(xi_)

	save_data(
		os.path.join(get_koopmode_dir(), filename),
		np.array(all_xi),
		metadata,
		backend="h5"
	)
	save_mode_png("xi_", xi_)

	# x.shape -> 400, 100
	# cause array are (y, x) in context of img
	# but for a meshgrid
	# it should be meshgrid(arange(100) arange(400)) starting at 1 to 400
	return xi, metadata


def save_mode_png(filename, xi_) -> None:
	"""Save the xi of a koopman mode to a png"""
	image_data = np.real(xi_.T)
	# TODO: also do that for the abs

	# Normalize the image data to the range [0, 255]
	normalized_data = ((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))) * 255
	normalized_data = normalized_data.astype(np.uint8)

	# Create a PIL Image object from the normalized image data
	image = Image.fromarray(normalized_data, mode="L")  # "L" mode represents grayscale images

	# Save the image as a PNG file
	image.save(f"{filename}.png")


def run(powers: list, plot: bool = True, filename: str = "cylinder_xi_v3_p.h5", dmd: str = "non-linear"):
	G_matrix, A_matrix, L_matrix, N, PSI_x = get_dict(dmd=dmd)

	x_pts = np.arange(-1.5, 1.55, 0.05)
	y_pts = np.arange(-1.5, 1.55, 0.05)
	X, Y = np.meshgrid(x_pts, y_pts)
	z_pts = X + 1j * Y
	z_pts = z_pts.flatten()

	RES = koop_pseudo_spec(G_matrix, A_matrix, L_matrix, z_pts, parallel=False)[0]
	RES = RES.reshape(len(y_pts), len(x_pts))

	D, V = np.linalg.eig(np.linalg.inv(G_matrix) @ A_matrix)
	# E = np.diag(D)

	if plot:
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

	if plot:
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
		# find the indices of eigenvalues close to t1^j
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

	if plot:
		plot_error(lam1, ang1, res1)
	# Energy L2 norm
	powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 18, 20] if powers is None else powers

	gen_koop_modes(V, PSI_x, t1, D, powers, plot, filename)


def main():
	plt.rcParams['text.usetex'] = True
	powers = [i for i in range(1, 51)]
	run(powers, plot=True, filename="cylinder_xi_1_50.h5", dmd="linear")


if __name__ == "__main__":
	main()
