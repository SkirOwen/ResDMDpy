from __future__ import annotations

import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import Literal

from rdp.koopman import koop_pseudo_spec

plt.rcParams['text.usetex'] = True


def plot_pseudospectra(D, RES, X, Y, x_pts, y_pts):
	# TODO: fix the following plot code
	# TODO: look at the 0,0 red dot on the plot
	# Define v
	v = np.power(10, np.arange(-2, 0.21, 0.2))

	# Create figure and plot
	fig, ax = plt.subplots()
	contourf = ax.contourf(
		X,
		Y,
		np.log10(np.real(RES)).reshape(len(y_pts), len(x_pts)),     # RES is already a real no?
		np.log10(v)
	)
	fig.colorbar(contourf)
	# cbh.set_ticks(np.log10([0.005, 0.01, 0.1, 1]))
	# cbh.ax.set_yticklabels([0, 0.01, 0.1, 1])
	# plt.clim(np.log10(0.01), np.log10(1))
	ax.set_ylim(y_pts[0], y_pts[-1])
	ax.set_xlim(x_pts[0], x_pts[-1])
	ax.set_aspect('equal')
	ax.plot(np.real(D), np.imag(D), '.r')
	ax.set_xlabel('Real axis')
	ax.set_ylabel('Imaginary axis')
	ax.set_title('Title')
	plt.show()


def plot_eig_res(D, RES2):
	fig, ax = plt.subplots()
	ax.semilogy(np.angle(D), RES2, ".r", markersize=5)
	plt.show()


def plot_error(lam1, ang1, res1):
	fig, ax = plt.subplots()
	ax.semilogy(lam1, "*-")
	ax.semilogy(res1, "d-")
	ax.semilogy(np.real(ang1), "s-")
	plt.legend(
		[r'$|\lambda_{j}-\lambda_{1}^{j}|$', r'$\mathrm{res}(\lambda_{j}, g_{j})$', r'eigenspace error'],
		fontsize=22
	)
	plt.ylim([10 ** (-14), 1])
	plt.yticks(10 ** (-np.arange(14, -1, 2)))
	plt.xlim([0, 100])
	plt.show()


def get_dict(dmd: Literal["linear", "combined", "pre-computed", "non-linear"]):
	if dmd == "linear":
		data = scipy.io.loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_DMD.mat")
		G_matrix = data["G_matrix"]
		A_matrix = data["A_matrix"]
		L_matrix = data["L_matrix"]
		N = data["N"][0, 0]
		PSI_x = data["PSI_x"]
	elif dmd == "combined":
		data_dmd = scipy.io.loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_DMD.mat")
		PSI_x0 = data_dmd["PSI_x"]
		PSI_y0 = data_dmd["PSI_y"]
		data_edmd = scipy.io.loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
		N = 2 * data_edmd["N"][0, 0]
		PSI_x = np.hstack([data_edmd["PSI_x"], PSI_x0])
		PSI_y = np.hstack([data_edmd["PSI_y"], PSI_y0])

		G_matrix = (PSI_x.conj().T @ PSI_x) / data_edmd["M2"]
		A_matrix = (PSI_x.conj().T @ PSI_y) / data_edmd["M2"]
		L_matrix = (PSI_y.conj().T @ PSI_y) / data_edmd["M2"]
	elif dmd == "non-linear":
		data_edmd = scipy.io.loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
		N = data_edmd["N"][0, 0]
		PSI_x = data_edmd["PSI_x"]
		PSI_y = data_edmd["PSI_y"]

		G_matrix = (PSI_x.conj().T @ PSI_x) / data_edmd["M2"]
		A_matrix = (PSI_x.conj().T @ PSI_y) / data_edmd["M2"]
		L_matrix = (PSI_y.conj().T @ PSI_y) / data_edmd["M2"]

	else:   # pre-computed
		data = scipy.io.loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_EDMD.mat")
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
	# TODO: see np.argmin

	lam1 = np.zeros(100)
	ang1 = np.zeros(100)
	res1 = np.zeros(100)

	for j in range(100):
		# find the indices of eigenvalues close to t1^j    (GPT???)
		I2 = np.where(np.abs(lam - t1 ** (j + 1)) < max(0.001, 0 * max(lam1)))[0]
		# TODO: max(0.001, 0) really useful, why is max(lam1) multiplied by 0 ?????
		# TODO: look at np.nonzero

		# check if only one eigenvalue was found
		if len(I2) == 1:
			# compute the error between the eigenspaces
			b1 = evec_x[:, np.abs(lam - t1) < max(0.001, 0 * max(lam1))] ** (j + 1)
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


if __name__ == "__main__":
	main()
