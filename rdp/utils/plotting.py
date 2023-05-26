from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# import cmocean as cm


def plot_pseudospectra(D, RES, X, Y, x_pts, y_pts):
	# TODO: fix the following plot code
	# Define v
	v = np.power(10, np.arange(-2, 0.21, 0.2))

	# Create figure and plot
	fig, ax = plt.subplots()
	contourf = ax.contourf(
		X,
		Y,
		np.log10(np.real(RES)).reshape(len(y_pts), len(x_pts)),     # RES is already a real no?
		np.log10(v),
		# cmap=cm.cm.thermal
	)
	fig.colorbar(contourf)
	# cbh.set_ticks(np.log10([0.005, 0.01, 0.1, 1]))
	# cbh.ax.set_yticklabels([0, 0.01, 0.1, 1])
	# plt.clim(np.log10(0.01), np.log10(1))
	ax.set_ylim(y_pts[0], y_pts[-1])
	ax.set_xlim(x_pts[0], x_pts[-1])
	ax.plot(np.real(D), np.imag(D), '.r')
	ax.set_xlabel('Real axis')
	ax.set_ylabel('Imaginary axis')
	ax.set_title('Title')
	fig.tight_layout()
	# TODO: I would like less padding on the right
	plt.show()


def plot_eig_res(D, RES2):
	fig, ax = plt.subplots()
	ax.semilogy(np.angle(D), RES2, ".r", markersize=5)
	ax.set_xlabel('Real axis')
	ax.set_ylabel(r"Angle")  # TODO: check this
	ax.set_title('Title')
	plt.show()


def plot_error(lam1, ang1, res1):
	fig, ax = plt.subplots()
	# TODO: look to change to a scatter log
	ax.semilogy(lam1, "*-")
	ax.semilogy(res1, "d-")
	ax.semilogy(np.real(ang1), "s-")
	plt.legend(
		[
			r'$|\lambda_{j}-\lambda_{1}^{j}|$',
			r'$\mathrm{res}(\lambda_{j}, g_{j})$',
			r'$\mathrm{eigenspace error}$'
		],
		fontsize=22
	)
	plt.ylim([10 ** (-14), 1])
	# plt.yticks(10 ** (np.arange(14, 2, 2))) TODO: fix this
	plt.xlim([0, 100])
	plt.show()
