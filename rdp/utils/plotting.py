from __future__ import annotations

import cmocean as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


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
		cmap=cm.cm.amp
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


def plot_koop_mode(
		xi_: np.ndarray,
		power: int,
		obst_x: float,
		obst_y: float,
		obst_r: float,
		x,
		y,
		cmap=cm.cm.ice,
		filename: None | str = None,
) -> None:
	d = 2 * obst_r

	contourp_1 = np.linspace(np.min(np.real(xi_)), np.max(np.real(xi_)), 21)
	contourp_2 = np.linspace(0, np.max(np.abs(xi_)), 21)

	# 	# Everything is normalized to the diameter
	# 	# TODO: would that not perturb the AI, as real data cannot be normalised to the size of the obstacle
	# 	# What if the data was normalized using a predicted size?
	# 	# What if normalising the data to the size of the tidal turbine
	fig, axs = plt.subplots(2, 1)

	c1 = axs[0].contourf(
		(x - obst_x) / d,
		(y - obst_y) / d,
		np.real(xi_),
		contourp_1,
		cmap=cmap,
		vmin=np.min(contourp_1),
		vmax=np.max(contourp_1)
	)
	cbar1 = fig.colorbar(c1, ax=axs[0])
	cbar1.formatter.set_powerlimits((-2, 2))  # Display colorbar tick labels in scientific notation
	cbar1.update_ticks()

	axs[0].fill(
		obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) / d,
		obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) / d,
		"r"
	)
	axs[0].set_xlim([-2, np.max((x - obst_x) / d)])
	T = f"Mode {power} (real part)"
	axs[0].set_title(T, fontsize=16)
	axs[0].set_frame_on(True)  # Enable the frame around the subplot
	axs[0].set_aspect('equal')  # Set the aspect ratio to equal

	c2 = axs[1].contourf(
			(x - obst_x) / d,
			(y - obst_y) / d,
			np.abs(xi_),
			contourp_2,
			cmap=cmap,
			vmin=np.min(contourp_2),
			vmax=np.max(contourp_2)
	)
	cbar2 = fig.colorbar(c2, ax=axs[1])
	cbar2.formatter.set_powerlimits((-2, 2))  # Display colorbar tick labels in scientific notation
	cbar2.update_ticks()

	axs[1].fill(
			obst_r * np.cos(np.arange(0, 2 * np.pi, 0.01)) / d,
			obst_r * np.sin(np.arange(0, 2 * np.pi, 0.01)) / d,
			"r"
	)
	axs[1].set_xlim([-2, np.max((x - obst_x) / d)])
	T = f"Mode {power} (absolute value)"
	axs[1].set_title(T, fontsize=16)
	axs[1].set_frame_on(True)  # Enable the frame around the subplot
	axs[1].set_aspect('equal')  # Set the aspect ratio to equal

	plt.tight_layout()

	if filename is not None:
		plt.savefig(filename, dpi=200)
	plt.show()
