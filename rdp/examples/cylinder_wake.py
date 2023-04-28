import numpy as np
from scipy.io import loadmat

from rdp.koopman import koop_pseudo_spec


def main():
	data = loadmat("G:\\PycharmProjects\\ai4er\\resdmd\\ResDMDpy\\rdp\\examples\\Cylinder_DMD.mat")
	G_matrix = data["G_matrix"]
	A_matrix = data["A_matrix"]
	L_matrix = data["L_matrix"]

	x_pts = np.arange(-1.5, 1.55, 0.05)
	y_pts = x_pts

	# TODO: see np.meshgrid
	z_pts = np.kron(x_pts, np.ones((len(y_pts), 1))) + 1j * np.kron(np.ones((1, len(x_pts))), y_pts.reshape(-1, 1))

	z_pts = z_pts.flatten()

	RES = koop_pseudo_spec(G_matrix, A_matrix, L_matrix, z_pts, parallel=False)[0]

	RES = RES.reshape(len(y_pts), len(x_pts))

	D, V = np.linalg.eig(np.linalg.inv(G_matrix) @ A_matrix)
	E = np.diag(D)

	# TODO: fix the following plot code
	# TODO: look at the 0,0 red dot on the plot
	import matplotlib.pyplot as plt
	from matplotlib.cm import ScalarMappable

	# Define v
	v = np.power(10, np.arange(-2, 0.21, 0.2))

	# Create figure and plot
	fig, ax = plt.subplots()
	contourf = ax.contourf(
		np.real(z_pts.reshape(len(y_pts), len(x_pts))),
		np.imag(z_pts.reshape(len(y_pts), len(x_pts))),
		np.log10(np.real(RES)).reshape(len(y_pts), len(x_pts)),
		np.log10(v)
	)
	fig.colorbar(contourf)
	# cbh.set_ticks(np.log10([0.005, 0.01, 0.1, 1]))
	# cbh.ax.set_yticklabels([0, 0.01, 0.1, 1])
	# plt.clim(np.log10(0.01), np.log10(1))
	ax.set_ylim(y_pts[0], y_pts[-1])
	ax.set_xlim(x_pts[0], x_pts[-1])
	ax.set_aspect('equal')
	ax.plot(np.real(E), np.imag(E), '.r')
	ax.set_xlabel('Real axis')
	ax.set_ylabel('Imaginary axis')
	ax.set_title('Title')
	plt.show()


if __name__ == "__main__":
	main()
