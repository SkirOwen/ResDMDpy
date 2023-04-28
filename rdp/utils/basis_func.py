import numpy as np
import scipy

from rdp import kernel_resdmd


def gen_from_file(
	filepath: str,
	n: int = 200, 
	m1: int = 500, 
	m2: int = 1000,
	use_dmd: int = 1  # TODO: change this to a bool maybe
):
	""""""
	data = scipy.io.loadmat(filepath)
	ind1 = np.arange(0, m1) + 6000
	ind2 = np.arange(0, m2) + ind1[-1] + 500
	
	if use_dmd != 1:
		psi_x, psi_y = kernel_resdmd(
			data[:, ind1],
			data[:, ind1 + 1],
			data[:, ind2],
			data[:, ind2 + 1],
			n=n,
			cut_off=True,		
		)
	else:
		_, S, V = np.linalg.svd(data[:, ind1].T / np.sqrt(m1), "econ")
		psi_x = data[:, ind2].T @ V[:, 0:n] * np.diag()
		
		