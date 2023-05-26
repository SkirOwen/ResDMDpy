import numpy as np
import scipy

from rdp import kernel_resdmd


def gen_from_file(
		filepath: str,
		n: int = 200,
		m1: int = 500,
		m2: int = 1000,
		use_dmd: int = 1,  # TODO: change this to a bool maybe
) -> tuple[np.ndarray, np.ndarray]:
	""""""
	data = scipy.io.loadmat(filepath)
	ind1 = np.arange(0, m1) + 6000    # slicing in matlab include the last item
	ind2 = np.arange(0, m2) + (m1 + 6000) + 500
	# TODO: this slicing returns the right thing, but I think there is a nicer way
	# I had to do a +1 for ind2

	if use_dmd != 1:
		psi_x, psi_y, _ = kernel_resdmd(
			data[:, ind1],
			data[:, ind1 + 1],
			data[:, ind2],
			data[:, ind2 + 1],
			n=n,
			cut_off=False,
		)
	else:
		_, s, vh = np.linalg.svd(data[:, ind1].T / np.sqrt(m1), full_matrices=False)
		# the vh returned by svd is the hermitian of the v in matlab
		psi_x = data[:, ind2].T @ vh[:n, :].T @ np.diag(1 / s[:n])
		# TODO: check if having vh instead of v is important
		psi_y = data[:, ind2 + 1].T @ vh[:n, :].T @ np.diag(1 / s[:n])

	return psi_x, psi_y
		
		