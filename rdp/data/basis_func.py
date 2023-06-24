import numpy as np
import scipy

from rdp import kernel_resdmd
from rdp.utils.mat_loader import loadmat


def gen_from_file(
		filepath: str,
		n: int = 200,
		m1: int = 500,
		m2: int = 1000,
		linear_dict: bool = False,  # eq to use_dmd = 0
) -> tuple[np.ndarray, np.ndarray]:
	""""""
	data = loadmat(filepath)
	ind1 = np.arange(0, m1) + 6000    # slicing in matlab include the last item
	ind2 = np.arange(0, m2) + (m1 + 6000) + 500
	# TODO: this slicing returns the right thing, but I think there is a nicer way
	# I had to do a +1 for ind2

	if linear_dict == 1:
		# Linear dictionary
		_, s, vh = np.linalg.svd(data["DATA"][:, ind1].T / np.sqrt(m1), full_matrices=False)
		# TODO: during the svd, at least two columns of vh (2, and 4) have their sign flipped compare to matlab
		# the vh returned by svd is the hermitian of the v in matlab
		# TODO: check if having vh instead of v is important
		psi_x = data["DATA"][:, ind2].T     @ vh.conj().T[:, :n] @ np.diag(1 / s[:n])
		psi_y = data["DATA"][:, ind2 + 1].T @ vh.conj().T[:, :n] @ np.diag(1 / s[:n])
	else:
		psi_x, psi_y, _ = kernel_resdmd(
			data["DATA"][:, ind1],
			data["DATA"][:, ind1 + 1],
			data["DATA"][:, ind2],
			data["DATA"][:, ind2 + 1],
			n=n,
			cut_off=False,
		)

	return psi_x, psi_y
		
		