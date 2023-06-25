import numpy as np
import scipy

from rdp import logger
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
	# this slicing returns the right thing, but I think there is a nicer way!
	# Since MATLAB index start at 1, but Python start at zero,
	# these are off by one, but it is expected .

	if linear_dict == 1:
		# Linear dictionary
		_, s, vh = np.linalg.svd(data["DATA"][:, ind1].T / np.sqrt(m1), full_matrices=False)
		# during the svd, at least two columns of vh (2, and 4) have their sign flipped compare to matlab
		# This is fine as the span from an eigen vector does not change if the sign changes
		# the vh returned by svd is the hermitian of the v in matlab
		psi_x = data["DATA"][:, ind2].T     @ vh.conj().T[:, :n] @ np.diag(1 / s[:n])
		psi_y = data["DATA"][:, ind2 + 1].T @ vh.conj().T[:, :n] @ np.diag(1 / s[:n])
	else:
		psi_x, psi_y, _ = kernel_resdmd(
			xa=data["DATA"][:, ind1],
			ya=data["DATA"][:, ind1 + 1],
			xb=data["DATA"][:, ind2],
			yb=data["DATA"][:, ind2 + 1],
			n=n,
			cut_off=False,
		)

	return psi_x, psi_y


def main():
	logger.setLevel("DEBUG")
	gen_from_file("./rdp/examples/Cylinder_data.mat", n=8, linear_dict=False)


if __name__ == "__main__":
	main()
