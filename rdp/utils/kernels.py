import numpy as np


def unitary_kernel(z: np.ndarray, epsilon: float) -> tuple:
	m = len(z)
	sigma = -np.conj(z) / (1 + epsilon * np.conj(z))
	V1 = np.array([sigma ** i for i in range(m)]).T
	V2 = np.array([z ** i for i in range(m)]).T
	rhs = np.eye(m, 1)
	# apparently gives the same results as
	# np.linalg.inv(V1.T) @ rhs, but is somewhat faster for large matrices
	c = np.linalg.solve(V1.T, rhs)
	d = np.linalg.solve(V2.T, rhs)
	return c, d
