from __future__ import annotations

import numpy as np
from tqdm import tqdm


def kernel_f(x, y, d):
	return np.exp(-np.linalg.norm(x - y) / d)


def kernel_resdmd(
		xa: np.ndarray,
		ya: np.ndarray,
		xb: np.ndarray,
		yb: np.ndarray,
		n: int,
		cut_off: bool,
		y2=None,
		sketch: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	""""""
	cut_off = 1e-12 if cut_off else 0

	m1 = xa.shape[1]
	m2 = xb.shape[1]
	# TODO: why mean(xa.H).H ?
	# vecnorm is the norm euclidian norm
	d = np.mean(np.linalg.norm(xa - np.mean(xa.conj().T).conj().T))

	s: int = max(np.ceil(5 * np.sqrt(m1 + m2) * np.log(m1 + m2)), 5000)

	if sketch:
		z = np.sqrt(2 / d**2) * np.random.randn([xa.shape[0], s])
		th = 2 * np.pi * np.random.randn(s, 1)

		psi_xa = np.sqrt(2 / s) * np.cos(th + z.conj().T * xa)
		psi_ya = np.sqrt(2 / s) * np.cos(th + z.conj().T * ya)

		g1 = psi_xa.conj().T * psi_xa
		a1 = psi_ya.conj().T * psi_xa
		g1 = np.max(g1, np.zeros(*g1.shape))
		# TODO: this should be equivalent to
		g1[g1 < 0] = 0
		# ditto
		a1[a1 < 0] = 0
	else:
		g1 = np.zeros((m1, m1))
		a1 = np.zeros((m1, m1))

		for i in tqdm(range(m1)):
			# TODO: I think I can vectorise this loop
			g1[i, :] = kernel_f(xa[:, i][:, np.newaxis], xa, d)
			a1[i, :] = kernel_f(ya[:, i][:, np.newaxis], xa, d)

	# TODO: have a function for this
	d0, u = np.linalg.eig(g1 + np.linalg.norm(g1, 2) * cut_off * np.eye(*g1.shape))
	d0[d0 < cut_off] = 0

	# TODO: something to do with ma here
	sig = np.sqrt(np.diag(d0))
	sig_dag = np.zeros(sig.shape)
	sig_dag[sig > 0] = 1 / sig[sig > 0]

	k_hat = sig_dag @ u.conj().T @ a1 @ u @ sig_dag
	d1, u1 = np.linalg.eig(k_hat)

	# TODO: what is this?
	# TODO: look at non-zero
	I = np.where(abs(d1) > cut_off)[0]

	if len(I) > n:
		_, I = np.sort(abs(np.diag(d1)), order="desc")
	else:
		n = len(I)

	p, _, _ = np.linalg.svd(u1[:, I[:n]], full_matrices=False)
	p = u @ sig_dag @ p

	if sketch:
		psi_xb = np.sqrt(2 / s) * np.cos(th + z.conj().T @ xb)
		psi_yb = np.sqrt(2 / s) * np.cos(th + z.conj().T @ yb)
		psi_x = psi_xb.conj().T @ psi_xa
		psi_y = psi_yb.conj().T @ psi_xa

		psi_x = np.max(psi_x, 0) @ p
		psi_y = np.max(psi_y, 0) @ p

		if y2 is not None:
			psi_y2b = np.sqrt(2 / s) * np.cos(th + z.conj().T * y2)
			psi_y2 = psi_y2b.conj().T @ psi_xa
			psi_y2 = np.max(psi_y2, 0) @ p

	else:
		psi_x = np.zeros((m2, m1))
		psi_y = np.zeros((m2, m1))
		# if y2 is not None:
		# 	psi_y2 = np.zeros(m2, m1)

		# I think I can vectorise this
		for i in tqdm(range(m2)):
			psi_x[i, :] = kernel_f(xb[:, i][:, np.newaxis], xa, d)
			psi_y[i, :] = kernel_f(yb[:, i][:, np.newaxis], xa, d)
		#
		# 	if y2 is not None:
		# 		psi_y2[i, :] = kernel_f(xb[:, i], xa)

		psi_x = kernel_f(xb.T, xa, d)
		psi_y = kernel_f(yb.T, xa, d)
		# if y2 is not None:
		# 	psi_y2 = kernel_f(xb.T, xa)

		psi_x = psi_x @ p  # @= (in-place) is not yet supported
		psi_y = psi_y @ p

		if y2 is not None:
			psi_y2 = kernel_f(xb.T, xa, d)
			psi_y2 = psi_y2 @ p

	return psi_x, psi_y, psi_y2


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
