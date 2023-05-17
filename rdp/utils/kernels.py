import numpy as np
from tqdm import tqdm


def kernel_resdmd(xa, ya, xb, yb, n, cut_off, y2=None, sketch=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	""""""
	cut_off = 1e-12 if cut_off else 0

	m1 = xa.shape[1]
	m2 = xb.shape[1]
	d = np.mean(np.vecnorm(xa - np.mean(xa.conj().T).conj().T))

	s: int = np.max(np.ceil(5 * np.sqrt(m1 + m2) * np.log(m1 + m2)), 5000)
	kernel_f = lambda x, y: np.exp(-np.vecnorm(x - y) / d)

	if sketch:
		# TODO: check if it is log10, log2, or logn
		# TODO: I'm pretty sure this shouldn't be a lamdba

		z = np.sqrt(2 / d**2) * np.random.randn([xa.shape[0], s])
		th = 2 * np.pi * np.random.randn(s, 1)

		psi_xa = np.sqrt(2 / s) * np.cos(th + z.conj().T * xa)
		psi_ya = np.sqrt(2 / s) * np.cos(th + z.conj().T * ya)

		g1 = psi_xa.conj().T * psi_xa
		a1 = psi_ya.conj().T * psi_xa
		g1 = np.max(g1, np.zeros(*g1.shape))
		# this should be equivalent to
		g1[g1 < 0] = 0
		# ditto
		a1[a1 < 0] = 0
	else:
		g1 = np.zeros(m1, m1)
		a1 = np.zeros(m1, m1)

		for i in tqdm(range(m1)):
			# TODO: I think I can vectorise this loop
			g1[i, :] = kernel_f(xa[:, i], xa)
			a1[i, :] = kernel_f(ya[:, i], xa)

	# TODO: have a function for this
	d0, u = np.linalg.eig(g1 + np.norm(g1, 2) * cut_off * np.eye(*g1.shape))
	d0[d0 < cut_off] = 0

	# TODO: something to do with ma here
	sig = np.sqrt(d0)
	sig_dag = np.zeros(sig.shape)
	sig_dag[sig > 0] = 1 / sig[sig > 0]

	k_hat = sig_dag @ u.conj().T @ a1 @ u @ sig_dag
	d1, u1 = np.linalg.eig(k_hat)

	# TODO: what is this?	
	I = np.where(abs(np.diag(d1)) > cut_off)

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
		psi_x = np.zeros(m2, m1)
		psi_y = np.zeros(m2, m1)
		if y2 is not None:
			psi_y2 = np.zeros(m2, m1)

		# I think I can vectorise this
		for i in tqdm(range(m2)):
			psi_x[i, :] = kernel_f(xb[:, i], xa)
			psi_y[i, :] = kernel_f(yb[:, i], xa)

			if y2 is not None:
				psi_y2[i, :] = kernel_f(xb[:, i], xa)

		psi_x = psi_x @ p  # test if @= works
		psi_y = psi_y @ p

		if y2 is not None:
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
