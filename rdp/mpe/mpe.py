from __future__ import annotations

import numpy as np
import scipy

from rdp.utils.linalg_op import guarantee_hermitian


def mpe(g: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	g = guarantee_hermitian(g)
	dg, vg = np.linalg.eig(g)
	g_sqrt = vg @ dg @ vg.conj().T
	g_sqrt_i = vg @ np.sqrt(1 / dg) @ vg.conj().T

	u, _, v = np.linalg.svd(g_sqrt_i @ a.conj().T @ g_sqrt_i)

	mp_d, mp_v, _ = scipy.linalg.schur((v.conj().T @ u.conj().T), output="complex")

	mp_v = g_sqrt_i @ mp_v
	mp_k = g_sqrt_i @ v.conj().T @ u.conj().T @ g_sqrt
	mp_d = np.diag(mp_d)

	return mp_k, mp_v, mp_d
