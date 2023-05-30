from __future__ import annotations

import numpy as np


def guarantee_hermitian(mat: np.ndarray):
	return (mat + mat.conj().T) / 2
