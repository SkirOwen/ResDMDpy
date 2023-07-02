from __future__ import annotations

import numpy as np
import scipy

from rdp import logger


def loadmat(filepath: str) -> dict:
	try:
		logger.debug(f"Loading {filepath} with scipy.")
		data = scipy.io.loadmat(filepath)
	except NotImplementedError:
		logger.debug("scipy did not work trying with h5py")
		import h5py
		data = {}
		f = h5py.File(filepath)
		for k, v in f.items():
			data[k] = np.array(v).swapaxes(0, 1)
	logger.debug("File loaded!")
	return data
