import numpy as np
import scipy


def loadmat(filepath: str) -> dict:
	try:
		data = scipy.io.loadmat(filepath)
	except NotImplementedError:
		import h5py
		data = {}
		f = h5py.File(filepath)
		for k, v in f.items():
			data[k] = np.array(v).swapaxes(0, 1)

	return data
