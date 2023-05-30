from __future__ import annotations

import os
from typing import Literal

import numpy as np
import h5py

from rdp import logger
from rdp.utils.downloader import downloader, get_url


def guarantee_existence(path: str) -> str:
	"""Function to guarantee the existence of a path, and returns its absolute path.

	Parameters
	----------
	path : str
		Path (in str) to guarantee the existence.

	Returns
	-------
	str
		The absolute path.
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	return os.path.abspath(path)


# TODO: should I use pathlib???
def guarantee_file_dl(filepath: str, url: None | list = None) -> None:
	# TODO: handle if url is not list???
	# TODO: checksum
	directory = os.path.dirname(filepath)
	filename = os.path.basename(filepath)

	if not os.path.exists(filepath):
		logger.info(f"{filename} not present.")
		url = get_url(filename) if url is None else url
		logger.info(f"Downloading from {url}")
		downloader(url, root=directory)


def save_np(filename: str, data: np.ndarray, **metadata: dict) -> None:
	"""Save data to a npz file alongside with the metadata"""
	np.savez(filename, data=data, **metadata)


def save_h5(filename: str, data: np.ndarray, **metadata: dict) -> None:
	"""Save data to a h5 file alongside the metadata"""
	with h5py.File(filename, "w") as f:
		f.create_dataset("data", data=data)
		for k, v in metadata.items():
			f.attrs[k] = v


def save_data(filename: str, data: np.ndarray, metadata: dict, backend: Literal["numpy", "h5"]) -> None:
	"""Helper function to save data with metadata to filename with either numpy or h5"""
	if backend == "numpy":
		save_np(filename, data, **metadata)
	elif backend == "h5":
		save_h5(filename, data, **metadata)
