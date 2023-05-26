from __future__ import annotations

import os

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
def guarantee_file_dl(filepath: str):
	directory = os.path.dirname(filepath)
	filename = os.path.basename(filepath)

	if not os.path.exists(filepath):
		url = get_url(filename)
		downloader(url, root=directory)
