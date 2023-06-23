import os

from typing import Iterable

from ..utils.mat_loader import loadmat
from ..utils.file_ops import guarantee_file_dl
from ..utils.directories import get_example_dir


def load_file(filename: str) -> dict:
	"""Helper function to load a .mat file. Attempt to download if not present."""
	filepath = os.path.join(get_example_dir(), filename)
	guarantee_file_dl(filepath, url=None)
	data = loadmat(filepath)
	return data


def load_cylinder_data() -> dict:
	"""Helper function to load `Cylinder_data.mat`"""
	return load_file("Cylinder_data.mat")


def load_cylinder_dmd() -> dict:
	"""Helper function to load `Cylinder_DMD.mat`"""
	return load_file("Cylinder_DMD.mat")


def load_cylinder_edmd() -> dict:
	"""Helper function to load `Cylinder_EDMD.mat`"""
	return load_file("Cylinder_EDMD.mat")
