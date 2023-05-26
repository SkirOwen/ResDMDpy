import os

from typing import Iterable

from ..utils.mat_loader import loadmat
from ..utils.file_ops import guarantee_file_dl
from ..utils.directories import get_example_dir


def load_file(filename: str) -> dict:
	filepath = os.path.join(get_example_dir(), filename)
	guarantee_file_dl(filepath, url=None)
	data = loadmat(filepath)
	return data


def load_cylinder_data() -> dict:
	return load_file("Cylinder_data.mat")


def load_cylinder_dmd() -> dict:
	return load_file("Cylinder_DMD.mat")


def load_cylinder_edmd() -> dict:
	return load_file("Cylinder_EDMD.mat")
