import os

from rdp.config import get_rdp_dir
from rdp.utils.file_ops import guarantee_existence


def get_example_dir() -> str:
	return guarantee_existence(os.path.join(get_rdp_dir(), "rdp", "examples"))


def get_outputs_dir() -> str:
	return guarantee_existence(os.path.join(get_rdp_dir(), "outputs"))


def get_koopmode_dir() -> str:
	return guarantee_existence(os.path.join(get_outputs_dir(), "koopmodes"))


def get_plots_dir() -> str:
	return guarantee_existence(os.path.join(get_outputs_dir(), "plots"))
