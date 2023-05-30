import os

from rdp.config import get_rdp_dir
from rdp.utils.file_ops import guarantee_existence


def get_example_dir() -> str:
	return guarantee_existence(os.path.join(get_rdp_dir(), "rdp", "examples"))
