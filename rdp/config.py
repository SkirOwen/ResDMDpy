import os
import importlib
import inspect


def get_rdp_dir() -> str:
	h3_module = importlib.import_module("rdp")
	h3_dir = os.path.dirname(inspect.getabsfile(h3_module))
	return os.path.abspath(os.path.join(h3_dir, ".."))
