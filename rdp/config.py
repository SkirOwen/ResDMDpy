import argparse
import importlib
import inspect
import os

from argparse import Namespace


def get_rdp_dir() -> str:
	h3_module = importlib.import_module("rdp")
	h3_dir = os.path.dirname(inspect.getabsfile(h3_module))
	return os.path.abspath(os.path.join(h3_dir, ".."))


def parse_args() -> Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"-m", "--modes",
		nargs="*",
		type=int,
		help="Modes to use for the koopman modes.",
	)

	parser.add_argument(
		"-o", "--output",
		help="Name of any output file for this run. The location and file extension will be handled."
	)

	# parser.add_argument(
	# 	"--output-dir",
	# 	default="output",
	# 	help="Path of specific output directory. The default is `./output`."
	# )

	parser.add_argument(
		"-e", "--example",
		help="Example to run. Currently only supports 'cylinder'.",
	)

	parser.add_argument(
		"-p", "--plot-graph",
		action="store_true",
		help="Option to plot results.",
	)

	parser.add_argument(
		"--log-level",
		help="Level of the logger, can be DEBUG / INFO / WARNING / ERROR / CRITICAL"
	)

	args = parser.parse_args()
	return args
