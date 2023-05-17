from __future__ import annotations

import os

from typing import Iterable

from rdp.utils.downloader import downloader


# TODO: should I use pathlib???
def guarantee_file_dl(filepath: str):
	directory = os.path.dirname(filepath)
	filename = os.path.basename(filepath)

	if not os.path.exists(filepath):
		url = get_url(filename)
		downloader(url, root=directory)


DL_URL = {
	"Cylinder_data.mat": "",
	"Cylinder_DMD.mat": "",
	"Cylinder_EDMD.mat": "",
	"dataset_1s_b.mat": "",
	"dataset_1s.mat": "",
	"double_pendulum_data.mat": "",
	"EDMD_canopy_final.mat": "",
	"HotWireData_Baseline.mat": "",
	"HotWireData_FlowInjection.mat": "",
	"LIP_times.mat": "",
	"LIP.mat": "",
	"mpEDMD_turbulent_data.mat": "",
	"pendulum_data.mat": "",

}


def get_url(filename: str) -> Iterable:
	# filename = filename.lower()
	try:
		url = DL_URL[filename]
	except KeyError:
		raise ValueError(f"{filename} is not known.")
	return [url]
