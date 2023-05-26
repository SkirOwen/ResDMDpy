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
