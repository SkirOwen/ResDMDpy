from ._loader import load_cylinder_data
from ._loader import load_cylinder_dmd
from ._loader import load_cylinder_edmd


__all__ = [
	"load_cylinder_data",
	"load_cylinder_dmd",
	"load_cylinder_edmd",
]


# def __getattr__(name: str):
# 	try:
# 		return globals()[name]
# 	except KeyError:
# 		raise AttributeError
