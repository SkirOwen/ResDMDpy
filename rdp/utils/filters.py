import numpy as np


def phi_fejer(x):
	return 1 - abs(x)


def phi_cosine(x):
	return (1 + np.cos(np.pi * x)) / 2


def phi_opt4(x):
	return 1 - x ** 4 * (-20 * abs(x) ** 3 + 70 * x ** 2 - 84 * abs(x) + 35)


def phi_sharp_cosine(x):
	return phi_cosine(x) ** 4 * (
			35 - 84 * phi_cosine(x) + 70 * phi_cosine(x) ** 2 - 20 * phi_cosine(x) ** 3)


def phi_inft(x):
	return np.exp(-2 / (1 - abs(x)) * np.exp(-0.109550455106347 / abs(x) ** 4))
