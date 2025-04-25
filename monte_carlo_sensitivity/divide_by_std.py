import numpy as np


def divide_by_std(perterbations: np.array, unperturbed_values: np.array) -> np.array:
    return perterbations / np.nanstd(unperturbed_values)
