import numpy as np


def divide_absolute_by_unperturbed(perterbations: np.array, unperturbed_values: np.array) -> np.array:
    unperturbed_values = np.array(unperturbed_values).astype(np.float64)
    unperturbed_values = np.where(np.isinf(unperturbed_values), np.nan, unperturbed_values)
    unperturbed_values = np.where(unperturbed_values == 0, np.nan, unperturbed_values)
    perterbations = np.abs(perterbations)
    normalized_values = perterbations / unperturbed_values

    return normalized_values
