from typing import Callable

import numpy as np
import pandas as pd

from .repeat_rows import repeat_rows
from .divide_by_std import divide_by_std

default_normalization_function = divide_by_std

def perturbed_run(
        input_df: pd.DataFrame,
        input_variable: str,
        output_variable: str,
        forward_process: Callable,
        perturbation_process: Callable = np.random.normal,
        normalization_function: Callable = default_normalization_function,
        n: int = 100,
        perturbation_mean: float = 0,
        perturbation_std: float = None,
        dropna: bool = True) -> pd.DataFrame:
    # calculate standard deviation of the input variable
    input_std = np.nanstd(input_df[input_variable])

    if input_std == 0:
        input_std = np.nan

    # use standard deviation of the input variable as the perturbation standard deviation if not given
    if perturbation_std is None:
        perturbation_std = input_std

    # forward process the unperturbed input
    unperturbed_output_df = forward_process(input_df)
    # calculate standard deviation of the output variable
    output_std = np.nanstd(unperturbed_output_df[output_variable])

    if output_std == 0:
        output_std = np.nan

    # extract output variable from unperturbed output
    unperturbed_output = unperturbed_output_df[output_variable]
    # repeat unperturbed output
    unperturbed_output = repeat_rows(unperturbed_output_df, n)[output_variable]
    # generate input perturbation
    input_perturbation = np.concatenate([perturbation_process(0, perturbation_std, n) for i in range(len(input_df))])

    # input_perturbation_std = input_perturbation / input_std

    # copy input for perturbation
    perturbed_input_df = input_df.copy()
    # repeat input for perturbation
    perturbed_input_df = repeat_rows(perturbed_input_df, n)
    # extract input variable from repeated unperturbed input
    unperturbed_input = perturbed_input_df[input_variable]

    # normalize input perturbations
    input_perturbation_std = normalization_function(input_perturbation, unperturbed_input)

    # add perturbation to input
    perturbed_input_df[input_variable] = perturbed_input_df[input_variable] + input_perturbation
    # extract perturbed input
    perturbed_input = perturbed_input_df[input_variable]
    # forward process the perturbed input
    perturbed_output_df = forward_process(perturbed_input_df)
    # extract output variable from perturbed output
    perturbed_output = perturbed_output_df[output_variable]
    # calculate output perturbation
    output_perturbation = perturbed_output - unperturbed_output

    # output_perturbation_std = output_perturbation / output_std

    # normalize output perturbations
    output_perturbation_std = normalization_function(output_perturbation, unperturbed_output)

    results_df = pd.DataFrame({
        "input_variable": input_variable,
        "output_variable": output_variable,
        "input_unperturbed": unperturbed_input,
        "input_perturbation": input_perturbation,
        "input_perturbation_std": input_perturbation_std,
        "input_perturbed": perturbed_input,
        "output_unperturbed": unperturbed_output,
        "output_perturbation": output_perturbation,
        "output_perturbation_std": output_perturbation_std,
        "output_perturbed": perturbed_output,
    })

    if dropna:
        results_df = results_df.dropna()

    return results_df
