from typing import Callable

import numpy as np
import pandas as pd

from monte_carlo_sensitivity import repeat_rows


def joint_varyingUQ_perturbed_run(
        input_df: pd.DataFrame,
        perturbed_variables: str,
        uncert_variables: str,
        perturbation_process: Callable = np.random.multivariate_normal,
        n: int = 100,
        perturbation_mean: float = None,
        perturbation_cor: float = None) -> pd.DataFrame:
    """
    Perform a joint perturbed run analysis on input data to evaluate the sensitivity of output variables
    to perturbations in input variables.

    Parameters:
        input_df (pd.DataFrame): The input DataFrame containing the input variables.
        perturbed_variables (str): The name(s) of the input variable(s) to perturb.
        perturbation_process (Callable, optional): A function to generate perturbations. Defaults to
                                                   np.random.multivariate_normal.
        n (int, optional): The number of perturbations to generate for each input. Defaults to 100.
        perturbation_mean (float, optional): The mean of the perturbation distribution. Defaults to None,
                                             which assumes zero mean.
        perturbation_cor (float, optional): The correlation matrix of the perturbation distribution. Defaults
                                            to None, which assumes diagonal covariance based on input standard
                                            deviations.

    Returns:
        pd.DataFrame: A DataFrame containing the unperturbed inputs, perturbed inputs, perturbations,
                      unperturbed outputs, perturbed outputs, and standardized perturbations for both inputs
                      and outputs.
    """
    # calculate standard deviation of the input variable

    n_input = len(perturbed_variables)

    input_std = np.nanstd(input_df[perturbed_variables],axis=0)

    if all(x == 0 for x in input_std):
        input_std = np.empty(n_input) * np.nan

    # use diagonal (independent) standard deviations of the input variables if not given
    if perturbation_cov is None:
        perturbation_cov = np.diag(input_std)

    if perturbation_mean is None:
        perturbation_mean = np.zeros(n_input)

    # generate input perturbation
    processed_rows = []
    # Enumerate over rows
    for index, row in input_df.iterrows():
        # Process each row (e.g., multiply by a factor based on row index)
        stds = np.diag(row[uncert_variables])
        covs =   np.array(stds @ perturbation_cor @ stds)
        samples = perturbation_process(perturbation_mean, covs, size=n)
        processed_rows.append(samples)

    # Concatenate the processed rows vertically
    input_perturbation = np.concatenate(processed_rows, axis=0)

    # copy input for perturbation
    perturbed_input_df = input_df.copy()
    # repeat input for perturbation
    perturbed_input_df = repeat_rows(perturbed_input_df, n)
    # extract input variable from repeated unperturbed input
    unperturbed_input = perturbed_input_df[perturbed_variables]
    # add perturbation to input
    perturbed_input_df[perturbed_variables] = perturbed_input_df[perturbed_variables] + input_perturbation
    
    # extract perturbed input
    perturbed_input = perturbed_input_df[perturbed_variables]
    input_perturbation_df = pd.DataFrame(input_perturbation, columns=[s+"_perturbation" for s in perturbed_variables])

    unperturbed_input.columns = [s+"_unperturbed" for s in perturbed_variables]
    perturbed_input.columns = [s+"_perturbed" for s in perturbed_variables]

    results_df = pd.concat([unperturbed_input,
                            input_perturbation_df,
                            perturbed_input], axis=1)

    return results_df