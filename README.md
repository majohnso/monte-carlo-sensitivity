# `monte-carlo-sensitivity` 

Monte-Carlo Sensitivity Analysis Python Package

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer and designer<br>
NASA Jet Propulsion Laboratory 329G

Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
Sensitivity and uncertainty analysis<br>
NASA Jet Propulsion Laboratory 398L

## Installation

This Python package is distributed using the pip package manager. Install it with the package name `monte-carlo-sensitivity` with dashes.

```
pip install monte-carlo-sensitivity
```

## Usage

The `monte-carlo-sensitivity` package provides tools for performing sensitivity analysis using Monte Carlo methods. 

Import this package in Python as `monte_carlo_sensitivity` with underscores.

Below are examples of how to use the key functions in the package:

### Sensitivity Analysis

The `sensitivity_analysis` function performs sensitivity analysis by perturbing input variables and observing the effect on output variables.

```python
import pandas as pd
from monte_carlo_sensitivity import sensitivity_analysis

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform sensitivity analysis
perturbation_df, sensitivity_metrics = sensitivity_analysis(
    input_df=input_df,
    input_variables=["input_var1", "input_var2"],
    output_variables=["output_var"],
    forward_process=forward_process,
    n=100
)

print(perturbation_df)
print(sensitivity_metrics)
```

### Perturbed Run

The `perturbed_run` function performs a Monte Carlo sensitivity analysis by perturbing an input variable and observing the effect on an output variable.

```Python
import pandas as pd
from monte_carlo_sensitivity.perturbed_run import perturbed_run

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform a perturbed run
results = perturbed_run(
    input_df=input_df,
    input_variable="input_var1",
    output_variable="output_var",
    forward_process=forward_process,
    n=100
)

print(results)
```

### Joint Perturbed Run

The `joint_perturbed_run` function evaluates the sensitivity of output variables to joint perturbations in input variables.

```Python
import pandas as pd
from monte_carlo_sensitivity.joint_perturbed_run import joint_perturbed_run

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform a joint perturbed run
results = joint_perturbed_run(
    input_df=input_df,
    input_variable=["input_var1", "input_var2"],
    output_variable=["output_var"],
    forward_process=forward_process,
    n=100
)

print(results)
```
