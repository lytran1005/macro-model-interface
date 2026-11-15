#%%
"""Loaders for the small reference tables in config/.

These files are the single source of truth for:
  - economies.csv                    canonical economy names and codes
  - gdp_model_parameters.csv         per-economy GDP model parameters
  - default_population_scenario.csv  per-economy population source selections
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CONFIG_DIR

#%%
# functions

def load_economies():
    """Return the canonical economy table (economy_code, economy)."""
    path = CONFIG_DIR / "economies.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing {path}. This file is tracked in git - "
            "check that config/economies.csv exists."
        )


def economy_code_map():
    """Return dict of canonical economy name -> economy code, e.g. 'Australia' -> '01_AUS'."""
    df = load_economies()
    return dict(zip(df["economy"], df["economy_code"]))


def economy_codes():
    """Return the sorted list of the 21 APEC economy codes."""
    return sorted(load_economies()["economy_code"].tolist())


def load_default_population_scenarios(filename="default_population_scenario.csv"):
    """Return the validated historical and projection choices for each economy."""
    path = CONFIG_DIR / filename
    df = pd.read_csv(path, dtype=str)
    columns = ["economy_code", "Historical to 2024", "Projection"]
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    df = df[columns].apply(lambda column: column.str.strip())
    if df["economy_code"].isna().any() or df["economy_code"].duplicated().any():
        raise ValueError(f"{path} must contain exactly one row per economy code.")
    expected_codes = set(economy_codes())
    got_codes = set(df["economy_code"])
    if expected_codes != got_codes:
        raise ValueError(
            f"{path} economy codes do not match config/economies.csv. "
            f"Missing: {sorted(expected_codes - got_codes)}, "
            f"unexpected: {sorted(got_codes - expected_codes)}"
        )
    for column, choices in [
        ("Historical to 2024", {"EGEDA", "Custom"}),
        ("Projection", {"MED", "LOW", "HIGH", "Custom"}),
    ]:
        invalid = df.loc[~df[column].isin(choices), ["economy_code", column]]
        if not invalid.empty:
            raise ValueError(
                f"{path} has invalid {column} choices: "
                f"{invalid.to_dict('records')}. Expected one of {sorted(choices)}."
            )
    return df.set_index("economy_code").to_dict(orient="index")


def load_gdp_model_parameters(filename="gdp_model_parameters.csv"):
    """Return dict of economy_code -> dict of model keyword arguments.

    The 'notes' column is dropped; everything else is passed straight to
    model.aperc_gdp_model as keyword arguments.
    """
    path = CONFIG_DIR / filename
    df = pd.read_csv(path)

    expected_codes = set(economy_codes())
    got_codes = set(df["economy_code"])
    if expected_codes != got_codes:
        raise ValueError(
            f"{path} economy codes do not match config/economies.csv. "
            f"Missing: {sorted(expected_codes - got_codes)}, "
            f"unexpected: {sorted(got_codes - expected_codes)}"
        )
    if "cap_compare" not in df.columns:
        raise ValueError(f"{path} is missing required column 'cap_compare'.")

    params = {}
    for _, row in df.iterrows():
        kwargs = row.drop(labels=["economy_code", "notes"], errors="ignore").to_dict()
        kwargs["lab_eff_periods"] = int(kwargs["lab_eff_periods"])
        if kwargs["lab_eff_periods"] < 1:
            raise ValueError(f"{path} lab_eff_periods must be at least 1 for {row['economy_code']}")
        if not 0 < kwargs["alpha"] < 1:
            raise ValueError(f"{path} alpha must be between 0 and 1 for {row['economy_code']}")
        try:
            cap_compare = float(kwargs["cap_compare"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{path} cap_compare must be finite and in [0, 1) "
                f"for {row['economy_code']}."
            ) from exc
        if not math.isfinite(cap_compare) or not 0 <= cap_compare < 1:
            raise ValueError(
                f"{path} cap_compare must be finite and in [0, 1) "
                f"for {row['economy_code']}."
            )
        kwargs["cap_compare"] = cap_compare
        params[row["economy_code"]] = kwargs
    return params
