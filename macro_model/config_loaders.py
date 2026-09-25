#%%
"""Loaders for the small reference tables in config/.

These files are the single source of truth for:
  - economies.csv                    canonical economy names and codes
  - economy_name_mappings.csv        source-specific name -> canonical name
  - population_variant_choices.csv   UN DESA variant used per economy
  - gdp_model_parameters.csv         per-economy GDP model parameters
"""

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


def load_name_mapping(source):
    """Return dict of source-specific name -> canonical name for one source.

    source must be one of 'UNDESA', 'IMF', 'PWT' (see economy_name_mappings.csv).
    """
    path = CONFIG_DIR / "economy_name_mappings.csv"
    df = pd.read_csv(path)
    known = df["source"].unique()
    if source not in known:
        raise ValueError(f"Unknown source '{source}'. Known sources: {sorted(known)}")
    df = df[df["source"] == source]
    return dict(zip(df["source_name"], df["economy"]))


def load_population_variant_choices():
    """Return dict of canonical economy name -> {'variant': str, 'extra_annual_growth': float}."""
    path = CONFIG_DIR / "population_variant_choices.csv"
    df = pd.read_csv(path)
    df["extra_annual_growth"] = df["extra_annual_growth"].fillna(0.0)
    return {
        row["economy"]: {
            "variant": row["variant"],
            "extra_annual_growth": row["extra_annual_growth"],
        }
        for _, row in df.iterrows()
    }


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

    params = {}
    for _, row in df.iterrows():
        kwargs = row.drop(labels=["economy_code", "notes"], errors="ignore").to_dict()
        kwargs["lab_eff_periods"] = int(kwargs["lab_eff_periods"])
        if kwargs["lab_eff_periods"] < 1:
            raise ValueError(f"{path} lab_eff_periods must be at least 1 for {row['economy_code']}")
        if not 0 < kwargs["alpha"] < 1:
            raise ValueError(f"{path} alpha must be between 0 and 1 for {row['economy_code']}")
        params[row["economy_code"]] = kwargs
    return params
