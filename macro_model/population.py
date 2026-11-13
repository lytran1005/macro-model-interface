"""Prepare configured default and sensitivity population paths for the sample.

History covers 1990–2024 and projections 2025–2060. The optional Custom
scenario column in the wide sample is separate from the configured Custom
source used here for the Default path.
"""

import numpy as np
import pandas as pd

from config_loaders import load_default_population_scenarios, load_economies
from model import FINAL_YEAR, HISTORY_START_YEAR
from paths import RESULTS_DATA_DIR

HISTORICAL_END_YEAR = 2024
EGEDA_FILE = "00APEC_preliminary_2026_1022_ind.xlsx"
CUSTOM_FILE = "custom_population_manually_consolidated.xlsx"
PROJECTION_FILES = {
    "MED": "undesa_pop_to2100_med.csv",
    "LOW": "undesa_pop_to2100_low.csv",
    "HIGH": "undesa_pop_to2100_high.csv",
}


def _read_egeda(path, economies):
    data = pd.read_excel(path, sheet_name="in")
    required = {"economy", "flows"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(data.columns))}")
    data = data[data["flows"] == "50 Population (millions)"].copy()
    year_columns = [
        column for column in data.columns
        if str(column).isdigit() and HISTORY_START_YEAR <= int(column) <= HISTORICAL_END_YEAR
    ]
    code_map = {
        code.replace("_", ""): code for code in economies["economy_code"]
    }
    data["economy_code"] = data["economy"].map(code_map)
    if data["economy_code"].isna().any():
        raise ValueError(f"{path} contains unrecognised EGEDA population economy codes.")
    data = data.melt(
        id_vars=["economy_code"], value_vars=year_columns,
        var_name="year", value_name="value",
    )
    data["value"] = pd.to_numeric(data["value"], errors="coerce") * 1000
    return data


def _read_long_population(path):
    if path.suffix.lower() == ".xlsx":
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)
    required = {"economy_code", "year", "variable", "value"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(data.columns))}")
    data = data.loc[
        data["variable"] == "population_1jan", ["economy_code", "year", "value"]
    ].copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    return data


def _population_segment(source, path, code, choice, start, end):
    rows = source[
        (source["economy_code"] == code) & source["year"].between(start, end)
    ].copy()
    if rows["year"].duplicated().any():
        raise ValueError(f"{path} has duplicate population years for {code} ({start}–{end}).")
    rows = rows.set_index("year").reindex(range(start, end + 1))
    invalid = rows["value"].isna() | ~np.isfinite(rows["value"]) | (rows["value"] <= 0)
    if invalid.any():
        years = rows.index[invalid].tolist()
        raise ValueError(
            f"{path} has missing or invalid {choice} population values for "
            f"{code} in years {years}."
        )
    return rows.reset_index(names="year")


def load_population_sensitivity_paths(default_population, data_dir=RESULTS_DATA_DIR):
    """Combine configured history with validated Low/Medium/High projections."""
    history = default_population.loc[
        default_population["year"].between(HISTORY_START_YEAR, HISTORICAL_END_YEAR),
        ["economy_code", "year", "value"],
    ]
    codes = sorted(default_population["economy_code"].unique())
    paths = []
    for choice, filename in PROJECTION_FILES.items():
        path = data_dir / filename
        source = _read_long_population(path)
        projections = []
        for code in codes:
            rows = _population_segment(
                source, path, code, choice, HISTORICAL_END_YEAR + 1, FINAL_YEAR
            )
            rows["economy_code"] = code
            projections.append(rows[["economy_code", "year", "value"]])
        combined = pd.concat([history, *projections], ignore_index=True)
        if combined.duplicated(["economy_code", "year"]).any():
            raise ValueError(f"Duplicate {choice} population paths in configured history.")
        column = "population_1jan_medium" if choice == "MED" else f"population_1jan_{choice.lower()}"
        paths.append(combined.rename(columns={"value": column}))
    result = paths[0]
    for path in paths[1:]:
        result = result.merge(path, on=["economy_code", "year"], validate="one_to_one")
    return result


def load_default_population(codes=None, data_dir=RESULTS_DATA_DIR):
    """Return one complete configured 1990–2060 population path per economy.

    Values are in thousands, as expected by the GDP model. Only selected sources
    are read; missing or invalid values in the selected periods are errors.
    """
    selections = load_default_population_scenarios()
    economies = load_economies()
    selected_codes = set(selections) if codes is None else set(codes)
    unknown = selected_codes - set(selections)
    if unknown:
        raise ValueError(f"No default population configuration for economies: {sorted(unknown)}")
    names = economies.set_index("economy_code")["economy"].to_dict()
    source_cache = {}
    frames = []
    for code in sorted(selected_codes):
        for choice, start, end in [
            (selections[code]["Historical to 2024"], HISTORY_START_YEAR, HISTORICAL_END_YEAR),
            (selections[code]["Projection"], HISTORICAL_END_YEAR + 1, FINAL_YEAR),
        ]:
            filename = EGEDA_FILE if choice == "EGEDA" else (
                CUSTOM_FILE if choice == "Custom" else PROJECTION_FILES[choice]
            )
            path = data_dir / filename
            if filename not in source_cache:
                source_cache[filename] = (
                    _read_egeda(path, economies) if choice == "EGEDA"
                    else _read_long_population(path)
                )
            rows = _population_segment(source_cache[filename], path, code, choice, start, end)
            rows["economy_code"] = code
            rows["economy"] = names[code]
            rows["variable"] = "population_1jan"
            frames.append(rows[["economy_code", "economy", "year", "variable", "value"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["economy_code", "economy", "year", "variable", "value"]
    )
