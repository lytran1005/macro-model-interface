"""Modeling logic for the Streamlit app (no streamlit dependency).

Derives the model inputs (capital stock, labour efficiency) from a raw
wide-format upload, rebuilds the inputs the APERC Solow-Swan model expects,
runs the projection for every economy, and assembles the combined result
table.

The model itself lives in the repo (workflow/scripts/Macro/model.py) so there
is a single authoritative copy; its functions and constants are imported
directly below. Sample/template generators live in generate_samples.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_DIR = PROJECT_ROOT / "macro_model"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"

REPO_MACRO = MACRO_DIR
sys.path.insert(0, str(MACRO_DIR))

from model import (  # noqa: E402
    aperc_gdp_model,
    jump_off_year_for_economy,
    HISTORY_START_YEAR,
    LAST_PWT_YEAR,
    LAST_WDI_YEAR,
    LAST_IMF_YEAR,
    FINAL_YEAR,
    LABEL_IMF,
    LABEL_WDI,
    LABEL_APERC,
    LABEL_9TH,
)
from config_loaders import load_economies, economy_code_map, load_gdp_model_parameters  # noqa: E402

# Raw variables in the uploaded CSV, with a short description. Everything else
# (capital stock, labour efficiency) is derived from these by the app.
REQUIRED_VARIABLES = {
    "population_1jan": "population, thousands (to 2060)",
    "Real GDP PPP 2021 USD": "historical real GDP, millions 2021 USD PPP (to 2025 for non-CT, to 2031 for CT)",
    "output_to_kstock": "PWT output-to-capital ratio (to 2023)",
    "delta": "PWT depreciation rate (to 2023)",
    "Gross national savings": "historical savings rate, percent of GDP (to 2025 for non-CT, to 2031 for CT)",
}

CRITICAL_VARIABLES = tuple(REQUIRED_VARIABLES)

# Wide-format upload: one column per variable.
WIDE_VARIABLE_COLUMNS = list(REQUIRED_VARIABLES.keys())

# Default capital share used when backing out labour efficiency; the per-economy
# 'alpha' from the parameters takes precedence (mirrors prepare_model_inputs.py).
ALPHA = 0.4

# PNG / Brunei assumptions, mirroring prepare_imf.py and prepare_pwt.py.
PNG_SAVINGS_RATE_PERCENT = 25

UNITS = {
    "real_GDP": "Millions (2021 USD PPP)",
    "population": "Thousands",
    "lab_efficiency": "Derived value (residual to model)",
    "depreciation": "Proportion",
    "savings": "Proportion",
    "k_stock": "Millions (2021 USD)",
    "GDP_per_capita": "USD PPP 2021",
    "gdp_growth": "Percent (%)",
    "real_GDP_9th": "Millions (2021 USD PPP)",
}

POPULATION_SCENARIO_FILES = {
    "Low": "undesa_pop_to2100_low.csv",
    "Medium": "undesa_pop_to2100_med.csv",
    "High": "undesa_pop_to2100_high.csv",
}
POPULATION_SCENARIO_COLUMNS = {
    scenario: f"population_1jan_{scenario.lower()}"
    for scenario in POPULATION_SCENARIO_FILES
}
CUSTOM_SCENARIO = "Custom"
CUSTOM_POPULATION_COLUMN = "population_1jan_custom"


def _with_economy_code(df_wide):
    frame = df_wide.copy()
    if "economy_code" not in frame.columns:
        frame["economy_code"] = frame["economy"].map(economy_code_map())
    return frame


def custom_population_status(df_wide):
    """Return (complete, incomplete) economy codes for the optional custom column.

    An economy is complete when the custom column has a value for every year in
    which the Medium UN DESA column has one. Economies with no custom values at
    all are ignored; partially filled economies are reported as incomplete.
    """
    if CUSTOM_POPULATION_COLUMN not in df_wide.columns:
        return [], []
    frame = _with_economy_code(df_wide)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    custom = pd.to_numeric(frame[CUSTOM_POPULATION_COLUMN], errors="coerce")
    reference_column = POPULATION_SCENARIO_COLUMNS["Medium"]
    reference = (
        pd.to_numeric(frame[reference_column], errors="coerce")
        if reference_column in frame.columns
        else frame["year"].where(frame["year"].notna())
    )
    in_horizon = frame["year"].between(HISTORY_START_YEAR, FINAL_YEAR)
    complete, incomplete = [], []
    for code, group in frame[in_horizon].groupby("economy_code"):
        provided = custom.loc[group.index].notna()
        if not provided.any():
            continue
        needed = reference.loc[group.index].notna()
        if (needed & ~provided).any():
            incomplete.append(code)
        else:
            complete.append(code)
    return sorted(complete), sorted(incomplete)


def available_population_scenarios(df_wide):
    """Official scenarios, plus Custom when any economy has a complete custom path."""
    scenarios = list(POPULATION_SCENARIO_COLUMNS)
    if custom_population_status(df_wide)[0]:
        scenarios.append(CUSTOM_SCENARIO)
    return scenarios


def population_scenarios_from_wide(df_wide):
    """Return population paths embedded in the currently loaded wide input."""
    missing = [
        column for column in POPULATION_SCENARIO_COLUMNS.values()
        if column not in df_wide.columns
    ]
    if missing:
        raise ValueError(
            "The loaded wide CSV is missing population scenario columns: "
            + ", ".join(missing)
        )

    frame = df_wide.copy()
    name_to_code = economy_code_map()
    code_to_name = {code: name for name, code in name_to_code.items()}
    if "economy_code" not in frame.columns:
        frame["economy_code"] = frame["economy"].map(name_to_code)
    if "economy" not in frame.columns:
        frame["economy"] = frame["economy_code"].map(code_to_name)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

    identifiers = ["economy_code", "economy", "year"]
    frames = []
    for scenario, column in POPULATION_SCENARIO_COLUMNS.items():
        population = frame[identifiers + [column]].rename(
            columns={column: "value"}
        )
        population["value"] = pd.to_numeric(population["value"], errors="coerce")
        population = population[
            population["year"].between(HISTORY_START_YEAR, FINAL_YEAR)
        ].dropna(subset=identifiers + ["value"])
        population["year"] = population["year"].astype(int)
        population["scenario"] = scenario
        frames.append(population)

    custom_codes, _ = custom_population_status(df_wide)
    if custom_codes:
        custom = frame[identifiers + [CUSTOM_POPULATION_COLUMN]].rename(
            columns={CUSTOM_POPULATION_COLUMN: "value"}
        )
        custom["value"] = pd.to_numeric(custom["value"], errors="coerce")
        custom = custom[
            custom["economy_code"].isin(custom_codes)
            & custom["year"].between(HISTORY_START_YEAR, FINAL_YEAR)
        ].dropna(subset=identifiers + ["value"])
        custom["year"] = custom["year"].astype(int)
        custom["scenario"] = CUSTOM_SCENARIO
        frames.append(custom)

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["economy_code", "year", "scenario"], keep="last"
    )


def run_uploaded_population_scenario(df_wide, scenario, params, economy=None):
    """Run the model using one population pathway from the uploaded wide CSV."""
    custom_codes = None
    if scenario == CUSTOM_SCENARIO:
        population_column = CUSTOM_POPULATION_COLUMN
        custom_codes, _ = custom_population_status(df_wide)
        if not custom_codes:
            raise ValueError("No economy has a complete custom population path.")
    else:
        population_column = POPULATION_SCENARIO_COLUMNS.get(scenario)
    if population_column is None:
        raise ValueError(f"Unknown population scenario: {scenario}")
    if population_column not in df_wide.columns:
        raise ValueError(
            f"The loaded wide CSV is missing population scenario column "
            f"'{population_column}'."
        )

    scenario_input = df_wide.copy()
    scenario_input["population_1jan"] = pd.to_numeric(
        scenario_input[population_column], errors="coerce"
    )
    long_input = wide_to_long(scenario_input)
    if custom_codes is not None:
        long_input = long_input[long_input["economy_code"].isin(custom_codes)]
    if economy is not None:
        long_input = long_input[long_input["economy_code"] == economy]
    results, _, errors = run_analysis(long_input, params)
    return results, errors


def run_uploaded_population_gdp_scenarios(df_wide, params):
    """Run GDP projections for Low, Medium, and High paths in the uploaded CSV."""
    frames = []
    errors = []
    for scenario in available_population_scenarios(df_wide):
        results, scenario_errors = run_uploaded_population_scenario(
            df_wide, scenario, params
        )
        errors.extend(
            (scenario, economy_code, message)
            for economy_code, message in scenario_errors
        )
        gdp = results[results["variable"] == "real_GDP"][
            ["economy_code", "economy", "year", "value"]
        ].copy()
        gdp["scenario"] = scenario
        frames.append(gdp)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["economy_code", "economy", "year", "value", "scenario"]
    )
    return _normalise_population_gdp_scenarios(combined), errors


def _normalise_population_gdp_scenarios(data):
    """Validate and normalise a combined population GDP scenario table."""
    required_columns = ["economy_code", "economy", "year", "value", "scenario"]
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Population GDP scenarios are missing columns: {sorted(missing_columns)}"
        )

    data = data[required_columns].copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[
        data["year"].between(HISTORY_START_YEAR, FINAL_YEAR)
    ].dropna(subset=["economy_code", "year", "value", "scenario"])
    data["year"] = data["year"].astype(int)
    return data.drop_duplicates(
        subset=["economy_code", "year", "scenario"], keep="last"
    ).reset_index(drop=True)


def ensure_percent(df):
    """Add a percent column (per economy, per variable) if the upload lacks one."""
    if "percent" in df.columns:
        return df
    df = df.copy()
    grp = df[df["variable"].isin(["Labour efficiency", "Capital stock"])].copy()
    df["percent"] = np.nan
    df.loc[grp.index, "percent"] = grp.groupby(["economy_code", "variable"])["value"]\
        .pct_change(fill_method=None).values
    return df


def build_input_table(df):
    """Rebuild the wide per-economy table the model consumes.

    Mirrors prepare_model_inputs.build_gdp_input_table: one row per
    (economy, year) with columns labour, efficiency, capital, real_output.
    """
    var_map = {
        "efficiency": "Labour efficiency",
        "capital": "Capital stock",
        "real_output": "Real GDP PPP 2021 USD",
    }
    frames = []
    for code in sorted(df["economy_code"].unique()):
        sub = df[df["economy_code"] == code]
        if sub.empty:
            continue
        economy = sub["economy"].iloc[0]
        base = sub[sub["variable"] == "population_1jan"][["year", "value"]]\
            .rename(columns={"value": "labour"})
        for col, variable in var_map.items():
            part = sub[sub["variable"] == variable][["year", "value"]]\
                .rename(columns={"value": col})
            base = base.merge(part, on="year", how="left")
        base.insert(0, "economy_code", code)
        base.insert(1, "economy", economy)
        frames.append(base)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_delta_snapshot(df, codes):
    """Depreciation value at the PWT jump-off year (2023), latest fallback."""
    return build_snapshot(df, codes, variable="delta", target_year=LAST_PWT_YEAR)


def build_snapshot(df, codes, variable, target_year):
    """One value per economy at target_year (or the last available year)."""
    rows = []
    for code in codes:
        sub = df[(df["economy_code"] == code) & (df["variable"] == variable)].sort_values("year")
        if sub.empty:
            continue
        snap = sub[sub["year"] == target_year]
        value = snap["value"].iloc[0] if not snap.empty else sub["value"].iloc[-1]
        rows.append({"economy_code": code, "economy": sub["economy"].iloc[0], "value": value})
    return pd.DataFrame(rows)


def build_savings_snapshot(df, codes):
    """Savings rate at each economy's jump-off year; PNG uses its flat assumption."""
    rows = []
    for code in codes:
        target_year = jump_off_year_for_economy(code)
        value, economy = None, None
        sub = df[
            (df["economy_code"] == code)
            & (df["variable"] == "Gross national savings")
        ].sort_values("year")
        if not sub.empty:
            snap = sub[sub["year"] == target_year]
            value = snap["value"].iloc[0] if not snap.empty else sub["value"].iloc[-1]
            economy = sub["economy"].iloc[0]
        if code == "13_PNG" and (value is None or np.isnan(value)):
            value = PNG_SAVINGS_RATE_PERCENT
            if economy is None:
                sub_code = df[df["economy_code"] == code]
                economy = sub_code["economy"].iloc[0] if not sub_code.empty else "Papua New Guinea"
        if value is not None:
            rows.append({"economy_code": code, "economy": economy, "value": value})
    return pd.DataFrame(rows)


def _fill_history(df, snap_data, missing_codes, variable, years):
    """Backfill a history series with the snapshot value for codes missing it."""
    fill_rows = []
    for code in missing_codes:
        matching = snap_data[snap_data["economy_code"] == code]
        if matching.empty:
            continue
        row = matching.iloc[0]
        fill_rows.append(pd.DataFrame({
            "economy_code": code,
            "economy": row["economy"],
            "variable": variable,
            "year": list(years),
            "value": row["value"],
        }))
    if not fill_rows:
        return df
    return pd.concat([df] + fill_rows, ignore_index=True)


def build_model_inputs(df):
    """Split the derived long dataset into the pieces aperc_gdp_model expects."""
    codes = sorted(df["economy_code"].unique())

    labour_data = df[df["variable"] == "Labour efficiency"].copy()
    cap_growth_data = df[df["variable"] == "Capital stock"][["economy_code", "year", "percent"]].copy()
    delta_hist = df[df["variable"] == "delta"].copy()
    save_invest_hist = df[df["variable"] == "Gross national savings"].copy()

    delta_data = build_delta_snapshot(df, codes)
    sav_data = build_savings_snapshot(df, codes)

    # Robustness: the model's reporting block needs a history series for every
    # economy, so backfill any that are missing from the history variables.
    delta_hist = _fill_history(delta_hist, delta_data,
                               set(delta_data["economy_code"]) - set(delta_hist["economy_code"]),
                               "delta", range(HISTORY_START_YEAR, LAST_PWT_YEAR + 1))
    save_invest_hist = _fill_history(save_invest_hist, sav_data,
                                     set(sav_data["economy_code"]) - set(save_invest_hist["economy_code"]),
                                     "Gross national savings", range(HISTORY_START_YEAR, LAST_IMF_YEAR + 1))

    has_9th = (df["variable"] == "9th Outlook").any()
    if has_9th:
        gdp_9th = df[df["variable"] == "9th Outlook"].copy()
        gdp_9th["variable"] = "real_GDP"
    else:
        gdp_9th = pd.DataFrame(
            columns=["economy_code", "economy", "variable", "year", "value"]
        )

    return {
        "input_data": build_input_table(df),
        "labour_data": labour_data,
        "cap_growth_data": cap_growth_data,
        "delta_data": delta_data,
        "sav_data": sav_data,
        "save_invest_hist": save_invest_hist,
        "delta_hist": delta_hist,
        "gdp_9th": gdp_9th,
    }


def wide_to_long(df):
    """Convert the wide upload (economy | year | <variable columns>) to the
    internal long schema (economy_code, economy, variable, year, value).

    Requires a 'year' column and an 'economy' and/or 'economy_code' column.
    Variable columns are named as in WIDE_VARIABLE_COLUMNS; an optional
    '9th Outlook' column is carried through for chart comparison.
    """
    df = df.copy()

    if "year" not in df.columns:
        raise ValueError("upload must have a 'year' column")
    if "economy" not in df.columns and "economy_code" not in df.columns:
        raise ValueError("upload must have an 'economy' or 'economy_code' column")

    name_to_code = economy_code_map()
    code_to_name = {v: k for k, v in name_to_code.items()}

    if "economy_code" not in df.columns:
        df["economy_code"] = df["economy"].map(name_to_code)
    if "economy" not in df.columns:
        df["economy"] = df["economy_code"].map(code_to_name)

    if df["economy_code"].isna().any():
        unknown = df.loc[df["economy_code"].isna(), "economy"].unique()[:5]
        raise ValueError(
            "economies not recognised: "
            + ", ".join(map(str, unknown))
            + ". Names/codes must match config/economies.csv.")

    value_cols = [c for c in WIDE_VARIABLE_COLUMNS if c in df.columns]
    if "9th Outlook" in df.columns:
        value_cols.append("9th Outlook")

    long = df.melt(id_vars=["economy_code", "economy", "year"],
                   value_vars=value_cols, var_name="variable", value_name="value")
    return long[["economy_code", "economy", "variable", "year", "value"]]


def derive_capital_stock(df, codes):
    """Derive the capital stock series: K = GDP / output_to_kstock, with the
    PWT ratio forward-filled from 2023 out to each economy's jump-off year.

    Mirrors prepare_model_inputs.build_capital_stock.
    """
    rows = []
    for code in codes:
        sub = df[df["economy_code"] == code]
        if sub.empty:
            continue
        economy = sub["economy"].iloc[0]
        gdp = sub[sub["variable"] == "Real GDP PPP 2021 USD"][["year", "value"]]\
            .rename(columns={"value": "gdp"}).set_index("year")
        ratio = sub[sub["variable"] == "output_to_kstock"][["year", "value"]]\
            .rename(columns={"value": "ratio"}).set_index("year")
        if gdp.empty or ratio.empty:
            continue
        merged = gdp.join(ratio, how="outer").sort_index()
        cutoff = jump_off_year_for_economy(code)
        merged = merged.loc[HISTORY_START_YEAR:cutoff].copy()
        merged["ratio"] = merged["ratio"].ffill()
        merged["capital"] = merged["gdp"] / merged["ratio"]
        for year, value in merged["capital"].dropna().items():
            rows.append({"economy_code": code, "economy": economy,
                         "variable": "Capital stock", "year": int(year), "value": value})
    return pd.DataFrame(rows)


def derive_labour_efficiency(df, capital_df, codes, alpha_by_code):
    """Back out labour efficiency E from the Cobb-Douglas production function:

        Y = K^alpha * (E*L)^(1-alpha)  =>  E = (Y / (L^(1-alpha) * K^alpha)) ** (1/(1-alpha))

    Mirrors prepare_model_inputs.build_labour_efficiency. Uses each economy's
    'alpha' from the model parameters (default ALPHA) from 1990 up to jump-off.
    """
    rows = []
    for code in codes:
        sub = df[df["economy_code"] == code]
        if sub.empty:
            continue
        economy = sub["economy"].iloc[0]
        alpha = alpha_by_code.get(code, ALPHA)
        labour = sub[sub["variable"] == "population_1jan"][["year", "value"]]\
            .rename(columns={"value": "L"}).set_index("year")
        output = sub[sub["variable"] == "Real GDP PPP 2021 USD"][["year", "value"]]\
            .rename(columns={"value": "Y"}).set_index("year")
        capital = capital_df[capital_df["economy_code"] == code][["year", "value"]]\
            .rename(columns={"value": "K"}).set_index("year")

        eqn = pd.concat([output, labour, capital], axis=1).dropna()
        cutoff = jump_off_year_for_economy(code)
        eqn = eqn.loc[HISTORY_START_YEAR:cutoff]
        if eqn.empty:
            continue
        eqn["E"] = (eqn["Y"] / (eqn["L"] ** (1 - alpha) * eqn["K"] ** alpha)) ** (1 / (1 - alpha))
        for year, value in eqn["E"].dropna().items():
            rows.append({"economy_code": code, "economy": economy,
                         "variable": "Labour efficiency", "year": int(year), "value": value})
    return pd.DataFrame(rows)


def run_analysis(df, params):
    """Derive capital and labour efficiency from the raw upload, run the model
    for every economy, and postprocess results.

    params: dict of economy_code -> model keyword arguments
            (e.g. config/gdp_model_parameters.csv). Unknown codes run with
            model defaults.

    Returns (results_long, inputs, errors):
      results_long : combined long result table (real_GDP, population, ...)
      inputs       : the model input pieces (for inspection in the UI)
      errors       : list of (economy_code, message) for economies that failed
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    codes = sorted(df["economy_code"].unique())
    capital_df = derive_capital_stock(df, codes)
    labour_eff_df = derive_labour_efficiency(
        df, capital_df, codes,
        {code: params.get(code, {}).get("alpha", ALPHA) for code in codes})

    df = pd.concat([df, capital_df, labour_eff_df], ignore_index=True)
    df = ensure_percent(df)

    inputs = build_model_inputs(df)
    codes = sorted(inputs["input_data"]["economy_code"].unique())

    combined, errors = [], []
    for code in codes:
        try:
            jump_off = jump_off_year_for_economy(code)
            gdp_long, _ = aperc_gdp_model(
                economy=code,
                jump_off_year=jump_off,
                **inputs,
                **params.get(code, {}),
            )
            combined.append(gdp_long)
        except Exception as exc:  # noqa: BLE001 - surface any per-economy failure in the UI
            errors.append((code, str(exc)))

    results_long = postprocess(pd.concat(combined, ignore_index=True)) if combined else pd.DataFrame()
    return results_long, inputs, errors


def postprocess(combined):
    """Assemble the combined long table (mirrors generate_results.combine_and_postprocess)."""
    combined = combined.drop_duplicates(
        subset=["economy_code", "economy", "year", "variable"], keep="last")

    wide = combined.pivot(columns="variable", values="value",
                          index=["economy_code", "economy", "year"]).reset_index()

    for col in wide.columns:
        if col not in ["economy_code", "economy", "year"]:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    # One continuous real GDP series: CT stays on IMF through 2031, all other
    # economies use WDI through 2025 and then continue via the APERC model.
    wide["real_GDP"] = np.nan
    for code in wide["economy_code"].unique():
        jump_off = jump_off_year_for_economy(code)
        mask = wide["economy_code"] == code
        historical_label = LABEL_IMF if code == "18_CT" else LABEL_WDI
        hist_mask = mask & (wide["year"] <= jump_off)
        proj_mask = mask & (wide["year"] > jump_off)
        if historical_label in wide.columns:
            wide.loc[hist_mask, "real_GDP"] = wide.loc[hist_mask, historical_label]
        if LABEL_APERC in wide.columns:
            wide.loc[proj_mask, "real_GDP"] = wide.loc[proj_mask, LABEL_APERC]

    keep_vars = ["economy_code", "economy", "year", "real_GDP"]
    rename_dict = {
        "Population": "population",
        "Labour efficiency": "lab_efficiency",
        "Depreciation": "depreciation",
        "Savings": "savings",
        "Capital stock": "k_stock",
    }
    for old_v, new_v in rename_dict.items():
        if old_v in wide.columns:
            wide[new_v] = wide[old_v]
            keep_vars.append(new_v)

    if LABEL_9TH in wide.columns and wide[LABEL_9TH].notna().any():
        wide["real_GDP_9th"] = wide[LABEL_9TH]
        keep_vars.append("real_GDP_9th")

    wide = wide[keep_vars]

    # GDP in millions / population in thousands * 1000 = USD per person
    if "population" in wide.columns and "real_GDP" in wide.columns:
        wide["GDP_per_capita"] = wide["real_GDP"] / wide["population"] * 1000

    # Calculate continuous annual real GDP growth rate (%)
    growth_df = wide[["economy_code", "year", "real_GDP"]].sort_values(["economy_code", "year"]).copy()
    growth_df["gdp_growth"] = growth_df.groupby("economy_code")["real_GDP"].pct_change().mul(100)
    wide["gdp_growth"] = growth_df["gdp_growth"]

    long = wide.melt(id_vars=["economy_code", "economy", "year"])
    long["units"] = long["variable"].map(UNITS)
    return long.sort_values(["economy_code", "variable", "year"]).reset_index(drop=True)


def parse_params_csv(df):
    """Parse user parameters while keeping pipeline cap_compare fixed."""
    params = {}
    configured = load_gdp_model_parameters("gdp_model_parameters.csv")
    for _, row in df.iterrows():
        code = row["economy_code"]
        kwargs = row.drop(
            labels=["economy_code", "notes", "cap_compare"], errors="ignore"
        ).to_dict()
        kwargs["lab_eff_periods"] = int(kwargs["lab_eff_periods"])
        kwargs["cap_compare"] = configured.get(code, {}).get("cap_compare", 0.05)
        params[code] = kwargs
    return params


def parse_uploaded_economy_params(df, economy_code):
    """Validate an app-exported parameter CSV and return one economy's settings."""
    parameter_columns = [
        "lab_eff_periods",
        "high_eff",
        "low_eff",
        "change_eff",
        "high_sav",
        "low_sav",
        "change_sav",
        "high_delta",
        "low_delta",
        "change_del",
        "alpha",
    ]
    required_columns = ["economy_code", *parameter_columns]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Parameter CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if df.empty:
        raise ValueError("Parameter CSV has no economy rows.")

    uploaded = df[required_columns].copy()
    uploaded["economy_code"] = uploaded["economy_code"].astype("string").str.strip()
    if uploaded["economy_code"].isna().any() or uploaded["economy_code"].eq("").any():
        raise ValueError("Parameter CSV contains a blank economy_code.")
    if uploaded["economy_code"].duplicated().any():
        duplicates = uploaded.loc[
            uploaded["economy_code"].duplicated(keep=False), "economy_code"
        ].unique()
        raise ValueError(
            "Parameter CSV contains duplicate economy rows: "
            + ", ".join(duplicates.tolist())
        )

    configured = load_gdp_model_parameters("gdp_model_parameters.csv")
    unknown_codes = sorted(set(uploaded["economy_code"]) - set(configured))
    if unknown_codes:
        raise ValueError(
            "Parameter CSV contains unknown economy_code values: "
            + ", ".join(unknown_codes)
        )

    for column in parameter_columns:
        uploaded[column] = pd.to_numeric(uploaded[column], errors="coerce")
        if uploaded[column].isna().any() or not np.isfinite(uploaded[column]).all():
            raise ValueError(
                f"Parameter CSV column '{column}' must contain finite numeric values."
            )

    if (
        (uploaded["lab_eff_periods"] < 1).any()
        or (uploaded["lab_eff_periods"] > 10).any()
        or (uploaded["lab_eff_periods"] % 1 != 0).any()
    ):
        raise ValueError("'lab_eff_periods' must be an integer from 1 to 10.")
    for low, high, label, minimum, maximum in [
        ("low_eff", "high_eff", "efficiency", 0.005, 0.1),
        ("low_sav", "high_sav", "savings", 0.15, 0.5),
        ("low_delta", "high_delta", "depreciation", 0.03, 0.07),
    ]:
        if (
            (uploaded[low] < minimum).any()
            or (uploaded[high] > maximum).any()
            or (uploaded[low] > uploaded[high]).any()
        ):
            raise ValueError(
                f"The {label} corridor must be ordered and within "
                f"{minimum * 100:g}%–{maximum * 100:g}%."
            )
    for column in ["change_eff", "change_sav", "change_del"]:
        if (uploaded[column] < 0).any():
            raise ValueError(f"'{column}' cannot be negative.")
    if (
        (uploaded["alpha"] < 0.3).any()
        or (uploaded["alpha"] > 0.5).any()
    ):
        raise ValueError("'alpha' must be between 0.3 and 0.5.")

    selected = uploaded[uploaded["economy_code"] == economy_code]
    if selected.empty:
        raise ValueError(
            f"Parameter CSV does not contain the currently selected economy "
            f"({economy_code}). No parameters were changed."
        )

    values = selected.iloc[0].drop(labels=["economy_code"]).to_dict()
    values["lab_eff_periods"] = int(values["lab_eff_periods"])
    values["cap_compare"] = configured[economy_code]["cap_compare"]
    return values
