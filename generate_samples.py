"""Sample / template data generators for the Streamlit app.

The default sample input should mirror the current pipeline output, refreshed
whenever the pipeline is re-run. A synthetic fallback remains only for
offline/demo use when the prepared result files are unavailable.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from core import (
    WIDE_VARIABLE_COLUMNS,
    HISTORY_START_YEAR,
    LAST_PWT_YEAR,
    FINAL_YEAR,
    POPULATION_SAMPLE_FILE,
    POPULATION_SCENARIO_FILES,
    jump_off_year_for_economy,
    load_economies,
)

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DATA_DIR = PROJECT_ROOT / "results" / "data"
SAMPLE_WIDE_INPUT_FILE = "sample_wide_input.csv"
SAMPLE_MODEL_PARAMETERS_FILE = "sample_model_parameters.csv"


def _pipeline_sample_data():
    """Build the sample wide CSV directly from the current pipeline outputs.

    This keeps the Streamlit sample aligned with the actual model input files
    written by the macro pipeline. If the underlying result files are missing,
    we fall back to the legacy synthetic sample for demo-only use.
    """
    required_files = [
        RESULTS_DATA_DIR / "undesa_pop_to2100.csv",
        RESULTS_DATA_DIR / "WDI_to2025.csv",
        RESULTS_DATA_DIR / "IMF_to2031.csv",
        RESULTS_DATA_DIR / "PWT_cap_labour_to2023.csv",
    ]
    if not all(path.exists() for path in required_files):
        return None

    economies = load_economies()
    pop = pd.read_csv(RESULTS_DATA_DIR / "undesa_pop_to2100.csv")
    pop = pop[pop["variable"] == "population_1jan"][
        ["economy_code", "economy", "year", "value"]
    ].copy()
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["value"] = pd.to_numeric(pop["value"], errors="coerce")

    gdp_frames = []
    for name in ["IMF_to2031.csv", "WDI_to2025.csv"]:
        path = RESULTS_DATA_DIR / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["variable"] == "Real GDP PPP 2021 USD"]
        # IMF_to2031.csv still carries legacy rows for all economies; only 18_CT
        # is authoritative there post-migration, all others use WDI (core.postprocess).
        if name == "IMF_to2031.csv":
            frame = frame[frame["economy_code"] == "18_CT"]
        else:
            frame = frame[frame["economy_code"] != "18_CT"]
        gdp_frames.append(frame)
    gdp = pd.concat(gdp_frames, ignore_index=True) if gdp_frames else pd.DataFrame(columns=["economy_code", "economy", "year", "variable", "value"])

    pwt = pd.read_csv(RESULTS_DATA_DIR / "PWT_cap_labour_to2023.csv")
    pwt = pwt[pwt["variable"].isin(["output_to_kstock", "delta"])][["economy_code", "economy", "year", "variable", "value"]].copy()
    pwt["year"] = pd.to_numeric(pwt["year"], errors="coerce")

    savings_frames = []
    for name in ["IMF_savings_2031.csv", "WDI_savings_2025.csv"]:
        path = RESULTS_DATA_DIR / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if name == "IMF_savings_2031.csv":
            frame = frame[frame["economy_code"] == "18_CT"]
        else:
            frame = frame[frame["economy_code"] != "18_CT"]
        savings_frames.append(frame)
    savings = pd.concat(savings_frames, ignore_index=True) if savings_frames else pd.DataFrame(columns=["economy_code", "economy", "year", "variable", "value"])
    if "variable" not in savings.columns:
        savings["variable"] = "Gross national savings"

    nine = pd.DataFrame(columns=["economy_code", "economy", "year", "value"])
    nine_path = RESULTS_DATA_DIR / "GDP_9th.csv"
    if nine_path.exists():
        nine = pd.read_csv(nine_path)
        nine = nine[nine["variable"] == "real_GDP"][ ["economy_code", "economy", "year", "value"] ].copy()
        nine["year"] = pd.to_numeric(nine["year"], errors="coerce")
        nine = nine.rename(columns={"value": "9th Outlook"})

    rows = []
    for _, row in economies.iterrows():
        code = row["economy_code"]
        name = row["economy"]
        years = set(pop[pop["economy_code"] == code]["year"].tolist())
        years |= set(gdp[gdp["economy_code"] == code]["year"].tolist())
        years |= set(pwt[pwt["economy_code"] == code]["year"].tolist())
        years |= set(savings[savings["economy_code"] == code]["year"].tolist())
        years |= set(nine[nine["economy_code"] == code]["year"].tolist())
        for year in sorted(
            int(y) for y in years
            if pd.notna(y) and HISTORY_START_YEAR <= y <= FINAL_YEAR
        ):
            record = {"economy_code": code, "economy": name, "year": year}
            pop_row = pop[(pop["economy_code"] == code) & (pop["year"] == year)]
            if not pop_row.empty:
                record["population_1jan"] = float(pop_row["value"].iloc[0])

            gdp_row = gdp[(gdp["economy_code"] == code) & (gdp["year"] == year)]
            if not gdp_row.empty:
                record["Real GDP PPP 2021 USD"] = float(gdp_row["value"].iloc[0])

            pwt_rows = pwt[(pwt["economy_code"] == code) & (pwt["year"] == year)]
            for _, pwt_row in pwt_rows.iterrows():
                record[pwt_row["variable"]] = float(pwt_row["value"])

            sav_rows = savings[(savings["economy_code"] == code) & (savings["year"] == year)]
            for _, sav_row in sav_rows.iterrows():
                if sav_row.get("variable") == "Gross national savings":
                    record[sav_row["variable"]] = float(sav_row["value"])

            nine_row = nine[(nine["economy_code"] == code) & (nine["year"] == year)]
            if not nine_row.empty:
                record["9th Outlook"] = float(nine_row["9th Outlook"].iloc[0])

            rows.append(record)

    if not rows:
        return None

    sample = pd.DataFrame(rows)
    all_cols = ["economy_code", "economy", "year"] + WIDE_VARIABLE_COLUMNS + (["9th Outlook"] if "9th Outlook" in sample.columns else [])
    for column in all_cols:
        if column not in sample.columns:
            sample[column] = np.nan
    return sample[all_cols].copy()


def make_sample_data(seed=7, include_9th=True):
    """Return the current pipeline-backed sample when available.

    For offline/demo work where the prepared result files do not exist yet,
    we keep the legacy synthetic generator as a fallback so the app still runs.
    """
    pipeline_sample = _pipeline_sample_data()
    if pipeline_sample is not None:
        return pipeline_sample

    rng = np.random.default_rng(seed)
    frames = []
    economies = load_economies()

    for _, row in economies.iterrows():
        code, name = row["economy_code"], row["economy"]
        jump_off = jump_off_year_for_economy(code)

        pop_1990 = rng.uniform(400, 1_150_000)
        pop_growth0 = rng.uniform(0.006, 0.024)
        per_capita_1990 = rng.uniform(2_000, 38_000)
        gdp_growth = rng.uniform(0.02, 0.065)
        ratio = rng.uniform(0.22, 0.42)
        delta = rng.uniform(0.042, 0.068)
        sav = rng.uniform(0.19, 0.36)

        def pop(year):
            return pop_1990 * (1 + pop_growth0 * (1 - (year - 1990) / 130)) ** (year - 1990)

        rows = []
        for year in range(HISTORY_START_YEAR, FINAL_YEAR + 1):
            rows.append((code, name, "population_1jan", year, pop(year)))

        for year in range(HISTORY_START_YEAR, jump_off + 1):
            gdp_pc = per_capita_1990 * (1 + gdp_growth) ** (year - 1990)
            gdp_millions = gdp_pc * pop(year) / 1000
            rows.append((code, name, "Real GDP PPP 2021 USD", year, gdp_millions))
            rows.append((code, name, "Gross national savings", year, sav * 100))

        for year in range(HISTORY_START_YEAR, LAST_PWT_YEAR + 1):
            rows.append((code, name, "output_to_kstock", year, ratio))
            rows.append((code, name, "delta", year, delta))

        if include_9th:
            for year in range(HISTORY_START_YEAR, FINAL_YEAR + 1):
                gdp_pc_9th = per_capita_1990 * (1 + gdp_growth * 0.95) ** (year - 1990)
                rows.append((code, name, "9th Outlook", year, gdp_pc_9th * pop(year) / 1000))

        frames.append(pd.DataFrame(rows, columns=["economy_code", "economy", "variable", "year", "value"]))

    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(index=["economy_code", "economy", "year"], columns="variable", values="value").reset_index()
    wide.columns.name = None
    all_cols = WIDE_VARIABLE_COLUMNS + (["9th Outlook"] if include_9th else [])
    cols = ["economy_code", "economy", "year"] + [v for v in all_cols if v in wide.columns]
    return wide[cols].copy()


def sample_params_csv():
    """Model parameters template pre-filled with the configured parameters
    from config/gdp_model_parameters.csv if present, or model defaults.

    ``cap_compare`` is intentionally omitted because it is fixed by the
    pipeline configuration and is not a user-editable app parameter.
    """
    default_csv = CONFIG_DIR / "gdp_model_parameters.csv"
    if default_csv.exists():
        return pd.read_csv(default_csv).drop(columns=["cap_compare"], errors="ignore")

    defaults = dict(
        lab_eff_periods=10, high_eff=0.015, low_eff=0.012, change_eff=0.002,
        high_sav=0.25, low_sav=0.22, change_sav=0.002, high_delta=0.046,
        low_delta=0.044, change_del=0.0005, alpha=0.4,
    )
    df = pd.DataFrame({"economy_code": load_economies()["economy_code"]})
    for key, value in defaults.items():
        df[key] = value
    df["notes"] = ""
    return df


def _population_scenarios_sample():
    """Build the population-scenarios fallback from the pipeline's UN DESA outputs.

    Mirrors what core.load_population_scenarios expects to find in
    sample_population_scenarios.csv: one row per economy/year/scenario.
    """
    frames = []
    for scenario, filename in POPULATION_SCENARIO_FILES.items():
        path = RESULTS_DATA_DIR / filename
        if not path.exists():
            return None
        population = pd.read_csv(path)
        population = population[population["variable"] == "population_1jan"].copy()
        population["year"] = pd.to_numeric(population["year"], errors="coerce")
        population["value"] = pd.to_numeric(population["value"], errors="coerce")
        population = population[
            population["year"].between(HISTORY_START_YEAR, FINAL_YEAR)
        ].dropna(subset=["economy_code", "year", "value"])
        population["year"] = population["year"].astype(int)
        population["scenario"] = scenario
        frames.append(population[["economy_code", "economy", "year", "value", "scenario"]])

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def refresh_streamlit_samples():
    """Regenerate the Streamlit app's bundled sample CSVs from pipeline outputs.

    Call this at the end of the main pipeline run so the sample files shipped
    with the app (used for demo/offline mode) stay in sync with the latest
    data. Silently skips a file if its required source outputs are not yet
    available (e.g. a partial pipeline run).
    """
    written = []

    wide_sample = _pipeline_sample_data()
    if wide_sample is not None:
        path = APP_DIR / SAMPLE_WIDE_INPUT_FILE
        wide_sample.to_csv(path, index=False)
        written.append(path.name)

    params_sample = sample_params_csv()
    if params_sample is not None:
        path = APP_DIR / SAMPLE_MODEL_PARAMETERS_FILE
        params_sample.to_csv(path, index=False)
        written.append(path.name)

    population_sample = _population_scenarios_sample()
    if population_sample is not None:
        path = APP_DIR / POPULATION_SAMPLE_FILE
        population_sample.to_csv(path, index=False)
        written.append(path.name)

    return written
    return df