r"""Standalone Streamlit app for the APERC Solow-Swan GDP model.

Run from this project root with:
  python -m streamlit run app.py
"""

from pathlib import Path
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

SCENARIO_ORDER = ["Low", "Medium", "High"]
SCENARIO_COLORS = ["#2a9d8f", "#e9c46a", "#e76f51"]
CUSTOM_SCENARIO_COLOR = "#7b4fa3"
DEFAULT_SCENARIO_COLOR = "#172b4d"
SCENARIO_DISPLAY_ORDER = ["High", "Medium", "Low", "Custom", "Default"]
WORLD_TOPOJSON_PATH = Path(__file__).resolve().parent / "assets" / "world-110m.json"
# ISO 3166-1 numeric codes used only to join model results to the basemap geometry.
APEC_MAP_IDS = {
    "01_AUS": "036", "02_BD": "096", "03_CDA": "124", "04_CHL": "152",
    "05_PRC": "156", "06_HKC": "344", "07_INA": "360", "08_JPN": "392",
    "09_ROK": "410", "10_MAS": "458", "11_MEX": "484", "12_NZ": "554",
    "13_PNG": "598", "14_PE": "604", "15_PHL": "608", "16_RUS": "643",
    "17_SGP": "702", "18_CT": "158", "19_THA": "764", "20_USA": "840",
    "21_VN": "704",
}
# Economies too small to be drawn at the basemap resolution get a point marker.
APEC_MAP_MARKERS = {
    "02_BD": (114.73, 4.54),
    "06_HKC": (114.17, 22.32),
    "17_SGP": (103.82, 1.35),
}
MAP_INDICATORS = {
    "Population": {
        "variable": "population", "factor": 1e-3,
        "unit": "Millions", "format": ",.1f", "scale": "sqrt",
    },
    "Real GDP": {
        "variable": "real_GDP", "factor": 1e-3,
        "unit": "Billion USD (2021 PPP)", "format": ",.0f", "scale": "sqrt",
    },
    "GDP per capita": {
        "variable": "GDP_per_capita", "factor": 1.0,
        "unit": "USD (2021 PPP)", "format": ",.0f", "scale": "log",
    },
}
MAP_BASE_YEAR = 2025
MAP_LAND_COLOR = "#e6e9ec"
MAP_APEC_BORDER_COLOR = "#475569"
MAP_SEQUENTIAL_SCHEME = alt.SchemeParams(
    name="plasma", extent=[0.0, 0.75]
)
INDEXED_BASE_YEAR = 2025
INDEXED_MODE = "Indexed (2025 = 100)"
CAGR_START_YEAR = 2025
CAGR_END_YEAR = 2060
INDEXED_COMPARISON_TITLES = {
    "real_GDP": "Real GDP Index (2025 = 100)",
    "GDP_per_capita": "GDP per Capita Index (2025 = 100)",
    "population": "Population Index (2025 = 100)",
}
ECONOMY_COLORS = {
    "01_AUS": "#8B4513",
    "02_BD": "#000000",
    "03_CDA": "#ED5A5F",
    "04_CHL": "#6189EF",
    "05_PRC": "#FF0000",
    "06_HKC": "#F8E277",
    "07_INA": "#E67A0F",
    "08_JPN": "#632DA9",
    "09_ROK": "#0B818E",
    "10_MAS": "#10B8CB",
    "11_MEX": "#20C557",
    "12_NZ": "#58E487",
    "13_PNG": "#F5A85C",
    "14_PE": "#C76ADC",
    "15_PHL": "#F4D125",
    "16_RUS": "#832499",
    "17_SGP": "#137634",
    "18_CT": "#00A86B",
    "19_THA": "#1345C3",
    "20_USA": "#12065F",
    "21_VN": "#B5141A",
}

from core import (
    WIDE_VARIABLE_COLUMNS,
    CRITICAL_VARIABLES,
    HISTORY_START_YEAR,
    LAST_PWT_YEAR,
    LAST_WDI_YEAR,
    LAST_IMF_YEAR,
    FINAL_YEAR,
    REPO_MACRO,  # noqa: F401  (kept explicit so the model path is obvious)
    jump_off_year_for_economy,
    CUSTOM_SCENARIO,
    DEFAULT_SCENARIO,
    build_leap_input_export,
    available_population_scenarios,
    custom_population_status,
    population_scenarios_from_wide,
    load_default_population_scenarios,
    load_repository_gdp_overrides,
    run_uploaded_population_gdp_scenarios,
    run_uploaded_population_scenario,
    uploaded_gdp_projection_overrides,
    load_gdp_model_parameters,
    load_economies,
    economy_code_map,
    run_analysis,
    wide_to_long,
    parse_params_csv,
    parse_uploaded_economy_params,
    validate_wide_model_input,
)
from generate_samples import sample_params_csv

SAMPLE_INPUT_PATH = Path(__file__).resolve().parent / "sample_wide_input.csv"
POPULATION_HISTORY_END_YEAR = 2024


def _indexed_comparison_data(chart_df, economy_codes):
    """Return chart-only indexed values and economies with invalid base values."""
    selected_data = chart_df.reindex(columns=economy_codes).apply(
        pd.to_numeric, errors="coerce"
    )
    if INDEXED_BASE_YEAR not in selected_data.index:
        return None, list(economy_codes)

    baseline = selected_data.loc[INDEXED_BASE_YEAR]
    valid_baseline = baseline.notna() & np.isfinite(baseline) & baseline.gt(0)
    invalid_codes = [
        code for code in economy_codes
        if code not in valid_baseline.index or not bool(valid_baseline.loc[code])
    ]
    if invalid_codes:
        return None, invalid_codes

    indexed_data = selected_data.div(baseline, axis="columns").mul(100)
    indexed_data = indexed_data.where(
        np.isfinite(indexed_data) & selected_data.gt(0)
    )
    return indexed_data, []


def _comparison_cagr_values(chart_df, economy_codes, variable):
    """Return display-ready CAGR values from the fixed 2025 and 2060 endpoints."""
    cagr_values = pd.Series("—", index=economy_codes, dtype=object)
    if variable == "gdp_growth":
        return cagr_values
    if (
        CAGR_START_YEAR not in chart_df.index
        or CAGR_END_YEAR not in chart_df.index
    ):
        return cagr_values

    start_values = pd.to_numeric(
        chart_df.loc[CAGR_START_YEAR].reindex(economy_codes), errors="coerce"
    )
    end_values = pd.to_numeric(
        chart_df.loc[CAGR_END_YEAR].reindex(economy_codes), errors="coerce"
    )
    valid_endpoints = (
        np.isfinite(start_values)
        & np.isfinite(end_values)
        & start_values.gt(0)
        & end_values.gt(0)
    )
    if not valid_endpoints.any():
        return cagr_values

    elapsed_years = CAGR_END_YEAR - CAGR_START_YEAR
    rates = np.expm1(
        (
            np.log(end_values[valid_endpoints])
            - np.log(start_values[valid_endpoints])
        )
        / elapsed_years
    ) * 100
    for economy_code, value in rates.items():
        if np.isfinite(value):
            cagr_values.loc[economy_code] = f"{value:.2f}%"
    return cagr_values


def _comparison_endpoint_ranks(chart_df, economy_codes, variable):
    """Rank selected economies by valid absolute values at 2025 and 2060."""
    rankings = pd.DataFrame(index=economy_codes)
    level_variable = variable != "gdp_growth"
    for year in (CAGR_START_YEAR, CAGR_END_YEAR):
        if year not in chart_df.index:
            values = pd.Series(np.nan, index=economy_codes, dtype=float)
        else:
            values = pd.to_numeric(
                chart_df.loc[year].reindex(economy_codes), errors="coerce"
            )
        valid_values = values.notna() & np.isfinite(values)
        if level_variable:
            valid_values &= values.gt(0)
        rankings[f"rank_{year}"] = values.where(valid_values).rank(
            method="min", ascending=False
        )

    rankings["movement"] = (
        rankings[f"rank_{CAGR_START_YEAR}"]
        - rankings[f"rank_{CAGR_END_YEAR}"]
    )
    return rankings


def _population_paths_caption(
    is_bundled,
    population_view,
    economy_code,
    economy_codes,
    custom_codes,
):
    custom_total_available = bool(economy_codes) and set(custom_codes) == set(
        economy_codes
    )
    if is_bundled:
        if population_view == "Total":
            selections = load_default_population_scenarios()
            loaded_selections = [selections[code] for code in economy_codes]
            historical_sources = []
            if any(
                item["Historical to 2024"] == "EGEDA"
                for item in loaded_selections
            ):
                historical_sources.append("EGEDA historical population")
            if any(
                item["Historical to 2024"] == "Custom"
                for item in loaded_selections
            ):
                historical_sources.append(
                    "researcher-provided custom historical population"
                )

            projection_names = []
            for choice, name in [
                ("MED", "Medium"),
                ("LOW", "Low"),
                ("HIGH", "High"),
            ]:
                if any(item["Projection"] == choice for item in loaded_selections):
                    projection_names.append(name)
            projection_sources = []
            if projection_names:
                projection_sources.append(
                    "UN DESA " + _join_population_sources(projection_names) + " projections"
                )
            if any(
                item["Projection"] == "Custom" for item in loaded_selections
            ):
                projection_sources.append(
                    "researcher-provided custom projections"
                )
            caption = (
                "Default combines the selected historical sources ("
                + _join_population_sources(historical_sources)
                + ") through 2024 with the selected projections ("
                + _join_population_sources(projection_sources)
                + ") from 2025 onward. Low, Medium and High share the same historical data "
                "and use their corresponding UN DESA projections from 2025 onward."
            )
        else:
            selections = load_default_population_scenarios()[economy_code]
            historical_source = (
                "EGEDA historical population"
                if selections["Historical to 2024"] == "EGEDA"
                else "researcher-provided custom historical population"
            )
            projection_choice = selections["Projection"]
            projection_source = {
                "MED": "UN DESA Medium projections",
                "LOW": "UN DESA Low projections",
                "HIGH": "UN DESA High projections",
                "Custom": "researcher-provided custom projections",
            }[projection_choice]
            caption = (
                f"Default combines {historical_source} through 2024 with {projection_source} "
                "from 2025 onward. Low, Medium and High share the same historical population "
                "data and use their corresponding UN DESA projections from 2025 onward."
            )
    else:
        if population_view == "Total":
            caption = (
                "Default, Low, Medium and High use population values supplied in the uploaded "
                "dataset; the underlying sources are not identified."
            )
        else:
            caption = (
                "Default, Low, Medium and High use population values supplied in the uploaded "
                "dataset; the underlying sources are not identified."
            )

    if population_view == "Total":
        if custom_total_available:
            caption += (
                " A separate Custom total is available because custom population paths "
                "cover all economies represented in the loaded data; it is distinct from "
                "the sources used for Default."
            )
    elif custom_codes:
        if economy_code in custom_codes:
            caption += (
                " A separate Custom path is available for this economy and is distinct "
                "from the sources used for Default."
            )
        else:
            caption += (
                " A separate Custom path is available for other economies with complete, "
                "valid custom data, but not for this economy; it does not affect Default."
            )
    return caption


def _join_population_sources(sources):
    if len(sources) == 1:
        return sources[0]
    return ", ".join(sources[:-1]) + " and " + sources[-1]


st.set_page_config(
    page_title="APERC 10th Outlook — Solow-Swan GDP Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("APERC 10th Outlook — Solow-Swan Macro Model")
st.markdown("Interactive front-end for the **APERC 10th Energy Demand and Supply Outlook** macroeconomic projections. ")


def validate(df):
    if "year" not in df.columns:
        return False, "Missing 'year' column"
    if "economy" not in df.columns and "economy_code" not in df.columns:
        return False, "Missing 'economy' or 'economy_code' column"
    have = [c for c in WIDE_VARIABLE_COLUMNS if c in df.columns]
    missing_critical = [v for v in CRITICAL_VARIABLES if v not in have]
    if missing_critical:
        return False, f"Missing critical columns: {missing_critical}"
    optional = [v for v in WIDE_VARIABLE_COLUMNS if v not in have]
    extra = sorted(set(df.columns) - set(WIDE_VARIABLE_COLUMNS)
                   - {"economy", "economy_code", "year"})
    return True, (f"OK. {len(have)} required variable columns present"
                  f" (optional missing: {optional or 'none'}; extra: {extra or 'none'})")


def _apec_results_status(results, errors, included_codes):
    expected_codes = set(
        load_economies()["economy_code"].astype(str)
    )
    included_codes = set(map(str, included_codes))
    if results.empty:
        return False, "No model results were produced."
    result_codes = set(results["economy_code"].astype(str))
    missing = sorted(expected_codes - included_codes)
    if missing:
        return False, "Missing economies: " + ", ".join(missing) + "."
    if errors:
        examples = "; ".join(
            f"{code}: {message}" for code, message in errors[:3]
        )
        return False, "One or more economies failed model calculations. " + examples
    missing_results = sorted(expected_codes - result_codes)
    if missing_results:
        return False, "Missing economies in model results: " + ", ".join(
            missing_results
        ) + "."

    expected_years = set(range(HISTORY_START_YEAR, FINAL_YEAR + 1))
    for code in sorted(expected_codes):
        for variable in ("real_GDP", "population"):
            series = results[
                results["economy_code"].astype(str).eq(code)
                & results["variable"].eq(variable)
            ]
            years = pd.to_numeric(series["year"], errors="coerce")
            if (
                series.empty
                or years.isna().any()
                or years.duplicated().any()
                or set(years.astype(int)) != expected_years
            ):
                return False, (
                    f"Incomplete {variable} result coverage for {code}; "
                    f"all years {HISTORY_START_YEAR}–{FINAL_YEAR} are required."
                )
            values = pd.to_numeric(series["value"], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all() or (values <= 0).any():
                return False, (
                    f"Invalid {variable} result values for {code}; "
                    "all values must be finite and positive."
                )
    return True, ""


def _population_scenario_coverage_error(df_wide):
    frame = df_wide.copy()
    if "economy_code" not in frame.columns:
        frame["economy_code"] = frame["economy"].map(economy_code_map())
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    expected_years = set(range(HISTORY_START_YEAR, FINAL_YEAR + 1))
    required_columns = {
        "Default": "population_1jan",
        "Low": "population_1jan_low",
        "Medium": "population_1jan_medium",
        "High": "population_1jan_high",
    }
    for scenario, column in required_columns.items():
        if column not in frame.columns:
            return f"The {scenario} population path requires column '{column}'."
        for code, group in frame.groupby("economy_code", sort=True):
            years = set(group["year"].dropna().astype(int))
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(
                dtype=float
            )
            if years != expected_years or len(values) != len(expected_years):
                return (
                    f"The {scenario} population path for {code} must contain "
                    f"one value for every year from {HISTORY_START_YEAR} "
                    f"through {FINAL_YEAR}."
                )
            if not np.isfinite(values).all() or (values <= 0).any():
                return (
                    f"The {scenario} population path for {code} must contain "
                    "only finite, positive values."
                )
    return ""


def _population_gdp_scenario_coverage_error(scenarios, df_wide):
    expected_years = set(range(HISTORY_START_YEAR, FINAL_YEAR + 1))
    expected_codes = set(load_economies()["economy_code"].astype(str))
    custom_codes, _ = custom_population_status(df_wide)
    for scenario in available_population_scenarios(df_wide):
        scenario_codes = (
            set(custom_codes) if scenario == CUSTOM_SCENARIO else expected_codes
        )
        for code in sorted(scenario_codes):
            series = scenarios[
                scenarios["scenario"].eq(scenario)
                & scenarios["economy_code"].astype(str).eq(code)
            ]
            years = pd.to_numeric(series["year"], errors="coerce")
            if (
                years.isna().any()
                or years.duplicated().any()
                or set(years.astype(int)) != expected_years
            ):
                return (
                    f"The {scenario} GDP sensitivity result for {code} is "
                    f"incomplete; all years {HISTORY_START_YEAR}–{FINAL_YEAR} "
                    "are required."
                )
            values = pd.to_numeric(series["value"], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all() or (values <= 0).any():
                return (
                    f"The {scenario} GDP sensitivity result for {code} must "
                    "contain only finite, positive values."
                )
    return ""


@st.cache_data(show_spinner="Deriving inputs and running Solow-Swan model for all economies...")
def cached_run(df, params, use_repository_gdp_overrides=False):
    return run_analysis(
        df, params, use_repository_gdp_overrides=use_repository_gdp_overrides
    )


@st.cache_data(show_spinner="Loading population scenarios...")
def cached_population_scenarios(df_wide):
    return population_scenarios_from_wide(df_wide)


@st.cache_data(show_spinner="Running GDP population-sensitivity projections...")
def cached_population_gdp_scenarios(
    df_wide, params, use_repository_gdp_overrides=False
):
    return run_uploaded_population_gdp_scenarios(
        df_wide, params, use_repository_gdp_overrides=use_repository_gdp_overrides
    )


def _gdp_projection_override_end_years(
    df_wide, economy_codes, use_repository_gdp_overrides
):
    economy_codes = sorted({str(code) for code in economy_codes})
    if not economy_codes:
        return {}

    if use_repository_gdp_overrides:
        overrides = load_repository_gdp_overrides()
    else:
        long_input = wide_to_long(df_wide)
        override_frames = [
            uploaded_gdp_projection_overrides(long_input, code)
            for code in economy_codes
        ]
        populated_frames = [frame for frame in override_frames if not frame.empty]
        overrides = (
            pd.concat(populated_frames, ignore_index=True)
            if populated_frames
            else pd.DataFrame(columns=["economy_code", "year", "value"])
        )

    overrides = overrides[overrides["economy_code"].astype(str).isin(economy_codes)]
    if overrides.empty:
        return {}
    return overrides.groupby("economy_code")["year"].max().astype(int).to_dict()


def _gdp_sensitivity_bottom_caption(
    is_bundled,
    population_view,
    economy_code,
    economy_codes,
    override_end_years,
):
    unit = "Real GDP (Millions 2021 USD PPP)."
    if population_view == "Total":
        standard_starts = []
        if any(code != "18_CT" for code in economy_codes):
            standard_starts.append(f"{LAST_WDI_YEAR + 1} for non-CT economies")
        if "18_CT" in economy_codes:
            standard_starts.append(f"{LAST_IMF_YEAR + 1} for CT")

        if is_bundled:
            historical_sources = []
            if any(code != "18_CT" for code in economy_codes):
                historical_sources.append(
                    f"World Bank WDI through {LAST_WDI_YEAR} for non-CT economies"
                )
            if "18_CT" in economy_codes:
                historical_sources.append(
                    f"IMF WEO through {LAST_IMF_YEAR} for CT"
                )
            return (
                unit
                + " The prepared sample uses "
                + " and ".join(historical_sources)
                + ". Researcher-supplied projections are retained where available; "
                "the model generates the remaining projections after each economy's "
                "final supplied year. Without additional supplied projections, model "
                "generation starts in "
                + " and ".join(standard_starts)
                + "."
            )

        return (
            unit
            + " GDP values come from the uploaded data. Supplied future projections "
            "are retained where available; the model generates the remaining "
            "projections after each economy's final supplied year. Without supplied "
            "future projections, model generation starts in "
            + " and ".join(standard_starts)
            + "."
        )

    jump_off_year = jump_off_year_for_economy(economy_code)
    if is_bundled:
        historical_source = (
            "IMF WEO" if economy_code == "18_CT" else "World Bank WDI"
        )
        caption = (
            f"{unit} In the prepared sample, historical GDP uses {historical_source} "
            f"through {jump_off_year}."
        )
    else:
        caption = f"{unit} GDP values come from the uploaded data."

    supplied_end_year = override_end_years.get(economy_code)
    if supplied_end_year is not None:
        supplied_source = (
            "Researcher-supplied projections"
            if is_bundled
            else "Supplied future projections"
        )
        return (
            f"{caption} {supplied_source} are retained through {supplied_end_year}; "
            f"the model generates the remaining projections from "
            f"{supplied_end_year + 1} onward."
        )

    if is_bundled:
        return (
            f"{caption} Without additional supplied projections, the model generates "
            f"projections from {jump_off_year + 1} onward."
        )
    return (
        f"{caption} No future GDP projections are supplied; the model generates "
        f"projections from {jump_off_year + 1} onward after the standard "
        f"{jump_off_year} anchor."
    )


def _deep_dive_gdp_caption(is_bundled, economy_code, override_end_years):
    unit = "Real GDP (Millions 2021 USD PPP)."
    jump_off_year = jump_off_year_for_economy(economy_code)
    supplied_end_year = override_end_years.get(economy_code)

    if is_bundled:
        if economy_code == "18_CT":
            historical_period = (
                f"IMF WEO data and projections extend through {jump_off_year}"
            )
        else:
            historical_period = (
                f"Historical GDP uses World Bank WDI through {jump_off_year}"
            )
        if supplied_end_year is not None:
            return (
                f"{unit} {historical_period}. Researcher-supplied projections are "
                f"retained through {supplied_end_year}, after which the model "
                f"generates the remaining projections from {supplied_end_year + 1} onward."
            )
        return (
            f"{unit} {historical_period}; the model generates projections from "
            f"{jump_off_year + 1} onward."
        )

    if supplied_end_year is not None:
        return (
            f"{unit} GDP values from the uploaded data are used as supplied. Future "
            f"GDP projections are retained through {supplied_end_year}, after which "
            f"the model generates the remaining projections from "
            f"{supplied_end_year + 1} onward."
        )
    return (
        f"{unit} GDP values from the uploaded data are used as supplied; the model "
        f"generates projections from {jump_off_year + 1} onward."
    )


def load_bundled_sample():
    """Load the checked-in sample without rebuilding it from pipeline outputs."""
    return pd.read_csv(SAMPLE_INPUT_PATH)


@st.cache_data(show_spinner="Running population scenario...")
def cached_uploaded_population_scenario(
    df_wide, scenario, params, economy, use_repository_gdp_overrides=False
):
    return run_uploaded_population_scenario(
        df_wide,
        scenario,
        params,
        economy=economy,
        use_repository_gdp_overrides=use_repository_gdp_overrides,
    )


def render_chart_download_button(chart, file_name):
    """Render a small PNG export button beside a chart."""
    try:
        png_bytes = chart.to_image(format="png")
    except Exception:
        return

    st.download_button(
        label="↓",
        data=png_bytes,
        file_name=file_name,
        mime="image/png",
        help="Download this chart as a PNG image.",
        use_container_width=True,
    )


def render_interactive_line_chart(
    data,
    y_title,
    series_title="Series",
    height=360,
    series_order=None,
    series_colors=None,
    value_format=",.2f",
    show_legend=True,
    dashed_series=None,
    fit_y_to_data=False,
    chart_title=None,
):
    """Render a line chart with a nearest-year guide, hover points, and tooltips."""
    chart_data = data.copy()
    if isinstance(chart_data, pd.Series):
        chart_data = chart_data.to_frame(chart_data.name or "Value")
    chart_data.index.name = "year"
    chart_data = chart_data.reset_index().melt(
        id_vars="year", var_name="series", value_name="plot_value"
    ).dropna(subset=["plot_value"])

    y_scale = alt.Undefined
    if fit_y_to_data and not chart_data.empty:
        y_min = float(chart_data["plot_value"].min())
        y_max = float(chart_data["plot_value"].max())
        padding = (y_max - y_min) * 0.05 or abs(y_max) * 0.05 or 1.0
        y_scale = alt.Scale(domain=[y_min - padding, y_max + padding], zero=False)

    hover = alt.selection_point(
        fields=["year"], nearest=True, on="pointerover", empty=False
    )
    legend = alt.Undefined if show_legend else None
    color_encoding = alt.Color(
        "series:N", title=series_title, sort=series_order, legend=legend
    )
    if series_colors:
        color_encoding = alt.Color(
            "series:N",
            title=series_title,
            sort=series_order,
            scale=alt.Scale(range=series_colors),
            legend=legend,
        )
    base = alt.Chart(chart_data).encode(
        x=alt.X("year:Q", title="Year"),
        y=alt.Y("plot_value:Q", title=y_title, scale=y_scale),
        color=color_encoding,
    )
    tooltip = [
        alt.Tooltip("year:Q", title="Year"),
        alt.Tooltip("plot_value:Q", title=y_title, format=value_format),
        alt.Tooltip("series:N", title=series_title),
    ]
    line = base.mark_line().encode(tooltip=alt.value(None))
    if dashed_series:
        line = line.encode(
            strokeDash=alt.condition(
                alt.FieldOneOfPredicate(field="series", oneOf=list(dashed_series)),
                alt.value([6, 4]),
                alt.value([1, 0]),
            )
        )
    selectors = base.mark_point(opacity=0).encode(
        tooltip=alt.value(None)
    ).add_params(hover)
    points = base.mark_circle(size=65).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=tooltip,
    )
    hover_rule = alt.Chart(chart_data).mark_rule(
        color="#a0a0a0", strokeDash=[3, 3]
    ).encode(x="year:Q").transform_filter(hover)
    chart = (line + selectors + points + hover_rule).properties(height=height)
    if chart_title:
        chart = chart.properties(title=chart_title)
    chart_col, download_col = st.columns([10, 1])
    with chart_col:
        st.altair_chart(chart, use_container_width=True)
    with download_col:
        render_chart_download_button(chart, f"{y_title.lower().replace(' ', '_')}.png")



def chart_series_order(series_values):
    """Return series in the requested High → Medium → Low order while keeping historical estimates first."""
    ordered = []
    ordered.extend(
        label for label in series_values
        if label == "Historical estimates" or str(label).endswith(" — Historical estimates")
    )
    for scenario in SCENARIO_DISPLAY_ORDER:
        for label in series_values:
            if (
                label == "Historical estimates"
                or str(label).endswith(" — Historical estimates")
            ):
                continue
            if label == scenario or str(label).endswith(f" — {scenario}"):
                ordered.append(label)
    remaining = [label for label in series_values if label not in ordered]
    return ordered + remaining


def chart_series_colors(series_values):
    """Map the active series labels to the matching scenario colours in the desired series order."""
    ordered = chart_series_order(series_values)
    if not ordered:
        return []
    color_map = {
        "High": SCENARIO_COLORS[2],
        "Medium": SCENARIO_COLORS[1],
        "Low": SCENARIO_COLORS[0],
        "Custom": CUSTOM_SCENARIO_COLOR,
        "Default": DEFAULT_SCENARIO_COLOR,
    }
    colors = []
    for label in ordered:
        if (
            label == "Historical estimates"
            or str(label).endswith(" — Historical estimates")
        ):
            colors.append("#7fb3d5")
            continue
        scenario = next(
            (s for s in SCENARIO_DISPLAY_ORDER if label == s or str(label).endswith(f" — {s}")),
            None,
        )
        colors.append(color_map.get(scenario, "#7fb3d5"))
    return colors


def calculate_cagr(start_value, end_value, years):
    """Return CAGR as a decimal fraction, given start/end values and the elapsed years."""
    if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0 or years <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


@st.cache_resource
def load_world_topojson():
    with open(WORLD_TOPOJSON_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def render_apec_summary_metrics(scenario_results):
    """Show APEC-wide totals for the selected population scenario."""
    gdp_rows = scenario_results[scenario_results["variable"] == "real_GDP"]
    pop_rows = scenario_results[scenario_results["variable"] == "population"]
    if gdp_rows.empty or pop_rows.empty:
        return

    gdp_by_yr = gdp_rows.groupby("year")["value"].sum()
    pop_by_yr = pop_rows.groupby("year")["value"].sum()
    gdp_start, gdp_end = gdp_by_yr.get(MAP_BASE_YEAR, np.nan), gdp_by_yr.get(FINAL_YEAR, np.nan)
    pop_start, pop_end = pop_by_yr.get(MAP_BASE_YEAR, np.nan), pop_by_yr.get(FINAL_YEAR, np.nan)

    def per_capita(gdp, pop):
        return gdp / pop * 1000 if pd.notna(gdp) and pd.notna(pop) and pop != 0 else np.nan

    gdp_pc_start, gdp_pc_end = per_capita(gdp_start, pop_start), per_capita(gdp_end, pop_end)
    gdp_cagr = calculate_cagr(gdp_start, gdp_end, FINAL_YEAR - MAP_BASE_YEAR)

    def delta(start, end, digits):
        if pd.isna(start) or pd.isna(end) or start == 0:
            return None
        return f"{(end / start - 1) * 100:+.{digits}f}% vs {MAP_BASE_YEAR}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            f"APEC GDP ({MAP_BASE_YEAR})",
            f"${gdp_start / 1_000_000:.1f} T" if pd.notna(gdp_start) else "N/A",
            help="Total APEC Real GDP in Trillions (2021 USD PPP)",
        )
    with c2:
        st.metric(
            f"APEC GDP ({FINAL_YEAR})",
            f"${gdp_end / 1_000_000:.1f} T" if pd.notna(gdp_end) else "N/A",
            delta=f"{gdp_cagr * 100:+.2f}% CAGR ({MAP_BASE_YEAR}-{FINAL_YEAR})" if pd.notna(gdp_cagr) else None,
            help=f"Total APEC Real GDP in Trillions (2021 USD PPP); CAGR for {MAP_BASE_YEAR}-{FINAL_YEAR}",
        )
    with c3:
        st.metric(
            f"APEC GDP per Capita ({FINAL_YEAR})",
            f"${gdp_pc_end:,.0f}" if pd.notna(gdp_pc_end) else "N/A",
            delta=delta(gdp_pc_start, gdp_pc_end, 0),
            help="Total APEC Real GDP per capita in 2021 USD PPP",
        )
    with c4:
        st.metric(
            f"APEC Population ({FINAL_YEAR})",
            f"{pop_end / 1_000_000:.2f} B" if pd.notna(pop_end) else "N/A",
            delta=delta(pop_start, pop_end, 1),
            help="Total APEC Population (Billions)",
        )


def _valid_map_values(values, scale):
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = numeric_values.notna() & np.isfinite(numeric_values)
    if scale == "log":
        valid_values &= numeric_values.gt(0)
    elif scale == "sqrt":
        valid_values &= numeric_values.ge(0)
    return numeric_values.loc[valid_values]


def _log_legend_values(low, high, max_ticks=4):
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
        return []

    min_exponent = int(np.floor(np.log10(low)))
    max_exponent = int(np.ceil(np.log10(high)))
    nice_ticks = []
    with np.errstate(over="ignore", under="ignore"):
        for exponent in range(min_exponent, max_exponent + 1):
            magnitude = np.power(10.0, exponent)
            if not np.isfinite(magnitude) or magnitude == 0:
                continue
            nice_ticks.extend(
                tick
                for tick in (magnitude, 2 * magnitude, 5 * magnitude)
                if low <= tick <= high
            )

    nice_ticks = sorted(set(nice_ticks))
    if len(nice_ticks) > max_ticks:
        log_ticks = np.log(nice_ticks)
        target_logs = np.linspace(log_ticks[0], log_ticks[-1], max_ticks)
        remaining_ticks = nice_ticks.copy()
        selected_ticks = []
        for target in target_logs:
            closest = min(
                remaining_ticks,
                key=lambda tick: abs(np.log(tick) - target),
            )
            selected_ticks.append(closest)
            remaining_ticks.remove(closest)
        nice_ticks = sorted(selected_ticks)
    elif not nice_ticks:
        nice_ticks = np.geomspace(low, high, num=max_ticks).tolist()

    unique_ticks = []
    seen_labels = set()
    for tick in nice_ticks:
        label = format(tick, ",.0f")
        if label not in seen_labels:
            unique_ticks.append(float(tick))
            seen_labels.add(label)
    return unique_ticks


def build_apec_map_data(scenario_results, indicator, year):
    """Return one row per APEC economy for the selected indicator and year."""
    spec = MAP_INDICATORS[indicator]
    rows = scenario_results[
        (scenario_results["variable"] == spec["variable"])
        & (scenario_results["year"] == year)
    ]
    values = _valid_map_values(rows["value"], spec["scale"]) * spec["factor"]
    rows = rows.loc[values.index]
    data = pd.DataFrame({
        "economy_code": rows["economy_code"].to_numpy(),
        "Economy": rows["economy"].to_numpy(),
        "value": values.to_numpy(),
    })
    data["id"] = data["economy_code"].map(APEC_MAP_IDS)
    data["Indicator"] = f"{indicator} ({spec['unit']})"
    data["Year"] = str(year)
    data["Value"] = data["value"].map(lambda v: format(v, spec["format"]))
    return data.dropna(subset=["id"])


def render_apec_map(scenario_results, indicator, year):
    """Draw a muted world basemap with the APEC economies shaded by value."""
    spec = MAP_INDICATORS[indicator]
    raw_series = scenario_results[
        scenario_results["variable"] == spec["variable"]
    ]["value"]
    series = _valid_map_values(raw_series, spec["scale"]) * spec["factor"]
    map_data = build_apec_map_data(scenario_results, indicator, year)
    if series.empty or map_data.empty:
        st.info("No values are available for this indicator and year.")
        return

    # Fixed range across all years so colours stay comparable when the year changes.
    low = 0.0 if spec["scale"] == "sqrt" else float(series.min())
    high = float(series.max())
    if spec["scale"] == "log" and low == high:
        padding = np.sqrt(10)
        low /= padding
        high *= padding
    color_scale = alt.Scale(
        type=spec["scale"],
        domain=[low, high],
        scheme=MAP_SEQUENTIAL_SCHEME,
    )
    legend_options = {
        "orient": "bottom",
        "direction": "horizontal",
        "format": ",.0f",
        "gradientLength": 240 if spec["scale"] == "log" else 320,
        "titleLimit": 400,
    }
    if spec["scale"] == "log":
        legend_options["values"] = _log_legend_values(low, high)
        legend_options["labelOverlap"] = False
    legend = alt.Legend(**legend_options)
    color = alt.Color(
        "value:Q",
        title=f"{indicator} ({spec['unit']})",
        scale=color_scale,
        legend=legend,
    )
    tooltip = [
        alt.Tooltip("Economy:N", title="Economy"),
        alt.Tooltip("Indicator:N", title="Indicator"),
        alt.Tooltip("Year:N", title="Year"),
        alt.Tooltip("Value:N", title="Value"),
    ]
    world = alt.NamedData(
        name="world",
        format=alt.DataFormat(type="topojson", feature="countries"),
    )
    # Land is filled and stroked in the same muted colour so that no borders show.
    land = alt.Chart(world).mark_geoshape(
        fill=MAP_LAND_COLOR, stroke=MAP_LAND_COLOR, strokeWidth=0.5
    ).transform_filter("datum.id !== '010'")
    economies = alt.Chart(world).mark_geoshape(
        stroke=MAP_APEC_BORDER_COLOR, strokeWidth=0.5
    ).transform_lookup(
        lookup="id",
        from_=alt.LookupData(map_data, "id", ["Economy", "Indicator", "Year", "Value", "value"]),
    ).transform_filter("isValid(datum.value)").encode(color=color, tooltip=tooltip)

    layers = [land, economies]
    markers = map_data[map_data["economy_code"].isin(APEC_MAP_MARKERS)].copy()
    if not markers.empty:
        markers["longitude"] = markers["economy_code"].map(lambda c: APEC_MAP_MARKERS[c][0])
        markers["latitude"] = markers["economy_code"].map(lambda c: APEC_MAP_MARKERS[c][1])
        layers.append(
            alt.Chart(markers).mark_circle(
                size=110, stroke=MAP_APEC_BORDER_COLOR, strokeWidth=0.8
            ).encode(
                longitude="longitude:Q", latitude="latitude:Q",
                color=alt.Color(
                    "value:Q",
                    title=f"{indicator} ({spec['unit']})",
                    scale=color_scale,
                    legend=legend,
                ), tooltip=tooltip,
            )
        )

    chart = alt.layer(*layers).properties(
        height=460, datasets={"world": load_world_topojson()}
    ).project(type="naturalEarth1", rotate=[-160, 0, 0]).configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True)


def set_deep_dive_parameter_widgets(economy_code, parameters):
    """Load model parameters into the visible Deep Dive controls."""
    widget_key = lambda parameter: f"deep_dive_{economy_code}_{parameter}"
    st.session_state[widget_key("lab_eff_periods")] = int(
        parameters["lab_eff_periods"]
    )
    st.session_state[widget_key("efficiency_range_pct")] = (
        float(parameters["low_eff"]) * 100,
        float(parameters["high_eff"]) * 100,
    )
    st.session_state[widget_key("change_eff_pct")] = (
        float(parameters["change_eff"]) * 100
    )
    st.session_state[widget_key("savings_range_pct")] = (
        float(parameters["low_sav"]) * 100,
        float(parameters["high_sav"]) * 100,
    )
    st.session_state[widget_key("change_sav_pct")] = (
        float(parameters["change_sav"]) * 100
    )
    st.session_state[widget_key("depreciation_range_pct")] = (
        float(parameters["low_delta"]) * 100,
        float(parameters["high_delta"]) * 100,
    )
    st.session_state[widget_key("change_del_pct")] = (
        float(parameters["change_del"]) * 100
    )
    st.session_state[widget_key("alpha_bounded")] = float(parameters["alpha"])


def load_uploaded_deep_dive_parameters():
    """Apply the uploaded parameter row for the currently selected economy."""
    uploaded_file = st.session_state.get("deep_dive_parameter_upload")
    economy_code = st.session_state.get("deep_dive_econ")
    if uploaded_file is None or economy_code is None:
        return

    try:
        uploaded_params = parse_uploaded_economy_params(
            pd.read_csv(io.BytesIO(uploaded_file.getvalue())),
            economy_code,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        st.session_state["deep_dive_parameter_upload_feedback"] = (
            "error",
            f"Could not load parameters: {exc}",
        )
        return

    st.session_state[f"deep_dive_applied_params_{economy_code}"] = (
        uploaded_params.copy()
    )
    set_deep_dive_parameter_widgets(economy_code, uploaded_params)
    st.session_state["deep_dive_parameter_upload_feedback"] = (
        "success",
        f"Parameters loaded successfully for {economy_code}.",
    )


# --- sidebar: data + parameters --------------------------------------------

data_input_error = None

with st.sidebar:
    has_data = "df_wide" in st.session_state

    with st.expander("⚙️ Data & Configuration", expanded=not has_data):
        st.caption(
            "Upload a wide-format model input CSV, or use the bundled sample. "
            "The summary below reflects the active dataset."
        )

        st.markdown("**📁 1. Model Input Data**")
        uploaded = st.file_uploader(
            "Upload Model Input Data CSV",
            type=["csv"],
            help=(
                "Include one row per economy and year from 1990 through 2060, "
                "with the required model inputs. Economy Deep Dive accepts a "
                "subset of APEC economies; Population Scenarios, Economy "
                "Comparison, APEC Overview, and Combined Results & Export "
                "require all 21. Download the sample file for reference."
            ),
        )

        use_sample = st.button("Use Pipeline Sample Data", use_container_width=True, type="primary" if not has_data else "secondary")
        st.download_button(
            "Download Sample Wide CSV",
            SAMPLE_INPUT_PATH.read_bytes(),
            file_name="sample_wide_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if uploaded is not None:
            try:
                uploaded_data = pd.read_csv(uploaded, keep_default_na=False)
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                data_input_error = f"Could not read uploaded model input: {exc}"
            else:
                st.session_state["df_wide"] = uploaded_data
                st.session_state["data_label"] = uploaded.name
                st.session_state["data_source"] = "uploaded"
        elif use_sample:
            st.session_state["df_wide"] = load_bundled_sample()
            st.session_state["data_label"] = "pipeline sample data (21 APEC economies)"
            st.session_state["data_source"] = "bundled"

        st.divider()
        st.markdown("**🛠️ 2. Model Parameters (Optional)**")
        st.caption("Defaults are pre-loaded from `config/gdp_model_parameters.csv`. Upload a CSV to customize tuning knobs.")
        params_file = st.file_uploader(
            "Upload Custom Parameters CSV",
            type=["csv"],
            help=(
                "Upload a custom parameter file to use your own model settings "
                "instead of the preloaded defaults. The file should follow the "
                "required format. Download the sample file for reference."
            ),
        )
        if params_file is not None:
            try:
                uploaded_params = parse_params_csv(
                    pd.read_csv(params_file, keep_default_na=False)
                )
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                st.error(f"Custom parameters were not applied: {exc}")
            else:
                st.session_state["params"] = uploaded_params
                st.success("Custom parameters loaded for all 21 APEC economies")
        st.download_button(
            "Download Parameter Template CSV",
            sample_params_csv().to_csv(index=False).encode(),
            file_name="sample_model_parameters.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()

        # --- data status ------------------------------------------------------

        df = pd.DataFrame(columns=["economy_code"])
        if data_input_error is None and "df_wide" in st.session_state:
            df_wide = st.session_state["df_wide"]
            ok, message = validate(df_wide)
            if not ok:
                data_input_error = message
            else:
                try:
                    validate_wide_model_input(df_wide)
                    df = wide_to_long(df_wide)
                except ValueError as exc:
                    data_input_error = str(exc)
        elif data_input_error is None:
            data_input_error = (
                "No model dataset is loaded. Upload a wide CSV or select "
                "the bundled sample."
            )
            st.info("👈 Upload a wide CSV or click **Use Pipeline Sample Data** to begin.")
        if data_input_error is not None:
            df = pd.DataFrame(columns=["economy_code"])

    use_repository_gdp_overrides = (
        data_input_error is None
        and st.session_state.get("data_source") == "bundled"
    )

    if data_input_error is not None:
        st.caption("**Data loaded:** No valid model dataset")
    elif use_repository_gdp_overrides:
        st.caption("**Data loaded:** Bundled sample")
        st.caption("**Population:** Prepared default historical and projection series")
        st.caption(
            "**GDP data and projections:**\n"
            f"- 20 APEC economies: World Bank WDI data through {LAST_WDI_YEAR}; "
            f"model-generated projections generally begin in {LAST_WDI_YEAR + 1}.\n"
            f"- Chinese Taipei: IMF WEO data through {LAST_IMF_YEAR}; "
            f"model-generated projections generally begin in {LAST_IMF_YEAR + 1}.\n"
            "Additional researcher-provided GDP projections may extend the period "
            "before model-generated projections begin."
        )
    else:
        data_label = st.session_state.get("data_label", "uploaded dataset")
        st.caption(f"**Data loaded:** Uploaded dataset — {data_label}")
        st.caption(
            "**Population:** Uses the population values provided in the uploaded dataset"
        )
        st.caption(
            "**GDP data and projections:** Uses uploaded values. "
            "Model-generated projections begin after "
            "the available GDP data for each economy."
        )

    if data_input_error is None:
        st.caption(f"**Economies:** {df['economy_code'].nunique()}")
    st.caption(f"**Model horizon:** {HISTORY_START_YEAR}–{FINAL_YEAR}")

# --- run model --------------------------------------------------------------

VIEW_DOCS = "📖 Model Methodology"

params = st.session_state.get("params") or load_gdp_model_parameters("gdp_model_parameters.csv")
if (
    data_input_error is not None
    or st.session_state.get("active_view") == VIEW_DOCS
):
    results, inputs, errors = pd.DataFrame(), {}, []
else:
    try:
        results, inputs, errors = cached_run(
            df, params, use_repository_gdp_overrides=use_repository_gdp_overrides
        )
    except ValueError as exc:
        data_input_error = f"Model calculation could not continue: {exc}"
        results, inputs, errors = pd.DataFrame(), {}, []

if data_input_error is not None:
    st.error(data_input_error)

if errors:
    with st.expander(f"⚠️ {len(errors)} economy/ies had warnings or issues - show details"):
        for code, err in errors:
            st.write(f"**{code}**: {err}")

# --- tabs -------------------------------------------------------------------

VIEW_POPULATION = "👥 Population Scenarios"
VIEW_DETAIL = "🔍 Economy Deep Dive"
VIEW_COMPARE = "📈 Economy Comparison"
VIEW_OVERVIEW = "🌏 APEC Overview"
VIEW_DATA = "📋 Combined Results & Export"
APEC_WIDE_VIEWS = {VIEW_POPULATION, VIEW_COMPARE, VIEW_OVERVIEW, VIEW_DATA}

view_labels = [VIEW_POPULATION, VIEW_DETAIL, VIEW_COMPARE, VIEW_OVERVIEW, VIEW_DATA, VIEW_DOCS]
tab_population, tab_detail, tab_compare, tab_overview, tab_data, tab_docs = st.tabs(
    view_labels,
    default=VIEW_POPULATION,
    key="active_view",
    on_change="rerun",
    width="stretch",
)

tabs_by_view = dict(zip(
    view_labels,
    [tab_population, tab_detail, tab_compare, tab_overview, tab_data, tab_docs],
))
active_view = st.session_state["active_view"]
active_tab = tabs_by_view[active_view]
apec_ready, apec_status_message = _apec_results_status(
    results,
    errors,
    df["economy_code"].astype(str).unique(),
)

with active_tab:
    if data_input_error is not None and active_view != VIEW_DOCS:
        st.stop()
    if results.empty and active_view != VIEW_DOCS:
        st.error("No results produced. Please check your uploaded dataset format.")
        st.stop()
    if active_view in APEC_WIDE_VIEWS and not apec_ready:
        st.warning(
            "This tab requires complete results for all 21 APEC economies. "
            + apec_status_message
            + " Economy Deep Dive remains available for the economies in the loaded dataset."
        )
        st.stop()

    # --- TAB 1: Economy Comparison -----------------------------------------
    if active_view == VIEW_COMPARE:
        st.subheader("Multi-Economy Projections Comparison")

        all_codes = sorted(results["economy_code"].unique())
        with st.sidebar:
            st.divider()
            st.markdown("### 📈 Economy Comparison")
            selected_codes = st.multiselect(
                "Select Economies to display",
                options=all_codes,
                default=all_codes[:6] if len(all_codes) >= 6 else all_codes,
                key="comparison_economies",
                help="Choose which economies to include in the comparison chart.",
            )
            metric_choice = st.selectbox(
                "Select Variable",
                options=[
                    ("real_GDP", "Real GDP (Millions 2021 USD PPP)"),
                    ("gdp_growth", "Annual Real GDP Growth Rate (%)"),
                    ("GDP_per_capita", "GDP Per Capita (2021 USD PPP)"),
                    ("population", "Population (Thousands)"),
                ],
                format_func=lambda x: x[1],
                key="comparison_variable",
            )

        var_key, var_title = metric_choice

        chart_df = None
        indexed_data = None
        invalid_codes = []
        if selected_codes:
            sub_comp = results[
                (results["variable"] == var_key)
                & (results["economy_code"].isin(selected_codes))
            ]
            if not sub_comp.empty:
                chart_df = sub_comp.pivot(index="year", columns="economy_code", values="value").sort_index()
                if var_key in INDEXED_COMPARISON_TITLES:
                    indexed_data, invalid_codes = _indexed_comparison_data(
                        chart_df, selected_codes
                    )

        mode_key = f"comparison_display_mode_{var_key}"
        mode_disabled = (
            var_key not in INDEXED_COMPARISON_TITLES
            or chart_df is None
            or bool(invalid_codes)
        )
        if mode_disabled and st.session_state.get(mode_key, "Absolute") != "Absolute":
            st.session_state[mode_key] = "Absolute"
        with st.sidebar:
            display_mode = st.radio(
                "Display Mode",
                options=["Absolute", INDEXED_MODE],
                key=mode_key,
                horizontal=True,
                disabled=mode_disabled,
            )

        if invalid_codes:
            st.warning(
                "Indexed mode is unavailable because the 2025 baseline "
                "is missing, non-finite, zero, or negative for: "
                f"{', '.join(invalid_codes)}."
            )

        if chart_df is not None:
            chart_data = chart_df
            chart_title = var_title
            chart_y_title = var_title
            chart_caption = f"{var_title} ({HISTORY_START_YEAR}–{FINAL_YEAR})"
            if (
                var_key in INDEXED_COMPARISON_TITLES
                and display_mode == INDEXED_MODE
            ):
                chart_data = indexed_data
                chart_title = INDEXED_COMPARISON_TITLES[var_key]
                chart_y_title = "Index (2025 = 100)"
                chart_caption = (
                    f"{var_title} indexed to each economy's 2025 value "
                    f"(= 100) ({HISTORY_START_YEAR}–{FINAL_YEAR}). "
                    "The benchmark-year table continues to show actual "
                    "values in original units."
                )

            render_interactive_line_chart(
                chart_data,
                chart_y_title,
                series_title="Economy",
                series_order=selected_codes,
                series_colors=[ECONOMY_COLORS[code] for code in selected_codes],
                chart_title=chart_title,
            )
            st.caption(chart_caption)

            # Benchmark year summary table
            benchmark_years = [y for y in [1990, 2025, 2030, 2040, 2050, FINAL_YEAR] if y in chart_df.index]
            if benchmark_years:
                st.markdown("##### Benchmark Year Values")
                bench_table = chart_df.loc[benchmark_years].T
                bench_table.columns = [str(year) for year in bench_table.columns]
                cagr_column = "CAGR (2025–2060)"
                bench_table[cagr_column] = _comparison_cagr_values(
                    chart_df, bench_table.index.tolist(), var_key
                )
                benchmark_formatters = {
                    str(year): "{:,.2f}" if var_key == "gdp_growth" else "{:,.0f}"
                    for year in benchmark_years
                }
                st.dataframe(
                    bench_table.style.format(benchmark_formatters),
                    use_container_width=True,
                )

            st.markdown("##### Ranking Changes (2025–2060)")
            st.caption(
                "Rankings compare each selected economy's relative position based on "
                "the selected indicator in 2025 and 2060. Positive movement indicates "
                "a rise in relative ranking among the selected economies, not "
                "necessarily faster growth or a larger absolute change in the indicator."
            )
            economy_names = (
                results.loc[
                    results["economy_code"].isin(selected_codes),
                    ["economy_code", "economy"],
                ]
                .drop_duplicates("economy_code")
                .set_index("economy_code")["economy"]
                .to_dict()
            )
            ranking_rows = _comparison_endpoint_ranks(
                chart_df, selected_codes, var_key
            )
            ranking_rows["economy_name"] = [
                str(economy_names.get(code, code)) for code in ranking_rows.index
            ]
            ranking_rows = ranking_rows.sort_values(
                [f"rank_{CAGR_END_YEAR}", "economy_name"],
                na_position="last",
                kind="stable",
            )

            rank_columns = st.columns([3, 1, 0.5, 1, 1.8])
            for column, heading in zip(
                rank_columns,
                ["Economy", "2025 rank", "→", "2060 rank", "Movement"],
            ):
                with column:
                    st.caption(heading)

            for _, row in ranking_rows.iterrows():
                economy_col, start_col, arrow_col, end_col, movement_col = st.columns(
                    [3, 1, 0.5, 1, 1.8]
                )
                with economy_col:
                    st.text(row["economy_name"])
                with start_col:
                    start_rank = row[f"rank_{CAGR_START_YEAR}"]
                    st.markdown(
                        f"**#{int(start_rank)}**"
                        if pd.notna(start_rank)
                        else "—"
                    )
                with arrow_col:
                    st.markdown("→")
                with end_col:
                    end_rank = row[f"rank_{CAGR_END_YEAR}"]
                    if pd.isna(end_rank):
                        end_rank_label = "—"
                    elif end_rank <= 3:
                        highlight_color = {
                            1: "orange",
                            2: "blue",
                            3: "gray",
                        }[int(end_rank)]
                        end_rank_label = (
                            f":{highlight_color}[**#{int(end_rank)}**]"
                        )
                    else:
                        end_rank_label = f"**#{int(end_rank)}**"
                    st.markdown(end_rank_label)
                with movement_col:
                    movement = row["movement"]
                    if pd.isna(movement):
                        movement_label = ":gray[—]"
                    elif movement > 0:
                        positions = int(movement)
                        suffix = "position" if positions == 1 else "positions"
                        movement_label = (
                            f":green[↑ Up {positions} {suffix}]"
                        )
                    elif movement < 0:
                        positions = abs(int(movement))
                        suffix = "position" if positions == 1 else "positions"
                        movement_label = (
                            f":red[↓ Down {positions} {suffix}]"
                        )
                    else:
                        movement_label = ":gray[→ No change]"
                    st.markdown(movement_label)
        if not selected_codes:
            st.warning("Please select at least one economy to compare.")

    # --- TAB 4: APEC Overview ----------------------------------------------
    if active_view == VIEW_OVERVIEW:
        st.subheader("APEC Overview")
        render_apec_summary_metrics(results)

        map_col1, map_col2 = st.columns([1, 2])
        with map_col1:
            map_indicator = st.radio(
                "Indicator",
                list(MAP_INDICATORS),
                horizontal=True,
                key="overview_indicator",
            )
        with map_col2:
            map_year = st.slider(
                "Model year",
                min_value=HISTORY_START_YEAR,
                max_value=FINAL_YEAR,
                value=MAP_BASE_YEAR,
                key="overview_year",
            )
        render_apec_map(results, map_indicator, map_year)
        st.caption(
            "Colour scale is fixed across all model years for the selected indicator. "
            "Hong Kong, China; Singapore; and Brunei Darussalam are shown as point markers. "
            "The map is for illustrating model results only; boundaries and names shown "
            "do not imply any position on the legal status of any territory."
        )

    # --- TAB 3: Population Scenarios -------------------------------------------
    if active_view == VIEW_POPULATION:
        st.subheader("Population Scenarios")
        st.caption(
            "Compare the Default, Low, Medium, High and available Custom population paths "
            "from the currently loaded wide CSV and examine their impact on long-term GDP."
        )

        try:
            population_scenarios = cached_population_scenarios(df_wide)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            st.info(
                "Add `population_1jan_low`, `population_1jan_medium`, and "
                "`population_1jan_high` columns to the loaded wide CSV."
            )
            population_scenarios = pd.DataFrame()

        if not population_scenarios.empty:
            coverage_error = _population_scenario_coverage_error(df_wide)
            if coverage_error:
                st.error(
                    "APEC population scenarios are unavailable. " + coverage_error
                )
                st.stop()
            scenario_order = available_population_scenarios(df_wide)
            custom_codes, incomplete_custom_codes = custom_population_status(df_wide)
            all_population_codes = sorted(population_scenarios["economy_code"].unique())

            population_labels = [
                population_scenarios.loc[
                    population_scenarios["economy_code"] == code, "economy"
                ].iloc[0] + f" ({code})"
                for code in all_population_codes
            ]
            population_label_to_code = dict(zip(population_labels, all_population_codes))
            scenario_choice = [DEFAULT_SCENARIO, "High", "Low", "Medium"]
            if CUSTOM_SCENARIO in scenario_order:
                scenario_choice.append(CUSTOM_SCENARIO)
            scenario_choice.append("All")
            if st.session_state.get("population_scenario_selected") not in scenario_choice:
                st.session_state.pop("population_scenario_selected", None)

            year_min = int(population_scenarios["year"].min())
            year_max = int(population_scenarios["year"].max())
            with st.sidebar:
                st.divider()
                st.markdown("### 👥 Population Scenarios")
                population_view = st.radio(
                    "View",
                    ["Total", "By Economy"],
                    index=0,
                    key="population_view",
                    horizontal=True,
                )
                selected_population_label = st.selectbox(
                    "Economy",
                    population_labels,
                    index=0,
                    key="population_economies_selected",
                    disabled=population_view == "Total",
                    help="Choose one economy for the economy-level view.",
                )
            with st.sidebar:
                selected_scenario = st.selectbox(
                    "Scenarios",
                    scenario_choice,
                    index=len(scenario_choice) - 1,
                    key="population_scenario_selected",
                    help="Choose Default, High, Low, Medium, Custom (if provided), or All scenarios.",
                )

                year_from, year_to = st.columns(2)
                with year_from:
                    population_year_from = st.number_input(
                        "From",
                        min_value=year_min,
                        max_value=year_max,
                        value=year_min,
                        step=1,
                        key="population_year_from_1990",
                    )
                with year_to:
                    population_year_to = st.number_input(
                        "To",
                        min_value=year_min,
                        max_value=year_max,
                        value=year_max,
                        step=1,
                        key="population_year_to_final",
                    )

            selected_scenarios = scenario_order if selected_scenario == "All" else [selected_scenario]
            if incomplete_custom_codes:
                st.warning(
                    "Custom population ignored for economies with missing or invalid values: "
                    + ", ".join(incomplete_custom_codes)
                )
            if (
                population_view == "Total"
                and CUSTOM_SCENARIO in selected_scenarios
                and set(custom_codes) != set(all_population_codes)
            ):
                selected_scenarios = [s for s in selected_scenarios if s != CUSTOM_SCENARIO]
                st.info(
                    "Custom is not shown in the total because it does not cover all economies in the loaded data. "
                    "Use the By Economy view."
                )

            if not selected_scenarios:
                st.warning("Select at least one population scenario.")
                st.stop()
            if population_year_from > population_year_to:
                st.warning("The From year must be earlier than or equal to the To year.")
                st.stop()

            population_scenarios_view = population_scenarios[
                population_scenarios["year"].between(population_year_from, population_year_to)
            ]

            if population_view == "Total":
                selected_population_codes = all_population_codes
            else:
                selected_population_codes = [population_label_to_code[selected_population_label]]
            common_population = population_scenarios_view[
                population_scenarios_view["economy_code"].isin(selected_population_codes)
                &
                population_scenarios_view["scenario"].isin(selected_scenarios)
            ]
            common_population_chart = common_population.pivot_table(
                index="year", columns=["economy", "scenario"], values="value"
            ).sort_index()
            economy_order = [
                population_scenarios_view.loc[
                    population_scenarios_view["economy_code"] == code, "economy"
                ].iloc[0]
                for code in selected_population_codes
            ]
            common_population_chart = common_population_chart.reindex(
                columns=pd.MultiIndex.from_product(
                    [economy_order, selected_scenarios], names=["economy", "scenario"]
                )
            )
            common_population_chart.columns = [
                f"{economy} — {scenario}"
                for economy, scenario in common_population_chart.columns
            ]
            apec_population = population_scenarios_view[
                population_scenarios_view["scenario"].isin(selected_scenarios)
            ]
            selected_total = apec_population.groupby(
                ["year", "scenario"], as_index=False
            )["value"].sum().pivot(index="year", columns="scenario", values="value").reindex(
                columns=scenario_order
            )
            selected_total = selected_total[
                [scenario for scenario in scenario_order if scenario in selected_scenarios]
            ]

            if population_view == "Total":
                chart_data = selected_total
                st.markdown("#### Total APEC Population Paths")
                st.caption(
                    "Aggregate population across the economies under the selected "
                    "population scenarios."
                )
            else:
                chart_data = common_population_chart
                st.markdown("#### Population Paths by Economy")
            population_long = chart_data.reset_index().melt(
                id_vars="year", var_name="series", value_name="value"
            ).dropna(subset=["value"])
            population_plot = population_long.copy()
            population_plot["is_default"] = (
                population_plot["series"].eq(DEFAULT_SCENARIO)
                | population_plot["series"].str.endswith(f" — {DEFAULT_SCENARIO}")
            )
            population_series_order = chart_series_order(population_plot["series"].unique())
            population_hover = alt.selection_point(
                fields=["year"], nearest=True, on="pointerover", empty=False
            )
            population_base = alt.Chart(population_plot).encode(
                x=alt.X("year:Q", title="Year"),
                y=alt.Y("value:Q", title="Population (thousands)"),
                color=alt.Color(
                    "series:N",
                    title="Path",
                    sort=population_series_order,
                    scale=alt.Scale(range=chart_series_colors(population_plot["series"].unique())),
                ),
            )
            population_tooltip = [
                alt.Tooltip("year:Q", title="Year"),
                alt.Tooltip("value:Q", title="Population", format=",.2f"),
                alt.Tooltip("series:N", title="Series"),
            ]
            population_line = population_base.transform_filter(
                "!datum.is_default"
            ).mark_line().encode(tooltip=alt.value(None))
            population_default_line = population_base.transform_filter(
                "datum.is_default"
            ).mark_line().encode(tooltip=alt.value(None))
            population_selectors = population_base.mark_point(opacity=0).encode(
                tooltip=alt.value(None)
            ).add_params(population_hover)
            population_points = population_base.mark_circle(size=65).encode(
                opacity=alt.condition(population_hover, alt.value(1), alt.value(0)),
                tooltip=population_tooltip,
            )
            population_hover_rule = alt.Chart(population_plot).mark_rule(
                color="#a0a0a0", strokeDash=[3, 3]
            ).encode(x="year:Q").transform_filter(population_hover)
            population_divider = alt.Chart(pd.DataFrame({"year": [POPULATION_HISTORY_END_YEAR + 1]})).mark_rule(
                color="#b8b8b8", strokeDash=[5, 5]
            ).encode(x="year:Q")
            population_chart = (
                population_line
                + population_default_line
                + population_selectors
                + population_points
                + population_hover_rule
                + population_divider
            ).properties(height=360)
            chart_col, download_col = st.columns([10, 1])
            with chart_col:
                st.altair_chart(population_chart, use_container_width=True)
            with download_col:
                render_chart_download_button(population_chart, "apec_population_paths.png")
            if population_view == "Total":
                caption_economy_code = None
            else:
                caption_economy_code = population_label_to_code[
                    selected_population_label
                ]
            population_caption = _population_paths_caption(
                is_bundled=st.session_state.get("data_source") == "bundled",
                population_view=population_view,
                economy_code=caption_economy_code,
                economy_codes=selected_population_codes,
                custom_codes=custom_codes,
            )
            st.caption(population_caption)
            st.download_button(
                "⬇️ Download population data",
                population_scenarios.to_csv(index=False).encode(),
                file_name="apec_population_scenarios.csv",
                mime="text/csv",
                help="Download all available population scenario paths used by this tab.",
            )

            st.markdown(f"#### Real GDP Population Sensitivity ({HISTORY_START_YEAR}–{FINAL_YEAR})")
            st.caption(
                "GDP projections reflect the selected population scenario and other model inputs."
            )
            try:
                population_gdp_scenarios, sensitivity_errors = (
                    cached_population_gdp_scenarios(
                        df_wide,
                        params,
                        use_repository_gdp_overrides=use_repository_gdp_overrides,
                    )
                )
                if population_gdp_scenarios.empty:
                    raise ValueError(
                        "No GDP sensitivity projections were produced from the loaded population paths."
                    )
                if sensitivity_errors:
                    error_examples = "; ".join(
                        f"{scenario}/{code}: {message}"
                        for scenario, code, message in sensitivity_errors[:5]
                    )
                    st.error(
                        f"Population sensitivity is unavailable because "
                        f"{len(sensitivity_errors)} economy/scenario run(s) failed. "
                        f"Examples: {error_examples}"
                    )
                    st.stop()
                scenario_coverage_error = _population_gdp_scenario_coverage_error(
                    population_gdp_scenarios, df_wide
                )
                if scenario_coverage_error:
                    st.error(
                        "APEC population sensitivity is unavailable. "
                        + scenario_coverage_error
                    )
                    st.stop()
                gdp_economy_codes = sorted(
                    population_gdp_scenarios["economy_code"].astype(str).unique()
                )
                override_end_years = _gdp_projection_override_end_years(
                    df_wide,
                    gdp_economy_codes,
                    use_repository_gdp_overrides,
                )
                population_gdp_view = population_gdp_scenarios[
                    population_gdp_scenarios["year"].between(
                        population_year_from, population_year_to
                    )
                    & population_gdp_scenarios["scenario"].isin(selected_scenarios)
                ]
                if population_view == "Total":
                    gdp_chart_data = population_gdp_view.groupby(
                        ["year", "scenario"], as_index=False
                    )["value"].sum().pivot(
                        index="year", columns="scenario", values="value"
                    ).reindex(columns=scenario_order)
                    gdp_chart_data = gdp_chart_data[
                        [scenario for scenario in scenario_order if scenario in selected_scenarios]
                    ]
                    st.caption(
                        "Total real GDP across the economies represented in the loaded "
                        "data under each selected population scenario."
                    )
                else:
                    population_gdp_view = population_gdp_view[
                        population_gdp_view["economy_code"].isin(selected_population_codes)
                    ]
                    gdp_chart_data = population_gdp_view.pivot_table(
                        index="year", columns=["economy", "scenario"], values="value"
                    ).sort_index()
                    economy_order = [
                        population_scenarios.loc[
                            population_scenarios["economy_code"] == code, "economy"
                        ].iloc[0]
                        for code in selected_population_codes
                    ]
                    gdp_chart_data = gdp_chart_data.reindex(
                        columns=pd.MultiIndex.from_product(
                            [economy_order, selected_scenarios],
                            names=["economy", "scenario"],
                        )
                    )
                    gdp_chart_data.columns = [
                        f"{economy} — {scenario}"
                        for economy, scenario in gdp_chart_data.columns
                    ]
                    st.caption(
                        "Real GDP paths for the selected economy under each population scenario."
                    )
                gdp_series_order = chart_series_order(list(gdp_chart_data.columns))
                gdp_plot = gdp_chart_data.reset_index().melt(
                    id_vars="year", var_name="series", value_name="value"
                ).dropna(subset=["value"])
                gdp_plot["is_default"] = (
                    gdp_plot["series"].eq(DEFAULT_SCENARIO)
                    | gdp_plot["series"].str.endswith(f" — {DEFAULT_SCENARIO}")
                )
                gdp_hover = alt.selection_point(
                    fields=["year"], nearest=True, on="pointerover", empty=False
                )
                gdp_base = alt.Chart(gdp_plot).encode(
                    x=alt.X("year:Q", title="Year"),
                    y=alt.Y("value:Q", title="Real GDP (millions 2021 USD PPP)"),
                    color=alt.Color(
                        "series:N",
                        title="Population scenario",
                        sort=gdp_series_order,
                        scale=alt.Scale(range=chart_series_colors(list(gdp_chart_data.columns))),
                    ),
                )
                gdp_tooltip = [
                    alt.Tooltip("year:Q", title="Year"),
                    alt.Tooltip("value:Q", title="Real GDP (millions 2021 USD PPP)", format=",.2f"),
                    alt.Tooltip("series:N", title="Population scenario"),
                ]
                gdp_line = gdp_base.transform_filter(
                    "!datum.is_default"
                ).mark_line().encode(tooltip=alt.value(None))
                gdp_default_line = gdp_base.transform_filter(
                    "datum.is_default"
                ).mark_line().encode(tooltip=alt.value(None))
                gdp_selectors = gdp_base.mark_point(opacity=0).encode(
                    tooltip=alt.value(None)
                ).add_params(gdp_hover)
                gdp_points = gdp_base.mark_circle(size=65).encode(
                    opacity=alt.condition(gdp_hover, alt.value(1), alt.value(0)),
                    tooltip=gdp_tooltip,
                )
                gdp_hover_rule = alt.Chart(gdp_plot).mark_rule(
                    color="#a0a0a0", strokeDash=[3, 3]
                ).encode(x="year:Q").transform_filter(gdp_hover)
                gdp_chart = (
                    gdp_line
                    + gdp_default_line
                    + gdp_selectors
                    + gdp_points
                    + gdp_hover_rule
                ).properties(height=360)
                chart_col, download_col = st.columns([10, 1])
                with chart_col:
                    st.altair_chart(gdp_chart, use_container_width=True)
                with download_col:
                    render_chart_download_button(gdp_chart, "apec_gdp_population_sensitivity.png")
                selected_gdp_code = (
                    None
                    if population_view == "Total"
                    else selected_population_codes[0]
                )
                caption_economy_codes = (
                    gdp_economy_codes
                    if population_view == "Total"
                    else selected_population_codes
                )
                st.caption(
                    _gdp_sensitivity_bottom_caption(
                        is_bundled=use_repository_gdp_overrides,
                        population_view=population_view,
                        economy_code=selected_gdp_code,
                        economy_codes=caption_economy_codes,
                        override_end_years=override_end_years,
                    )
                )
                st.download_button(
                    "⬇️ Download GDP sensitivity data",
                    population_gdp_view.to_csv(index=False).encode(),
                    file_name="apec_gdp_population_sensitivity.csv",
                    mime="text/csv",
                    help="Download the GDP sensitivity data currently selected for this chart.",
                )
            except (FileNotFoundError, ValueError) as exc:
                st.warning(str(exc))

    # --- TAB 2: Economy Deep Dive ----------------------------------------------
    if active_view == VIEW_DETAIL:
        st.subheader("Economy Deep Dive")
        included_economies = sorted(df["economy_code"].astype(str).unique())
        st.caption(
            f"Loaded dataset includes {len(included_economies)} of 21 APEC "
            "economies. This deep dive is limited to the economies included."
        )
        st.caption(
            "Refine the GDP projection by modifying key macroeconomic parameters, including savings rates, "
            "depreciation rates, and labour efficiency growth. Compare the resulting projections against the "
            "rebased 9th Outlook (2021 USD PPP)."
        )
        with st.sidebar:
            st.divider()
            st.markdown("### 🔍 Economy Deep Dive")
            sel_code = st.selectbox(
                "Choose Economy",
                sorted(results["economy_code"].unique()),
                key="deep_dive_econ",
                on_change=load_uploaded_deep_dive_parameters,
            )
            jump_off = jump_off_year_for_economy(sel_code)
            source_name = "IMF" if sel_code == "18_CT" else "WDI"

            deep_dive_options = [DEFAULT_SCENARIO, "Medium", "Low", "High"]
            if sel_code in custom_population_status(df_wide)[0]:
                deep_dive_options.append(CUSTOM_SCENARIO)
            if st.session_state.get("deep_dive_population_scenario") not in deep_dive_options:
                st.session_state.pop("deep_dive_population_scenario", None)
            deep_dive_scenario = st.radio(
                "Population Scenario",
                deep_dive_options,
                horizontal=True,
                key="deep_dive_population_scenario",
                help="Default uses population_1jan from the loaded CSV. Each selection reruns this economy using its population column and current Deep Dive parameters.",
            )
            show_9th_outlook = st.toggle(
                "Show rebased 9th Outlook",
                value=False,
                key="deep_dive_show_9th_outlook",
                help="Display the rebased 9th Outlook GDP comparison line.",
            )

            st.markdown(f"**Historical Anchor:** {jump_off} ({source_name})")
            st.markdown(f"**Projection Period:** {jump_off + 1}–{FINAL_YEAR}")

            econ_params = params.get(sel_code, {}).copy()
            applied_params_key = f"deep_dive_applied_params_{sel_code}"
            if applied_params_key not in st.session_state:
                st.session_state[applied_params_key] = econ_params.copy()

            with st.expander("Model Parameters", expanded=True):
                widget_key = lambda parameter: f"deep_dive_{sel_code}_{parameter}"
                if st.button(
                    "↻ Reset to Preloaded Defaults",
                    key=f"deep_dive_reset_params_{sel_code}",
                    help="Restore this economy's values from config/gdp_model_parameters.csv and apply them immediately.",
                    use_container_width=True,
                ):
                    configured_params = load_gdp_model_parameters(
                        "gdp_model_parameters.csv"
                    )[sel_code]
                    st.session_state[applied_params_key] = configured_params.copy()
                    set_deep_dive_parameter_widgets(sel_code, configured_params)

                with st.form(
                    f"deep_dive_parameter_form_{sel_code}",
                    clear_on_submit=False,
                    border=False,
                ):
                    run_model = st.form_submit_button(
                        "Run Model",
                        type="primary",
                        width="stretch",
                        help="Apply all parameter changes and rerun this economy.",
                    )

                    lab_eff_periods = st.slider(
                        "Labour efficiency look-back period",
                        min_value=1,
                        max_value=10,
                        value=int(econ_params["lab_eff_periods"]),
                        key=widget_key("lab_eff_periods"),
                        help="Number of recent years used to calculate the average labour-efficiency growth rate for each projection year. The average is recalculated each year using the preceding years in the look-back period. Longer periods produce more stable estimates, while shorter periods place greater weight on recent trends.",
                    )

                    low_eff_pct, high_eff_pct = st.slider(
                        "Labour efficiency growth corridor (%)",
                        min_value=0.5,
                        max_value=10.0,
                        value=(float(econ_params["low_eff"]) * 100, float(econ_params["high_eff"]) * 100),
                        step=0.01,
                        format="%.1f",
                        key=widget_key("efficiency_range_pct"),
                        help="Lower and upper bounds for future labour-efficiency growth. Labour efficiency captures improvements in productivity that cannot be explained by labour or capital alone.",
                    )
                    change_eff = st.number_input(
                        "Annual efficiency adjustment (%)",
                        min_value=0.0,
                        value=float(econ_params["change_eff"]) * 100,
                        step=0.01,
                        format="%.4f",
                        key=widget_key("change_eff_pct"),
                        help="Determines how much the labour-efficiency growth rate changes each year toward the specified corridor. A larger value moves the rate toward the corridor more quickly, and a smaller value moves it more slowly.",
                    )

                    low_sav_pct, high_sav_pct = st.slider(
                        "Savings rate corridor (%)",
                        min_value=15.0,
                        max_value=50.0,
                        value=(float(econ_params["low_sav"]) * 100, float(econ_params["high_sav"]) * 100),
                        step=0.1,
                        format="%.1f",
                        key=widget_key("savings_range_pct"),
                        help="Lower and upper bounds for the savings rate. Savings are assumed to equal investment, which increases the capital stock available for future production.",
                    )
                    change_sav = st.number_input(
                        "Annual savings adjustment (%)",
                        min_value=0.0,
                        value=float(econ_params["change_sav"]) * 100,
                        step=0.01,
                        format="%.4f",
                        key=widget_key("change_sav_pct"),
                        help="Determines how much the savings rate changes each year toward the specified corridor. A larger value moves the rate toward the corridor more quickly, and a smaller value moves it more slowly.",
                    )

                    low_delta_pct, high_delta_pct = st.slider(
                        "Depreciation rate corridor (%)",
                        min_value=3.0,
                        max_value=7.0,
                        value=(float(econ_params["low_delta"]) * 100, float(econ_params["high_delta"]) * 100),
                        step=0.01,
                        format="%.1f",
                        key=widget_key("depreciation_range_pct"),
                        help="Lower and upper bounds for the depreciation rate. Depreciation reduces the productive capital stock through wear and tear, ageing, and technological obsolescence.",
                    )
                    change_del = st.number_input(
                        "Annual depreciation adjustment (%)",
                        min_value=0.0,
                        value=float(econ_params["change_del"]) * 100,
                        step=0.01,
                        format="%.4f",
                        key=widget_key("change_del_pct"),
                        help="Determines how much the depreciation rate changes each year toward the specified corridor. A larger value moves the rate toward the corridor more quickly, and a smaller value moves it more slowly.",
                    )
                    alpha = st.number_input(
                        "Capital share (alpha)",
                        min_value=0.3,
                        max_value=0.5,
                        value=float(econ_params["alpha"]),
                        step=0.05,
                        format="%.2f",
                        key=widget_key("alpha_bounded"),
                        help="Capital share parameter in the Cobb-Douglas production function. Higher values imply economic output is more sensitive to changes in capital stock relative to labour.",
                    )

                    pending_params = econ_params.copy()
                    pending_params.update({
                        "lab_eff_periods": lab_eff_periods,
                        "low_eff": low_eff_pct / 100,
                        "high_eff": high_eff_pct / 100,
                        "change_eff": change_eff / 100,
                        "low_sav": low_sav_pct / 100,
                        "high_sav": high_sav_pct / 100,
                        "change_sav": change_sav / 100,
                        "low_delta": low_delta_pct / 100,
                        "high_delta": high_delta_pct / 100,
                        "change_del": change_del / 100,
                        "alpha": alpha,
                    })

                if run_model:
                    st.session_state[applied_params_key] = pending_params

                export_columns = sample_params_csv().columns.tolist()
                export_row = {
                    "economy_code": sel_code,
                    **st.session_state[applied_params_key],
                    "notes": "",
                }
                economy_params_csv = pd.DataFrame([export_row]).reindex(columns=export_columns)
                st.file_uploader(
                    "Upload Parameters CSV",
                    type=["csv"],
                    key="deep_dive_parameter_upload",
                    on_change=load_uploaded_deep_dive_parameters,
                    help="Upload custom parameter settings for the selected economy.",
                )
                st.download_button(
                    f"Download {sel_code} parameters CSV",
                    economy_params_csv.to_csv(index=False).encode(),
                    file_name=f"{sel_code}_model_parameters.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                    help="Download the parameters currently applied to this economy.",
                )
                upload_feedback = st.session_state.get(
                    "deep_dive_parameter_upload_feedback"
                )
                if upload_feedback is not None:
                    if upload_feedback[0] == "success":
                        st.success(upload_feedback[1])
                    else:
                        st.error(upload_feedback[1])

            deep_dive_params = {
                **params,
                sel_code: st.session_state[applied_params_key],
            }

        try:
            deep_dive_results, deep_dive_errors = cached_uploaded_population_scenario(
                df_wide,
                deep_dive_scenario,
                deep_dive_params,
                sel_code,
                use_repository_gdp_overrides=use_repository_gdp_overrides,
            )
            economy_error = next(
                (
                    message for code, message in deep_dive_errors
                    if code == sel_code
                ),
                None,
            )
            if economy_error:
                st.error(
                    f"{deep_dive_scenario} population scenario failed for "
                    f"{sel_code}: {economy_error}"
                )
                deep_dive_results = pd.DataFrame(columns=results.columns)
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Population scenario could not be applied: {exc}")
            deep_dive_results = pd.DataFrame(columns=results.columns)

        with st.container():
            econ_results = deep_dive_results

            # Primary GDP Chart
            st.markdown(f"#### Real GDP Trajectory ({HISTORY_START_YEAR}–{FINAL_YEAR})")
            displayed_gdp_variables = ["real_GDP"]
            if show_9th_outlook:
                displayed_gdp_variables.append("real_GDP_9th")
            gdp_series = econ_results[econ_results["variable"].isin(displayed_gdp_variables)]
            if not gdp_series.empty:
                gdp_chart = gdp_series.pivot(index="year", columns="variable", values="value").reset_index()
                historical_gdp = gdp_chart[
                    gdp_chart["year"] <= jump_off
                ][["year", "real_GDP"]].rename(columns={"real_GDP": "value"})
                historical_gdp["series"] = "10th Outlook historical"
                projected_gdp = gdp_chart[
                    gdp_chart["year"] > jump_off
                ][["year", "real_GDP"]].rename(columns={"real_GDP": "value"})
                projected_gdp["series"] = "10th Outlook projection"
                gdp_plot = pd.concat([historical_gdp, projected_gdp], ignore_index=True)
                if show_9th_outlook and "real_GDP_9th" in gdp_chart.columns:
                    ninth_gdp = gdp_chart[["year", "real_GDP_9th"]].dropna().rename(
                        columns={"real_GDP_9th": "value"}
                    )
                    ninth_gdp["series"] = "9th Outlook (Rebased)"
                    gdp_plot = pd.concat([gdp_plot, ninth_gdp], ignore_index=True)
                gdp_hover = alt.selection_point(
                    fields=["year"], nearest=True, on="pointerover", empty=False
                )
                gdp_base = alt.Chart(gdp_plot).encode(
                    x=alt.X("year:Q", title="Year"),
                    y=alt.Y("value:Q", title="Real GDP (millions 2021 USD PPP)"),
                    color=alt.Color(
                        "series:N",
                        title="Series",
                        sort=[
                            "10th Outlook historical",
                            "10th Outlook projection",
                            "9th Outlook (Rebased)",
                        ],
                        scale=alt.Scale(range=["#7fb3d5", "#1f4e79", "#c0392b"]),
                    ),
                    strokeDash=alt.StrokeDash(
                        "series:N",
                        sort=[
                            "10th Outlook historical",
                            "10th Outlook projection",
                            "9th Outlook (Rebased)",
                        ],
                        scale=alt.Scale(range=[[1, 0], [6, 4], [1, 0]]),
                        legend=None,
                    ),
                )
                gdp_tooltip = [
                    alt.Tooltip("year:Q", title="Year"),
                    alt.Tooltip("value:Q", title="Real GDP", format=",.2f"),
                    alt.Tooltip("series:N", title="Series"),
                ]
                gdp_line = gdp_base.mark_line().encode(tooltip=alt.value(None))
                gdp_selectors = gdp_base.mark_point(opacity=0).encode(
                    tooltip=alt.value(None)
                ).add_params(gdp_hover)
                gdp_points = gdp_base.mark_circle(size=65).encode(
                    opacity=alt.condition(gdp_hover, alt.value(1), alt.value(0)),
                    tooltip=gdp_tooltip,
                )
                gdp_hover_rule = alt.Chart(gdp_plot).mark_rule(
                    color="#a0a0a0", strokeDash=[3, 3]
                ).encode(x="year:Q").transform_filter(gdp_hover)
                st.altair_chart(
                    (gdp_line + gdp_selectors + gdp_points + gdp_hover_rule).properties(height=360),
                    use_container_width=True,
                )
                override_end_years = _gdp_projection_override_end_years(
                    df_wide,
                    [sel_code],
                    use_repository_gdp_overrides,
                )
                st.caption(
                    _deep_dive_gdp_caption(
                        use_repository_gdp_overrides,
                        sel_code,
                        override_end_years,
                    )
                )

            # Population path used by the current model run
            st.markdown("#### Population Projection")
            st.caption(f"{deep_dive_scenario} scenario")
            population_path = econ_results[
                econ_results["variable"] == "population"
            ].set_index("year")["value"].sort_index()
            if not population_path.empty:
                if deep_dive_scenario == CUSTOM_SCENARIO:
                    population_plot = pd.DataFrame({CUSTOM_SCENARIO: population_path})
                    population_order = [CUSTOM_SCENARIO]
                    population_colors = ["#1f4e79"]
                    population_dashed = None
                else:
                    population_plot = pd.DataFrame({
                        "Historical": population_path[population_path.index <= POPULATION_HISTORY_END_YEAR],
                        "Projection": population_path[population_path.index >= POPULATION_HISTORY_END_YEAR],
                    })
                    population_order = ["Historical", "Projection"]
                    population_colors = (
                        [DEFAULT_SCENARIO_COLOR, DEFAULT_SCENARIO_COLOR]
                        if deep_dive_scenario == DEFAULT_SCENARIO
                        else ["#7fb3d5", "#1f4e79"]
                    )
                    population_dashed = ["Projection"]
                render_interactive_line_chart(
                    population_plot,
                    "Population (thousands)",
                    series_order=population_order,
                    series_colors=population_colors,
                    value_format=",.0f",
                    show_legend=False,
                    dashed_series=population_dashed,
                    fit_y_to_data=True,
                )
                st.caption(
                    "Shows the population path selected for the current GDP projection."
                )

            # 4-Quadrant Decomposition Grid
            st.markdown("#### Model Drivers Decomposition")
            grid1, grid2 = st.columns(2)

            with grid1:
                # Labour efficiency
                eff_data = econ_results[econ_results["variable"] == "lab_efficiency"].set_index("year")["value"]
                st.markdown("##### 1. Labour Efficiency ($E$)")
                render_interactive_line_chart(
                    eff_data, "Labour efficiency", value_format=",.7f"
                )

                # Savings & Depreciation
                st.markdown("##### 3. Rates: Savings ($s$) & Depreciation ($\\delta$)")
                rates_df = econ_results[econ_results["variable"].isin(["savings", "depreciation"])].pivot(
                    index="year", columns="variable", values="value"
                ).sort_index()
                render_interactive_line_chart(rates_df, "Rate", value_format=",.7f")

            with grid2:
                # Capital Stock
                k_data = econ_results[econ_results["variable"] == "k_stock"].set_index("year")["value"]
                st.markdown("##### 2. Capital Stock ($K$)")
                render_interactive_line_chart(k_data, "Capital stock (millions 2021 USD)")

                # GDP per capita
                gdp_pc = econ_results[econ_results["variable"] == "GDP_per_capita"].set_index("year")["value"]
                st.markdown("##### 4. GDP Per Capita (USD PPP)")
                render_interactive_line_chart(gdp_pc, "GDP per capita (2021 USD PPP)")

            st.markdown("#### Compound Annual Growth Rate of GDP")
            gdp_series = (
                econ_results[econ_results["variable"] == "real_GDP"]
                .loc[:, ["year", "value"]]
                .dropna()
                .sort_values("year")
                .set_index("year")["value"]
            )
            cagr_rows = []
            for start_year, end_year in [(2025, 2040), (2040, 2060), (2025, 2060)]:
                if start_year not in gdp_series.index or end_year not in gdp_series.index:
                    continue
                years = end_year - start_year
                cagr = calculate_cagr(float(gdp_series[start_year]), float(gdp_series[end_year]), years)
                cagr_rows.append(
                    {
                        "Period": f"{start_year}–{end_year}",
                        "CAGR": cagr,
                    }
                )

            if cagr_rows:
                cagr_df = pd.DataFrame(cagr_rows)
                cagr_df["CAGR"] = cagr_df["CAGR"].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")
                st.dataframe(cagr_df, use_container_width=True, hide_index=True)
            else:
                st.info("GDP values for the requested CAGR windows are not available for this economy.")

    # --- TAB 3: Combined Results & Export ---------------------------------------
    if active_view == VIEW_DATA:
        st.subheader("Data Export & Full Results Table")
        st.markdown("#### Full Model Results")
        st.caption(
            "Inspect comprehensive model outputs. Long format has one row per "
            "economy, year, and variable; wide format places variables in columns."
        )

        # Filter controls
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            f_econs = st.multiselect("Filter by Economy", options=sorted(results["economy_code"].unique()), default=[])
        with f_c2:
            f_vars = st.multiselect("Filter by Variable", options=sorted(results["variable"].unique()), default=[])

        filtered_res = results.copy()
        if f_econs:
            filtered_res = filtered_res[filtered_res["economy_code"].isin(f_econs)]
        if f_vars:
            filtered_res = filtered_res[filtered_res["variable"].isin(f_vars)]

        st.dataframe(filtered_res, use_container_width=True, height=450)

        # Wide format table for download
        wide_results = results.pivot(
            index=["economy_code", "economy", "year"],
            columns="variable",
            values="value",
        ).reset_index()

        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.download_button(
                "⬇️ Download Combined Results (Long Format CSV)",
                results.to_csv(index=False).encode(),
                file_name="apec_gdp_results_long.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d_c2:
            st.download_button(
                "⬇️ Download Combined Results (Wide Format CSV)",
                wide_results.to_csv(index=False).encode(),
                file_name="apec_gdp_results_wide.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### LEAP Modelling Input")
        st.caption(
            "Download the prepared macroeconomic projections in the format "
            "required by the LEAP modelling team."
        )
        try:
            leap_results = build_leap_input_export(results)
        except ValueError as exc:
            st.error(f"LEAP input download unavailable: {exc}")
        else:
            st.download_button(
                "⬇️ Download LEAP Input CSV",
                leap_results.to_csv(index=False).encode("utf-8"),
                file_name="apec_21_economies_leap_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --- TAB 4: Model Methodology ----------------------------------------------
    if active_view == VIEW_DOCS:
        st.subheader("APERC Solow-Swan Macro Model Methodology")
        st.markdown(r"""
### Model Formulation

The macroeconomic projections utilize an augmented **Solow-Swan neoclassical growth model** with a Cobb-Douglas production function:

$$Y(t) = K(t)^\alpha \cdot \big(E(t) \cdot L(t)\big)^{1-\alpha}$$

Where:
- $Y(t)$: Real GDP in millions of 2021 USD (PPP).
- $K(t)$: Capital stock, accumulated via $K(t) = K(t-1) \cdot (1 - \delta(t)) + s(t) \cdot Y(t-1)$.
- $E(t)$: Labour efficiency (technology factor).
- $L(t)$: Labour supply represented by total population (thousands).
- $\alpha$: Capital share parameter (default $0.4$).
- $s(t)$: Gross national savings rate (% of GDP / 100).
- $\delta(t)$: Capital depreciation rate.

---

### Jump-off Architecture (10th Outlook)

1. **Penn World Table (PWT 11.0):** Provides the historical output-to-capital ratio ($Y/K$) and depreciation rate ($\delta$) through **2023**.
   - The app derives historical capital stock as $K = GDP / (Y/K)$; it does not use a PWT capital-stock series directly. The latest available ratio is carried forward through the historical GDP source cutoff (2025 for non-Chinese Taipei economies and 2031 for Chinese Taipei).
   - The 2023 PWT depreciation value is carried forward through that cutoff. Capital and depreciation then evolve during model years, including years with supplied GDP overrides; overrides may delay the GDP output handoff but do not extend the historical carry-forward.
2. **Population Data and Scenarios:**
   - **Historical population** (through **2024**)**:** Sourced from EGEDA or custom historical data provided by researcher.
   - **Population projection** (from **2025**)**:** Sourced from UN DESA Medium, Low or High projection datasets, or custom projections provided by researcher.
   - **Default population:** Combines the historical and projection sources selected by the researcher for each economy.
   - **Sensitivity scenarios:** Low, Medium and High population scenarios use the selected historical source and their corresponding UN DESA projections. An optional Custom scenario is available where valid custom population data are provided.
3. **GDP Historical Data and Projection Anchors:**  
   - **World Bank WDI:** Provides historical GDP data through **2025** for 20 APEC member economies, excluding Chinese Taipei.
   - **IMF WEO:** Provides GDP data and projections through **2031** for Chinese Taipei (`18_CT`).
   - **Special cases:** Where available, researcher-supplied partial GDP projections are incorporated before the model's own GDP calculations begin.
   - **Model calculation start:** GDP projections generated by the model begin immediately after the final year of available historical or researcher-supplied GDP data. In the absence of partial projections, calculations start in **2026** for the 20 economies and **2032** for Chinese Taipei.
4. **Gross National Savings:**
   - The bundled sample uses WDI only for the 20 economies other than Chinese Taipei, through 2025. Chinese Taipei uses IMF only: historical observations through 2030 and its 2031 jump-off value. WDI gaps are not filled with IMF observations.
   - The accepted 45 historical gaps remain blank: Brunei Darussalam (1990-2000), Hong Kong, China (1990-1997), Japan (1990-1995), New Zealand (1990-1999), Russia (1990-1993), and Viet Nam (1990-1995). These are not missing projected savings values or model execution blockers; the full list is also in `config/gdp_model_parameters.md`.
5. **Labour Efficiency & Rate Convergence:**
   - Labour efficiency growth is calculated over a rolling look-back window and guided towards the long-term target corridor $[low\_eff, high\_eff]$ at the adjustment rate $change\_eff$.
   - Savings and depreciation rates transition dynamically towards their respective parameter corridors $[low\_sav, high\_sav]$ and $[low\_delta, high\_delta]$.
6. **Capital-Growth Guardrail:** For non-zero prior-year capital growth, the model compares the ratio of raw current growth to prior applied growth against the band $[1-c, 1+c]$, where $c$ is `cap_compare`. Below the band it applies $g_{prior}(1-c)$; above it applies $g_{prior}(1+c)$; inside it retains raw growth. Comparing the ratio preserves the intended behavior for negative prior growth. At exact-zero prior growth, raw growth passes through for that year and the guardrail resumes once prior growth is non-zero. `cap_compare = 0` is a zero-width band, not a disabled guardrail; configured values must be finite and in $[0,1)$.
7. **Horizon:** Historical data begin in 1990, with the main model projection horizon extending through 2060.
""")
