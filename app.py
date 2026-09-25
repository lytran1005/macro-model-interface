r"""Standalone Streamlit app for the APERC Solow-Swan GDP model.

Run from this project root with:
  python -m streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

SCENARIO_ORDER = ["Low", "Medium", "High"]
SCENARIO_COLORS = ["#2a9d8f", "#e9c46a", "#e76f51"]
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
    load_population_scenarios,
    load_population_gdp_scenarios,
    load_gdp_model_parameters,
    run_analysis,
    wide_to_long,
    run_pipeline_population_scenario,
    parse_params_csv,
)
from generate_samples import make_sample_data, sample_params_csv

st.set_page_config(
    page_title="APERC 10th Outlook — Solow-Swan GDP Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("APERC 10th Outlook — Solow-Swan Macro Model")
st.markdown(
    "Interactive front-end for the **APERC 10th Energy Demand and Supply Outlook** macroeconomic projections. "
    "Upload a single wide-format CSV containing observed data across the 21 APEC economies to derive capital stock ($K$) "
    "and labour efficiency ($E$), simulate Solow-Swan growth paths to **2100**, and explore or export results."
)


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


@st.cache_data(show_spinner="Deriving inputs and running Solow-Swan model for all economies...")
def cached_run(df, params):
    return run_analysis(df, params)


@st.cache_data(show_spinner="Loading population scenarios...")
def cached_population_scenarios():
    return load_population_scenarios()


@st.cache_data(show_spinner="Loading GDP population-sensitivity projections...")
def cached_population_gdp_scenarios():
    return load_population_gdp_scenarios()


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
):
    """Render a line chart with a nearest-year guide, hover points, and tooltips."""
    chart_data = data.copy()
    if isinstance(chart_data, pd.Series):
        chart_data = chart_data.to_frame(chart_data.name or "Value")
    chart_data.index.name = "year"
    chart_data = chart_data.reset_index().melt(
        id_vars="year", var_name="series", value_name="plot_value"
    ).dropna(subset=["plot_value"])

    hover = alt.selection_point(
        fields=["year"], nearest=True, on="pointerover", empty=False
    )
    color_encoding = alt.Color(
        "series:N", title=series_title, sort=series_order
    )
    if series_colors:
        color_encoding = alt.Color(
            "series:N",
            title=series_title,
            sort=series_order,
            scale=alt.Scale(range=series_colors),
        )
    base = alt.Chart(chart_data).encode(
        x=alt.X("year:Q", title="Year"),
        y=alt.Y("plot_value:Q", title=y_title),
        color=color_encoding,
    )
    tooltip = [
        alt.Tooltip("year:Q", title="Year"),
        alt.Tooltip("plot_value:Q", title=y_title, format=value_format),
        alt.Tooltip("series:N", title=series_title),
    ]
    line = base.mark_line().encode(tooltip=alt.value(None))
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
    chart_col, download_col = st.columns([10, 1])
    with chart_col:
        st.altair_chart(chart, use_container_width=True)
    with download_col:
        render_chart_download_button(chart, f"{y_title.lower().replace(' ', '_')}.png")



def chart_series_order(series_values):
    """Return series in the requested High → Medium → Low order while keeping historical estimates first."""
    ordered = []
    if "Historical estimates" in series_values:
        ordered.append("Historical estimates")
    for scenario in ["High", "Medium", "Low"]:
        for label in series_values:
            if label == "Historical estimates":
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
    }
    colors = []
    for label in ordered:
        if label == "Historical estimates":
            colors.append("#7fb3d5")
            continue
        scenario = next(
            (s for s in ["High", "Medium", "Low"] if label == s or str(label).endswith(f" — {s}")),
            None,
        )
        colors.append(color_map.get(scenario, "#7fb3d5"))
    return colors


def calculate_cagr(start_value, end_value, years):
    """Return CAGR as a decimal fraction, given start/end values and the elapsed years."""
    if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0 or years <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


# --- sidebar: data + parameters --------------------------------------------

with st.sidebar:
    st.header("⚙️ Data & Configuration")
    st.caption(
        "Upload a wide CSV (one row per economy-year). Required columns: `economy` or `economy_code`, `year`, and: "
        "`population_1jan`, `Real GDP PPP 2021 USD`, `output_to_kstock`, `delta`, `Gross national savings`."
    )

    has_data = "df_wide" in st.session_state

    inputs_label = "📁 1. Model Input Data"
    if has_data:
        inputs_label = f"📁 1. Data ({st.session_state.get('data_label', 'loaded')})"

    with st.expander(inputs_label, expanded=not has_data):
        uploaded = st.file_uploader("Upload Wide CSV", type=["csv"], help="Upload observed data for 21 APEC economies.")

        use_sample = st.button("Use Pipeline Sample Data", use_container_width=True, type="primary" if not has_data else "secondary")
        sample_csv = make_sample_data()
        st.download_button(
            "Download Sample Wide CSV",
            sample_csv.to_csv(index=False).encode(),
            file_name="sample_wide_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if uploaded is not None:
            st.session_state["df_wide"] = pd.read_csv(uploaded)
            st.session_state["data_label"] = uploaded.name
        elif use_sample:
            st.session_state["df_wide"] = sample_csv
            st.session_state["data_label"] = "pipeline sample data (21 APEC economies)"

    with st.expander("🛠️ 2. Model Parameters (Optional)", expanded=False):
        st.caption("Defaults are pre-loaded from `config/gdp_model_parameters.csv`. Upload a CSV to customize tuning knobs.")
        params_file = st.file_uploader("Upload custom parameters.csv", type=["csv"])
        if params_file is not None:
            st.session_state["params"] = parse_params_csv(pd.read_csv(params_file))
            st.success("Custom parameters loaded")
        st.download_button(
            "Download Parameter Template CSV",
            sample_params_csv().to_csv(index=False).encode(),
            file_name="sample_model_parameters.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # --- data status ----------------------------------------------------------

    if "df_wide" in st.session_state:
        df_wide = st.session_state["df_wide"]
        ok, message = validate(df_wide)
        if not ok:
            st.error(message)
            st.stop()
        st.success(f"✓ Loaded {st.session_state.get('data_label', 'data')}")
        df = wide_to_long(df_wide)
        st.caption(
            f"**Economies:** {df['economy_code'].nunique()} | "
            f"**Years:** {df['year'].min()}–{df['year'].max()} | "
            f"**Rows:** {len(df):,}"
        )
    else:
        st.info("👈 Upload a wide CSV or click **Use Pipeline Sample Data** to begin.")
        st.stop()

    st.caption(
        f"**Model Horizon:** {HISTORY_START_YEAR}–{FINAL_YEAR}\n\n"
        f"**Jump-off Years:**\n"
        f"- Non-CT Economies (WDI): {LAST_WDI_YEAR}\n"
        f"- 18_CT (IMF): {LAST_IMF_YEAR}"
    )

# --- run model --------------------------------------------------------------

params = st.session_state.get("params") or load_gdp_model_parameters("gdp_model_parameters.csv")
results, inputs, errors = cached_run(df, params)

if errors:
    with st.expander(f"⚠️ {len(errors)} economy/ies had warnings or issues - show details"):
        for code, err in errors:
            st.write(f"**{code}**: {err}")

if results.empty:
    st.error("No results produced. Please check your uploaded dataset format.")
    st.stop()

# --- top summary metrics ----------------------------------------------------

gdp_rows = results[results["variable"] == "real_GDP"]
pop_rows = results[results["variable"] == "population"]

if not gdp_rows.empty and not pop_rows.empty:
    gdp_by_yr = gdp_rows.groupby("year")["value"].sum()
    pop_by_yr = pop_rows.groupby("year")["value"].sum()

    gdp_2025 = gdp_by_yr.get(2025, np.nan)
    gdp_2060 = gdp_by_yr.get(2060, np.nan)

    pop_2025 = pop_by_yr.get(2025, np.nan)
    pop_2060 = pop_by_yr.get(2060, np.nan)

    val_2025_gdp = gdp_2025 / 1_000_000  # Trillion USD PPP
    val_2060_gdp = gdp_2060 / 1_000_000

    val_2025_pop = pop_2025 / 1_000_000  # Billion people
    val_2060_pop = pop_2060 / 1_000_000

    val_2025_gdp_pc = (gdp_2025 / pop_2025) * 1000 if pd.notna(gdp_2025) and pd.notna(pop_2025) and pop_2025 != 0 else np.nan
    val_2060_gdp_pc = (gdp_2060 / pop_2060) * 1000 if pd.notna(gdp_2060) and pd.notna(pop_2060) and pop_2060 != 0 else np.nan

    gdp_cagr_2025_2060 = calculate_cagr(gdp_2025, gdp_2060, 2060 - 2025)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "APEC GDP (2025)",
            f"${val_2025_gdp:.1f} T" if pd.notna(val_2025_gdp) else "N/A",
            help="Total APEC Real GDP in Trillions (2021 USD PPP)",
        )
    with c2:
        st.metric(
            "APEC GDP (2060)",
            f"${val_2060_gdp:.1f} T" if pd.notna(val_2060_gdp) else "N/A",
            delta=f"{gdp_cagr_2025_2060 * 100:+.2f}% CAGR (2025-2060)" if pd.notna(gdp_cagr_2025_2060) else None,
            help="Total APEC Real GDP in Trillions (2021 USD PPP); CAGR for 2025-2060",
        )
    with c3:
        st.metric(
            "APEC GDP per Capita (2060)",
            f"${val_2060_gdp_pc:,.0f}" if pd.notna(val_2060_gdp_pc) else "N/A",
            delta=f"{(val_2060_gdp_pc/val_2025_gdp_pc - 1)*100:+.0f}% vs 2025" if pd.notna(val_2025_gdp_pc) and pd.notna(val_2060_gdp_pc) else None,
            help="Total APEC Real GDP per capita in 2021 USD PPP",
        )
    with c4:
        st.metric(
            "APEC Population (2060)",
            f"{val_2060_pop:.2f} B" if pd.notna(val_2060_pop) else "N/A",
            delta=f"{(val_2060_pop/val_2025_pop - 1)*100:+.1f}% vs 2025" if pd.notna(val_2025_pop) and pd.notna(val_2060_pop) else None,
            help="Total APEC Population (Billions)",
        )

# --- tabs -------------------------------------------------------------------

tab_population, tab_detail, tab_compare, tab_data, tab_docs = st.tabs([
    "👥 Population Scenarios",
    "🔍 Economy Deep Dive",
    "📈 Economy Comparison",
    "📋 Combined Results & Export",
    "📖 Model Methodology",
])

# --- TAB 1: Economy Comparison ---------------------------------------------
with tab_compare:
    st.subheader("Multi-Economy Projections Comparison")

    all_codes = sorted(results["economy_code"].unique())
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_codes = st.multiselect(
            "Select Economies to display",
            options=all_codes,
            default=all_codes[:6] if len(all_codes) >= 6 else all_codes,
            help="Choose which economies to include in the comparison chart.",
        )
    with col_sel2:
        metric_choice = st.selectbox(
            "Select Variable",
            options=[
                ("real_GDP", "Real GDP (Millions 2021 USD PPP)"),
                ("gdp_growth", "Annual Real GDP Growth Rate (%)"),
                ("GDP_per_capita", "GDP Per Capita (2021 USD PPP)"),
                ("population", "Population (Thousands)"),
                ("k_stock", "Capital Stock (Millions 2021 USD)"),
                ("lab_efficiency", "Labour Efficiency (E)"),
            ],
            format_func=lambda x: x[1],
        )

    var_key, var_title = metric_choice

    if selected_codes:
        sub_comp = results[
            (results["variable"] == var_key)
            & (results["economy_code"].isin(selected_codes))
        ]
        if not sub_comp.empty:
            chart_df = sub_comp.pivot(index="year", columns="economy_code", values="value").sort_index()
            render_interactive_line_chart(
                chart_df,
                var_title,
                series_title="Economy",
                series_order=selected_codes,
                series_colors=[ECONOMY_COLORS[code] for code in selected_codes],
            )
            st.caption(f"{var_title} ({HISTORY_START_YEAR}–{FINAL_YEAR})")

            # Benchmark year summary table
            benchmark_years = [y for y in [1990, 2025, 2030, 2050, 2070, 2100] if y in chart_df.index]
            if benchmark_years:
                st.markdown("##### Benchmark Year Values")
                bench_table = chart_df.loc[benchmark_years].T
                st.dataframe(
                    bench_table.style.format("{:,.2f}" if var_key in ["gdp_growth", "lab_efficiency"] else "{:,.0f}"),
                    use_container_width=True,
                )
    else:
        st.warning("Please select at least one economy to compare.")

# --- TAB 3: Population Scenarios -------------------------------------------
with tab_population:
    st.subheader("Population Scenarios")
    st.caption(
        "Compare UN DESA Low, Medium, and High population projections for your economy and examine "
        "their impact on long-term GDP. Select the population pathway that best reflects your economy's "
        "expectations before proceeding to macroeconomic assumptions in the next tab."
    )

    try:
        population_scenarios = cached_population_scenarios()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Run the main pipeline with the population step enabled, then refresh this page.")
        population_scenarios = pd.DataFrame()

    if not population_scenarios.empty:
        if "source" in population_scenarios.columns and population_scenarios["source"].eq("Sample fallback").all():
            st.warning(
                "Official population scenario outputs were not found. "
                "This tab is using the bundled pipeline sample fallback."
            )
        scenario_order = SCENARIO_ORDER
        all_population_codes = sorted(population_scenarios["economy_code"].unique())

        population_labels = [
            population_scenarios.loc[
                population_scenarios["economy_code"] == code, "economy"
            ].iloc[0] + f" ({code})"
            for code in all_population_codes
        ]
        population_label_to_code = dict(zip(population_labels, all_population_codes))
        scenario_choice = ["High", "Low", "Medium", "All"]

        control_view, control_economies, control_scenarios = st.columns([1, 2, 1])
        with control_view:
            population_view = st.radio(
                "View",
                ["Total", "By Economy"],
                index=0,
                key="population_view",
                horizontal=True,
            )
        with control_economies:
            selected_population_label = st.selectbox(
                "Economy",
                population_labels,
                index=0,
                key="population_economies_selected",
                disabled=population_view == "Total",
                help="Choose one economy for the economy-level view.",
            )
        with control_scenarios:
            selected_scenario = st.selectbox(
                "Scenarios",
                scenario_choice,
                index=3,
                key="population_scenario_selected",
                help="Choose High, Low, Medium, or All scenarios.",
            )
        selected_scenarios = scenario_order if selected_scenario == "All" else [selected_scenario]

        year_min = int(population_scenarios["year"].min())
        year_max = int(population_scenarios["year"].max())
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
                key="population_year_to_2100",
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
            st.caption("Total population across all 21 APEC economies for each selected scenario.")
        else:
            chart_data = common_population_chart
            st.markdown("#### Population Paths by Economy")
            st.caption("The selected economy is shown for each selected scenario.")
        population_long = chart_data.reset_index().melt(
            id_vars="year", var_name="series", value_name="value"
        ).dropna(subset=["value"])
        historical_population = population_long[
            population_long["year"] <= LAST_PWT_YEAR
        ].drop_duplicates(subset=["year"], keep="first")
        projected_population = population_long[
            population_long["year"] > LAST_PWT_YEAR
        ]
        historical_population["series"] = "Historical estimates"
        population_plot = pd.concat(
            [historical_population, projected_population], ignore_index=True
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
        population_line = population_base.mark_line().encode(tooltip=alt.value(None))
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
        population_divider = alt.Chart(pd.DataFrame({"year": [LAST_PWT_YEAR + 1]})).mark_rule(
            color="#b8b8b8", strokeDash=[5, 5]
        ).encode(x="year:Q")
        population_chart = (
            population_line
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
        st.caption(
            "UN DESA historical estimates extend through 2023; scenario projections begin in 2024."
        )
        st.download_button(
            "⬇️ Download population data",
            population_scenarios.to_csv(index=False).encode(),
            file_name="apec_population_scenarios.csv",
            mime="text/csv",
            help="Download the complete Low, Medium, and High population scenario dataset used by this tab.",
        )

        st.markdown("#### Real GDP Population Sensitivity (1990–2100)")
        st.caption(
            "GDP projections calculated by the main macro pipeline using the selected population scenarios."
        )
        try:
            population_gdp_scenarios = cached_population_gdp_scenarios()
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
                st.caption("Total APEC real GDP under each selected population scenario.")
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
                st.caption("Real GDP paths for each selected economy and population scenario.")
            gdp_series_order = chart_series_order(list(gdp_chart_data.columns))
            gdp_plot = gdp_chart_data.reset_index().melt(
                id_vars="year", var_name="series", value_name="value"
            ).dropna(subset=["value"])
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
            gdp_chart = (
                gdp_line
                + gdp_selectors
                + gdp_points
                + gdp_hover_rule
            ).properties(height=360)
            chart_col, download_col = st.columns([10, 1])
            with chart_col:
                st.altair_chart(gdp_chart, use_container_width=True)
            with download_col:
                render_chart_download_button(gdp_chart, "apec_gdp_population_sensitivity.png")
            if population_view == "Total":
                st.caption(
                    "Real GDP (Millions 2021 USD PPP). Historical source: World Bank WDI "
                    "through 2025 for non-CT economies and IMF WEO through 2031 for CT; "
                    "population-sensitivity projections begin in 2026 and 2032 respectively."
                )
            else:
                selected_gdp_code = selected_population_codes[0]
                selected_gdp_source = "IMF WEO" if selected_gdp_code == "18_CT" else "World Bank WDI"
                selected_gdp_jump_off = (
                    LAST_IMF_YEAR if selected_gdp_code == "18_CT" else LAST_WDI_YEAR
                )
                st.caption(
                    f"Real GDP (Millions 2021 USD PPP). Jump-off from {selected_gdp_source} "
                    f"occurs at year {selected_gdp_jump_off}; population-sensitivity projections "
                    f"begin in {selected_gdp_jump_off + 1}."
                )
            st.download_button(
                "⬇️ Download GDP sensitivity data",
                population_gdp_view.to_csv(index=False).encode(),
                file_name="apec_gdp_population_sensitivity.csv",
                mime="text/csv",
                help="Download the GDP sensitivity data currently selected for this chart.",
            )
        except FileNotFoundError as exc:
            st.warning(str(exc))

# --- TAB 2: Economy Deep Dive ----------------------------------------------
with tab_detail:
    st.subheader("Economy Deep Dive")
    st.caption(
        "Refine the GDP projection by modifying key macroeconomic parameters, including savings rates, "
        "depreciation rates, and labour efficiency growth. Compare the resulting projections against the "
        "rebased 9th Outlook (2021 USD PPP)."
    )
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        sel_code = st.selectbox("Choose Economy", sorted(results["economy_code"].unique()), key="deep_dive_econ")
        sel_name = results.loc[results["economy_code"] == sel_code, "economy"].iloc[0]
        jump_off = jump_off_year_for_economy(sel_code)
        source_name = "IMF WEO" if sel_code == "18_CT" else "World Bank WDI"

        deep_dive_scenario = st.radio(
            "Population Scenario",
            ["Medium", "Low", "High"],
            horizontal=True,
            key="deep_dive_population_scenario",
            help="Re-runs this economy with the selected UN DESA population path and the current model parameters.",
        )
        show_9th_outlook = st.toggle(
            "Show rebased 9th Outlook",
            value=False,
            key="deep_dive_show_9th_outlook",
            help="Display the rebased 9th Outlook GDP comparison line.",
        )

        st.markdown(f"### {sel_name}")
        st.markdown(f"**Code:** `{sel_code}`")
        st.markdown(f"**Historical Anchor:** {jump_off} ({source_name})")
        st.markdown(f"**Projection Period:** {jump_off + 1}–{FINAL_YEAR}")

        with st.expander("Model Parameters Applied", expanded=True):
            econ_params = params.get(sel_code, {})
            p_df = pd.DataFrame(
                [
                    (key, value)
                    for key, value in econ_params.items()
                    if key != "cap_compare"
                ],
                columns=["Parameter", "Value"],
            ).set_index("Parameter")
            st.dataframe(p_df, use_container_width=True)

    try:
        deep_dive_results = run_pipeline_population_scenario(
            sel_code, deep_dive_scenario, params
        )
    except (FileNotFoundError, ValueError) as exc:
        st.warning(
            f"Population scenario could not be applied: {exc}. Showing the loaded input instead."
        )
        deep_dive_results = results[results["economy_code"] == sel_code]

    with col_d2:
        econ_results = deep_dive_results

        # Primary GDP Chart
        st.markdown("#### Real GDP Trajectory (1990–2100)")
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
                gdp_chart["year"] >= jump_off
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
            st.caption(f"Real GDP (Millions 2021 USD PPP). Jump-off from {source_name} occurs at year {jump_off}.")

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
with tab_data:
    st.subheader("Data Export & Full Results Table")

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

# --- TAB 4: Model Methodology ----------------------------------------------
with tab_docs:
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

1. **Penn World Table (PWT 11.0):** Provides historical output-to-capital ratio ($Y/K$) and depreciation rate ($\delta$) through **2023**.
2. **Historical Anchor Hand-off:**
   - **World Bank WDI:** Historical observed GDP and savings data through **2025** for 20 APEC economies.
   - **IMF WEO:** Historical observed GDP and projections through **2031** for Chinese Taipei (`18_CT`).
3. **Labour Efficiency & Rate Convergence:**
   - Labour efficiency growth is computed over a rolling window and guided towards the long-term target corridor $[low\_eff, high\_eff]$ at step rate $change\_eff$.
   - Savings and depreciation rates transition dynamically toward policy corridors $[low\_sav, high\_sav]$ and $[low\_delta, high\_delta]$.
4. **Horizon:** 1990 historical start through **2100** projection end.
""")
