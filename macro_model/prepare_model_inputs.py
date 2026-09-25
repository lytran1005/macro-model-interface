#%%
"""Build capital stock and labour efficiency inputs (step 4 of the pipeline).

Combines the outputs of the population, IMF, and PWT prep steps to produce:

  results/data/capital_stock.csv
    capital = GDP / output_to_kstock, with the PWT ratio forward-filled
    after the last PWT year (2023) to each economy's GDP source cutoff.

    results/data/labour_efficiency_estimate.csv
    Labour efficiency E backed out of the Cobb-Douglas production function
    from 1990 onward. CT continues through 2031; other economies end in 2025.

Also provides build_gdp_input_table(), the wide per-economy input table
(labour, efficiency, capital, real_output) consumed by the GDP model.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS_DIR, RESULTS_DATA_DIR, ensure_dir
from config_loaders import economy_codes, load_gdp_model_parameters
from model import HISTORY_START_YEAR, LAST_PWT_YEAR, LAST_IMF_YEAR, LAST_WDI_YEAR, FINAL_YEAR

#%%
# constants

# Fallback capital share used only if an economy-specific alpha is not supplied.
ALPHA = 0.4
INPUT_START_YEAR = 1990

# Accepted GDP variable names; if both are present, 2017 vintage is preferred
# to mimic the 9th Outlook pipeline.
GDP_VAR_CANDIDATES = ["Real GDP PPP 2017 USD", "Real GDP PPP 2021 USD"]
GDP_PRIORITY = {"Real GDP PPP 2017 USD": 0, "Real GDP PPP 2021 USD": 1}

def source_cutoff_for_economy(economy_code, wdi_available=True):
    """Return the last source year represented in the prepared input tables."""
    if economy_code != "18_CT" and wdi_available:
        return LAST_WDI_YEAR
    return LAST_IMF_YEAR


def input_source_label(economy_code, wdi_available=True):
    """Describe the GDP source and PWT-derived calculation used in an output."""
    if economy_code != "18_CT" and wdi_available:
        return "WDI and PWT calculation"
    return "IMF and PWT calculation"


#%%
# functions

def dedupe_gdp_rows(df, extra_keys=None):
    """Keep one GDP row per year (per economy), preferring the 2017 vintage."""
    keys = (extra_keys or []) + ["year"]
    df = df.copy()
    df["gdp_priority"] = df["variable"].map(GDP_PRIORITY)
    df = df.sort_values(keys + ["gdp_priority"]).drop_duplicates(subset=keys, keep="first")
    return df.drop(columns="gdp_priority")


def build_capital_stock(imf_long, pwt_long, codes, wdi_long=None):
    """Capital stock per economy: GDP / output_to_kstock, ratio forward-filled."""
    capital_df = pd.DataFrame()

    for economy in codes:
        source_df = imf_long if economy == "18_CT" else (wdi_long if wdi_long is not None else imf_long)
        source_temp = source_df[
            (source_df["economy_code"] == economy)
            & (source_df["variable"].isin(GDP_VAR_CANDIDATES))
        ].copy()
        source_temp = dedupe_gdp_rows(source_temp).set_index("year")

        pwt_temp = pwt_long[
            (pwt_long["economy_code"] == economy)
            & (pwt_long["variable"] == "output_to_kstock")
        ][["year", "value"]].rename(columns={"value": "ratio"}).set_index("year")

        if source_temp.empty or pwt_temp.empty:
            print(f"Warning: no capital stock built for {economy} "
                  "(missing source GDP or PWT output/capital ratio)")
            continue

        capital = pd.concat([source_temp, pwt_temp], axis=1).sort_index()
        cutoff = source_cutoff_for_economy(economy, wdi_available=wdi_long is not None)
        capital = capital.loc[INPUT_START_YEAR:cutoff, :].copy()

        # Continue the last known PWT ratio forward to the last historical GDP year
        capital["ratio"] = capital["ratio"].ffill()

        capital["capital"] = capital["value"] / capital["ratio"]
        capital["variable"] = "Capital stock"
        capital["source"] = input_source_label(economy, wdi_available=wdi_long is not None)

        capital = capital.reset_index()
        capital = capital[["economy_code", "economy", "year", "variable", "capital", "source"]]
        capital = capital.rename(columns={"capital": "value"})
        capital["percent"] = capital.groupby(["economy", "variable"], group_keys=False)["value"]\
            .apply(pd.Series.pct_change)

        capital_df = pd.concat([capital_df, capital], axis=0).reset_index(drop=True)

    return capital_df


def build_labour_efficiency(
    imf_long, pop_long, capital_df, codes, alpha=ALPHA, wdi_long=None,
    alpha_by_economy=None,
):
    """Back labour efficiency out of the Cobb-Douglas production function."""
    gdp_source = []
    for economy in codes:
        gdp_source.append(
            (imf_long if economy == "18_CT" else (wdi_long if wdi_long is not None else imf_long))
            [["economy_code", "economy", "year", "variable", "value"]]
            .copy()
        )
    source_long = pd.concat(gdp_source, ignore_index=True)

    gdp_hist = dedupe_gdp_rows(
        source_long[source_long["variable"].isin(GDP_VAR_CANDIDATES)],
        extra_keys=["economy_code"],
    ).reset_index(drop=True)

    pop = pop_long[pop_long["variable"] == "population_1jan"].copy()
    non_ct_cutoff = source_cutoff_for_economy("01_AUS", wdi_available=wdi_long is not None)
    ct_cutoff = source_cutoff_for_economy("18_CT", wdi_available=wdi_long is not None)
    pop = pop[
        (
            (pop["economy_code"] != "18_CT") & (pop["year"] <= non_ct_cutoff)
        ) | (
            (pop["economy_code"] == "18_CT") & (pop["year"] <= ct_cutoff)
        )
    ]
    pop = pop[pop["year"] >= INPUT_START_YEAR].copy().reset_index(drop=True)

    e_calc = pd.concat([gdp_hist, pop, capital_df], ignore_index=True)
    e_estimate = pd.DataFrame()

    for economy in codes:
        economy_alpha = (alpha_by_economy or {}).get(economy, alpha)
        labour = e_calc[
            (e_calc["economy_code"] == economy) & (e_calc["variable"] == "population_1jan")
        ].copy().set_index("year")
        labour["L^1-alpha"] = labour["value"] ** (1 - economy_alpha)
        labour = labour[["economy_code", "economy", "L^1-alpha"]]

        k = e_calc[
            (e_calc["economy_code"] == economy) & (e_calc["variable"] == "Capital stock")
        ].copy().set_index("year")
        k["K^alpha"] = k["value"] ** economy_alpha
        k = k[["K^alpha"]]

        y = dedupe_gdp_rows(e_calc[
            (e_calc["economy_code"] == economy) & (e_calc["variable"].isin(GDP_VAR_CANDIDATES))
        ]).set_index("year")[["value"]].rename(columns={"value": "Output_y"})

        eqn = pd.concat([y, labour, k], axis=1)
        eqn["E"] = (eqn["Output_y"] / (eqn["L^1-alpha"] * eqn["K^alpha"])) ** (1 / (1 - economy_alpha))

        eqn = eqn.reset_index()[["year", "economy_code", "economy", "E"]]
        cutoff = source_cutoff_for_economy(economy, wdi_available=wdi_long is not None)
        eqn = eqn[(eqn["year"] >= INPUT_START_YEAR) & (eqn["year"] <= cutoff)]
        e_estimate = pd.concat([e_estimate, eqn]).reset_index(drop=True)

    e_df = e_estimate.rename(columns={"E": "value"})
    e_df["variable"] = "Labour efficiency"
    e_df = e_df[["economy_code", "economy", "year", "variable", "value"]].reset_index(drop=True)
    e_df["percent"] = e_df.groupby(["economy", "variable"], group_keys=False)["value"]\
        .apply(pd.Series.pct_change)
    e_df["source"] = e_df["economy_code"].map(
        lambda economy: input_source_label(economy, wdi_available=wdi_long is not None)
    )
    return e_df


def build_gdp_input_table(pop_long, imf_long, lab_eff, cap_df, codes, wdi_long=None):
    """Assemble the wide per-economy table the GDP model consumes.

    Columns: economy_code, economy, year, labour, efficiency, capital, real_output.
    Used by both the main results run and the population sensitivity run
    (which passes different population datasets).
    """
    def source_for_economy(economy_code):
        if economy_code == "18_CT":
            return imf_long
        if wdi_long is not None:
            return wdi_long
        return imf_long

    gdp_input = pd.DataFrame()
    for economy in codes:
        source_hist = source_for_economy(economy)
        input_df = pd.concat([pop_long, source_hist, lab_eff, cap_df]).reset_index(drop=True)
        input_df["year"] = pd.to_numeric(input_df["year"], errors="coerce")
        input_df = input_df[
            input_df["variable"].isin(
                ["population_1jan", "Real GDP PPP 2021 USD", "Labour efficiency", "Capital stock"]
            )
            & (input_df["year"] >= HISTORY_START_YEAR)
            & (input_df["year"] <= FINAL_YEAR)
        ].copy().reset_index(drop=True)

        def one_variable(variable, colname):
            return input_df[
                (input_df["economy_code"] == economy) & (input_df["variable"] == variable)
            ][["year", "value"]].rename(columns={"value": colname}).reset_index(drop=True)

        base = input_df[
            (input_df["economy_code"] == economy) & (input_df["variable"] == "population_1jan")
        ][["economy_code", "economy", "year", "value"]]\
            .rename(columns={"value": "labour"}).reset_index(drop=True)

        base = base.merge(one_variable("Labour efficiency", "efficiency"), how="left", on="year")
        base = base.merge(one_variable("Capital stock", "capital"), how="left", on="year")
        base = base.merge(one_variable("Real GDP PPP 2021 USD", "real_output"), how="left", on="year")
        gdp_input = pd.concat([gdp_input, base]).reset_index(drop=True)

    gdp_input["year"] = gdp_input["year"].astype(int)
    return gdp_input


def make_capital_charts(capital_df, codes):
    """Save capital stock level + growth charts per economy."""
    sns.set_theme(style="ticks")
    chart_dir = ensure_dir(RESULTS_DIR / "capital")
    for economy in codes:
        chart_df = capital_df[capital_df["economy_code"] == economy].reset_index(drop=True)
        if chart_df.empty or chart_df["value"].isna().all():
            continue
        fig, axs = plt.subplots(2, 1, figsize=(9, 6))
        sns.lineplot(ax=axs[0], data=chart_df, x="year", y="value")
        axs[0].set(title=f"{economy} capital stock", xlabel="Year", ylabel="Capital stock")
        axs[0].grid(True)
        sns.lineplot(ax=axs[1], data=chart_df, x="year", y="percent")
        axs[1].set(title=f"{economy} capital stock growth", xlabel="Year", ylabel="Capital stock growth")
        axs[1].grid(True)
        plt.tight_layout()
        fig.savefig(chart_dir / f"{economy}_capital.png")
        plt.close(fig)


def make_labour_efficiency_charts(e_df, codes):
    """Save labour efficiency level + growth charts per economy."""
    sns.set_theme(style="ticks")
    chart_dir = ensure_dir(RESULTS_DIR / "labour_efficiency")
    for economy in codes:
        chart_df = e_df[e_df["economy_code"] == economy].reset_index(drop=True)
        if chart_df.empty or chart_df["value"].isna().all():
            continue
        fig, axs = plt.subplots(2, 1, figsize=(9, 6))
        sns.lineplot(ax=axs[0], data=chart_df, x="year", y="value")
        axs[0].set(title=f"{economy} labour efficiency estimate", xlabel="Year", ylabel="Labour efficiency")
        axs[0].grid(True)
        sns.lineplot(ax=axs[1], data=chart_df, x="year", y="percent")
        axs[1].set(title=f"{economy} labour efficiency growth", xlabel="Year", ylabel="Labour efficiency growth")
        axs[1].grid(True)
        plt.tight_layout()
        fig.savefig(chart_dir / f"{economy}_labour_efficiency.png")
        plt.close(fig)


def run_model_input_prep(save_charts=True):
    """Run the full model-input preparation step (capital + labour efficiency)."""
    ensure_dir(RESULTS_DATA_DIR)
    pop_long = pd.read_csv(RESULTS_DATA_DIR / "undesa_pop_to2100.csv")
    imf_long = pd.read_csv(RESULTS_DATA_DIR / "IMF_to2031.csv")
    wdi_long = None
    if (RESULTS_DATA_DIR / "WDI_to2025.csv").exists():
        wdi_long = pd.read_csv(RESULTS_DATA_DIR / "WDI_to2025.csv")

    codes = sorted(economy_codes())
    model_params = load_gdp_model_parameters("gdp_model_parameters.csv")
    alpha_by_economy = {code: values["alpha"] for code, values in model_params.items()}

    capital_df = build_capital_stock(
        imf_long,
        pd.read_csv(RESULTS_DATA_DIR / "PWT_cap_labour_to2023.csv"),
        codes,
        wdi_long=wdi_long,
    )
    capital_df.to_csv(RESULTS_DATA_DIR / "capital_stock.csv", index=False)

    e_df = build_labour_efficiency(
        imf_long, pop_long, capital_df, codes, wdi_long=wdi_long,
        alpha_by_economy=alpha_by_economy,
    )
    e_df.to_csv(RESULTS_DATA_DIR / "labour_efficiency_estimate.csv", index=False)

    if save_charts:
        make_capital_charts(capital_df, codes)
        make_labour_efficiency_charts(e_df, codes)

    print("Model input prep complete: capital_stock.csv, labour_efficiency_estimate.csv")
