#%%
"""The APERC Solow-Swan GDP model (single authoritative copy).

This module defines the model calculation only. It does no file I/O and no
plotting, so it is safe to import from anywhere. The runner scripts
(generate_results.py, run_sensitivity.py) load the data, call
aperc_gdp_model() per economy, and handle saving/plotting.

Model outline, per economy:
  1. Project labour efficiency growth beyond each economy's jump-off year by nudging the recent
     average growth rate toward a [low_eff, high_eff] band.
  2. Project savings and depreciation rates beyond each economy's jump-off year.
  3. Accumulate capital: K(t) = K(t-1)*(1-delta) + s*Y(t-1), with the growth
     rate capped relative to recent capital growth (cap_compare).
  4. Output from Cobb-Douglas: Y = K^alpha * (E*L)^(1-alpha).

Parameter defaults below are the model-wide defaults; per-economy values live
in config/gdp_model_parameters.csv.
"""

import numpy as np
import pandas as pd

#%%
# constants

HISTORY_START_YEAR = 1990   # first year of model/output history
LAST_PWT_YEAR = 2023        # last year of PWT 11.0 data
LAST_IMF_YEAR = 2031        # last year of IMF WEO projections (CT default historical anchor)
LAST_WDI_YEAR = 2025        # WDI jump-off year for all non-CT economies
PROJECTION_START_YEAR = LAST_IMF_YEAR + 1   # 2032: first APERC-modelled year for CT
FINAL_YEAR = 2100           # last projected year
SAVINGS_HIGH = 0.25         # upper stability corridor for savings rate
SAVINGS_LOW = 0.22          # lower stability corridor for savings rate
SAVINGS_STEP = 0.002        # annual movement toward the corridor


def jump_off_year_for_economy(economy_code):
    """Return the historical GDP jump-off year for an economy.

    CT stays on IMF through 2031; all other APEC economies use WDI through 2025.
    """
    if economy_code == "18_CT":
        return LAST_IMF_YEAR
    return LAST_WDI_YEAR

LABEL_IMF = "IMF GDP through 2031"
LABEL_WDI = "WDI GDP through 2025"
LABEL_APERC = "APERC GDP projections"
LABEL_9TH = "9th Outlook projections (rebased to 2021 PPP USD)"

#%%
# functions

def _band_walk(current, high, low, step):
    """Move a rate one step toward the [low, high] band; unchanged if inside."""
    if current > high:
        return current - step
    elif current < low:
        return current + step
    return current


def aperc_gdp_model(economy,
                    input_data,
                    labour_data,
                    cap_growth_data,
                    delta_data,
                    sav_data,
                    save_invest_hist,
                    delta_hist,
                    gdp_9th,
                    wdi_savings_hist=None,
                    jump_off_year=None,
                    lab_eff_periods=10,
                    high_eff=0.015,
                    low_eff=0.012,
                    change_eff=0.002,
                    high_sav=SAVINGS_HIGH,
                    low_sav=SAVINGS_LOW,
                    change_sav=SAVINGS_STEP,
                    high_delta=0.046,
                    low_delta=0.044,
                    change_del=0.0005,
                    alpha=0.4,
                    cap_compare=0.05):
    """Run the Solow-Swan projection for one economy.

    Parameters
    ----------
    economy : str
        Economy code, e.g. '01_AUS'.
    input_data : pd.DataFrame
        Wide input table from prepare_model_inputs.build_gdp_input_table
        (columns: economy_code, economy, year, labour, efficiency, capital,
        real_output).
    labour_data : pd.DataFrame
        Labour efficiency history (labour_efficiency_estimate.csv).
    cap_growth_data : pd.DataFrame
        Capital stock growth history (columns economy_code, year, percent).
    delta_data : pd.DataFrame
        Depreciation snapshot at 2023 (PWT_delta_2023.csv).
    sav_data : pd.DataFrame
        Economy-specific savings anchor: WDI 2025 for non-CT, IMF 2031 for CT.
    save_invest_hist : pd.DataFrame
        IMF history (IMF_to2031.csv) for historical savings/investment rates.
    wdi_savings_hist : pd.DataFrame, optional
        WDI savings history and 2025 anchor for non-CT economies.
    delta_hist : pd.DataFrame
        PWT history (PWT_cap_labour_to2023.csv) for historical delta.
    gdp_9th : pd.DataFrame
        9th Outlook data (long format) for comparison; must contain
        variable == 'real_GDP' rows in millions USD PPP.
    lab_eff_periods, high_eff, ... cap_compare
        Model parameters; per-economy values in config/gdp_model_parameters.csv.

    Returns
    -------
    (GDP_estimates_long, labour_efficiency_df)
        GDP_estimates_long : long dataframe of all model series with display
        labels, ready for saving/plotting.
        labour_efficiency_df : labour + efficiency paths to FINAL_YEAR.
    """
    if jump_off_year is None:
        jump_off_year = jump_off_year_for_economy(economy)
    projection_start_year = jump_off_year + 1

    # --- 1. Labour efficiency growth beyond the historical jump-off year ---
    eff_df = labour_data[labour_data["economy_code"] == economy][["year", "percent"]]\
        .set_index("year").iloc[-lab_eff_periods:]

    for year in range(projection_start_year, FINAL_YEAR + 1):
        rolling_avg = eff_df["percent"].iloc[-lab_eff_periods:].sum() / lab_eff_periods
        if rolling_avg > high_eff or rolling_avg < low_eff:
            eff_df.loc[year, "percent"] = _band_walk(rolling_avg, high_eff, low_eff, change_eff)
        else:
            eff_df.loc[year, "percent"] = eff_df.loc[year - 1, "percent"]

    GDP_df2 = input_data[input_data["economy_code"] == economy].copy().set_index("year")

    for year in range(projection_start_year, FINAL_YEAR + 1):
        GDP_df2.at[year, "efficiency"] = GDP_df2.loc[year - 1, "efficiency"] \
            * (1 + eff_df.loc[year, "percent"])

    labour_efficiency_df = GDP_df2[["economy_code", "economy", "labour", "efficiency"]].copy()

    # --- 2. Savings and depreciation jump-off values ---
    # sav_data holds savings in percent of GDP; convert to a fraction here.
    savings = sav_data[sav_data["economy_code"] == economy]["value"].values[0] / 100
    delta = delta_data[delta_data["economy_code"] == economy]["value"].values[0]

    years_index = range(HISTORY_START_YEAR, FINAL_YEAR + 1)
    dyn_savings = pd.DataFrame(index=years_index, columns=["savings"])
    dyn_savings.index.name = "year"
    dyn_savings.loc[projection_start_year, "savings"] = savings

    dyn_delta = pd.DataFrame(index=years_index, columns=["delta"])
    dyn_delta.index.name = "year"
    dyn_delta.loc[projection_start_year, "delta"] = delta

    cap_growth = cap_growth_data[cap_growth_data["economy_code"] == economy][["year", "percent"]]\
        .set_index("year")

    # --- 3. Solow-Swan projection loop ---
    for year in range(projection_start_year, FINAL_YEAR + 1):
        if year < FINAL_YEAR:
            dyn_savings.loc[year + 1, "savings"] = _band_walk(
                dyn_savings.loc[year, "savings"], high_sav, low_sav, change_sav)
            dyn_delta.loc[year + 1, "delta"] = _band_walk(
                dyn_delta.loc[year, "delta"], high_delta, low_delta, change_del)

        # Capital accumulation
        new_cap_calc = GDP_df2.loc[year - 1, "capital"] \
            - (GDP_df2.loc[year - 1, "capital"] * dyn_delta.loc[year, "delta"]) \
            + (GDP_df2.loc[year - 1, "real_output"] * dyn_savings.loc[year, "savings"])

        cap_prev = GDP_df2.loc[year - 1, "capital"]
        cap_diff = (new_cap_calc / cap_prev) - 1

        # Cap the change in capital growth relative to last year's growth
        growth_ratio = cap_diff / cap_growth.loc[year - 1, "percent"]
        if (growth_ratio < (1 - cap_compare)) | (growth_ratio > (1 + cap_compare)):
            GDP_df2.at[year, "capital"] = GDP_df2.loc[year - 1, "capital"] \
                * (1 + (cap_growth.loc[year - 1, "percent"] * (1 - cap_compare)))
            cap_growth.loc[year, "percent"] = cap_growth.loc[year - 1, "percent"] * (1 - cap_compare)
        else:
            GDP_df2.at[year, "capital"] = new_cap_calc
            cap_growth.loc[year, "percent"] = cap_diff

        # Cobb-Douglas output
        GDP_df2.at[year, "real_output"] = (GDP_df2.loc[year, "capital"]) ** alpha \
            * ((GDP_df2.loc[year, "labour"] * GDP_df2.loc[year, "efficiency"]) ** (1 - alpha))

    # --- 4. Historical savings and depreciation (for reporting) ---
    if economy == "02_BD":
        # Brunei: prefer WDI savings; retain the IMF investment proxy only as
        # a fallback for years absent from the WDI history.
        hist_sav = save_invest_hist[
            (save_invest_hist["variable"] == "Total investment")
            & (save_invest_hist["economy_code"] == economy)
        ].set_index("year")["value"] / 100
        if wdi_savings_hist is not None:
            wdi_hist = wdi_savings_hist[
                wdi_savings_hist["economy_code"] == economy
            ].set_index("year")["value"] / 100
            hist_sav = wdi_hist.combine_first(hist_sav)
    elif economy == "13_PNG":
        # PNG: use the WDI seed/corridor path where available; use its anchor
        # for earlier years without a WDI observation.
        hist_sav = pd.Series(savings, index=range(HISTORY_START_YEAR, jump_off_year + 1))
        if wdi_savings_hist is not None:
            wdi_hist = wdi_savings_hist[
                wdi_savings_hist["economy_code"] == economy
            ].set_index("year")["value"] / 100
            hist_sav = wdi_hist.combine_first(hist_sav)
    else:
        hist_sav = save_invest_hist[
            (save_invest_hist["variable"] == "Gross national savings")
            & (save_invest_hist["economy_code"] == economy)
        ].set_index("year")["value"] / 100
        if economy != "18_CT" and wdi_savings_hist is not None:
            wdi_hist = wdi_savings_hist[
                wdi_savings_hist["economy_code"] == economy
            ].set_index("year")["value"] / 100
            hist_sav = wdi_hist.combine_first(hist_sav)

    hist_years = range(HISTORY_START_YEAR, jump_off_year + 1)
    dyn_savings.loc[hist_years, "savings"] = hist_sav.reindex(hist_years).values

    hist_delta = delta_hist[
        (delta_hist["variable"] == "delta") & (delta_hist["economy_code"] == economy)
    ].set_index("year")["value"]
    pwt_years = range(HISTORY_START_YEAR, LAST_PWT_YEAR + 1)
    dyn_delta.loc[pwt_years, "delta"] = hist_delta.reindex(pwt_years).values
    dyn_delta.loc[range(LAST_PWT_YEAR + 1, jump_off_year + 1), "delta"] = delta

    GDP_df2 = pd.concat([GDP_df2, dyn_savings, dyn_delta], axis=1)

    # --- 5. Finalise and label ---
    GDP_estimates = GDP_df2.reset_index()
    GDP_estimates["real_output_historical"] = np.where(
        GDP_estimates["year"] <= jump_off_year, GDP_estimates["real_output"], np.nan)
    GDP_estimates["real_output_projection"] = np.where(
        GDP_estimates["year"] > jump_off_year, GDP_estimates["real_output"], np.nan)

    GDP_estimates["economy_code"] = GDP_estimates["economy_code"].astype(str)
    gdp_9th = gdp_9th.copy()
    gdp_9th["economy_code"] = gdp_9th["economy_code"].astype(str)
    gdp_9th_filtered = gdp_9th[gdp_9th["variable"] == "real_GDP"][["economy_code", "year", "value"]]

    GDP_estimates = GDP_estimates.merge(gdp_9th_filtered, on=["economy_code", "year"], how="left")\
        .rename(columns={"value": "real_output_9th"})

    # Drop the original real_output column since we've split it into IMF/projection versions
    GDP_estimates = GDP_estimates.drop(columns=["real_output"])

    GDP_estimates_long = GDP_estimates.melt(id_vars=["economy_code", "economy", "year"])
    historical_label = LABEL_IMF if economy == "18_CT" else LABEL_WDI
    GDP_estimates_long["variable"] = GDP_estimates_long["variable"].map({
        "real_output_historical": historical_label,
        "real_output_projection": LABEL_APERC,
        "real_output_9th": LABEL_9TH,
        "labour": "Population",
        "capital": "Capital stock",
        "efficiency": "Labour efficiency",
        "savings": "Savings",
        "delta": "Depreciation",
    })

    return GDP_estimates_long, labour_efficiency_df
