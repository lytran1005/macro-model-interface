# `gdp_model_parameters.csv` — what the parameters mean

This file holds the per-economy Solow-Swan model assumptions used by the main
calibrated run (`generate_results.py`, step 6) and, via the twin file
`gdp_model_parameters_sensitivity.csv`, by the population sensitivity run
(`run_sensitivity.py`, step 7). Every column (except `economy_code` and
`notes`) is passed straight to `model.aperc_gdp_model` as a keyword argument,
so the column set must match that function's signature.

The bands below are **assumed ranges**, not hard facts. They were set to make
the projections reasonable and should be revisited whenever there is evidence
to change them. Any change to an assumption has knock-on effects — see
[If you change an assumption](#if-you-change-an-assumption).

**Units used in this file:** rates are stored as decimal fractions. For
example, `0.015` means 1.5% per year, `0.002` means 0.2 percentage points per
year, and `0.0001` means 0.01 percentage points per year. The savings and
depreciation values shown in the output are proportions: `0.25` means 25%.

| Parameter | Role | Notes |
| --- | --- | --- |
| `lab_eff_periods` | Look-back window (years) for labour-efficiency growth | The model averages efficiency growth over this many recent historical years and nudges that average toward the `[low_eff, high_eff]` band. A shorter period reacts more to recent history; a longer period smooths shocks. |
| `high_eff` / `low_eff` | Labour-efficiency growth band | Default corridor **1.2–1.5%/yr**. It matters only when the rolling efficiency-growth average is outside the corridor; inside it, the current rate is retained. Some economies widen it where historical or structural evidence supports faster growth, such as PNG 6–8% and Indonesia 1.2–3%. |
| `change_eff` | Annual step toward the efficiency band | `0.0001` means 0.01 percentage points per year; `0.002` means 0.2 percentage points. A larger step removes an above-band or below-band deviation faster; a smaller step preserves the starting trajectory longer. |
| `high_sav` / `low_sav` | Savings-rate corridor | Default corridor **22–25%**, seeded at the economy's source jump-off year: IMF 2031 for CT and WDI 2025 for other economies. Exceptions: PRC 18–20%, THA 22–34%. |
| `change_sav` | Annual step toward the savings corridor | `0.002` means 0.2 percentage points per year. The effect depends on the starting value: if savings starts above the corridor, a larger step lowers GDP sooner; if it starts below, a larger step raises GDP sooner. |
| `high_delta` / `low_delta` | Depreciation corridor | Default corridor **4.4–4.6%**, seeded from 2023 PWT data (`PWT_delta_2023.csv`) and carried to the jump-off year. Higher depreciation reduces capital accumulation; lower depreciation increases it. Exceptions: PRC 5.2–5.4%, SGP high 5%. |
| `change_del` | Annual step toward the depreciation corridor | `0.0005` means 0.05 percentage points per year. It controls how quickly depreciation moves toward the corridor, not the eventual corridor itself. Its GDP effect is conditional on whether the starting depreciation rate is above or below the corridor. |
| `alpha` | Capital share in the Cobb-Douglas production function `Y = K^alpha · (E·L)^(1-alpha)` | Fixed at **0.4** for all economies — the standard middle-ground capital share, consistent with national-accounting labour shares of roughly 60–70% for APEC. |
| `cap_compare` | Series-break guardrail on capital growth | Caps how much year-on-year capital growth may jump relative to the previous year (default 0.05). Model-stability tuning, not an economics assumption. |
| `notes` | Free-text per-economy caveats | Not used by the model; documents exceptions such as Brunei's IMF investment proxy and PNG's IMF-fallback savings assumption. WDI savings are preferred for non-CT economies when available. |

### Interpreting `alpha`

`alpha` is the capital share in production, not a measure of how wealthy or
developed an economy is. A capital-intensive economy may justify a higher
value, while a labour-intensive economy may justify a lower value. This can
occur in both developed and developing economies: manufacturing, mining,
energy, and infrastructure-heavy economies may have higher capital shares,
whereas agriculture, informal activity, or labour-intensive services may have
lower shares.

Use national-accounts labour compensation shares, sector composition, capital
intensity, resource dependence, and comparable empirical estimates to justify
an economy-specific value. The common `0.4` value is a central assumption, not
a rule that every economy must use. Test modest alternatives such as `0.35`,
`0.40`, and `0.45`, and rerun model-input preparation because changing
`alpha` also changes the historically derived labour-efficiency series.

## How the bands are applied

Savings, depreciation, and labour-efficiency growth all follow the same
pattern: start from a seeded/historical value and, each projection year, walk
one `change_*` step toward the `[low, high]` band until inside it
(`_band_walk` in `model.py`). Inside the band the value stays where it is, so
economies retain their historical level differences.

- **Efficiency**: seeded from the recent average of labour-efficiency growth.
- **Savings**: seeded at the economy's source jump-off value: WDI 2025 for non-CT economies and IMF 2031 for CT.
- **Depreciation**: seeded at the 2023 PWT value and held forward to the economy's jump-off year.

## How parameters affect GDP

The model is a Solow-Swan calculation. Its production function is:

`Y = K^alpha * (E * L)^(1 - alpha)`

where `Y` is GDP, `K` is capital, `E` is labour efficiency, and `L` is
population. Capital evolves as:

`K(t) = K(t-1) * (1 - delta) + savings(t) * Y(t-1)`

The practical implications are:

| To make projected GDP... | Usually change... | Main mechanism |
| --- | --- | --- |
| Higher | Shorten `lab_eff_periods`, raise the efficiency corridor, or slow `change_eff` when the starting rate is above the corridor | Higher efficiency raises output directly and then supports more capital accumulation |
| Higher | Raise the savings corridor, or slow `change_sav` when the starting savings rate is above the corridor | More of previous output becomes investment rather than consumption |
| Higher | Lower the depreciation corridor, or move toward it faster when the starting depreciation rate is above the corridor | More capital survives each year |
| Lower | Lengthen `lab_eff_periods`, lower the efficiency corridor, or speed `change_eff` when the starting rate is above the corridor | Recent high growth is smoothed or reduced |
| Lower | Lower the savings corridor, or speed `change_sav` when the starting savings rate is above the corridor | Capital accumulation slows |
| Lower | Raise the depreciation corridor, or move toward it faster when the starting depreciation rate is below the corridor | Capital is lost more quickly |

These directions are conditional. For example, increasing `change_sav` does
not always lower GDP: it lowers GDP faster only when the starting savings rate
is above the target corridor. If it starts below the corridor, the same change
raises GDP faster.

The strongest long-run levers are usually `lab_eff_periods`, the efficiency
corridor, and the savings corridor. `change_*` parameters mainly control the
speed of transition, so they often have little effect in the first few years
but compound over a longer horizon. Population is not controlled by this file;
it comes from the UN DESA population inputs.

## How to justify a corridor

Treat each corridor as a transparent scenario assumption, not as a precisely
estimated fact. A defensible corridor should be supported by:

1. **Historical evidence:** calculate the economy's recent growth distribution
  and show whether the proposed band is near its median, recent average, or a
  deliberately conservative percentile.
2. **Economic structure:** explain persistent differences such as productivity
  catch-up, ageing, capital intensity, investment needs, or limited capacity
  to absorb investment.
3. **Source consistency:** document whether the seed comes from IMF, WDI, PWT,
  or an APERC assumption, and avoid treating a source change as an economic
  shock without checking the level and trend first.
4. **Scenario purpose:** label the corridor as central, lower, or higher rather
  than implying that its endpoints are forecasts with equal probability.
5. **Sensitivity evidence:** report how changing the corridor or look-back
  period changes GDP at the intended planning horizon, such as 2040, 2050, or
  2060.

The width of a corridor represents uncertainty. A narrow band says that the
long-run rate is believed to be stable; a wider band preserves more historical
or structural uncertainty. The midpoint should have an economic rationale, and
the width should be large enough to cover plausible outcomes without being so
wide that it becomes an unbounded tuning knob.

## A practical tuning workflow

When an economy's projected GDP seems too high or too low:

1. Check the source level at the jump-off year and the growth continuity from
  the historical source into the first APERC year.
2. Check the final `lab_eff_periods` years of efficiency growth; do not rely on
  the full 1990–2025 average when the model uses a shorter window.
3. Change one parameter family at a time and record GDP at the planning
  horizon. Start with a modest alternative, such as 5/7/10 efficiency years
  or a corridor shifted by 0.2 percentage points.
4. Use `change_*` to control the transition speed, not to redefine the
  long-run assumption. Use the corridor endpoints to change the long-run
  direction.
5. Compare levels, annual growth, and drivers together. A smooth but widening
  difference is compounding sensitivity; a one-year jump is a handoff or data
  problem.
6. Keep the chosen value only when it can be justified independently of the
  desired GDP outcome. Record the reason in the row's `notes` field.

## If you change an assumption

- **Change the CSV, not the model.** Per-economy values in this file fully
  override the defaults in the `aperc_gdp_model` signature (`model.py`); the
  function itself is the single authoritative calculation and should not be
  edited for a parameter tweak. Editing one cell changes only that economy.
- **Keep the twin files in sync.** `gdp_model_parameters.csv` and
  `gdp_model_parameters_sensitivity.csv` must stay identical so the
  population sensitivity run isolates population uncertainty only (Low/Medium/
  High UN DESA variants). Edit the main file, then run
  `workflow/scripts/Other/sync_sensitivity_parameters.py` (set `REPORT_ONLY = False`,
  `APPLY_SYNC = True`) — it backs up the old sensitivity file to
  `gdp_model_parameters_sensitivity.backup.csv` first.
- **PNG's depreciation is not in this file.** PNG is missing from PWT, so its
  assumed depreciation/output-capital ratio is a constant at the top of
  `prepare_pwt.py`. The 25% PNG savings value in `prepare_imf.py` is only an
  IMF fallback; WDI savings are preferred for PNG when available.
- **Do not add, rename, or drop columns.** `config_loaders.load_gdp_model_parameters`
  drops only `economy_code` and `notes` and passes everything else to
  `aperc_gdp_model` as keyword arguments; an extra column raises a `TypeError`,
  and a renamed column silently changes what is passed. Keep `economy_code`
  values equal to `config/economies.csv` (the loader validates this).
- **Re-run and validate.** After any change, re-run the pipeline
  (`python workflow/scripts/run_pipeline.py`) and check that every row of
  `results/validation_report.csv` says `PASS`.
