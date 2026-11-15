# `gdp_model_parameters.csv` — what the parameters mean

This file holds the per-economy Solow-Swan assumptions loaded as defaults by
the standalone app. Standard and population-sensitivity runs use the loaded
model parameters; Economy Deep Dive can apply session-specific parameters to
its selected economy. Every column (except `economy_code` and `notes`) is
defined in the model configuration. `cap_compare` remains fixed from the
configured CSV and is not accepted as an upload override.

The global parameter upload requires exactly one row for each of the 21 APEC
economies. The Economy Deep Dive uploader continues to apply only the row for
the selected economy. Both upload paths validate finite values and the bounds
and relationships below; these are input-validation limits, not claims that
every economy should share the same parameter values.

**Units used in this file:** rates are stored as decimal fractions. For
example, `0.015` means 1.5% per year, `0.002` means 0.2 percentage points per
year, and `0.0001` means 0.01 percentage points per year. The savings and
depreciation values shown in the output are proportions: `0.25` means 25%.

| Parameter | Role | Notes |
| --- | --- | --- |
| `lab_eff_periods` | Look-back window (years) for labour-efficiency growth | Uploads accept an integer from **1 to 10**. The model averages efficiency growth over this many recent historical years and nudges that average toward the `[low_eff, high_eff]` band. |
| `high_eff` / `low_eff` | Labour-efficiency growth band | Each bound must be from **0.5% to 10%/yr**, with `low_eff` no greater than `high_eff`. The economy-specific values in the CSV are retained. |
| `change_eff` | Annual step toward the efficiency band | Must be finite and non-negative. `0.0001` means 0.01 percentage points per year; `0.002` means 0.2 percentage points. |
| `high_sav` / `low_sav` | Savings-rate corridor | Each bound must be from **15% to 50%**, with `low_sav` no greater than `high_sav`. The starting value continues to use the existing economy-specific savings workflow. |
| `change_sav` | Annual step toward the savings corridor | Must be finite and non-negative. `0.002` means 0.2 percentage points per year. |
| `high_delta` / `low_delta` | Depreciation corridor | Each bound must be from **3% to 7%**, with `low_delta` no greater than `high_delta`. |
| `change_del` | Annual step toward the depreciation corridor | Must be finite and non-negative. `0.0005` means 0.05 percentage points per year. |
| `alpha` | Capital share in the Cobb-Douglas production function `Y = K^alpha · (E·L)^(1-alpha)` | Must be from **0.3 to 0.5**. Economy-specific values are allowed and affect the derived labour-efficiency series. |
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

For bundled researcher overrides and future GDP supplied in uploaded CSVs, the
provided GDP values take precedence through their final supplied year. Capital,
savings, depreciation and efficiency continue evolving during those years, but
the GDP model resumes calculating output only in the following year. See the
README for the different bundled-data and upload rules.

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
but compound over a longer horizon. The bundled sample's default population
is prepared separately using
[`default_population_scenario.csv`](./default_population_scenario.csv).
The app uses the population columns in the loaded CSV, including for uploads;
see the [default population guide](./default_population_scenario.md).

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
- **Compare population paths with consistent parameters.** The app's
  population sensitivity run uses the same loaded parameter set for the
  Default, Low, Medium, High and available Custom population paths. Deep Dive
  parameter controls apply only to its selected economy and view.
- **PNG's depreciation is not in this file.** PNG is missing from PWT, so its
  assumed depreciation/output-capital ratio is a constant at the top of
  `prepare_pwt.py`. The 25% PNG savings value in `prepare_imf.py` is only an
  IMF fallback; WDI savings are preferred for PNG when available.
- **Do not add, rename, or drop columns.** `config_loaders.load_gdp_model_parameters`
  drops only `economy_code` and `notes` and passes everything else to
  `aperc_gdp_model` as keyword arguments; an extra column raises a `TypeError`,
  and a renamed column silently changes what is passed. Keep `economy_code`
  values equal to `config/economies.csv` (the loader validates this).
- **Re-run and validate.** Restart the app after editing the parameter file,
  run the model and compare the affected economy's outputs. The full pipeline
  and its validation scripts are maintained outside this standalone project.
