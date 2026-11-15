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
| `cap_compare` | Relative-growth guardrail on capital growth | Must be finite and in `[0, 1)`. For non-zero prior growth, it bounds the ratio of raw to prior growth between `1 - cap_compare` and `1 + cap_compare`. A value of `0` creates a zero-width band; exact-zero prior growth passes raw growth through for that year. It is a technical guardrail, not a target growth-rate assumption. |
| `notes` | Free-text per-economy caveats | Not used by the model. The Brunei and PNG notes in the CSV describe their current source treatment; they are not instructions to substitute another savings series. |

### Savings source policy

- The 20 non-Chinese Taipei economies use World Bank WDI gross national
  savings only, through the 2025 jump-off year. IMF observations are not used
  to fill missing WDI values.
- Chinese Taipei uses IMF gross national savings only: historical observations
  through 2030 and the existing 2031 jump-off value. No later IMF observations
  are included.
- Missing source observations remain missing, except for the existing
  conditional Papua New Guinea (PNG) jump-off fallback described below.
- The app consumes the selected `Gross national savings` values from its
  bundled or uploaded wide input. It does not load source files at model-run
  time or pass the optional `wdi_savings_hist` argument to the model.

The Brunei CSV note says it no longer uses savings proxied by IMF Total
Investment. That variable is not a substitute for gross national savings.

The bundled sample retains 45 accepted missing historical observations under
this policy:

| Economy | Missing years | Observations |
| --- | --- | ---: |
| Brunei Darussalam | 1990-2000 | 11 |
| Hong Kong, China | 1990-1997 | 8 |
| Japan | 1990-1995 | 6 |
| New Zealand | 1990-1999 | 10 |
| Russia | 1990-1993 | 4 |
| Viet Nam | 1990-1995 | 6 |
| **Total** |  | **45** |

These are accepted historical source-data limitations. They remain missing
rather than being filled with another source; they are not missing projected
savings values and do not prevent model execution.

### Capital-growth guardrail (`cap_compare`)

Let `g_raw` be the raw capital growth calculated for the current year, `g_prior`
the previous year's applied capital growth, and `c = cap_compare`. When
`g_prior` is non-zero, the model calculates `q = g_raw / g_prior` and permits
the relative-growth band:

`1 - c <= q <= 1 + c`

If `q` is below the band, applied growth is `g_prior * (1 - c)`. If `q` is
above it, applied growth is `g_prior * (1 + c)`. Within the band, the raw
growth is retained. Comparing the ratio is important when prior growth is
negative: it keeps the relative bounds in the intended order and prevents the
guardrail from treating a contraction as a positive-growth case.

When `g_prior` is exactly zero, a percentage-relative band is undefined, so
raw growth passes through for that year. If that produces non-zero applied
growth, the guardrail resumes in the next year. A `cap_compare` value of zero
does not disable the guardrail: it creates a zero-width band and holds applied
growth at the prior year's rate whenever prior growth is non-zero. Configured
values must be finite and in `[0, 1)`.

### Historical capital stock and depreciation

The app does not use a PWT capital-stock series directly. Historical capital
stock is derived as `K = GDP / output_to_kstock`, using the PWT output-to-capital
ratio. The latest available PWT output-to-capital ratio is carried forward
beyond 2023 through the historical GDP source cutoff: 2025 for non-Chinese
Taipei economies and 2031 for Chinese Taipei. These cutoffs do not extend to
cover later GDP overrides.

The 2023 PWT depreciation value is carried forward through the same historical
GDP source cutoff. Capital and depreciation then evolve during model years,
including years with supplied GDP overrides. Overrides can delay the GDP
output handoff without extending the historical capital-stock derivation or
depreciation carry-forward. In the current bundled sample, for example,
Australia has a GDP override through 2030 and Chile through 2031; most other
economies have supplied GDP through 2025.

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
  assumed depreciation/output-capital ratio is a constant in the upstream
  preparation pipeline. PNG's WDI value is the 2025 savings anchor when
  available. The app retains the conditional 25% fallback only when the PNG
  jump-off value is missing or NaN; it is not the normal PNG savings
  source or a replacement for missing historical WDI observations.
- **Do not add, rename, or drop columns.** `config_loaders.load_gdp_model_parameters`
  drops only `economy_code` and `notes` and passes everything else to
  `aperc_gdp_model` as keyword arguments; an extra column raises a `TypeError`,
  and a renamed column silently changes what is passed. Keep `economy_code`
  values equal to `config/economies.csv` (the loader validates this).
- **Re-run and validate.** Restart the app after editing the parameter file,
  run the model and compare the affected economy's outputs. The full pipeline
  and its validation scripts are maintained outside this standalone project.
