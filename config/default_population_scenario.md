# `default_population_scenario.csv` — population source selections

This file chooses the source used to prepare the bundled sample's default
population path for each APEC economy. It is separate from
[`gdp_model_parameters.csv`](./gdp_model_parameters.md): this file selects
population data, while the parameter file sets GDP-model assumptions.

The app reads the population paths from the loaded wide CSV. It does not read
this configuration or the source workbooks at runtime. To apply a configuration
change to the bundled sample, regenerate `sample_wide_input.csv` using
`generate_samples.refresh_streamlit_samples()`.

## Configuration columns

`default_population_scenario.csv` must contain one row for each economy code
in [`economies.csv`](./economies.csv), with these columns:

| Column | Accepted values | Years selected |
| --- | --- | --- |
| `economy_code` | One canonical code from `economies.csv` | Identifies the economy |
| `Historical to 2024` | `EGEDA` or `Custom` | 1990–2024, inclusive |
| `Projection` | `MED`, `LOW`, `HIGH`, or `Custom` | 2025–2060, inclusive |

Choices are case-sensitive. The loader checks that codes are unique and match
the economy list, and that each source choice is allowed. During population
preparation, each selected source must contain exactly one finite, positive
value for every year in its selected period. Missing files, missing or invalid
values, or duplicate years raise an error; the loader does not silently
substitute another source.

## Source files and units

Population values passed to the model are in thousands:

- **Historical `EGEDA`** reads `00APEC_preliminary_2026_1022_ind.xlsx`, sheet
  `in`, the rows with flow `50 Population (millions)`, and year columns
  1990–2024. Economy codes are mapped from the workbook's `economy` values to
  the canonical codes in `economies.csv`. The workbook's millions are
  multiplied by 1,000.
- **Historical `Custom`** reads `population_1jan` rows from
  `custom_population_manually_consolidated.xlsx` for 1990–2024.
- **Projection `MED`, `LOW`, or `HIGH`** reads `population_1jan` rows from the
  matching `undesa_pop_to2100_med.csv`, `undesa_pop_to2100_low.csv`, or
  `undesa_pop_to2100_high.csv` for 2025–2060.
- **Projection `Custom`** reads `population_1jan` rows from
  `custom_population_manually_consolidated.xlsx` for 2025–2060.

The long-format custom workbook and UN DESA CSVs contain `economy_code`,
`year`, `variable`, and `value` fields. Their population values are already in
thousands and are not rescaled. The selected historical and projection
segments are concatenated at the 2024/2025 boundary as supplied, without
interpolation or rebasing.

The EGEDA workbook and the consolidated custom population workbook named
above are tracked bundled inputs in this standalone app. Researcher-supplied
replacement or additional source workbooks should be placed under
`results/data/` where sample preparation or model-input preparation will run.
Only files required by the configured default sources need to be read for the
Default path. Sample refresh also requires all three UN DESA Low/Medium/High
CSVs to prepare the sensitivity columns.

## Prepared wide-input scenarios

When the bundled sample is refreshed:

| Scenario | Wide-input column | Prepared values |
| --- | --- | --- |
| Default | `population_1jan` | The configured historical and projection selections |
| Low | `population_1jan_low` | The configured historical selection, then UN DESA Low from 2025 |
| Medium | `population_1jan_medium` | The configured historical selection, then UN DESA Medium from 2025 |
| High | `population_1jan_high` | The configured historical selection, then UN DESA High from 2025 |
| Custom | `population_1jan_custom` | Existing optional scenario values are preserved from the sample; this column is not generated from the Custom source selection |

Thus Low, Medium, and High share the configured history through 2024, but use
their respective UN DESA projections starting in 2025. The Default projection
uses the `Projection` choice in the configuration and may overlap one of those
comparison scenarios.

The configured population also feeds `run_model_input_prep()` when that
pipeline helper is run. At app runtime, the standard GDP calculation and the
Default comparison use `population_1jan` from the currently loaded CSV. An
uploaded CSV's existing `population_1jan` is preserved; its data is not
replaced by this configuration. Uploaded comparison columns are likewise
used as supplied.

## Custom default source vs. Custom scenario

The `Custom` value in this configuration selects a source for either the
Default path's history or projection. It does not populate or enable the
separate `population_1jan_custom` comparison scenario.

The optional Custom scenario is determined from the `population_1jan_custom`
column in the loaded wide CSV. An economy is eligible when it has exactly one
finite, positive custom value for every year from 1990 through 2060,
independently of the Medium path. Economies with entirely blank custom values
have no Custom scenario; partially blank paths are not offered. Supplied
non-numeric, non-finite, or non-positive custom values are rejected during
wide-input validation.

The Population Scenarios tab offers Custom in its By Economy view only for an
eligible selected economy; its APEC total includes Custom only when every
economy in the loaded population data is eligible, and does not offer it
otherwise. Economy Deep Dive also offers Custom only for an eligible selected
economy. When refreshing the bundled sample, existing `population_1jan_custom`
values are preserved by economy and year; the configured Custom source workbook
is not copied into that scenario column.

## Updating the configuration

1. Edit the source choices in `default_population_scenario.csv`, keeping one
   row per canonical economy code and using only the accepted values above.
2. Ensure the corresponding source workbooks or CSVs are present under
   `results/data/`.
3. Regenerate the bundled sample:

   ```powershell
   python -c "import generate_samples; print(generate_samples.refresh_streamlit_samples())"
   ```

4. Reload the refreshed sample in the app. Uploaded files continue to use their
   own population columns and are unaffected by the configuration.
