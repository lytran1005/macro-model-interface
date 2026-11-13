# APERC Solow-Swan GDP Model — standalone interface

This project is a self-contained Streamlit front end for the APERC 10th Outlook
macro model. It copies the UI and the required model helpers into a standalone
root so it can run without depending on the original repository layout.

The app loads model-parameter defaults from the local `config` directory. The
bundled sample's population paths are prepared using the source selections in
`config/default_population_scenario.csv` and source files in `results/data/`.
Default and sensitivity model runs use population columns from the currently
loaded wide CSV; uploaded population values are not replaced.

---

## 1. Project layout

- `app.py` — Streamlit UI
- `core.py` — model execution and data loading
- `generate_samples.py` — refreshes the bundled sample CSV from prepared pipeline outputs
- `macro_model/` — copied model helper modules
- `config/` — economy metadata, model parameters and default population selections
- `results/data/` — source data for preparing the bundled sample and model inputs
- `.streamlit/config.toml` — light theme configuration

---

## 2. Setup

Create and activate the conda environment (named `macro10-interface-env`):

```bash
conda env create -f envs/environment.yml
conda activate macro10-interface-env
streamlit run app.py
```

Dependencies are listed in `envs/environment.yml` (local conda setup) and
`requirements.txt` (pip, used by Streamlit Cloud deployment). Keep them in sync.

> **Deployment note:** deploy with pip, not conda. Keep `requirements.txt` in the
> repo root and do not put an `environment.yml` there. A root-level
> `environment.yml` made Streamlit Cloud builds much slower (conda solve), so the
> conda file lives in `envs/` where the deployment ignores it.

Update an existing environment with
`conda env update -n macro10-interface-env -f envs/environment.yml --prune`.

### Run order

For normal use, start only the Streamlit app:

1. Install the project dependencies once.
2. From this project folder, run `python -m streamlit run app.py`.
3. `app.py` imports `core.py` and the helper modules under `macro_model/`;
   you do not need to run those files separately. The app reads the bundled
   `sample_wide_input.csv` or a CSV uploaded in the interface.

After changing `config/default_population_scenario.csv` or the prepared source
files in `results/data/`, refresh the bundled sample from the project folder:

```powershell
python -c "import generate_samples; print(generate_samples.refresh_streamlit_samples())"
```

The refresh function rebuilds `sample_wide_input.csv` from the prepared outputs
and reports the files written. It returns an empty list when its required
pipeline CSVs are unavailable; selected population sources that are missing or
invalid raise an error. In the full data pipeline, refresh after upstream
scripts have finished writing `results/`. Those scripts are not part of this
standalone interface. To use the current sample without changing its defaults,
skip the refresh. Reload the sample in the app or restart the app after a
refresh; editing configuration alone does not update an already loaded sample.

---

## 3. What is included

The project is intentionally packaged with the files needed for a local run:

- `config/` with economy codes, GDP model parameters and default population source selections
- `results/` copied from the main project as source data for regenerating the sample
- `macro_model/` copied from the main project to preserve the same calculation logic
- the bundled `sample_wide_input.csv` for offline/demo use when the prepared outputs are unavailable

### Population data

The bundled sample's default population sources are configured per economy in
[`default_population_scenario.csv`](./config/default_population_scenario.csv).
For source-file formats, year ranges, preparation steps and scenario details,
see the [configuration guide](./config/default_population_scenario.md).
The app uses the population columns in the loaded wide CSV: `population_1jan`
for Default and the Low/Medium/High columns for comparisons and sensitivity.
Uploads keep their own population values. The separate optional Custom scenario
uses `population_1jan_custom` when a complete valid path is available.

The app now resolves all paths relative to this project root instead of assuming the original repository structure.

---

## 4. Notes

### APEC Overview tab

Shows APEC-wide summary metrics and an interactive map of the 21 economies
(GDP per capita, Real GDP or Population, by model year). It appears after
Economy Comparison and uses the standard model results, like that tab; it does
not have a population-scenario selector. Population and population-dependent
results use `population_1jan` from the loaded CSV (Default). The
Default/Low/Medium/High/Custom choices are available in the Population
Scenarios and Economy Deep Dive workflows when their required columns and
Custom coverage are present. The overview uses the default model parameters,
not the Economy Deep Dive controls. The basemap is
`assets/world-110m.json` (Natural Earth-derived, via `world-atlas`); non-APEC land
is muted and borders are not drawn. Hong Kong, China; Singapore; and Brunei
Darussalam are too small to draw at this resolution and appear as point markers.

### Model parameter files

In the Economy Deep Dive sidebar, upload a parameter CSV previously downloaded
from the app to restore settings. The file may include one or more economies;
only the currently selected economy's row is applied. A successful upload
updates that economy's controls and projection for the current app session
without changing `config/gdp_model_parameters.csv`. Use **Reset to preloaded
defaults** to restore that economy's configured values.

This folder is intended to be portable and runnable from its own directory. The
main repo still remains the source of truth for the full pipeline; this project
contains the app and the minimum supporting files required to run the UI in isolation.
