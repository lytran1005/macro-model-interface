# APERC Solow-Swan GDP Model — standalone interface

This project is a self-contained Streamlit front end for the APERC 10th Outlook
macro model. It copies the UI and the required model helpers into a standalone
root so it can run without depending on the original repository layout.

The app reads its configuration from the local `config` directory. Population
scenario paths and the corresponding GDP sensitivity runs use the population
columns in the currently loaded wide CSV.

---

## 1. Project layout

- `app.py` — Streamlit UI
- `core.py` — model execution and data loading
- `generate_samples.py` — refreshes the bundled sample CSV from prepared pipeline outputs
- `macro_model/` — copied model helper modules
- `config/` — economy metadata and model parameter tables
- `results/data/` — prepared source data used only to refresh the bundled sample
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
Update an existing environment with
`conda env update -n macro10-interface-env -f envs/environment.yml --prune`.

### Run order

For normal use, start only the Streamlit app:

1. Install the project dependencies once.
2. From this project folder, run `python -m streamlit run app.py`.
3. `app.py` imports `core.py` and the helper modules under `macro_model/`;
   you do not need to run those files separately. The app reads the bundled
   `sample_wide_input.csv` or a CSV uploaded in the interface.

To refresh the bundled samples after the prepared source files in
`results/data/` change, run this from the project folder before launching (or
relaunching) the app:

```powershell
python -c "import generate_samples; print(generate_samples.refresh_streamlit_samples())"
```

The refresh function rebuilds `sample_wide_input.csv` from the available prepared outputs. In the full data pipeline, this refresh
belongs **after** the upstream scripts have finished writing the required
`results/` files. Those upstream pipeline scripts are not part of this
standalone interface, so there is no additional sequence of pipeline `.py`
files to run here. If you only want to run the interface using its current
sample, skip the refresh step.

---

## 3. What is included

The project is intentionally packaged with the files needed for a local run:

- `config/` copied from the main project, including economy codes and GDP model parameters
- `results/` copied from the main project as source data for regenerating the sample
- `macro_model/` copied from the main project to preserve the same calculation logic
- the bundled `sample_wide_input.csv` for offline/demo use when the prepared outputs are unavailable

The wide input may include `population_1jan_low`, `population_1jan_medium`,
and `population_1jan_high`. The Population Scenarios tab and Economy Deep Dive
use these columns from the currently loaded file; the main model input column
`population_1jan` remains the population path used for the standard results.
When refreshing the bundled sample, these three columns are populated from
the matching prepared UN DESA outputs. GDP for each population scenario is recalculated by the app from the loaded wide CSV; nothing is read from `results/` at runtime.

**Optional custom population.** Add a `population_1jan_custom` column to the
wide CSV to include a user-defined "Custom" scenario (it is not a UN DESA
variant). The official Low/Medium/High columns are never modified. Custom is
shown only when at least one economy has a value in every year where
`population_1jan_medium` has one; economies with a partial path are ignored
with a warning, and economies left blank simply have no Custom line. The
Custom population and GDP lines appear alongside the official scenarios in the
Population Scenarios tab (By Economy view; the APEC total only includes Custom
when every economy is covered) and as an extra option in Economy Deep Dive for
covered economies. Without the column the app behaves as before.

The app now resolves all paths relative to this project root instead of assuming the original repository structure.

---

## 4. Notes

### Summary metrics

The top-of-page summary metrics are currently disabled. To show them again
after the underlying data has been reviewed and agreed on, open `app.py` and
change `SHOW_SUMMARY_METRICS = False` to `SHOW_SUMMARY_METRICS = True`.

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
