# APERC Solow-Swan GDP Model — standalone interface

This project is a self-contained Streamlit front end for the APERC 10th Outlook
macro model. It copies the UI and the required model helpers into a standalone
root so it can run without depending on the original repository layout.

The app reads its configuration from the local `config` directory and loads the
prepared scenario outputs from the local `results` directory.

---

## 1. Project layout

- `app.py` — Streamlit UI
- `core.py` — model execution and data loading
- `generate_samples.py` — sample fallback generation
- `macro_model/` — copied model helper modules
- `config/` — economy metadata and model parameter tables
- `results/` — prepared data used by the scenario charts
- `.streamlit/config.toml` — light theme configuration

---

## 2. Setup

From this folder:

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you are using PowerShell on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

---

## 3. What is included

The project is intentionally packaged with the files needed for a local run:

- `config/` copied from the main project, including economy codes and GDP model parameters
- `results/` copied from the main project, including population scenario and GDP sensitivity outputs
- `macro_model/` copied from the main project to preserve the same calculation logic
- bundled sample CSVs for offline/demo use when the prepared outputs are unavailable

The app now resolves all paths relative to this project root instead of assuming the original repository structure.

---

## 4. Notes

This folder is intended to be portable and runnable from its own directory. The
main repo still remains the source of truth for the full pipeline; this project
contains the app and the minimum supporting files required to run the UI in isolation.
