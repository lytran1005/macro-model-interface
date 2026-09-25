#%%
"""Central path definitions for the macro_variables_10th workflow.

Every script imports its paths from here so nothing depends on the
current working directory. Do not use os.chdir() anywhere in the repo.
"""

from pathlib import Path

#%%
# constants

# This file lives at <project>/macro_model/paths.py and resolves to the
# standalone app root directory rather than the original repo layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_SOURCE_DIR = PROJECT_ROOT / "data_source"
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"

#%%
# functions

def ensure_dir(path):
    """Create a directory (and parents) if it does not exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
