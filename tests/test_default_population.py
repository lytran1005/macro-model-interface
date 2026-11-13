import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import config_loaders
import core
import population
import generate_samples


class PopulationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data_dir = Path(self.directory.name)
        self.economies = pd.DataFrame({
            "economy_code": ["01_AUS", "02_BD"],
            "economy": ["Australia", "Brunei Darussalam"],
        })
        years = range(1990, 2025)
        egeda = pd.DataFrame([
            {"economy": "01AUS", "flows": "50 Population (millions)",
             **{year: 10 + (year - 1990) / 100 for year in years}},
            {"economy": "02BD", "flows": "50 Population (millions)",
             **{year: 1 for year in years}},
        ])
        with pd.ExcelWriter(self.data_dir / population.EGEDA_FILE) as writer:
            egeda.to_excel(writer, sheet_name="in", index=False)
        for choice, filename in population.PROJECTION_FILES.items():
            value = {"LOW": 20, "MED": 30, "HIGH": 40}[choice]
            pd.DataFrame([
                {"economy_code": code, "year": year,
                 "variable": "population_1jan", "value": value}
                for code in self.economies["economy_code"]
                for year in range(2025, 2061)
            ]).to_csv(self.data_dir / filename, index=False)
        pd.DataFrame([
            {"economy_code": code, "year": year,
             "variable": "population_1jan", "value": 50 + year - 1990}
            for code in self.economies["economy_code"]
            for year in range(1990, 2061)
        ]).to_excel(self.data_dir / population.CUSTOM_FILE, index=False)
        self.selections = {
            "01_AUS": {"Historical to 2024": "EGEDA", "Projection": "HIGH"},
            "02_BD": {"Historical to 2024": "Custom", "Projection": "LOW"},
        }

    def load(self):
        with patch.object(population, "load_economies", return_value=self.economies), \
             patch.object(population, "load_default_population_scenarios", return_value=self.selections):
            return population.load_default_population(data_dir=self.data_dir)

    def test_selects_independently_and_converts_millions_to_thousands(self):
        result = self.load().set_index(["economy_code", "year"])["value"]
        self.assertEqual(len(result), 2 * 71)
        self.assertEqual(result["01_AUS", 2024], 10340)
        self.assertEqual(result["01_AUS", 2025], 40)
        self.assertEqual(result["02_BD", 2024], 84)
        self.assertEqual(result["02_BD", 2025], 20)
        self.selections["02_BD"]["Projection"] = "Custom"
        result = self.load().set_index(["economy_code", "year"])["value"]
        self.assertEqual(result["02_BD", 2025], 85)
        self.assertEqual(result["02_BD", 2060], 120)

    def test_sensitivity_paths_share_configured_history(self):
        default = self.load()
        paths = population.load_population_sensitivity_paths(default, self.data_dir)
        paths = paths.set_index(["economy_code", "year"])
        for column in core.POPULATION_SCENARIO_COLUMNS.values():
            self.assertEqual(paths.loc[("01_AUS", 2024), column], 10340)
            self.assertEqual(paths.loc[("02_BD", 2024), column], 84)
        self.assertEqual(paths.loc[("01_AUS", 2025), "population_1jan_low"], 20)
        self.assertEqual(paths.loc[("01_AUS", 2025), "population_1jan_medium"], 30)
        self.assertEqual(paths.loc[("01_AUS", 2025), "population_1jan_high"], 40)
        self.assertEqual(len(paths), 2 * 71)
        self.assertFalse(paths.index.duplicated().any())

    def test_missing_and_duplicate_selected_values_fail_explicitly(self):
        path = self.data_dir / population.PROJECTION_FILES["HIGH"]
        data = pd.read_csv(path)
        data = data[~((data["economy_code"] == "01_AUS") & (data["year"] == 2030))]
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "01_AUS.*2030"):
            self.load()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate population years"):
            self.load()

    def test_incomplete_custom_and_missing_selected_file_fail(self):
        self.selections["01_AUS"]["Historical to 2024"] = "Custom"
        path = self.data_dir / population.CUSTOM_FILE
        data = pd.read_excel(path)
        data.loc[(data["economy_code"] == "01_AUS") & (data["year"] == 1995), "value"] = None
        data.to_excel(path, index=False)
        with self.assertRaisesRegex(ValueError, "01_AUS.*1995"):
            self.load()
        self.selections["01_AUS"]["Historical to 2024"] = "EGEDA"
        (self.data_dir / population.PROJECTION_FILES["HIGH"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_config_requires_unique_economies_and_valid_choices(self):
        self.economies.to_csv(self.data_dir / "economies.csv", index=False)
        path = self.data_dir / "default_population_scenario.csv"
        config = pd.DataFrame([
            {"economy_code": code, **choice}
            for code, choice in self.selections.items()
        ])
        with patch.object(config_loaders, "CONFIG_DIR", self.data_dir):
            config.to_csv(path, index=False)
            self.assertEqual(config_loaders.load_default_population_scenarios(), self.selections)
            pd.concat([config, config.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                config_loaders.load_default_population_scenarios()
            config.loc[0, "Projection"] = "UNKNOWN"
            config.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid Projection"):
                config_loaders.load_default_population_scenarios()
            config.iloc[:1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing"):
                config_loaders.load_default_population_scenarios()


class DefaultModelIntegrationTests(unittest.TestCase):
    def test_sample_refresh_preserves_optional_custom_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_wide_input.csv"
            sample = pd.read_csv(ROOT / "sample_wide_input.csv")
            sample.loc[0, "population_1jan_custom"] = 12345.6
            sample.to_csv(path, index=False)
            default = sample[["economy_code", "economy", "year", "population_1jan"]].rename(
                columns={"population_1jan": "value"}
            )
            paths = sample[["economy_code", "year", *core.POPULATION_SCENARIO_COLUMNS.values()]]
            with patch.object(generate_samples, "APP_DIR", Path(directory)), \
                 patch.object(generate_samples, "load_default_population", return_value=default), \
                 patch.object(generate_samples, "load_population_sensitivity_paths", return_value=paths):
                refreshed = generate_samples._pipeline_sample_data()
            self.assertEqual(refreshed.loc[0, "population_1jan_custom"], 12345.6)
            self.assertEqual(refreshed["population_1jan_custom"].notna().sum(), 1)

    def test_default_model_replaces_upload_but_sensitivity_keeps_its_path(self):
        wide = pd.read_csv(ROOT / "sample_wide_input.csv")
        wide = wide[wide["economy_code"] == "01_AUS"].copy()
        params = core.load_gdp_model_parameters()
        configured = wide[["economy_code", "economy", "year", "population_1jan"]].copy()
        configured = configured.rename(columns={"population_1jan": "value"})
        configured["variable"] = "population_1jan"
        configured["value"] *= 1.5
        with patch.object(core, "load_default_population", return_value=configured) as loader:
            result, _, errors = core.run_analysis(core.wide_to_long(wide), params, use_default_population=True)
            self.assertEqual(errors, [])
            loader.assert_called_once_with(["01_AUS"])
            explicit_wide = wide.copy()
            explicit_wide["population_1jan"] = configured["value"]
            explicit, _, explicit_errors = core.run_analysis(
                core.wide_to_long(explicit_wide), params, use_default_population=False
            )
            self.assertEqual(explicit_errors, [])
            pd.testing.assert_frame_equal(result, explicit)
            actual = result[(result["variable"] == "population") & (result["year"] == 2060)]
            self.assertAlmostEqual(actual["value"].iloc[0], configured.loc[configured["year"] == 2060, "value"].iloc[0])
            scenario, errors = core.run_uploaded_population_scenario(wide, "High", params, economy="01_AUS")
            self.assertEqual(errors, [])
            loader.assert_called_once()
            actual = scenario[(scenario["variable"] == "population") & (scenario["year"] == 2060)]
            self.assertAlmostEqual(actual["value"].iloc[0], wide.loc[wide["year"] == 2060, "population_1jan_high"].iloc[0])
            default_gdp = result[(result["variable"] == "real_GDP") & (result["year"] == 2060)]["value"].iloc[0]
            high_gdp = scenario[(scenario["variable"] == "real_GDP") & (scenario["year"] == 2060)]["value"].iloc[0]
            self.assertNotAlmostEqual(default_gdp, high_gdp)

    def test_default_run_preserves_uploaded_population(self):
        wide = pd.read_csv(ROOT / "sample_wide_input.csv")
        wide = wide[wide["economy_code"] == "01_AUS"].copy()
        wide["population_1jan"] *= 1.1
        with patch.object(core, "load_default_population") as loader:
            result, _, errors = core.run_analysis(
                core.wide_to_long(wide), core.load_gdp_model_parameters()
            )
        self.assertEqual(errors, [])
        loader.assert_not_called()
        actual = result[(result["variable"] == "population") & (result["year"] == 2060)]
        self.assertAlmostEqual(actual["value"].iloc[0], wide.loc[wide["year"] == 2060, "population_1jan"].iloc[0])

    def test_medium_column_remains_supported(self):
        wide = pd.read_csv(ROOT / "sample_wide_input.csv")
        wide = wide[wide["economy_code"] == "01_AUS"].copy()
        paths = core.population_scenarios_from_wide(wide)
        medium = paths[(paths["scenario"] == "Medium") & (paths["year"] == 2060)]
        self.assertAlmostEqual(medium["value"].iloc[0], wide.loc[wide["year"] == 2060, "population_1jan_medium"].iloc[0])

    def test_default_scenario_matches_standard_model_and_preserves_input(self):
        wide = pd.read_csv(ROOT / "sample_wide_input.csv")
        wide = wide[wide["economy_code"] == "01_AUS"].copy()
        wide["population_1jan"] *= 1.1
        original = wide.copy(deep=True)
        params = core.load_gdp_model_parameters()
        default, errors = core.run_uploaded_population_scenario(wide, "Default", params, economy="01_AUS")
        standard, _, standard_errors = core.run_analysis(core.wide_to_long(wide), params)
        self.assertEqual(errors, [])
        self.assertEqual(standard_errors, [])
        pd.testing.assert_frame_equal(default, standard)
        pd.testing.assert_frame_equal(wide, original)
        scenarios = core.population_scenarios_from_wide(wide)
        default_path = scenarios[scenarios["scenario"] == "Default"].set_index("year")["value"]
        pd.testing.assert_series_equal(
            default_path.sort_index(), wide.set_index("year")["population_1jan"].sort_index(),
            check_names=False,
        )

    def test_custom_availability_requires_complete_positive_finite_values(self):
        wide = pd.read_csv(ROOT / "sample_wide_input.csv")
        wide = wide[wide["economy_code"] == "01_AUS"].copy()
        wide["population_1jan_custom"] = float("inf")
        self.assertNotIn("Custom", core.available_population_scenarios(wide))
        wide["population_1jan_custom"] = wide["population_1jan"]
        self.assertIn("Custom", core.available_population_scenarios(wide))
        wide.loc[wide.index[0], "population_1jan_custom"] = 0
        self.assertNotIn("Custom", core.available_population_scenarios(wide))


if __name__ == "__main__":
    unittest.main()
