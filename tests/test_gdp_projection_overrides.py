import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import core


class GdpProjectionOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pd.read_csv(ROOT / "sample_wide_input.csv")
        cls.params = core.load_gdp_model_parameters()
        cls.gdp_column = "Real GDP PPP 2021 USD"

    def economy_input(self, code):
        return self.sample[self.sample["economy_code"] == code].copy()

    @staticmethod
    def real_gdp_at(results, code, year):
        row = results[
            (results["economy_code"] == code)
            & (results["variable"] == "real_GDP")
            & (results["year"] == year)
        ]
        return row["value"].iloc[0]

    def test_repository_overrides_are_loaded_only_when_explicitly_enabled(self):
        wide = self.economy_input("01_AUS")
        source = core.load_repository_gdp_overrides()
        australia = source[source["economy_code"] == "01_AUS"].set_index("year")
        self.assertEqual(australia.index.tolist(), list(range(2026, 2031)))

        baseline, _, baseline_errors = core.run_analysis(
            core.wide_to_long(wide), self.params
        )
        bundled, _, bundled_errors = core.run_analysis(
            core.wide_to_long(wide),
            self.params,
            use_repository_gdp_overrides=True,
        )
        self.assertEqual(baseline_errors, [])
        self.assertEqual(bundled_errors, [])
        self.assertNotAlmostEqual(
            self.real_gdp_at(baseline, "01_AUS", 2026),
            australia.loc[2026, "value"],
        )
        self.assertAlmostEqual(
            self.real_gdp_at(bundled, "01_AUS", 2026),
            australia.loc[2026, "value"],
        )
        scenarios, scenario_errors = core.run_uploaded_population_gdp_scenarios(
            wide, self.params, use_repository_gdp_overrides=True
        )
        self.assertEqual(scenario_errors, [])
        for scenario in ["Default", "Low", "Medium", "High"]:
            value = scenarios[
                (scenarios["economy_code"] == "01_AUS")
                & (scenarios["scenario"] == scenario)
                & (scenarios["year"] == 2026)
            ]["value"].iloc[0]
            self.assertAlmostEqual(value, australia.loc[2026, "value"])

    def test_uploaded_values_are_preserved_across_population_scenarios(self):
        wide = self.economy_input("01_AUS")
        manual_values = {2026: 1_800_000.0, 2027: 1_850_000.0, 2028: 1_900_000.0}
        for year, value in manual_values.items():
            wide.loc[wide["year"] == year, self.gdp_column] = value

        default, _, default_errors = core.run_analysis(
            core.wide_to_long(wide), self.params
        )
        high, high_errors = core.run_uploaded_population_scenario(
            wide, "High", self.params, economy="01_AUS"
        )
        self.assertEqual(default_errors, [])
        self.assertEqual(high_errors, [])
        for year, value in manual_values.items():
            self.assertAlmostEqual(self.real_gdp_at(default, "01_AUS", year), value)
            self.assertAlmostEqual(self.real_gdp_at(high, "01_AUS", year), value)

        self.assertNotAlmostEqual(
            self.real_gdp_at(default, "01_AUS", 2029),
            self.real_gdp_at(high, "01_AUS", 2029),
        )
        baseline, _, baseline_errors = core.run_analysis(
            core.wide_to_long(self.economy_input("01_AUS")), self.params
        )
        self.assertEqual(baseline_errors, [])
        self.assertNotAlmostEqual(
            self.real_gdp_at(default, "01_AUS", 2029),
            self.real_gdp_at(baseline, "01_AUS", 2029),
        )

    def test_uploaded_ct_projections_extend_after_imf_cutoff(self):
        wide = self.economy_input("18_CT")
        manual_values = {2032: 2_500_000.0, 2033: 2_600_000.0}
        for year, value in manual_values.items():
            wide.loc[wide["year"] == year, self.gdp_column] = value

        results, _, errors = core.run_analysis(core.wide_to_long(wide), self.params)
        self.assertEqual(errors, [])
        for year, value in manual_values.items():
            self.assertAlmostEqual(self.real_gdp_at(results, "18_CT", year), value)

    def test_gapped_uploaded_projection_is_reported_as_an_economy_error(self):
        wide = self.economy_input("01_AUS")
        wide.loc[wide["year"] == 2026, self.gdp_column] = 1_800_000.0
        wide.loc[wide["year"] == 2028, self.gdp_column] = 1_900_000.0

        results, _, errors = core.run_analysis(core.wide_to_long(wide), self.params)
        self.assertTrue(results.empty)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "01_AUS")
        self.assertIn("continuous annual values", errors[0][1])

    def test_invalid_values_and_duplicate_projection_years_are_rejected(self):
        wide = self.economy_input("01_AUS")
        wide.loc[wide["year"] == 2026, self.gdp_column] = 0
        results, _, errors = core.run_analysis(core.wide_to_long(wide), self.params)
        self.assertTrue(results.empty)
        self.assertIn("finite and positive", errors[0][1])

        wide = self.economy_input("01_AUS")
        wide.loc[wide["year"] == 2026, self.gdp_column] = 1_800_000.0
        duplicated = wide[wide["year"] == 2026].copy()
        wide = pd.concat([wide, duplicated], ignore_index=True)
        results, _, errors = core.run_analysis(core.wide_to_long(wide), self.params)
        self.assertTrue(results.empty)
        self.assertIn("duplicate years", errors[0][1])


if __name__ == "__main__":
    unittest.main()
