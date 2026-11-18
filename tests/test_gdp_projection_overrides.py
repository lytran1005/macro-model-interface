import io
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
        with self.assertRaisesRegex(ValueError, "01_AUS 2026.*positive"):
            core.run_analysis(core.wide_to_long(wide), self.params)

        wide = self.economy_input("01_AUS")
        wide.loc[wide["year"] == 2026, self.gdp_column] = 1_800_000.0
        duplicated = wide[wide["year"] == 2026].copy()
        wide = pd.concat([wide, duplicated], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            core.run_analysis(core.wide_to_long(wide), self.params)

    def test_uploaded_blank_gdp_and_custom_population_scenarios(self):
        wide = self.sample[self.sample["economy_code"].isin(["01_AUS", "04_CHL"])].copy()
        wide["population_1jan_custom"] = ""
        australia = wide["economy_code"].eq("01_AUS")
        wide.loc[australia, "population_1jan_custom"] = (
            wide.loc[australia, "population_1jan"].astype(str)
        )
        future_australia = australia & wide["year"].ge(2026)
        wide.loc[future_australia, "population_1jan_custom"] = (
            wide.loc[future_australia, "population_1jan"].mul(1.1).astype(str)
        )
        wide = pd.read_csv(io.StringIO(wide.to_csv(index=False)), keep_default_na=False)
        australia = wide["economy_code"].eq("01_AUS")
        core.validate_wide_model_input(wide)
        self.assertEqual(core.custom_population_status(wide), (["01_AUS"], []))
        paths = core.population_scenarios_from_wide(wide)
        custom = paths[paths["scenario"].eq("Custom")]
        self.assertEqual(set(custom["economy_code"]), {"01_AUS"})
        expected = wide[wide["economy_code"].eq("01_AUS")].set_index("year")
        pd.testing.assert_series_equal(
            custom.set_index("year")["value"].sort_index(),
            pd.to_numeric(expected["population_1jan_custom"]).sort_index(),
            check_names=False,
        )

        for overrides in ({}, {2026: 1_800_000.0, 2027: 1_850_000.0}):
            with self.subTest(overrides=overrides):
                data = wide.copy()
                for year, value in overrides.items():
                    data.loc[australia & data["year"].eq(year), self.gdp_column] = str(value)
                long = core.wide_to_long(data)
                extracted = core.uploaded_gdp_projection_overrides(long, "01_AUS")
                self.assertEqual(extracted["year"].tolist(), list(overrides))
                self.assertEqual(extracted["value"].tolist(), list(overrides.values()))
                standard, _, errors = core.run_analysis(long, self.params)
                self.assertEqual(errors, [])
                default, errors = core.run_uploaded_population_scenario(
                    data, "Default", self.params, economy="01_AUS"
                )
                self.assertEqual(errors, [])
                custom_result, errors = core.run_uploaded_population_scenario(
                    data, "Custom", self.params, economy="01_AUS"
                )
                self.assertEqual(errors, [])
                pd.testing.assert_frame_equal(
                    default.reset_index(drop=True),
                    standard[standard["economy_code"].eq("01_AUS")].reset_index(drop=True),
                )
                actual_population = custom_result[
                    custom_result["variable"].eq("population")
                ].set_index("year")["value"]
                pd.testing.assert_series_equal(
                    actual_population.sort_index(),
                    pd.to_numeric(expected["population_1jan_custom"]).sort_index(),
                    check_names=False,
                )
                explicit = data[australia].copy()
                explicit["population_1jan"] = pd.to_numeric(explicit["population_1jan_custom"])
                explicit_result, _, errors = core.run_analysis(
                    core.wide_to_long(explicit), self.params
                )
                self.assertEqual(errors, [])
                pd.testing.assert_frame_equal(custom_result, explicit_result)
                for year, value in overrides.items():
                    self.assertAlmostEqual(self.real_gdp_at(custom_result, "01_AUS", year), value)
                self.assertNotAlmostEqual(
                    self.real_gdp_at(custom_result, "01_AUS", 2060),
                    self.real_gdp_at(default, "01_AUS", 2060),
                )
                sensitivity, errors = core.run_uploaded_population_gdp_scenarios(data, self.params)
                self.assertEqual(errors, [])
                for scenario, result in [("Default", default), ("Custom", custom_result)]:
                    value = sensitivity.loc[
                        sensitivity["economy_code"].eq("01_AUS")
                        & sensitivity["scenario"].eq(scenario)
                        & sensitivity["year"].eq(2060), "value"
                    ].iloc[0]
                    self.assertAlmostEqual(value, self.real_gdp_at(result, "01_AUS", 2060))

    def test_blank_custom_and_invalid_uploaded_gdp_values(self):
        wide = self.economy_input("01_AUS")
        wide["population_1jan_custom"] = ""
        wide = pd.read_csv(io.StringIO(wide.to_csv(index=False)), keep_default_na=False)
        self.assertEqual(core.custom_population_status(wide), ([], []))
        core.validate_wide_model_input(wide)
        wide["population_1jan_custom"] = wide["population_1jan"].astype(str)
        wide.loc[wide["year"].eq(2040), "population_1jan_custom"] = " "
        core.validate_wide_model_input(wide)
        self.assertEqual(core.custom_population_status(wide), ([], ["01_AUS"]))
        with self.assertRaisesRegex(ValueError, "No economy has a complete custom"):
            core.run_uploaded_population_scenario(wide, "Custom", self.params)
        for value in ["bad", "0", "-1", "inf"]:
            with self.subTest(value=value):
                data = wide.copy()
                data.loc[data["year"].eq(2026), self.gdp_column] = value
                with self.assertRaises(ValueError):
                    core.validate_wide_model_input(data)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    core.uploaded_gdp_projection_overrides(core.wide_to_long(data), "01_AUS")
        data = wide.copy()
        data.loc[data["year"].eq(2026), self.gdp_column] = "1800000"
        data.loc[data["year"].eq(2027), self.gdp_column] = " "
        data.loc[data["year"].eq(2028), self.gdp_column] = "1900000"
        with self.assertRaisesRegex(ValueError, "continuous annual"):
            core.uploaded_gdp_projection_overrides(core.wide_to_long(data), "01_AUS")


if __name__ == "__main__":
    unittest.main()
