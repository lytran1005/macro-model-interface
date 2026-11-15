import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import core
import generate_samples


class SavingsSampleSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "data"
        self.app_dir = self.root / "app"
        self.data_dir.mkdir()
        self.app_dir.mkdir()

        self.economies = pd.DataFrame(
            [
                {"economy_code": "01_AUS", "economy": "Australia"},
                {"economy_code": "13_PNG", "economy": "Papua New Guinea"},
                {"economy_code": "18_CT", "economy": "Chinese Taipei"},
            ]
        )
        self.population = pd.DataFrame(
            [
                {
                    "economy_code": economy.economy_code,
                    "economy": economy.economy,
                    "year": year,
                    "value": 1000 + year,
                }
                for economy in self.economies.itertuples(index=False)
                for year in range(1990, 2061)
            ]
        )
        scenario_columns = list(core.POPULATION_SCENARIO_COLUMNS.values())
        self.population_paths = pd.DataFrame(
            [
                {
                    "economy_code": economy.economy_code,
                    "year": year,
                    **{column: 1000 + year for column in scenario_columns},
                }
                for economy in self.economies.itertuples(index=False)
                for year in range(1990, 2061)
            ]
        )

        imf_rows = [
            {
                "economy_code": code,
                "economy": name,
                "year": year,
                "variable": "Gross national savings",
                "value": value,
                "source": "IMF",
            }
            for code, name, year, value in [
                ("01_AUS", "Australia", 1990, 90.0),
                ("01_AUS", "Australia", 1991, 91.0),
                ("01_AUS", "Australia", 2025, 95.0),
                ("13_PNG", "Papua New Guinea", 1990, 98.0),
                ("13_PNG", "Papua New Guinea", 2025, 97.0),
                ("18_CT", "Chinese Taipei", 1990, 31.907),
                ("18_CT", "Chinese Taipei", 1991, 32.419),
                ("18_CT", "Chinese Taipei", 1992, None),
                ("18_CT", "Chinese Taipei", 2030, 40.379),
                ("18_CT", "Chinese Taipei", 2031, 99.0),
                ("18_CT", "Chinese Taipei", 2032, 100.0),
            ]
        ]
        imf_rows.append(
            {
                "economy_code": "01_AUS",
                "economy": "Australia",
                "year": 1992,
                "variable": "Total investment",
                "value": 77.0,
                "source": "IMF",
            }
        )
        imf_rows.extend(
            [
                {
                    "economy_code": "18_CT",
                    "economy": "Chinese Taipei",
                    "year": 2031,
                    "variable": "Real GDP PPP 2021 USD",
                    "value": 800.0,
                    "source": "IMF",
                },
                {
                    "economy_code": "01_AUS",
                    "economy": "Australia",
                    "year": 2025,
                    "variable": "Real GDP PPP 2021 USD",
                    "value": 500.0,
                    "source": "IMF",
                },
            ]
        )
        pd.DataFrame(imf_rows).to_csv(
            self.data_dir / "IMF_to2031.csv", index=False
        )

        pd.DataFrame(
            [
                {
                    "economy_code": code,
                    "economy": name,
                    "year": 2025,
                    "variable": "Real GDP PPP 2021 USD",
                    "value": 600.0,
                }
                for code, name in [
                    ("01_AUS", "Australia"),
                    ("13_PNG", "Papua New Guinea"),
                    ("18_CT", "Chinese Taipei"),
                ]
            ]
        ).to_csv(self.data_dir / "WDI_to2025.csv", index=False)

        pd.DataFrame(
            [
                {
                    "economy_code": code,
                    "economy": name,
                    "year": year,
                    "value": value,
                    "source": "WDI",
                }
                for code, name, year, value in [
                    ("01_AUS", "Australia", 1990, 12.0),
                    ("01_AUS", "Australia", 1991, None),
                    ("01_AUS", "Australia", 2025, 22.0),
                    ("01_AUS", "Australia", 2026, 99.0),
                    ("13_PNG", "Papua New Guinea", 1990, 18.0),
                    ("13_PNG", "Papua New Guinea", 2025, 26.221),
                    ("18_CT", "Chinese Taipei", 1990, 77.0),
                ]
            ]
        ).to_csv(self.data_dir / "WDI_savings_2025.csv", index=False)

        pd.DataFrame(
            [
                {
                    "economy_code": code,
                    "economy": name,
                    "year": year,
                    "value": value,
                    "source": source,
                }
                for code, name, year, value, source in [
                    ("01_AUS", "Australia", 2031, 88.0, "IMF"),
                    ("13_PNG", "Papua New Guinea", 2031, 25.0, "APERC assumption"),
                    ("18_CT", "Chinese Taipei", 2031, 39.814, "IMF"),
                    ("18_CT", "Chinese Taipei", 2032, 45.0, "IMF"),
                ]
            ]
        ).to_csv(self.data_dir / "IMF_savings_2031.csv", index=False)

        pwt_rows = [
            {
                "economy_code": economy.economy_code,
                "economy": economy.economy,
                "year": 2023,
                "variable": variable,
                "value": 0.05,
            }
            for economy in self.economies.itertuples(index=False)
            for variable in ["output_to_kstock", "delta"]
        ]
        pd.DataFrame(pwt_rows).to_csv(
            self.data_dir / "PWT_cap_labour_to2023.csv", index=False
        )
        for filename in core.POPULATION_SCENARIO_FILES.values():
            (self.data_dir / filename).write_text("", encoding="utf-8")

    def build_sample(self):
        with patch.object(generate_samples, "APP_DIR", self.app_dir), \
             patch.object(generate_samples, "RESULTS_DATA_DIR", self.data_dir), \
             patch.object(generate_samples, "load_economies", return_value=self.economies), \
             patch.object(generate_samples, "load_default_population", return_value=self.population), \
             patch.object(
                 generate_samples,
                 "load_population_sensitivity_paths",
                 return_value=self.population_paths,
             ):
            return generate_samples._pipeline_sample_data()

    @staticmethod
    def value_at(sample, code, year):
        value = sample.loc[
            sample["economy_code"].eq(code) & sample["year"].eq(year),
            "Gross national savings",
        ]
        return value.iloc[0]

    def test_pipeline_uses_only_approved_savings_sources_and_cutoffs(self):
        sample = self.build_sample()

        self.assertEqual(self.value_at(sample, "01_AUS", 1990), 12.0)
        self.assertTrue(pd.isna(self.value_at(sample, "01_AUS", 1991)))
        self.assertTrue(pd.isna(self.value_at(sample, "01_AUS", 1992)))
        self.assertEqual(self.value_at(sample, "01_AUS", 2025), 22.0)
        self.assertTrue(pd.isna(self.value_at(sample, "01_AUS", 2026)))
        self.assertEqual(self.value_at(sample, "13_PNG", 1990), 18.0)
        self.assertEqual(self.value_at(sample, "13_PNG", 2025), 26.221)

        self.assertEqual(self.value_at(sample, "18_CT", 1990), 31.907)
        self.assertEqual(self.value_at(sample, "18_CT", 1991), 32.419)
        self.assertTrue(pd.isna(self.value_at(sample, "18_CT", 1992)))
        self.assertEqual(self.value_at(sample, "18_CT", 2030), 40.379)
        self.assertEqual(self.value_at(sample, "18_CT", 2031), 39.814)
        self.assertTrue(pd.isna(self.value_at(sample, "18_CT", 2032)))

        non_ct = sample[sample["economy_code"].isin(["01_AUS", "13_PNG"])]
        for code, year, expected in [
            ("01_AUS", 1990, 12.0),
            ("01_AUS", 1991, None),
            ("01_AUS", 2025, 22.0),
            ("13_PNG", 1990, 18.0),
            ("13_PNG", 2025, 26.221),
        ]:
            actual = self.value_at(non_ct, code, year)
            if expected is None:
                self.assertTrue(pd.isna(actual))
            else:
                self.assertEqual(actual, expected)

    def test_existing_chinese_taipei_values_are_verified_before_refresh(self):
        existing_path = self.app_dir / generate_samples.SAMPLE_WIDE_INPUT_FILE
        pd.DataFrame(
            [
                {
                    "economy_code": "18_CT",
                    "year": 1990,
                    "Gross national savings": 31.907,
                }
            ]
        ).to_csv(existing_path, index=False)
        self.assertEqual(self.value_at(self.build_sample(), "18_CT", 1990), 31.907)

        for year in [1990, 2031]:
            with self.subTest(year=year):
                pd.DataFrame(
                    [
                        {
                            "economy_code": "18_CT",
                            "year": year,
                            "Gross national savings": 99.0,
                        }
                    ]
                ).to_csv(existing_path, index=False)
                with self.assertRaisesRegex(
                    ValueError, "do not match the approved IMF source"
                ):
                    self.build_sample()

        pd.DataFrame(
            [
                {
                    "economy_code": "18_CT",
                    "year": 2032,
                    "Gross national savings": 30.0,
                }
            ]
        ).to_csv(existing_path, index=False)
        with self.assertRaisesRegex(ValueError, "after the approved IMF cutoff"):
            self.build_sample()

    def test_png_fallback_is_conditional_and_unchanged(self):
        empty_history = pd.DataFrame(
            columns=["economy_code", "economy", "year", "variable", "value"]
        )
        fallback = core.build_savings_snapshot(empty_history, ["13_PNG"])
        self.assertEqual(fallback.loc[0, "value"], 25)

        observed_history = pd.DataFrame(
            [
                {
                    "economy_code": "13_PNG",
                    "economy": "Papua New Guinea",
                    "year": 2025,
                    "variable": "Gross national savings",
                    "value": 26.221,
                }
            ]
        )
        observed = core.build_savings_snapshot(observed_history, ["13_PNG"])
        self.assertEqual(observed.loc[0, "value"], 26.221)


class SavingsModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pd.read_csv(ROOT / "sample_wide_input.csv")
        cls.params = core.load_gdp_model_parameters()

    def test_uploaded_savings_values_are_preserved(self):
        wide = self.sample[self.sample["economy_code"].eq("01_AUS")].copy()
        wide.loc[wide["year"].eq(1990), "Gross national savings"] = 35.5
        original = wide.copy(deep=True)

        results, _, errors = core.run_analysis(core.wide_to_long(wide), self.params)

        self.assertEqual(errors, [])
        assert_frame_equal(wide, original)
        value = results.loc[
            results["variable"].eq("savings") & results["year"].eq(1990),
            "value",
        ].iloc[0]
        self.assertEqual(value, 0.355)

    def test_ct_historical_backfill_preserves_gdp_anchor_and_projections(self):
        original = self.sample[self.sample["economy_code"].eq("18_CT")].copy()
        source = pd.read_csv(ROOT / "results" / "data" / "IMF_to2031.csv")
        source = source[
            source["economy_code"].eq("18_CT")
            & source["variable"].eq("Gross national savings")
            & source["year"].between(1990, 2030)
            & source["value"].notna()
        ]
        self.assertEqual(len(source), 41)

        corrected = original.copy(deep=True)
        source_by_year = source.set_index("year")["value"]
        for index, row in corrected[
            corrected["year"].between(1990, 2030)
        ].iterrows():
            source_value = source_by_year.loc[row["year"]]
            old_value = row["Gross national savings"]
            if pd.notna(old_value):
                self.assertEqual(old_value, source_value)
            else:
                corrected.at[index, "Gross national savings"] = source_value

        self.assertEqual(
            original.loc[original["year"].eq(2031), "Gross national savings"].iloc[0],
            39.814,
        )
        baseline, baseline_inputs, baseline_errors = core.run_analysis(
            core.wide_to_long(original), self.params
        )
        updated, updated_inputs, updated_errors = core.run_analysis(
            core.wide_to_long(corrected), self.params
        )
        self.assertEqual(baseline_errors, [])
        self.assertEqual(updated_errors, [])

        baseline_gdp = baseline[baseline["variable"].eq("real_GDP")].reset_index(drop=True)
        updated_gdp = updated[updated["variable"].eq("real_GDP")].reset_index(drop=True)
        assert_frame_equal(baseline_gdp, updated_gdp)
        baseline_projected_savings = baseline[
            baseline["variable"].eq("savings") & baseline["year"].ge(2031)
        ].reset_index(drop=True)
        updated_projected_savings = updated[
            updated["variable"].eq("savings") & updated["year"].ge(2031)
        ].reset_index(drop=True)
        assert_frame_equal(baseline_projected_savings, updated_projected_savings)
        assert_frame_equal(baseline_inputs["sav_data"], updated_inputs["sav_data"])


if __name__ == "__main__":
    unittest.main()
