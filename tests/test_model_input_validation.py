import sys
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import core


class ModelInputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pd.read_csv(ROOT / "sample_wide_input.csv")
        cls.params = core.load_gdp_model_parameters()
        cls.gdp_column = "Real GDP PPP 2021 USD"

    def test_shuffled_economy_rows_produce_identical_results(self):
        wide = self.sample[
            self.sample["economy_code"].isin(["01_AUS", "02_BD", "18_CT"])
        ].copy()
        original = wide.copy(deep=True)

        baseline, _, baseline_errors = core.run_analysis(
            core.wide_to_long(wide), self.params
        )
        self.assertEqual(baseline_errors, [])

        baseline = baseline.sort_values(
            ["economy_code", "variable", "year"]
        ).reset_index(drop=True)
        for seed in [11, 42, 2026]:
            shuffled = wide.sample(frac=1, random_state=seed).reset_index(drop=True)
            results, _, errors = core.run_analysis(
                core.wide_to_long(shuffled), self.params
            )
            self.assertEqual(errors, [])
            results = results.sort_values(
                ["economy_code", "variable", "year"]
            ).reset_index(drop=True)
            assert_frame_equal(
                baseline,
                results,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-10,
            )

        pd.testing.assert_frame_equal(wide, original)
        self.assertEqual(
            wide.groupby("economy_code")["year"].nunique().to_dict(),
            {"01_AUS": 71, "02_BD": 71, "18_CT": 71},
        )
        self.assertFalse(wide.duplicated(["economy_code", "year"]).any())

    def test_valid_economy_subset_keeps_complete_year_coverage(self):
        subset = self.sample[
            self.sample["economy_code"].isin(["01_AUS", "18_CT"])
        ].copy()

        core.validate_wide_model_input(subset)

        self.assertEqual(set(subset["economy_code"]), {"01_AUS", "18_CT"})
        self.assertTrue(
            subset.groupby("economy_code")["year"].apply(
                lambda years: set(years) == set(range(1990, 2061))
            ).all()
        )

    def test_rejects_unknown_or_mismatched_economy_identifiers(self):
        unknown = self.sample.copy()
        unknown.loc[unknown.index[0], "economy_code"] = "99_UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown economy codes"):
            core.validate_wide_model_input(unknown)

        mismatch = self.sample.copy()
        mismatch.loc[mismatch["economy_code"].eq("01_AUS"), "economy"] = (
            "Brunei Darussalam"
        )
        with self.assertRaisesRegex(ValueError, "names must match"):
            core.validate_wide_model_input(mismatch)

    def test_rejects_duplicate_rows_and_incomplete_year_coverage(self):
        duplicate = pd.concat(
            [self.sample, self.sample[self.sample["year"].eq(2025)].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate economy/year rows"):
            core.validate_wide_model_input(duplicate)

        incomplete = self.sample[
            ~(
                self.sample["economy_code"].eq("01_AUS")
                & self.sample["year"].eq(2042)
            )
        ]
        with self.assertRaisesRegex(
            ValueError, "01_AUS is missing years 2042"
        ):
            core.validate_wide_model_input(incomplete)

    def test_rejects_invalid_population_in_history_and_projection(self):
        for year, value in [
            (2000, 0),
            (2001, -1),
            (2025, float("inf")),
            (2060, None),
        ]:
            with self.subTest(year=year, value=value):
                wide = self.sample.copy()
                mask = wide["economy_code"].eq("01_AUS") & wide["year"].eq(year)
                wide.loc[mask, "population_1jan"] = value
                with self.assertRaisesRegex(
                    ValueError, rf"01_AUS {year} population_1jan"
                ):
                    core.validate_wide_model_input(wide)

    def test_rejects_invalid_gdp_in_history_and_projection(self):
        for year, value in [
            (2000, 0),
            (2001, -1),
            (2025, float("inf")),
            (2026, 0),
            (2027, -1),
            (2028, float("inf")),
        ]:
            with self.subTest(year=year, value=value):
                wide = self.sample.copy()
                mask = wide["economy_code"].eq("01_AUS") & wide["year"].eq(year)
                wide.loc[mask, self.gdp_column] = value
                with self.assertRaisesRegex(
                    ValueError, rf"01_AUS {year} .*Real GDP PPP 2021 USD"
                ):
                    core.validate_wide_model_input(wide)

    def test_rejects_missing_historical_population_and_gdp(self):
        wide = self.sample.copy()
        mask = wide["economy_code"].eq("01_AUS") & wide["year"].eq(2000)
        wide.loc[mask, "population_1jan"] = None
        with self.assertRaisesRegex(
            ValueError, r"01_AUS 2000 population_1jan"
        ):
            core.validate_wide_model_input(wide)

        wide = self.sample.copy()
        mask = wide["economy_code"].eq("01_AUS") & wide["year"].eq(2000)
        wide.loc[mask, self.gdp_column] = None
        with self.assertRaisesRegex(
            ValueError, r"01_AUS 2000 Real GDP PPP 2021 USD"
        ):
            core.run_analysis(core.wide_to_long(wide), self.params)

    def test_rejects_non_numeric_and_non_finite_uploaded_values(self):
        for value in ["not-a-number", "NaN", float("-inf")]:
            with self.subTest(value=value):
                wide = self.sample.copy()
                wide[self.gdp_column] = wide[self.gdp_column].astype(object)
                mask = wide["economy_code"].eq("01_AUS") & wide["year"].eq(2060)
                wide.loc[mask, self.gdp_column] = value
                with self.assertRaisesRegex(ValueError, "01_AUS 2060"):
                    core.validate_wide_model_input(wide)


if __name__ == "__main__":
    unittest.main()
