import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import core


class LeapExportTests(unittest.TestCase):
    def make_results(self):
        rows = []
        economies = core.load_economies().sort_values("economy_code")
        for economy in economies.itertuples(index=False):
            for variable in core.LEAP_EXPORT_VARIABLES:
                base = 1000.126 if variable == "population" else 2_000_000.126
                for year in range(
                    core.LEAP_EXPORT_START_YEAR,
                    core.LEAP_EXPORT_END_YEAR + 1,
                ):
                    rows.append(
                        {
                            "economy_code": economy.economy_code,
                            "economy": economy.economy,
                            "variable": variable,
                            "year": year,
                            "value": base + year - core.LEAP_EXPORT_START_YEAR,
                            "units": core.UNITS[variable],
                        }
                    )
        return pd.DataFrame(rows)

    def test_formats_all_economies_in_reference_schema_without_mutating_results(self):
        results = self.make_results()
        original = results.copy(deep=True)

        export = core.build_leap_input_export(results)

        self.assertEqual(
            export.columns.tolist(),
            [
                "economy_code",
                "economy",
                "variable",
                "units",
                "Data_2022_2060_round2",
            ],
        )
        self.assertEqual(export.shape, (42, 5))
        self.assertEqual(export.iloc[0]["economy"], "Australia")
        self.assertEqual(
            export[["economy_code", "variable"]].head(4).values.tolist(),
            [
                ["01_AUS", "population"],
                ["01_AUS", "real_GDP"],
                ["02_BD", "population"],
                ["02_BD", "real_GDP"],
            ],
        )
        self.assertEqual(
            export[["variable", "units"]].drop_duplicates().values.tolist(),
            [
                ["population", "Thousands"],
                ["real_GDP", "Millions (2021 USD PPP)"],
            ],
        )

        population_series = export.iloc[0]["Data_2022_2060_round2"]
        self.assertTrue(population_series.startswith("Data(2022, 1000.13,"))
        pairs = re.findall(r"(\d{4}), (-?\d+\.\d+)", population_series)
        years = [int(year) for year, _ in pairs]
        self.assertEqual(years, list(range(2022, 2061)))
        self.assertTrue(
            all(float(value) == round(float(value), 2) for _, value in pairs)
        )

        gdp_series = export.iloc[1]["Data_2022_2060_round2"]
        gdp_pairs = re.findall(r"(\d{4}), (-?\d+\.\d+)", gdp_series)
        self.assertEqual(len(gdp_pairs), 39)
        self.assertEqual(float(gdp_pairs[0][1]), 2_000_000.13)
        pd.testing.assert_frame_equal(results, original)

    def test_rejects_missing_economy_indicator_or_year(self):
        results = self.make_results()

        with self.subTest("missing economy"):
            incomplete = results[results["economy_code"] != "21_VN"]
            with self.assertRaisesRegex(ValueError, "missing economies: 21_VN"):
                core.build_leap_input_export(incomplete)

        with self.subTest("missing indicator"):
            incomplete = results[
                ~(
                    results["economy_code"].eq("03_CDA")
                    & results["variable"].eq("real_GDP")
                )
            ]
            with self.assertRaisesRegex(
                ValueError, "03_CDA real_GDP missing years 2022–2060"
            ):
                core.build_leap_input_export(incomplete)

        with self.subTest("missing year"):
            incomplete = results[
                ~(
                    results["economy_code"].eq("01_AUS")
                    & results["variable"].eq("population")
                    & results["year"].eq(2025)
                )
            ]
            with self.assertRaisesRegex(
                ValueError, "01_AUS population missing years 2025"
            ):
                core.build_leap_input_export(incomplete)

    def test_rejects_non_finite_values(self):
        results = self.make_results()
        mask = (
            results["economy_code"].eq("01_AUS")
            & results["variable"].eq("population")
            & results["year"].eq(2025)
        )
        results.loc[mask, "value"] = float("inf")

        with self.assertRaisesRegex(
            ValueError, "01_AUS population has non-finite values in years 2025"
        ):
            core.build_leap_input_export(results)

    def test_rejects_zero_and_negative_population_or_gdp(self):
        cases = [
            ("population", 0),
            ("population", -1),
            ("real_GDP", 0),
            ("real_GDP", -1),
        ]
        for variable, value in cases:
            with self.subTest(variable=variable, value=value):
                results = self.make_results()
                mask = (
                    results["economy_code"].eq("01_AUS")
                    & results["variable"].eq(variable)
                    & results["year"].eq(2025)
                )
                results.loc[mask, "value"] = value
                with self.assertRaisesRegex(
                    ValueError,
                    rf"01_AUS {variable} has non-positive values in years 2025",
                ):
                    core.build_leap_input_export(results)


if __name__ == "__main__":
    unittest.main()
