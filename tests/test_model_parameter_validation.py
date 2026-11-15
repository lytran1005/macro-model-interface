import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import core
from generate_samples import sample_params_csv


class ModelParameterValidationTests(unittest.TestCase):
    def setUp(self):
        self.template = sample_params_csv()

    def test_global_upload_requires_all_21_unique_known_economies(self):
        valid = core.parse_params_csv(self.template)
        self.assertEqual(len(valid), 21)
        self.assertEqual(set(valid), set(core.load_gdp_model_parameters()))

        with self.subTest("missing economy"):
            with self.assertRaisesRegex(ValueError, "Missing economy codes"):
                core.parse_params_csv(self.template.iloc[:-1].copy())

        with self.subTest("duplicate economy"):
            duplicate = pd.concat(
                [self.template, self.template.iloc[[0]]], ignore_index=True
            )
            with self.assertRaisesRegex(ValueError, "duplicate economy rows"):
                core.parse_params_csv(duplicate)

        with self.subTest("unknown economy"):
            unknown = self.template.copy()
            unknown.loc[0, "economy_code"] = "99_UNKNOWN"
            with self.assertRaisesRegex(ValueError, "unknown economy_code"):
                core.parse_params_csv(unknown)

    def test_global_upload_preserves_each_economys_configured_values(self):
        params = core.parse_params_csv(self.template)
        configured = core.load_gdp_model_parameters()
        for code, values in configured.items():
            for parameter in core.PARAMETER_COLUMNS:
                self.assertEqual(params[code][parameter], values[parameter])
            self.assertEqual(params[code]["cap_compare"], values["cap_compare"])

    def test_global_upload_rejects_missing_and_non_finite_values(self):
        for value in [None, float("inf"), float("-inf")]:
            with self.subTest(value=value):
                invalid = self.template.copy()
                invalid.loc[0, "alpha"] = value
                with self.assertRaisesRegex(
                    ValueError, "01_AUS parameter 'alpha'"
                ):
                    core.parse_params_csv(invalid)

        missing_column = self.template.drop(columns=["high_eff"])
        with self.assertRaisesRegex(ValueError, "missing required columns.*high_eff"):
            core.parse_params_csv(missing_column)

    def test_global_upload_rejects_invalid_ranges_and_relationships(self):
        invalid_cases = [
            ("alpha", 1.5, "alpha"),
            ("lab_eff_periods", 0, "lab_eff_periods"),
            ("low_sav", 0.4, "savings corridor"),
            ("change_eff", -0.01, "change_eff"),
        ]
        for column, value, message in invalid_cases:
            with self.subTest(column=column):
                invalid = self.template.copy()
                invalid.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    core.parse_params_csv(invalid)

    def test_deep_dive_upload_still_applies_only_the_selected_economy(self):
        selected = core.parse_uploaded_economy_params(
            self.template, "01_AUS"
        )
        self.assertEqual(
            selected["alpha"],
            core.load_gdp_model_parameters()["01_AUS"]["alpha"],
        )
        self.assertEqual(selected["cap_compare"], 0.05)

        one_economy = self.template[
            self.template["economy_code"].eq("01_AUS")
        ].copy()
        selected = core.parse_uploaded_economy_params(
            one_economy, "01_AUS"
        )
        self.assertEqual(
            selected["alpha"],
            core.load_gdp_model_parameters()["01_AUS"]["alpha"],
        )
        with self.assertRaisesRegex(ValueError, "does not contain.*18_CT"):
            core.parse_uploaded_economy_params(one_economy, "18_CT")


if __name__ == "__main__":
    unittest.main()
