import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "macro_model"))

import config_loaders
import core
from model import _apply_capital_growth_guardrail


class CapitalGrowthGuardrailTests(unittest.TestCase):
    def apply_guardrail(self, raw_growth, prior_growth, cap_compare=0.1):
        return _apply_capital_growth_guardrail(
            raw_growth, prior_growth, cap_compare, "01_AUS", 2026
        )

    def test_positive_prior_growth_lower_upper_in_band_and_boundaries(self):
        cases = [
            ("lower breach", 0.016, 0.02, 0.018),
            ("upper breach", 0.024, 0.02, 0.022),
            ("inside band", 0.021, 0.02, 0.021),
            ("lower boundary", 0.018, 0.02, 0.018),
            ("upper boundary", 0.022, 0.02, 0.022),
        ]
        for label, raw, prior, expected in cases:
            with self.subTest(label=label):
                self.assertAlmostEqual(self.apply_guardrail(raw, prior), expected)

    def test_negative_prior_growth_clips_by_ratio_and_prevents_sign_reversal(self):
        cases = [
            ("lower ratio breach", -0.016, -0.02, -0.018),
            ("upper ratio breach", -0.024, -0.02, -0.022),
            ("inside band", -0.021, -0.02, -0.021),
            ("attempted sign reversal", 0.01, -0.02, -0.018),
            ("lower ratio boundary", -0.018, -0.02, -0.018),
            ("upper ratio boundary", -0.022, -0.02, -0.022),
        ]
        for label, raw, prior, expected in cases:
            with self.subTest(label=label):
                self.assertAlmostEqual(self.apply_guardrail(raw, prior), expected)

    def test_exact_zero_prior_passes_positive_and_negative_raw_growth(self):
        self.assertAlmostEqual(self.apply_guardrail(0.03, 0), 0.03)
        self.assertAlmostEqual(self.apply_guardrail(-0.02, 0), -0.02)

    def test_small_nonzero_prior_is_not_treated_as_zero(self):
        self.assertAlmostEqual(
            self.apply_guardrail(1.2e-12, 1e-12),
            1.1e-12,
        )

    def test_economy_specific_cap_compare_values_produce_distinct_bounds(self):
        raw, prior = 0.03, 0.02
        self.assertAlmostEqual(
            _apply_capital_growth_guardrail(raw, prior, 0.05, "01_AUS", 2026),
            0.021,
        )
        self.assertAlmostEqual(
            _apply_capital_growth_guardrail(raw, prior, 0.2, "02_BD", 2026),
            0.024,
        )

    def test_non_finite_guardrail_inputs_and_ratio_are_rejected(self):
        invalid_values = [None, np.nan, np.inf, -np.inf, "not numeric"]
        for name in ("raw", "prior", "cap_compare"):
            for value in invalid_values:
                args = {"raw": 0.02, "prior": 0.01, "cap_compare": 0.1}
                args[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        _apply_capital_growth_guardrail(
                            args["raw"],
                            args["prior"],
                            args["cap_compare"],
                            "01_AUS",
                            2026,
                        )

        with self.assertRaisesRegex(ValueError, "ratio must be finite"):
            self.apply_guardrail(1.0, 1e-320)

    def test_unsupported_cap_compare_values_are_rejected(self):
        for value in [-0.01, 1.0]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "less than one"):
                    self.apply_guardrail(0.02, 0.01, value)


class CapitalGrowthModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample = pd.read_csv(ROOT / "sample_wide_input.csv")
        sample = sample[sample["economy_code"].eq("01_AUS")].copy()
        _, cls.inputs, errors = core.run_analysis(
            core.wide_to_long(sample),
            core.load_gdp_model_parameters(),
        )
        if errors:
            raise AssertionError(f"Bundled Australia inputs failed: {errors}")

    def run_projection(self, prior_growth, first_raw, second_raw, cap_compare=0.1):
        inputs = {
            key: value.copy(deep=True)
            for key, value in self.inputs.items()
        }
        code = "01_AUS"
        capital_row = inputs["input_data"][
            inputs["input_data"]["economy_code"].eq(code)
            & inputs["input_data"]["year"].eq(2025)
        ].iloc[0]
        savings_fraction = (first_raw + 0.04) * (
            capital_row["capital"] / capital_row["real_output"]
        )

        inputs["cap_growth_data"].loc[
            inputs["cap_growth_data"]["economy_code"].eq(code)
            & inputs["cap_growth_data"]["year"].eq(2025),
            "percent",
        ] = prior_growth
        inputs["sav_data"].loc[
            inputs["sav_data"]["economy_code"].eq(code), "value"
        ] = savings_fraction * 100
        inputs["delta_data"].loc[
            inputs["delta_data"]["economy_code"].eq(code), "value"
        ] = 0.04

        capital_2026 = capital_row["capital"] * (1 + first_raw)
        gdp_2026 = capital_2026 * (second_raw + 0.04) / savings_fraction
        raw_capital_2026 = (
            capital_row["capital"]
            - capital_row["capital"] * 0.04
            + capital_row["real_output"] * savings_fraction
        )
        output, _ = core.aperc_gdp_model(
            economy=code,
            jump_off_year=2025,
            gdp_overrides=pd.DataFrame(
                [{"year": 2026, "value": gdp_2026}]
            ),
            cap_compare=cap_compare,
            high_sav=savings_fraction,
            low_sav=savings_fraction,
            change_sav=0,
            high_delta=0.04,
            low_delta=0.04,
            change_del=0,
            **inputs,
        )
        capital = (
            output[output["variable"].eq("Capital stock")]
            .set_index("year")["value"]
        )
        return capital, raw_capital_2026

    def test_model_applies_upper_bound_and_resumes_after_zero_prior(self):
        zero_prior, _ = self.run_projection(
            prior_growth=0,
            first_raw=0.02,
            second_raw=0.05,
        )
        self.assertAlmostEqual(zero_prior.loc[2026] / zero_prior.loc[2025] - 1, 0.02)
        self.assertAlmostEqual(zero_prior.loc[2027] / zero_prior.loc[2026] - 1, 0.022)

        positive_prior, _ = self.run_projection(
            prior_growth=0.02,
            first_raw=0.03,
            second_raw=0.05,
        )
        self.assertAlmostEqual(
            positive_prior.loc[2026] / positive_prior.loc[2025] - 1,
            0.022,
        )

    def test_model_preserves_raw_capital_calculation_within_band(self):
        capital, raw_capital = self.run_projection(
            prior_growth=0.02,
            first_raw=0.021,
            second_raw=0.02,
        )
        self.assertEqual(capital.loc[2026], raw_capital)


class CapCompareConfigurationTests(unittest.TestCase):
    def test_current_economy_specific_values_are_loaded_unchanged(self):
        configured = config_loaders.load_gdp_model_parameters()
        csv = pd.read_csv(ROOT / "config" / "gdp_model_parameters.csv")
        for row in csv.itertuples(index=False):
            with self.subTest(economy=row.economy_code):
                self.assertEqual(
                    configured[row.economy_code]["cap_compare"],
                    row.cap_compare,
                )

    def test_missing_and_invalid_config_values_are_rejected(self):
        source = ROOT / "config"
        invalid_values = [None, np.nan, np.inf, -0.01, 1.0, "invalid"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir)
            shutil.copyfile(source / "economies.csv", temp_config / "economies.csv")
            params_path = temp_config / "gdp_model_parameters.csv"
            original = pd.read_csv(source / "gdp_model_parameters.csv")

            with mock.patch.object(config_loaders, "CONFIG_DIR", temp_config):
                missing_column = original.drop(columns=["cap_compare"])
                missing_column.to_csv(params_path, index=False)
                with self.assertRaisesRegex(ValueError, "missing required column"):
                    config_loaders.load_gdp_model_parameters()

                for value in invalid_values:
                    with self.subTest(value=value):
                        invalid = original.copy()
                        if isinstance(value, str):
                            invalid["cap_compare"] = invalid[
                                "cap_compare"
                            ].astype(object)
                        invalid.loc[0, "cap_compare"] = value
                        invalid.to_csv(params_path, index=False)
                        with self.assertRaisesRegex(ValueError, "cap_compare"):
                            config_loaders.load_gdp_model_parameters()


if __name__ == "__main__":
    unittest.main()
