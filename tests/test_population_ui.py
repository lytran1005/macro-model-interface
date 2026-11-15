import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core


class PopulationUITests(unittest.TestCase):
    def chart_values(self, app):
        chart = app.get("vega_lite_chart")[0]
        arrow_data = chart.proto.datasets[0].data.data
        return pa.ipc.open_stream(arrow_data).read_all().to_pandas()

    def apec_map_data(self, app):
        chart = app.get("vega_lite_chart")[0]
        for dataset in chart.proto.datasets:
            try:
                data = pa.ipc.open_stream(dataset.data.data).read_all().to_pandas()
            except pa.ArrowInvalid:
                continue
            if {"economy_code", "value", "Indicator", "Year", "Value"} <= set(
                data.columns
            ):
                return data
        self.fail("APEC map data was not found in the chart datasets")

    def ranking_markdown_values(self, app, economy_count):
        markdown_values = [element.value for element in app.markdown]
        heading = "##### Ranking Changes (2025–2060)"
        start = markdown_values.index(heading) + 1
        return markdown_values[start:start + 4 * economy_count]

    def app_with_population(
        self, view, custom, economy_code=None, data_source="uploaded",
        population_view=None, low_missing_year=None, partial_upload=False,
    ):
        data = pd.read_csv(ROOT / "sample_wide_input.csv")
        requested_codes = None
        if view == "📈 Economy Comparison" and economy_code is not None:
            requested_codes = (
                list(economy_code)
                if isinstance(economy_code, (list, tuple, set))
                else [economy_code]
            )
        elif view == "👥 Population Scenarios" and economy_code is not None:
            requested_codes = (
                list(economy_code)
                if isinstance(economy_code, (list, tuple, set))
                else [economy_code]
            )
        elif economy_code is not None or partial_upload:
            if economy_code is None:
                economy_code = "01_AUS"
            if isinstance(economy_code, (list, tuple, set)):
                data = data[data["economy_code"].isin(economy_code)].copy()
            else:
                data = data[data["economy_code"] == economy_code].copy()
        if custom:
            data["population_1jan_custom"] = data["population_1jan"]
        if low_missing_year is not None:
            data.loc[
                data["year"] == low_missing_year, "population_1jan_low"
            ] = None
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        app.session_state["df_wide"] = data
        app.session_state["data_label"] = "uploaded test data"
        app.session_state["active_view"] = view
        app.session_state["data_source"] = data_source
        if population_view is not None:
            app.session_state["population_view"] = population_view
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        if requested_codes and view == "📈 Economy Comparison":
            economy_select = next(
                element
                for element in app.multiselect
                if element.label == "Select Economies to display"
            )
            economy_select.set_value(requested_codes).run(timeout=120)
            self.assertFalse(app.exception, [element.value for element in app.exception])
        elif (
            requested_codes
            and view == "👥 Population Scenarios"
            and population_view == "By Economy"
        ):
            economy_select = next(
                element for element in app.selectbox if element.label == "Economy"
            )
            economy_names = (
                data.drop_duplicates("economy_code")
                .set_index("economy_code")["economy"]
                .to_dict()
            )
            requested_code = requested_codes[0]
            economy_select.set_value(
                f"{economy_names[requested_code]} ({requested_code})"
            ).run(timeout=120)
            self.assertFalse(app.exception, [element.value for element in app.exception])
        return app

    def app_with_view(self, view, data=None, data_source="uploaded"):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        if data is not None:
            app.session_state["df_wide"] = data
            app.session_state["data_label"] = "uploaded test data"
        app.session_state["active_view"] = view
        app.session_state["data_source"] = data_source
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        return app

    def file_uploader(self, app, label):
        return next(
            element for element in app.get("file_uploader")
            if element.label == label
        )

    def test_model_csv_upload_errors_are_reported(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception, [element.value for element in app.exception])

        invalid_files = [
            ("empty.csv", b"", "No columns to parse"),
            ("invalid-utf8.csv", b"\xff\xfe\xfa", "codec can't decode"),
        ]
        for filename, content, expected_message in invalid_files:
            with self.subTest(filename=filename):
                self.file_uploader(app, "Upload Model Input Data CSV").set_value(
                    (filename, content, "text/csv")
                ).run(timeout=120)
                self.assertFalse(
                    app.exception, [element.value for element in app.exception]
                )
                self.assertTrue(
                    any(expected_message in item.value for item in app.error)
                )
                self.assertEqual(app.get("vega_lite_chart"), [])

    def test_model_csv_upload_switches_dataset_without_reusing_results(self):
        app = self.app_with_view("🌏 APEC Overview")
        sample_path = ROOT / "sample_wide_input.csv"
        sample = pd.read_csv(sample_path)

        self.file_uploader(app, "Upload Model Input Data CSV").set_value(
            ("sample_wide_input.csv", sample_path.read_bytes(), "text/csv")
        )
        app.session_state["active_view"] = "🌏 APEC Overview"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(app.get("vega_lite_chart"))

        partial = sample[sample["economy_code"].isin(["01_AUS", "18_CT"])]
        self.file_uploader(app, "Upload Model Input Data CSV").set_value(
            (
                "partial_wide_input.csv",
                partial.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        )
        app.session_state["active_view"] = "🌏 APEC Overview"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any("Missing economies" in item.value for item in app.warning)
        )
        self.assertEqual(app.get("vega_lite_chart"), [])
        self.assertEqual(app.dataframe, [])
        self.assertFalse(
            {
                "⬇️ Download Combined Results (Long Format CSV)",
                "⬇️ Download Combined Results (Wide Format CSV)",
                "⬇️ Download LEAP Input CSV",
            }
            & {item.label for item in app.get("download_button")}
        )

        invalid = sample.copy()
        invalid.loc[
            invalid["economy_code"].eq("01_AUS")
            & invalid["year"].eq(2025),
            "population_1jan",
        ] = 0
        self.file_uploader(app, "Upload Model Input Data CSV").set_value(
            (
                "invalid_population.csv",
                invalid.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        )
        app.session_state["active_view"] = "🌏 APEC Overview"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any("01_AUS 2025 population_1jan" in item.value for item in app.error)
        )
        self.assertEqual(app.get("vega_lite_chart"), [])
        self.assertEqual(app.dataframe, [])

        self.file_uploader(app, "Upload Model Input Data CSV").set_value(
            ("malformed.csv", b'"unterminated\n1,2\n', "text/csv")
        )
        app.session_state["active_view"] = "🌏 APEC Overview"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any("Could not read uploaded model input" in item.value for item in app.error)
        )
        self.assertEqual(app.get("vega_lite_chart"), [])
        self.assertFalse(
            any(
                item.label in {
                    "⬇️ Download Combined Results (Long Format CSV)",
                    "⬇️ Download Combined Results (Wide Format CSV)",
                    "⬇️ Download LEAP Input CSV",
                }
                for item in app.get("download_button")
            )
        )

    def test_global_parameter_csv_upload_errors_and_valid_upload(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        invalid_files = [
            ("empty-parameters.csv", b"", "No columns to parse"),
            (
                "malformed-parameters.csv",
                b'"unterminated\n1,2\n',
                "EOF inside string",
            ),
            ("invalid-utf8-parameters.csv", b"\xff\xfe\xfa", "codec can't decode"),
        ]
        for filename, content, expected_message in invalid_files:
            with self.subTest(filename=filename):
                self.file_uploader(app, "Upload Custom Parameters CSV").set_value(
                    (filename, content, "text/csv")
                ).run(timeout=120)
                self.assertFalse(
                    app.exception, [element.value for element in app.exception]
                )
                self.assertTrue(
                    any(
                        "Custom parameters were not applied" in item.value
                        and expected_message in item.value
                        for item in app.error
                    )
                )

        valid_parameters = (
            ROOT / "config" / "gdp_model_parameters.csv"
        ).read_bytes()
        self.file_uploader(app, "Upload Custom Parameters CSV").set_value(
            ("valid-parameters.csv", valid_parameters, "text/csv")
        ).run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any("Custom parameters loaded for all 21" in item.value
                for item in app.success)
        )

    def test_deep_dive_parameter_csv_upload_errors_and_valid_upload(self):
        sample = pd.read_csv(ROOT / "sample_wide_input.csv")
        app = self.app_with_view(
            "🔍 Economy Deep Dive",
            sample[sample["economy_code"].eq("01_AUS")].copy(),
        )

        invalid_files = [
            ("empty-australia.csv", b"", "No columns to parse"),
            (
                "malformed-australia.csv",
                b'"unterminated\n1,2\n',
                "EOF inside string",
            ),
            ("invalid-utf8-australia.csv", b"\xff\xfe\xfa", "codec can't decode"),
        ]
        for filename, content, expected_message in invalid_files:
            with self.subTest(filename=filename):
                self.file_uploader(app, "Upload Parameters CSV").set_value(
                    (filename, content, "text/csv")
                )
                app.run(timeout=120)
                self.assertFalse(
                    app.exception, [element.value for element in app.exception]
                )
                self.assertTrue(
                    any(
                        "Could not load parameters" in item.value
                        and expected_message in item.value
                        for item in app.error
                    )
                )
                self.assertTrue(app.get("vega_lite_chart"))

        valid_parameters = pd.read_csv(
            ROOT / "config" / "gdp_model_parameters.csv"
        )
        valid_parameters = valid_parameters[
            valid_parameters["economy_code"].eq("01_AUS")
        ]
        self.file_uploader(app, "Upload Parameters CSV").set_value(
            (
                "australia-parameters.csv",
                valid_parameters.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        )
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any("Parameters loaded successfully for 01_AUS" in item.value
                for item in app.success)
        )

    def test_full_results_and_leap_export_sections(self):
        app = self.app_with_population(
            "📋 Combined Results & Export", False, economy_code=None
        )
        self.assertIn("#### Full Model Results", {item.value for item in app.markdown})
        self.assertIn("#### LEAP Modelling Input", {item.value for item in app.markdown})
        self.assertIn(
            "Download the prepared macroeconomic projections in the format "
            "required by the LEAP modelling team.",
            {item.value for item in app.caption},
        )

        download_labels = {item.label for item in app.get("download_button")}
        self.assertTrue(
            {
                "⬇️ Download Combined Results (Long Format CSV)",
                "⬇️ Download Combined Results (Wide Format CSV)",
                "⬇️ Download LEAP Input CSV",
            }.issubset(download_labels)
        )

        economy_filter = next(
            item for item in app.multiselect if item.label == "Filter by Economy"
        )
        variable_filter = next(
            item for item in app.multiselect if item.label == "Filter by Variable"
        )
        economy_filter.set_value(["01_AUS"])
        variable_filter.set_value(["population"]).run(timeout=120)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual(
            set(app.dataframe[0].value["economy_code"]),
            {"01_AUS"},
        )
        self.assertEqual(
            set(app.dataframe[0].value["variable"]),
            {"population"},
        )
        self.assertIn(
            "⬇️ Download LEAP Input CSV",
            {item.label for item in app.get("download_button")},
        )

    def test_leap_export_ui_blocks_incomplete_results(self):
        app = self.app_with_population(
            "📋 Combined Results & Export", False, economy_code="01_AUS"
        )
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(
            any("Missing economies" in item.value for item in app.warning)
        )
        self.assertNotIn(
            "⬇️ Download LEAP Input CSV",
            {item.label for item in app.get("download_button")},
        )

    def test_partial_dataset_blocks_all_apec_wide_tabs_but_not_deep_dive(self):
        data = pd.read_csv(ROOT / "sample_wide_input.csv")
        partial = data[data["economy_code"].isin(["01_AUS", "18_CT"])].copy()
        app = self.app_with_view("🔍 Economy Deep Dive", partial)
        self.assertTrue(app.get("vega_lite_chart"))

        apec_wide_views = [
            "👥 Population Scenarios",
            "📈 Economy Comparison",
            "🌏 APEC Overview",
            "📋 Combined Results & Export",
        ]
        result_downloads = {
            "⬇️ Download Combined Results (Long Format CSV)",
            "⬇️ Download Combined Results (Wide Format CSV)",
            "⬇️ Download LEAP Input CSV",
        }
        for view in apec_wide_views:
            with self.subTest(view=view):
                app.session_state["active_view"] = view
                app.run(timeout=120)
                self.assertFalse(
                    app.exception, [element.value for element in app.exception]
                )
                self.assertTrue(
                    any(
                        "complete results for all 21 APEC economies" in item.value
                        and "Missing economies" in item.value
                        for item in app.warning
                    )
                )
                self.assertEqual(app.get("vega_lite_chart"), [])
                self.assertEqual(app.dataframe, [])
                self.assertFalse(
                    result_downloads
                    & {item.label for item in app.get("download_button")}
                )

        app.session_state["active_view"] = "🔍 Economy Deep Dive"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(app.get("vega_lite_chart"))
        self.assertTrue(
            any("Loaded dataset includes 2 of 21" in item.value for item in app.caption)
        )

    def test_deep_dive_resets_selection_and_blocks_invalid_replacement(self):
        complete = pd.read_csv(ROOT / "sample_wide_input.csv")
        app = self.app_with_view("🔍 Economy Deep Dive", complete)

        economy_select = next(
            item for item in app.selectbox if item.label == "Choose Economy"
        )
        economy_select.set_value("01_AUS").run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])

        partial = complete[complete["economy_code"].eq("18_CT")].copy()
        app.session_state["df_wide"] = partial
        app.session_state["data_label"] = "partial Chinese Taipei data"
        app.session_state["active_view"] = "🔍 Economy Deep Dive"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertEqual(app.session_state["deep_dive_econ"], "18_CT")
        economy_select = next(
            item for item in app.selectbox if item.label == "Choose Economy"
        )
        self.assertEqual(economy_select.value, "18_CT")
        self.assertEqual(economy_select.options, ["18_CT"])
        self.assertTrue(app.get("vega_lite_chart"))

        selected_parameters = pd.read_csv(
            ROOT / "config" / "gdp_model_parameters.csv"
        )
        selected_parameters = selected_parameters[
            selected_parameters["economy_code"].eq("18_CT")
        ]
        self.file_uploader(app, "Upload Parameters CSV").set_value(
            (
                "chinese-taipei-parameters.csv",
                selected_parameters.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        )
        app.session_state["active_view"] = "🔍 Economy Deep Dive"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any(
                "Parameters loaded successfully for 18_CT" in item.value
                for item in app.success
            )
        )

        invalid = partial.copy()
        invalid.loc[invalid["year"].eq(2025), "population_1jan"] = 0
        app.session_state["df_wide"] = invalid
        app.session_state["data_label"] = "invalid Chinese Taipei data"
        app.session_state["active_view"] = "🔍 Economy Deep Dive"
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any(
                "18_CT 2025 population_1jan" in item.value
                for item in app.error
            )
        )
        self.assertEqual(app.get("vega_lite_chart"), [])
        self.assertEqual(app.dataframe, [])

    def test_methodology_is_accessible_without_model_execution(self):
        complete = pd.read_csv(ROOT / "sample_wide_input.csv")
        # Ensure this complete input cannot hit a cached model result.
        unique_row = complete["economy_code"].eq("01_AUS") & complete["year"].eq(1990)
        complete.loc[unique_row, "population_1jan"] += 0.125
        invalid = complete.copy()
        invalid.loc[
            invalid["economy_code"].eq("01_AUS")
            & invalid["year"].eq(2025),
            "population_1jan",
        ] = 0
        partial = complete[complete["economy_code"].eq("01_AUS")].copy()

        def assert_methodology(app):
            self.assertFalse(
                app.exception, [element.value for element in app.exception]
            )
            self.assertIn(
                "APERC Solow-Swan Macro Model Methodology",
                {element.value for element in app.subheader},
            )
            self.assertEqual(app.get("vega_lite_chart"), [])

        with patch.object(
            core,
            "run_analysis",
            side_effect=AssertionError("Model calculation was triggered"),
        ) as run_analysis:
            app = self.app_with_view("📖 Model Methodology", complete)
            assert_methodology(app)
            run_analysis.assert_not_called()

            app.session_state["df_wide"] = invalid
            app.session_state["active_view"] = "📖 Model Methodology"
            app.run(timeout=120)
            assert_methodology(app)
            self.assertTrue(
                any(
                    "01_AUS 2025 population_1jan" in item.value
                    for item in app.error
                )
            )
            run_analysis.assert_not_called()

            app.session_state["df_wide"] = partial
            app.session_state["active_view"] = "📖 Model Methodology"
            app.run(timeout=120)
            assert_methodology(app)
            run_analysis.assert_not_called()

            del app.session_state["df_wide"]
            app.session_state["active_view"] = "📖 Model Methodology"
            app.run(timeout=120)
            assert_methodology(app)
            self.assertTrue(
                any("No model dataset is loaded" in item.value for item in app.error)
            )
            run_analysis.assert_not_called()

    def test_methodology_is_accessible_without_data_or_valid_input(self):
        app = self.app_with_view("📖 Model Methodology")
        self.assertIn(
            "APERC Solow-Swan Macro Model Methodology",
            {element.value for element in app.subheader},
        )
        self.assertTrue(
            any("No model dataset is loaded" in item.value for item in app.error)
        )
        self.assertEqual(app.get("vega_lite_chart"), [])

        no_data_overview = self.app_with_view("🌏 APEC Overview")
        self.assertTrue(
            any("No model dataset is loaded" in item.value
                for item in no_data_overview.error)
        )
        self.assertEqual(no_data_overview.get("vega_lite_chart"), [])

        invalid = pd.read_csv(ROOT / "sample_wide_input.csv")
        invalid.loc[
            invalid["economy_code"].eq("01_AUS")
            & invalid["year"].eq(2025),
            "population_1jan",
        ] = 0
        invalid_methodology = self.app_with_view("📖 Model Methodology", invalid)
        self.assertTrue(
            any("01_AUS 2025 population_1jan" in item.value
                for item in invalid_methodology.error)
        )
        self.assertIn(
            "APERC Solow-Swan Macro Model Methodology",
            {element.value for element in invalid_methodology.subheader},
        )
        self.assertEqual(invalid_methodology.get("vega_lite_chart"), [])

        invalid_overview = self.app_with_view("🌏 APEC Overview", invalid)
        self.assertTrue(
            any("01_AUS 2025 population_1jan" in item.value
                for item in invalid_overview.error)
        )
        self.assertEqual(invalid_overview.get("vega_lite_chart"), [])

        partial = pd.read_csv(ROOT / "sample_wide_input.csv")
        partial = partial[partial["economy_code"].eq("01_AUS")].copy()
        partial_methodology = self.app_with_view("📖 Model Methodology", partial)
        self.assertIn(
            "APERC Solow-Swan Macro Model Methodology",
            {element.value for element in partial_methodology.subheader},
        )
        self.assertFalse(
            any("01_AUS" in item.value for item in partial_methodology.error)
        )
        self.assertEqual(partial_methodology.get("vega_lite_chart"), [])

        partial_overview = self.app_with_view("🌏 APEC Overview", partial)
        self.assertTrue(
            any("Missing economies" in item.value
                for item in partial_overview.warning)
        )
        self.assertEqual(partial_overview.get("vega_lite_chart"), [])

    def test_data_configuration_summary_tracks_active_dataset(self):
        app = self.app_with_population(
            "🌏 APEC Overview", False, economy_code=None, data_source="bundled"
        )

        bundled_captions = [item.value for item in app.sidebar.caption]
        self.assertIn("**Data loaded:** Bundled sample", bundled_captions)
        self.assertTrue(any(
            "Population:** Prepared default historical and projection series"
            in caption
            for caption in bundled_captions
        ))
        self.assertTrue(any(
            "World Bank WDI data through 2025; model-generated projections "
            "generally begin in 2026"
            in caption
            for caption in bundled_captions
        ))
        self.assertTrue(any(
            "IMF WEO data through 2031; model-generated projections generally "
            "begin in 2032"
            in caption
            for caption in bundled_captions
        ))
        self.assertTrue(any(
            "Additional researcher-provided GDP projections may extend the period "
            "before model-generated projections begin."
            in caption
            for caption in bundled_captions
        ))
        self.assertTrue(any(
            caption == "**Economies:** 21" for caption in bundled_captions
        ))
        self.assertFalse(any(
            "Available input years:" in caption for caption in bundled_captions
        ))
        self.assertIn("**Model horizon:** 1990–2060", bundled_captions)
        self.assertFalse(any("Rows:" in caption for caption in bundled_captions))

        uploaded_data = app.session_state["df_wide"]
        uploaded_data = uploaded_data[
            uploaded_data["economy_code"].isin(["01_AUS", "18_CT"])
        ].copy()
        app.session_state["df_wide"] = uploaded_data
        app.session_state["data_source"] = "uploaded"
        app.session_state["data_label"] = "researcher.csv"
        app.run(timeout=120)
        self.assertFalse(app.exception, [item.value for item in app.exception])

        uploaded_captions = [item.value for item in app.sidebar.caption]
        self.assertIn(
            "**Data loaded:** Uploaded dataset — researcher.csv",
            uploaded_captions,
        )
        self.assertTrue(any(
            "Population:** Uses the population values provided in the uploaded dataset"
            in caption
            for caption in uploaded_captions
        ))
        self.assertTrue(any(
            "GDP data and projections:** Uses uploaded values. "
            "Model-generated projections begin after "
            "the available GDP data for each economy."
            in caption
            for caption in uploaded_captions
        ))
        self.assertFalse(any(
            "WDI" in caption or "IMF" in caption for caption in uploaded_captions
        ))
        self.assertTrue(any(
            caption == "**Economies:** 2" for caption in uploaded_captions
        ))
        self.assertFalse(any(
            "Available input years:" in caption for caption in uploaded_captions
        ))
        self.assertIn("**Model horizon:** 1990–2060", uploaded_captions)
        self.assertFalse(any("Rows:" in caption for caption in uploaded_captions))

    def test_apec_map_scales_and_actual_tooltip_values_across_years(self):
        app = self.app_with_population(
            "🌏 APEC Overview", False, economy_code=None
        )
        expected_scales = {
            "Population": "sqrt",
            "Real GDP": "sqrt",
            "GDP per capita": "log",
        }
        expected_formats = {
            "Population": ",.1f",
            "Real GDP": ",.0f",
            "GDP per capita": ",.0f",
        }
        expected_legend_titles = {
            "Population": "Population (Millions)",
            "Real GDP": "Real GDP (Billion USD (2021 PPP))",
            "GDP per capita": "GDP per capita (USD (2021 PPP))",
        }
        scheme = {"name": "plasma", "extent": [0.0, 0.75]}

        for indicator, expected_scale in expected_scales.items():
            domain_2025 = None
            for year in [2025, 2060]:
                next(
                    element for element in app.radio
                    if element.label == "Indicator"
                ).set_value(indicator)
                next(
                    element for element in app.slider
                    if element.label == "Model year"
                ).set_value(year)
                app.run(timeout=120)
                self.assertFalse(app.exception, [e.value for e in app.exception])

                spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
                polygon_layer, marker_layer = spec["layer"][1:3]
                polygon_color = polygon_layer["encoding"]["color"]
                color_scale = polygon_color["scale"]
                self.assertEqual(color_scale["type"], expected_scale)
                self.assertEqual(color_scale["scheme"], scheme)
                self.assertEqual(
                    polygon_color["title"], expected_legend_titles[indicator]
                )
                self.assertEqual(
                    polygon_color["legend"]["format"], ",.0f"
                )
                self.assertEqual(polygon_color["field"], "value")
                if indicator == "GDP per capita":
                    self.assertEqual(
                        polygon_color["legend"]["values"],
                        [2000.0, 10000.0, 50000.0, 200000.0],
                    )
                    self.assertEqual(
                        polygon_color["legend"]["gradientLength"], 240
                    )
                    self.assertFalse(
                        polygon_color["legend"]["labelOverlap"]
                    )
                self.assertEqual(
                    marker_layer["encoding"]["color"]["scale"], color_scale
                )
                self.assertEqual(
                    marker_layer["encoding"]["color"]["legend"],
                    polygon_color["legend"],
                )
                self.assertEqual(
                    polygon_layer["mark"]["stroke"], "#475569"
                )
                self.assertEqual(spec["layer"][0]["mark"]["fill"], "#e6e9ec")
                self.assertEqual(
                    polygon_layer["encoding"]["tooltip"][-1]["field"], "Value"
                )
                self.assertEqual(
                    marker_layer["encoding"]["tooltip"][-1]["field"], "Value"
                )

                if year == 2025:
                    domain_2025 = color_scale["domain"]
                else:
                    self.assertEqual(color_scale["domain"], domain_2025)

                map_data = self.apec_map_data(app)
                self.assertEqual(set(map_data["Year"]), {str(year)})
                self.assertTrue(
                    all(
                        formatted == format(value, expected_formats[indicator])
                        for value, formatted in zip(map_data["value"], map_data["Value"])
                    )
                )
                if indicator == "GDP per capita":
                    self.assertGreater(color_scale["domain"][0], 0)
                    self.assertTrue(
                        map_data["value"].map(
                            lambda value: pd.notna(value) and value > 0
                        ).all()
                    )

    def test_apec_overview_rejects_invalid_population_before_rendering(self):
        data = pd.read_csv(ROOT / "sample_wide_input.csv")
        economy_rows = data["economy_code"].eq("01_AUS")
        for year, value in [
            (2025, 0),
            (2030, -100),
            (2040, None),
            (2060, float("inf")),
        ]:
            data.loc[economy_rows & data["year"].eq(year), "population_1jan"] = value
        app = self.app_with_view("🌏 APEC Overview", data)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertTrue(
            any(
                "01_AUS 2025 population_1jan" in item.value
                for item in app.error
            )
        )
        self.assertEqual(app.get("vega_lite_chart"), [])

    def test_population_tab_options_and_default_line(self):
        app = self.app_with_population("👥 Population Scenarios", False)
        self.assertEqual(
            [element.label for element in app.sidebar.radio],
            ["View"],
        )
        self.assertEqual(
            [element.label for element in app.sidebar.selectbox],
            ["Economy", "Scenarios"],
        )
        self.assertEqual(
            [element.label for element in app.sidebar.number_input],
            ["From", "To"],
        )
        self.assertEqual([element.label for element in app.radio], ["View"])
        self.assertEqual(
            [element.label for element in app.selectbox],
            ["Economy", "Scenarios"],
        )
        self.assertEqual(
            [element.label for element in app.number_input],
            ["From", "To"],
        )
        self.assertEqual(app.sidebar.radio[0].value, "Total")
        self.assertEqual(app.sidebar.selectbox[1].value, "All")
        self.assertTrue(app.sidebar.selectbox[0].disabled)

        select = next(
            element for element in app.sidebar.selectbox
            if element.label == "Scenarios"
        )
        self.assertIn("Default", select.options)
        self.assertNotIn("Custom", select.options)
        charts = app.get("vega_lite_chart")
        self.assertGreaterEqual(len(charts), 2)
        for chart in charts[:2]:
            layers = json.loads(chart.proto.spec)["layer"]
            other, default = layers[:2]
            self.assertEqual(default["mark"], other["mark"])
            self.assertNotIn("strokeDash", default.get("encoding", {}))
            self.assertIn("#172b4d", default["encoding"]["color"]["scale"]["range"])
        app.session_state["df_wide"]["population_1jan_custom"] = (
            app.session_state["df_wide"]["population_1jan"]
        )
        app.run(timeout=120)
        self.assertFalse(app.exception)
        select = next(
            element for element in app.sidebar.selectbox
            if element.label == "Scenarios"
        )
        self.assertIn("Custom", select.options)

    def test_economy_comparison_offers_only_four_variables(self):
        app = self.app_with_population("📈 Economy Comparison", False)
        self.assertEqual(
            [element.label for element in app.sidebar.multiselect],
            ["Select Economies to display"],
        )
        self.assertEqual(
            app.sidebar.multiselect[0].value,
            sorted(app.session_state["df_wide"]["economy_code"].unique())[:6],
        )
        self.assertEqual(
            [element.label for element in app.sidebar.selectbox],
            ["Select Variable"],
        )
        self.assertEqual(
            [element.label for element in app.sidebar.radio],
            ["Display Mode"],
        )
        self.assertEqual(list(app.main.multiselect), [])
        self.assertEqual(list(app.main.selectbox), [])
        self.assertEqual(list(app.main.radio), [])

        variable_select = next(
            element for element in app.selectbox if element.label == "Select Variable"
        )
        option_labels = {
            option[1] if isinstance(option, tuple) else option
            for option in variable_select.options
        }
        expected_labels = {
            "Real GDP (Millions 2021 USD PPP)",
            "Annual Real GDP Growth Rate (%)",
            "GDP Per Capita (2021 USD PPP)",
            "Population (Thousands)",
        }
        self.assertEqual(option_labels, expected_labels)
        self.assertGreaterEqual(len(app.get("vega_lite_chart")), 1)
        self.assertTrue(
            any("Benchmark Year Values" in item.value for item in app.markdown)
        )
        display_mode = next(
            element for element in app.radio if element.label == "Display Mode"
        )
        self.assertEqual(display_mode.value, "Absolute")
        self.assertEqual(
            display_mode.options, ["Absolute", "Indexed (2025 = 100)"]
        )
        self.assertTrue(display_mode.proto.horizontal)
        self.assertFalse(display_mode.disabled)

        variable_select.set_value(
            ("gdp_growth", "Annual Real GDP Growth Rate (%)")
        ).run(timeout=120)
        self.assertFalse(app.exception)
        display_mode = next(
            element for element in app.radio if element.label == "Display Mode"
        )
        self.assertEqual(
            display_mode.options, ["Absolute", "Indexed (2025 = 100)"]
        )
        self.assertEqual(display_mode.value, "Absolute")
        self.assertTrue(display_mode.disabled)

    def test_indexed_comparison_transforms_only_the_chart(self):
        app = self.app_with_population(
            "📈 Economy Comparison",
            False,
            economy_code=["01_AUS", "02_BD"],
        )
        variable_options = {
            "real_GDP": (
                "Real GDP (Millions 2021 USD PPP)",
                "Real GDP Index (2025 = 100)",
            ),
            "GDP_per_capita": (
                "GDP Per Capita (2021 USD PPP)",
                "GDP per Capita Index (2025 = 100)",
            ),
            "population": (
                "Population (Thousands)",
                "Population Index (2025 = 100)",
            ),
        }

        for variable, (variable_label, indexed_title) in variable_options.items():
            variable_select = next(
                element
                for element in app.selectbox
                if element.label == "Select Variable"
            )
            variable_select.set_value((variable, variable_label)).run(timeout=120)
            self.assertFalse(app.exception)

            display_mode = next(
                element for element in app.radio if element.label == "Display Mode"
            )
            self.assertEqual(display_mode.value, "Absolute")
            self.assertFalse(display_mode.disabled)
            self.assertTrue(display_mode.proto.horizontal)
            absolute_values = self.chart_values(app)
            absolute_spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
            self.assertEqual(absolute_spec["title"], variable_label)
            self.assertEqual(
                absolute_spec["layer"][0]["encoding"]["y"]["title"],
                variable_label,
            )
            self.assertEqual(
                absolute_spec["layer"][2]["encoding"]["tooltip"][1]["title"],
                variable_label,
            )
            self.assertTrue(
                any(
                    item.value == f"{variable_label} (1990–2060)"
                    for item in app.caption
                )
            )
            benchmark_values = app.dataframe[0].value.copy()
            source_data = app.session_state["df_wide"].copy()

            display_mode.set_value("Indexed (2025 = 100)").run(timeout=120)
            self.assertFalse(app.exception)
            indexed_values = self.chart_values(app)
            spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)

            self.assertEqual(spec["title"], indexed_title)
            self.assertEqual(
                spec["layer"][0]["encoding"]["y"]["title"],
                "Index (2025 = 100)",
            )
            tooltip = spec["layer"][2]["encoding"]["tooltip"]
            self.assertEqual(tooltip[1]["title"], "Index (2025 = 100)")
            self.assertTrue(
                any(
                    "indexed to each economy's 2025 value (= 100)" in item.value
                    for item in app.caption
                )
            )
            self.assertEqual(
                set(indexed_values["year"]), set(absolute_values["year"])
            )
            self.assertEqual(
                set(indexed_values["series"]), set(absolute_values["series"])
            )
            for economy_code in absolute_values["series"].unique():
                absolute_series = absolute_values[
                    absolute_values["series"] == economy_code
                ].set_index("year")["plot_value"]
                indexed_series = indexed_values[
                    indexed_values["series"] == economy_code
                ].set_index("year")["plot_value"]
                self.assertAlmostEqual(indexed_series.loc[2025], 100.0)
                self.assertAlmostEqual(
                    indexed_series.loc[2030],
                    absolute_series.loc[2030] / absolute_series.loc[2025] * 100,
                )

            pd.testing.assert_frame_equal(app.dataframe[0].value, benchmark_values)
            pd.testing.assert_frame_equal(app.session_state["df_wide"], source_data)

    def test_indexed_mode_requires_a_valid_2025_baseline(self):
        source_data = pd.read_csv(ROOT / "sample_wide_input.csv")
        baseline_row = (source_data["economy_code"] == "01_AUS") & (
            source_data["year"] == 2025
        )
        source_data.loc[baseline_row, "population_1jan"] = 0
        app = self.app_with_view("📈 Economy Comparison", source_data)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(
            any(
                "01_AUS 2025 population_1jan" in item.value
                for item in app.error
            )
        )
        self.assertEqual(app.get("vega_lite_chart"), [])

    def test_comparison_cagr_column_for_levels_and_growth(self):
        app = self.app_with_population(
            "📈 Economy Comparison",
            False,
            economy_code=["01_AUS", "02_BD"],
        )
        variable_options = {
            "real_GDP": "Real GDP (Millions 2021 USD PPP)",
            "GDP_per_capita": "GDP Per Capita (2021 USD PPP)",
            "population": "Population (Thousands)",
            "gdp_growth": "Annual Real GDP Growth Rate (%)",
        }
        cagr_column = "CAGR (2025–2060)"

        for variable, variable_label in variable_options.items():
            variable_select = next(
                element
                for element in app.selectbox
                if element.label == "Select Variable"
            )
            variable_select.set_value((variable, variable_label)).run(timeout=120)
            self.assertFalse(app.exception)

            table = app.dataframe[0].value
            chart_values = self.chart_values(app)
            self.assertEqual(table.columns[-1], cagr_column)
            for economy_code in ["01_AUS", "02_BD"]:
                if variable == "gdp_growth":
                    self.assertEqual(table.loc[economy_code, cagr_column], "—")
                else:
                    endpoints = chart_values[
                        (chart_values["series"] == economy_code)
                        & chart_values["year"].isin([2025, 2060])
                    ].set_index("year")["plot_value"]
                    expected = (
                        (endpoints.loc[2060] / endpoints.loc[2025]) ** (1 / 35)
                        - 1
                    ) * 100
                    self.assertEqual(
                        table.loc[economy_code, cagr_column],
                        f"{expected:.2f}%",
                    )

                for year in [1990, 2025, 2030, 2040, 2050, 2060]:
                    table_value = table.loc[economy_code, str(year)]
                    plotted_values = chart_values.loc[
                        (chart_values["series"] == economy_code)
                        & (chart_values["year"] == year),
                        "plot_value",
                    ]
                    if pd.isna(table_value):
                        self.assertTrue(plotted_values.empty)
                    else:
                        self.assertAlmostEqual(
                            table_value,
                            plotted_values.iloc[0],
                        )

    def test_invalid_population_endpoints_are_rejected_before_results(self):
        source_data = pd.read_csv(ROOT / "sample_wide_input.csv")
        end_year_row = (source_data["economy_code"] == "01_AUS") & (
            source_data["year"] == 2060
        )
        source_data.loc[end_year_row, "population_1jan"] = -1
        app = self.app_with_view("📈 Economy Comparison", source_data)
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertTrue(
            any(
                "01_AUS 2060 population_1jan" in item.value
                for item in app.error
            )
        )
        self.assertEqual(app.get("vega_lite_chart"), [])

    def test_ranking_ties_movement_and_indexed_mode_independence(self):
        economy_codes = ["01_AUS", "02_BD", "03_CDA", "04_CHL"]
        app = self.app_with_population(
            "📈 Economy Comparison", False, economy_code=economy_codes
        )
        source_data = app.session_state["df_wide"]
        endpoints = {
            "01_AUS": (40, 20),
            "02_BD": (30, 40),
            "03_CDA": (20, 15),
            "04_CHL": (10, 15),
        }
        for economy_code, (start_value, end_value) in endpoints.items():
            for year, value in [(2025, start_value), (2060, end_value)]:
                source_data.loc[
                    (source_data["economy_code"] == economy_code)
                    & (source_data["year"] == year),
                    "population_1jan",
                ] = value

        variable_select = next(
            element for element in app.selectbox if element.label == "Select Variable"
        )
        variable_select.set_value(
            ("population", "Population (Thousands)")
        ).run(timeout=120)
        self.assertFalse(app.exception)
        ranking_values = self.ranking_markdown_values(app, 4)
        self.assertEqual(
            ranking_values,
            [
                "**#2**", "→", ":orange[**#1**]", ":green[↑ Up 1 position]",
                "**#1**", "→", ":blue[**#2**]", ":red[↓ Down 1 position]",
                "**#3**", "→", ":gray[**#3**]", ":gray[→ No change]",
                "**#4**", "→", ":gray[**#3**]", ":green[↑ Up 1 position]",
            ],
        )
        self.assertEqual(
            {element.value for element in app.text},
            {"Australia", "Brunei Darussalam", "Canada", "Chile"},
        )
        ranking_heading = next(
            element for element in app.markdown
            if element.value == "##### Ranking Changes (2025–2060)"
        )
        ranking_caption = (
            "Rankings compare each selected economy's relative position based on the "
            "selected indicator in 2025 and 2060. Positive movement indicates a rise "
            "in relative ranking among the selected economies, not necessarily faster "
            "growth or a larger absolute change in the indicator."
        )
        self.assertFalse(ranking_heading.proto.help)
        self.assertIn(ranking_caption, {element.value for element in app.caption})
        self.assertFalse(
            any(
                "selected economies' indicator values in 2025 and 2060"
                in element.value
                for element in app.caption
            )
        )

        benchmark_values = app.dataframe[0].value.copy()
        display_mode = next(
            element for element in app.radio if element.label == "Display Mode"
        )
        display_mode.set_value("Indexed (2025 = 100)").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(self.ranking_markdown_values(app, 4), ranking_values)
        pd.testing.assert_frame_equal(app.dataframe[0].value, benchmark_values)

    def test_annual_growth_ranks_higher_raw_rates_first(self):
        economy_codes = ["01_AUS", "02_BD", "03_CDA"]
        app = self.app_with_population(
            "📈 Economy Comparison", False, economy_code=economy_codes
        )
        source_data = app.session_state["df_wide"]
        final_values = {
            "01_AUS": 90_000,
            "02_BD": 80_000,
            "03_CDA": 110_000,
        }
        for economy_code, final_value in final_values.items():
            projection_rows = (
                source_data["economy_code"].eq(economy_code)
                & source_data["year"].between(2026, 2060)
            )
            source_data.loc[
                projection_rows, "Real GDP PPP 2021 USD"
            ] = 100_000
            source_data.loc[
                source_data["economy_code"].eq(economy_code)
                & source_data["year"].eq(2060),
                "Real GDP PPP 2021 USD",
            ] = final_value

        variable_select = next(
            element for element in app.selectbox if element.label == "Select Variable"
        )
        variable_select.set_value(
            ("gdp_growth", "Annual Real GDP Growth Rate (%)")
        ).run(timeout=120)
        self.assertFalse(app.exception)

        chart_values = self.chart_values(app)
        final_growth = chart_values[chart_values["year"] == 2060].set_index(
            "series"
        )["plot_value"]
        self.assertLess(final_growth["01_AUS"], 0)
        self.assertLess(final_growth["02_BD"], final_growth["01_AUS"])
        self.assertGreater(final_growth["03_CDA"], 0)
        economy_names = (
            source_data.drop_duplicates("economy_code")
            .set_index("economy_code")["economy"]
            .to_dict()
        )
        expected_order = [
            economy_names[code]
            for code in final_growth.sort_values(ascending=False).index
        ]
        self.assertEqual([element.value for element in app.text], expected_order)
        ranking_values = self.ranking_markdown_values(app, 3)
        self.assertEqual(
            [ranking_values[index] for index in [2, 6, 10]],
            [":orange[**#1**]", ":blue[**#2**]", ":gray[**#3**]"],
        )

    def test_ranking_includes_all_selected_economies(self):
        app = self.app_with_population(
            "📈 Economy Comparison", False, economy_code=None
        )
        economy_select = next(
            element
            for element in app.multiselect
            if element.label == "Select Economies to display"
        )
        economy_select.set_value(economy_select.options).run(timeout=120)
        self.assertFalse(app.exception)

        selected_names = set(
            app.session_state["df_wide"]["economy"].dropna().unique()
        )
        self.assertEqual(set(element.value for element in app.text), selected_names)
        self.assertEqual(len(app.text), 21)
        self.assertEqual(len(self.ranking_markdown_values(app, 21)), 84)
        self.assertEqual(app.dataframe[0].value.shape[0], 21)

    def test_deep_dive_default_and_custom_options(self):
        app = self.app_with_population("🔍 Economy Deep Dive", False)
        radio = next(element for element in app.radio if element.label == "Population Scenario")
        self.assertIn("Default", radio.options)
        self.assertNotIn("Custom", radio.options)
        radio.set_value("Default").run(timeout=120)
        self.assertFalse(app.exception)
        default_chart = json.loads(app.get("vega_lite_chart")[1].proto.spec)["layer"][0]
        self.assertEqual(
            default_chart["encoding"]["strokeDash"],
            {
                "condition": {"test": {"field": "series", "oneOf": ["Projection"]}, "value": [6, 4]},
                "value": [1, 0],
            },
        )
        self.assertEqual(default_chart["encoding"]["color"]["scale"]["range"], ["#172b4d", "#172b4d"])
        radio = next(element for element in app.radio if element.label == "Population Scenario")
        radio.set_value("Medium").run(timeout=120)
        self.assertFalse(app.exception)
        medium_chart = json.loads(app.get("vega_lite_chart")[1].proto.spec)["layer"][0]
        self.assertEqual(default_chart["encoding"]["strokeDash"], medium_chart["encoding"]["strokeDash"])
        app.session_state["df_wide"]["population_1jan_custom"] = (
            app.session_state["df_wide"]["population_1jan"]
        )
        app.run(timeout=120)
        self.assertFalse(app.exception)
        radio = next(element for element in app.radio if element.label == "Population Scenario")
        self.assertIn("Custom", radio.options)

    def test_population_caption_is_concise_for_uploaded_data(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            population_view="By Economy",
        )
        self.assertNotIn(
            "The selected economy is shown for each selected scenario.",
            [element.value for element in app.caption],
        )
        caption = next(
            element.value
            for element in app.caption
            if "population values supplied in the uploaded dataset" in element.value
        )
        self.assertIn("uploaded dataset", caption)
        self.assertNotIn("EGEDA", caption)
        self.assertNotIn("UN DESA", caption)
        self.assertNotIn("Displayed data cover", caption)
        self.assertNotIn("1990–2060", caption)
        self.assertNotIn("For Australia", caption)
        self.assertNotIn("Refresh", caption)
        self.assertNotIn("separate Custom path", caption)

        app.session_state["df_wide"]["population_1jan_custom"] = (
            app.session_state["df_wide"]["population_1jan"]
        )
        app.run(timeout=120)
        caption = next(
            element.value
            for element in app.caption
            if "population values supplied in the uploaded dataset" in element.value
        )
        self.assertIn("separate Custom path is available", caption)
        self.assertIn("distinct from the sources used for Default", caption)

    def test_population_tab_blocks_incomplete_official_apec_paths(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            low_missing_year=2000,
        )
        self.assertTrue(
            any(
                "APEC population scenarios are unavailable" in item.value
                and "Low" in item.value
                for item in app.error
            )
        )
        self.assertEqual(app.get("vega_lite_chart"), [])

    def test_bundled_population_caption_uses_economy_specific_selections(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="01_AUS",
            data_source="bundled",
            population_view="By Economy",
        )
        caption = next(
            element.value
            for element in app.caption
            if "researcher-provided custom projections" in element.value
        )
        self.assertEqual(
            caption,
            "Default combines EGEDA historical population through 2024 with "
            "researcher-provided custom projections from 2025 onward. Low, Medium "
            "and High share the same historical population data and use their "
            "corresponding UN DESA projections from 2025 onward.",
        )
        self.assertNotIn("For Australia", caption)
        self.assertNotIn("Displayed data", caption)
        self.assertNotIn("Refresh", caption)

        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="05_PRC",
            data_source="bundled",
            population_view="By Economy",
        )
        caption = next(
            element.value
            for element in app.caption
            if "UN DESA Medium projections" in element.value
        )
        self.assertIn("UN DESA Medium projections from 2025 onward", caption)

        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="16_RUS",
            data_source="bundled",
            population_view="By Economy",
        )
        caption = next(
            element.value
            for element in app.caption
            if "Default combines researcher-provided custom historical population" in element.value
        )
        self.assertIn("researcher-provided custom projections from 2025 onward", caption)
        self.assertNotIn("A separate Custom path", caption)

    def test_total_population_caption_summarizes_sources_and_custom_coverage(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code=None,
            data_source="bundled",
            population_view="Total",
        )
        caption = next(
            element.value
            for element in app.caption
            if "Default combines the selected historical sources" in element.value
        )
        self.assertIn(
            "Aggregate population across the economies under the selected population scenarios.",
            [element.value for element in app.caption],
        )
        self.assertEqual(
            caption,
            "Default combines the selected historical sources "
            "(EGEDA historical population and "
            "researcher-provided custom historical population) through 2024 "
            "with the selected projections (UN DESA Medium, Low and High projections "
            "and researcher-provided custom projections) from 2025 onward. Low, "
            "Medium and High share the same historical data and use their "
            "corresponding UN DESA projections from 2025 onward.",
        )
        self.assertIn("EGEDA historical population", caption)
        self.assertIn("researcher-provided custom historical population", caption)
        self.assertNotIn("Aggregate population", caption)
        self.assertNotIn("separate Custom total", caption)

        data = app.session_state["df_wide"]
        first_economy = data["economy_code"].iloc[0]
        first_economy_rows = data["economy_code"] == first_economy
        data.loc[first_economy_rows, "population_1jan_custom"] = data.loc[
            first_economy_rows, "population_1jan"
        ]
        app.run(timeout=120)
        self.assertTrue(
            any("Custom is not shown in the total" in item.value for item in app.info)
        )
        caption = next(
            element.value
            for element in app.caption
            if "Default combines the selected historical sources" in element.value
        )
        self.assertNotIn("separate Custom total", caption)

        app.session_state["df_wide"]["population_1jan_custom"] = app.session_state[
            "df_wide"
        ]["population_1jan"]
        app.run(timeout=120)
        caption = next(
            element.value
            for element in app.caption
            if "Default combines the selected historical sources" in element.value
        )
        self.assertIn("separate Custom total is available", caption)
        self.assertIn("cover all economies represented", caption)

    def test_uploaded_total_population_caption_does_not_infer_sources(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            data_source="uploaded",
            population_view="Total",
        )
        caption = next(
            element.value
            for element in app.caption
            if "population values supplied in the uploaded dataset" in element.value
        )
        self.assertIn(
            "Aggregate population across the economies under the selected population scenarios.",
            [element.value for element in app.caption],
        )
        self.assertIn("uploaded dataset", caption)
        self.assertIn("underlying sources are not identified", caption)
        self.assertNotIn("EGEDA", caption)
        self.assertNotIn("UN DESA", caption)

    def test_gdp_sensitivity_captions_for_bundled_economies(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="01_AUS",
            data_source="bundled",
            population_view="By Economy",
        )
        captions = [element.value for element in app.caption]
        self.assertIn(
            "GDP projections reflect the selected population scenario and other model inputs.",
            captions,
        )
        self.assertIn(
            "Real GDP paths for the selected economy under each population scenario.",
            captions,
        )
        bottom = next(
            caption for caption in captions if caption.startswith("Real GDP (Millions")
        )
        self.assertIn("World Bank WDI through 2025", bottom)
        self.assertIn("Researcher-supplied projections are retained through 2030", bottom)
        self.assertIn("remaining projections from 2031 onward", bottom)

        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="18_CT",
            data_source="bundled",
            population_view="By Economy",
        )
        bottom = next(
            element.value
            for element in app.caption
            if element.value.startswith("Real GDP (Millions")
        )
        self.assertIn("IMF WEO through 2031", bottom)
        self.assertIn("projections from 2032 onward", bottom)
        self.assertNotIn("Researcher-supplied projections are retained through", bottom)

    def test_gdp_sensitivity_aggregate_caption_summarizes_represented_economies(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code=["01_AUS", "18_CT"],
            data_source="bundled",
            population_view="Total",
        )
        captions = [element.value for element in app.caption]
        self.assertIn(
            "GDP projections reflect the selected population scenario and other model inputs.",
            captions,
        )
        self.assertIn(
            "Total real GDP across the economies represented in the loaded data under each selected population scenario.",
            captions,
        )
        bottom = next(
            caption for caption in captions if caption.startswith("Real GDP (Millions")
        )
        self.assertIn("World Bank WDI through 2025 for non-CT economies", bottom)
        self.assertIn("IMF WEO through 2031 for CT", bottom)
        self.assertIn("after each economy's final supplied year", bottom)
        self.assertIn("2026 for non-CT economies and 2032 for CT", bottom)

    def test_uploaded_gdp_sensitivity_caption_tracks_supplied_projection_end(self):
        app = self.app_with_population(
            "👥 Population Scenarios",
            False,
            economy_code="01_AUS",
            data_source="uploaded",
            population_view="By Economy",
        )
        bottom = next(
            element.value
            for element in app.caption
            if element.value.startswith("Real GDP (Millions")
        )
        self.assertIn("GDP values come from the uploaded data", bottom)
        self.assertIn("No future GDP projections are supplied", bottom)
        self.assertIn("projections from 2026 onward", bottom)
        self.assertNotIn("WDI", bottom)
        self.assertNotIn("IMF", bottom)

        data = app.session_state["df_wide"]
        data.loc[data["year"] == 2026, "Real GDP PPP 2021 USD"] = 1_800_000.0
        data.loc[data["year"] == 2027, "Real GDP PPP 2021 USD"] = 1_850_000.0
        app.run(timeout=120)
        bottom = next(
            element.value
            for element in app.caption
            if element.value.startswith("Real GDP (Millions")
        )
        self.assertIn("Supplied future projections are retained through 2027", bottom)
        self.assertIn("remaining projections from 2028 onward", bottom)

    def test_deep_dive_gdp_caption_tracks_economy_dataset_and_override_end(self):
        app = self.app_with_population(
            "🔍 Economy Deep Dive",
            False,
            economy_code=["01_AUS", "05_PRC", "18_CT"],
            data_source="bundled",
        )

        def trajectory_caption():
            return next(
                element.value
                for element in app.caption
                if element.value.startswith("Real GDP (Millions 2021 USD PPP)")
            )

        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). Historical GDP uses World Bank WDI "
            "through 2025. Researcher-supplied projections are retained through "
            "2030, after which the model generates the remaining projections from "
            "2031 onward.",
        )

        economy_picker = next(
            element for element in app.selectbox if element.label == "Choose Economy"
        )
        economy_picker.set_value("05_PRC").run(timeout=120)
        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). Historical GDP uses World Bank WDI "
            "through 2025; the model generates projections from 2026 onward.",
        )

        economy_picker = next(
            element for element in app.selectbox if element.label == "Choose Economy"
        )
        economy_picker.set_value("18_CT").run(timeout=120)
        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). IMF WEO data and projections extend "
            "through 2031; the model generates projections from 2032 onward.",
        )

        app.session_state["data_source"] = "uploaded"
        app.run(timeout=120)
        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). GDP values from the uploaded data are "
            "used as supplied; the model generates projections from 2032 onward.",
        )

        economy_picker = next(
            element for element in app.selectbox if element.label == "Choose Economy"
        )
        economy_picker.set_value("01_AUS").run(timeout=120)
        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). GDP values from the uploaded data are "
            "used as supplied; the model generates projections from 2026 onward.",
        )

        data = app.session_state["df_wide"]
        australia = data["economy_code"] == "01_AUS"
        data.loc[australia & data["year"].eq(2026), "Real GDP PPP 2021 USD"] = (
            1_800_000.0
        )
        data.loc[australia & data["year"].eq(2027), "Real GDP PPP 2021 USD"] = (
            1_850_000.0
        )
        app.run(timeout=120)
        self.assertEqual(
            trajectory_caption(),
            "Real GDP (Millions 2021 USD PPP). GDP values from the uploaded data are "
            "used as supplied. Future GDP projections are retained through 2027, "
            "after which the model generates the remaining projections from 2028 onward.",
        )


if __name__ == "__main__":
    unittest.main()
