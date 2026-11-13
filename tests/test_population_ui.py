import json
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class PopulationUITests(unittest.TestCase):
    def app_with_population(self, view, custom):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        data = pd.read_csv(ROOT / "sample_wide_input.csv")
        data = data[data["economy_code"] == "01_AUS"].copy()
        if custom:
            data["population_1jan_custom"] = data["population_1jan"]
        app.session_state["df_wide"] = data
        app.session_state["data_label"] = "uploaded test data"
        app.session_state["active_view"] = view
        app.run(timeout=120)
        self.assertFalse(app.exception, [element.value for element in app.exception])
        return app

    def test_population_tab_options_and_default_line(self):
        app = self.app_with_population("👥 Population Scenarios", False)
        select = next(element for element in app.selectbox if element.label == "Scenarios")
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
        select = next(element for element in app.selectbox if element.label == "Scenarios")
        self.assertIn("Custom", select.options)

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


if __name__ == "__main__":
    unittest.main()
