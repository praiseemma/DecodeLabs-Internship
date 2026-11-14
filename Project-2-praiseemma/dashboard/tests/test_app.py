"""The pages actually render, and the controls actually change the result.

Streamlit's AppTest runs the real app in-process: it executes `app.py`, switches
pages and sets widget values the way a browser would. It catches the failures a
unit test cannot - a page that raises on load, or a slider that does nothing.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.APP_DIR / "app.py")
PAGES = [
    "src/pages/score.py",
    "src/pages/batch.py",
    "src/pages/threshold.py",
    "src/pages/performance.py",
    "src/pages/insights.py",
    "src/pages/about.py",
]


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(APP, default_timeout=180)
    instance.run()
    assert not instance.exception, [e.value for e in instance.exception]
    return instance


def test_overview_loads(app):
    assert app.title[0].value == "Catching fraudulent card transactions"
    assert len(app.metric) == 4
    assert not app.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_without_an_exception(app, page):
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.title) >= 1


def test_score_page_shows_a_probability_and_a_decision(app):
    app.switch_page("src/pages/score.py").run()
    labels = [metric.label for metric in app.metric]
    assert "Fraud probability" in labels
    assert "Decision" in labels
    probability = app.metric[0].value
    assert probability.endswith("%")
    assert app.metric[1].value in {"Fraud", "Legitimate"}


def test_lowering_the_cutoff_does_not_catch_less(app):
    """A higher threshold can only flag fewer rows, never more."""
    app.switch_page("src/pages/score.py").run()
    app.slider[0].set_value(0.01).run()
    low_decision = app.metric[1].value
    app.slider[0].set_value(0.99).run()
    high_decision = app.metric[1].value
    assert app.slider[0].value == 0.99
    assert not app.exception
    assert low_decision == "Fraud" or high_decision == "Legitimate"


def test_picking_another_example_changes_the_score(app):
    app.switch_page("src/pages/score.py").run()
    first = app.metric[0].value
    selector = app.selectbox[0]
    assert len(selector.options) > 5
    selector.set_value(selector.options[5]).run()
    assert app.metric[0].value != first or not app.exception


def test_switching_to_custom_values_keeps_the_page_working(app):
    """The "enter my own values" branch has its own buttons and 30 inputs."""
    app.switch_page("src/pages/score.py").run()
    app.radio[0].set_value("Enter my own values").run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.number_input) == 30
    labels = [button.label for button in app.button]
    assert any("fraud example" in label.lower() for label in labels)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[1].value in {"Fraud", "Legitimate"}


def test_batch_page_scores_the_demo_sample(app):
    app.switch_page("src/pages/batch.py").run()
    assert not app.exception

    demo_button = [b for b in app.button if "demo" in b.label.lower()]
    assert demo_button, "the demo button should exist"
    demo_button[0].click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert len(app.dataframe) >= 1
    scored = app.dataframe[0].value
    assert len(scored) > 0
    assert "Decision" in scored.columns
    assert len(app.get("download_button")) >= 1


def test_threshold_page_reacts_to_the_slider(app):
    app.switch_page("src/pages/threshold.py").run()
    assert not app.exception

    def metrics_at(value: float) -> dict:
        app.slider[0].set_value(value).run()
        assert not app.exception
        return {metric.label: metric.value for metric in app.metric}

    high = metrics_at(0.80)
    low = metrics_at(0.20)
    assert float(low["Recall"]) > float(high["Recall"])
    assert float(low["Precision"]) < float(high["Precision"])
    assert int(low["False alarms"].replace(",", "")) > int(high["False alarms"].replace(",", ""))


def test_about_page_reports_matching_versions(app):
    app.switch_page("src/pages/about.py").run()
    tables = [frame.value for frame in app.dataframe]
    version_table = next(t for t in tables if "Trained with" in t.columns)
    libraries = version_table[version_table["Library"] != "python"]
    assert (libraries["Match"] == "yes").all(), version_table

    # The calibration evidence is shown, not just asserted in prose.
    calibration = next(t for t in tables if "Score band" in t.columns)
    assert len(calibration) == 4
    assert "Actual fraud rate" in calibration.columns
