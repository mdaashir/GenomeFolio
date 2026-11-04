"""Headless smoke test of the Streamlit app (skipped when streamlit is not installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_runs_pipeline_and_shows_every_phase():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception

    run_buttons = [b for b in at.button if b.label == "Run pipeline"]
    assert run_buttons
    run_buttons[0].click().run()
    assert not at.exception

    headers = [h.value for h in at.header]
    for phase in ("1. Data", "2. Symbolization", "3. Alignment", "4. Clustering", "5. Forecast"):
        assert any(h.startswith(phase) for h in headers), headers
