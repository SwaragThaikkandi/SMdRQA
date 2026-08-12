"""
Tests for the SMdRQA UI support modules.

The ScriptRecorder and window_size_sensitivity tests run without any UI
dependency; the app import smoke test is skipped unless streamlit and
plotly are installed.
"""

import numpy as np
import pytest

from SMdRQA.RQA2 import RQA2
from SMdRQA.ui import ScriptRecorder, window_size_sensitivity
from SMdRQA.ui.sensitivity import MEASURES


# ---------------------------------------------------------------------------
# ScriptRecorder
# ---------------------------------------------------------------------------

class TestScriptRecorder:

    def test_header_contains_seed(self):
        rec = ScriptRecorder(seed=123)
        assert "SEED = 123" in rec.script()
        assert "np.random.seed(SEED)" in rec.script()

    def test_record_appends_blocks(self):
        rec = ScriptRecorder(seed=1)
        rec.record("x = 1", comment="set x")
        rec.record("y = x + 1")
        script = rec.script()
        assert "# set x" in script
        assert script.index("x = 1") < script.index("y = x + 1")
        assert rec.n_blocks == 2

    def test_clear_keeps_header(self):
        rec = ScriptRecorder(seed=7)
        rec.record("a = 0")
        rec.clear()
        assert rec.n_blocks == 0
        assert "SEED = 7" in rec.script()

    def test_script_is_valid_python(self):
        rec = ScriptRecorder(seed=42)
        rec.record("data = np.zeros(5)", comment="allocate")
        compile(rec.script(), "<generated>", "exec")


# ---------------------------------------------------------------------------
# window_size_sensitivity
# ---------------------------------------------------------------------------

class TestWindowSizeSensitivity:

    @pytest.fixture(scope="class")
    def rp(self):
        signal = np.sin(np.linspace(0, 12 * np.pi, 300))
        return RQA2(signal).recurrence_plot

    def test_returns_expected_columns(self, rp):
        df = window_size_sensitivity(
            rp, 'percent_det', min_size=20, max_size=60, step=20,
            n_boot=50, seed=0)
        assert list(df.columns) == ['window_size', 'ci_width']
        assert list(df['window_size']) == [20, 40]
        assert np.all(df['ci_width'] >= 0)

    def test_reproducible_with_seed(self, rp):
        kwargs = dict(min_size=20, max_size=50, step=10, n_boot=50)
        a = window_size_sensitivity(rp, 'percent_lam', seed=5, **kwargs)
        b = window_size_sensitivity(rp, 'percent_lam', seed=5, **kwargs)
        assert np.allclose(a['ci_width'], b['ci_width'])

    def test_all_measures_run(self, rp):
        for measure in MEASURES:
            df = window_size_sensitivity(
                rp, measure, min_size=30, max_size=50, step=20,
                n_boot=20, seed=0)
            assert len(df) == 1

    def test_unknown_measure_raises(self, rp):
        with pytest.raises(ValueError, match="Unknown measure"):
            window_size_sensitivity(rp, 'nope')

    def test_progress_callback_called(self, rp):
        calls = []
        window_size_sensitivity(
            rp, 'avg_diag', min_size=20, max_size=60, step=20,
            n_boot=10, seed=0,
            progress_callback=lambda i, n, w: calls.append((i, n, w)))
        assert calls == [(0, 2, 20), (1, 2, 40)]


# ---------------------------------------------------------------------------
# Streamlit app (smoke test, optional dependency)
# ---------------------------------------------------------------------------

class TestAppSmoke:

    def test_app_module_imports(self):
        pytest.importorskip("streamlit")
        pytest.importorskip("plotly")
        import SMdRQA.ui.app  # noqa: F401

    def test_launch_without_streamlit_raises_cleanly(self, monkeypatch):
        import builtins
        import SMdRQA.ui as ui

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("streamlit"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(SystemExit, match="pip install SMdRQA"):
            ui.launch()
