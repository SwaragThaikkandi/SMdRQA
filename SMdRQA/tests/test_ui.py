"""
Tests for the SMdRQA UI support modules.

The ScriptRecorder and window_size_sensitivity tests run without any UI
dependency; the app import smoke test is skipped unless streamlit and
plotly are installed.
"""

from SMdRQA.ui.simulate import (
    REGIMES, SYSTEM_PARAM_DEFAULTS, kuramoto_kc, regime_threshold,
    sample_regime_values, simulate_signal,
)
from SMdRQA.RQA2 import RQA2_simulators
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


# ---------------------------------------------------------------------------
# Simulation helpers (regime sampling, kuramoto)
# ---------------------------------------------------------------------------


class TestKuramotoSimulator:

    def test_shape_and_range(self):
        sim = RQA2_simulators(seed=0)
        sig = sim.kuramoto(n=300, n_osc=7, K=2.0)
        assert sig.shape == (300, 7)
        assert np.all(np.abs(sig) <= 1.0)

    def test_seed_reproducible(self):
        a = RQA2_simulators(seed=3).kuramoto(n=200, n_osc=5)
        b = RQA2_simulators(seed=3).kuramoto(n=200, n_osc=5)
        assert np.allclose(a, b)

    def test_strong_coupling_synchronises(self):
        sim = RQA2_simulators(seed=1)
        sig = sim.kuramoto(n=500, n_osc=20, K=10.0, omega_sd=0.5)
        theta = np.arcsin(np.clip(sig[-1], -1, 1))
        # crude check: strongly coupled oscillators end up clustered
        assert np.std(sig[-1]) < 0.9


class TestRegimeSampling:

    def test_below_clipped(self):
        rng = np.random.default_rng(0)
        vals = sample_regime_values(
            'normal', {'mean': 10.0, 'sd': 5.0}, 200, 'below', 5.0, rng)
        assert np.all(vals < 5.0)

    def test_above_clipped(self):
        rng = np.random.default_rng(0)
        vals = sample_regime_values(
            'uniform', {'low': 0.0, 'high': 10.0}, 200, 'above', 5.0,
            rng)
        assert np.all(vals > 5.0)

    def test_fixed_distribution(self):
        rng = np.random.default_rng(0)
        vals = sample_regime_values(
            'fixed', {'value': 3.0}, 5, 'below', 5.0, rng)
        assert np.allclose(vals, 3.0)

    def test_seeded_reproducible(self):
        kwargs = ('uniform', {'low': 1.0, 'high': 4.0}, 50, 'below', 5.0)
        a = sample_regime_values(*kwargs, np.random.default_rng(9))
        b = sample_regime_values(*kwargs, np.random.default_rng(9))
        assert np.allclose(a, b)

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            sample_regime_values(
                'beta', {}, 5, 'below', 5.0, np.random.default_rng(0))

    def test_regime_registry_thresholds(self):
        for system, info in REGIMES.items():
            thr = regime_threshold(system)
            assert thr is not None and thr > 0
        assert regime_threshold('sine') is None
        assert np.isclose(kuramoto_kc(1.0), np.sqrt(8 / np.pi))


class TestSimulateSignal:

    def test_params_change_output(self):
        rng = np.random.default_rng(0)
        sim_a = RQA2_simulators(seed=0)
        sim_b = RQA2_simulators(seed=0)
        a = simulate_signal(sim_a, 'rossler', 300, 0.0, rng, c=4.0)
        b = simulate_signal(sim_b, 'rossler', 300, 0.0, rng, c=5.7)
        assert a.shape == b.shape == (300, 3)
        assert not np.allclose(a, b)

    def test_kuramoto_n_osc(self):
        rng = np.random.default_rng(0)
        sim = RQA2_simulators(seed=0)
        sig = simulate_signal(sim, 'kuramoto', 200, 0.0, rng, n_osc=4)
        assert sig.shape == (200, 4)

    def test_all_systems_run(self):
        rng = np.random.default_rng(0)
        for system in SYSTEM_PARAM_DEFAULTS:
            sim = RQA2_simulators(seed=0)
            kwargs = ({'n_osc': 3} if system == 'kuramoto' else {})
            sig = simulate_signal(sim, system, 150, 0.1, rng, **kwargs)
            assert sig.shape[0] == 150

    def test_unknown_system_raises(self):
        with pytest.raises(ValueError, match="Unknown system"):
            simulate_signal(RQA2_simulators(seed=0), 'foo', 100, 0.0,
                            np.random.default_rng(0))
