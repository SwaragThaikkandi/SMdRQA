"""
SMdRQA interactive UI (Streamlit).

Run with::

    smdrqa-ui                       # console script
    python -m SMdRQA.ui             # module launcher
    streamlit run SMdRQA/ui/app.py  # directly

Every action performed in the UI is mirrored into a reproducible Python
script (see the "Script" tab), parameterised by the seed chosen in the
sidebar.
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from SMdRQA.RQA2 import RQA2, RQA2_ml, RQA2_simulators
from SMdRQA.ui.recorder import ScriptRecorder
from SMdRQA.ui.sensitivity import MEASURES, window_size_sensitivity
from SMdRQA.ui.simulate import (
    DISTRIBUTIONS, REGIMES, SYSTEM_PARAM_DEFAULTS,
    regime_threshold, sample_regime_values, simulate_signal,
)

st.set_page_config(page_title="SMdRQA", layout="wide")

SIMULATOR_SYSTEMS = ('rossler', 'lorenz', 'henon', 'chua', 'kuramoto',
                     'sine', 'white_noise')


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    ss = st.session_state
    ss.setdefault('seed', 42)
    if 'recorder' not in ss:
        ss.recorder = ScriptRecorder(ss.seed)
    ss.setdefault('signals', [])
    ss.setdefault('ids', [])
    ss.setdefault('labels', [])
    ss.setdefault('rqa_measures', None)
    ss.setdefault('rqa_windowed', None)
    ss.setdefault('sensitivity', None)
    ss.setdefault('features', None)
    ss.setdefault('ml_out', None)


def _seed_widget():
    ss = st.session_state
    seed = st.sidebar.number_input(
        "Random seed (required, recorded in script)",
        min_value=0, max_value=2**31 - 1, value=int(ss.seed), step=1)
    if seed != ss.seed:
        ss.seed = int(seed)
        ss.recorder.seed = int(seed)
        ss.recorder.record(
            f"SEED = {int(seed)}\nnp.random.seed(SEED)",
            comment="Seed changed in the UI")
    return int(seed)


def _df_download(df, label, filename, key):
    st.download_button(
        label, df.to_csv(index=True).encode(),
        file_name=filename, mime="text/csv", key=key)


def _fig_download(fig, label, filename, key):
    st.download_button(
        label, fig.to_html(include_plotlyjs='cdn').encode(),
        file_name=filename, mime="text/html", key=key)


# ---------------------------------------------------------------------------
# Tab 1 — Data
# ---------------------------------------------------------------------------

def _system_param_inputs(system, key_prefix):
    """Editable parameter inputs for a system; returns a params dict."""
    defaults = SYSTEM_PARAM_DEFAULTS.get(system, {})
    params = {}
    if not defaults:
        return params
    with st.expander("System parameters", expanded=False):
        cols = st.columns(min(len(defaults), 4))
        for i, (name, default) in enumerate(defaults.items()):
            if name == 'n_osc':
                params[name] = cols[i % 4].number_input(
                    name, 2, 200, int(default),
                    key=f"{key_prefix}_{name}")
            else:
                params[name] = cols[i % 4].number_input(
                    name, value=float(default), format="%.4f",
                    key=f"{key_prefix}_{name}")
    return params


def _distribution_inputs(side, threshold, key_prefix):
    """Distribution picker + parameters for one regime side."""
    dist = st.selectbox(
        f"Distribution ({side})", DISTRIBUTIONS,
        key=f"{key_prefix}_dist")
    if dist == 'uniform':
        low = st.number_input(
            "low", value=float(threshold * (0.5 if side == 'below'
                                            else 1.0)),
            format="%.4f", key=f"{key_prefix}_low")
        high = st.number_input(
            "high", value=float(threshold * (1.0 if side == 'below'
                                             else 1.5)),
            format="%.4f", key=f"{key_prefix}_high")
        return dist, {'low': low, 'high': high}
    if dist == 'normal':
        mean = st.number_input(
            "mean", value=float(threshold * (0.75 if side == 'below'
                                             else 1.25)),
            format="%.4f", key=f"{key_prefix}_mean")
        sd = st.number_input(
            "sd", value=float(abs(threshold) * 0.1 or 0.1),
            format="%.4f", key=f"{key_prefix}_sd")
        return dist, {'mean': mean, 'sd': sd}
    value = st.number_input(
        "value", value=float(threshold * (0.75 if side == 'below'
                                          else 1.25)),
        format="%.4f", key=f"{key_prefix}_value")
    return dist, {'value': value}


def tab_data(seed):
    ss = st.session_state
    st.subheader("Import or simulate signals")
    source = st.radio(
        "Data source", ("Load from folder", "Simulate"),
        horizontal=True)

    if source == "Load from folder":
        folder = st.text_input(
            "Folder containing .npy / .csv signal files")
        label_from_name = st.checkbox(
            "Derive label from filename prefix (text before first '-')",
            value=True)
        if st.button("Load folder") and folder:
            if not os.path.isdir(folder):
                st.error(f"Not a directory: {folder}")
            else:
                files = sorted(
                    f for f in os.listdir(folder)
                    if f.endswith(('.npy', '.csv')))
                if not files:
                    st.error("No .npy or .csv files found.")
                else:
                    signals, ids, labels = [], [], []
                    for fname in files:
                        path = os.path.join(folder, fname)
                        if fname.endswith('.npy'):
                            data = np.load(path)
                        else:
                            data = pd.read_csv(path).to_numpy()
                        signals.append(np.asarray(data, dtype=float))
                        ids.append(fname)
                        labels.append(
                            fname.split('-')[0] if label_from_name
                            else None)
                    ss.signals, ss.ids = signals, ids
                    ss.labels = (labels if label_from_name
                                 else [None] * len(signals))
                    ss.recorder.record(
                        f'''folder = r"{folder}"
files = sorted(f for f in os.listdir(folder)
               if f.endswith((".npy", ".csv")))
signals, ids, labels = [], [], []
for fname in files:
    path = os.path.join(folder, fname)
    data = (np.load(path) if fname.endswith(".npy")
            else pd.read_csv(path).to_numpy())
    signals.append(np.asarray(data, dtype=float))
    ids.append(fname)
    labels.append(fname.split("-")[0] if {label_from_name}
                  else None)''',
                        comment="Load signals from folder")
                    st.success(
                        f"Loaded {len(signals)} signals from {folder}")

    else:  # Simulate
        col1, col2, col3 = st.columns(3)
        system = col1.selectbox("System", SIMULATOR_SYSTEMS)
        length = col2.number_input("Signal length", 50, 20000, 1000)
        noise_sd = col3.number_input(
            "Additive noise SD", 0.0, 10.0, 0.0, step=0.05)

        params = _system_param_inputs(system, f"prm_{system}")

        osc_range = None
        if system == 'kuramoto':
            if st.checkbox("Sample oscillator count from a range"):
                o1, o2 = st.columns(2)
                osc_lo = o1.number_input("n_osc min", 2, 200, 5)
                osc_hi = o2.number_input("n_osc max", 2, 200, 20)
                osc_range = (int(osc_lo), int(max(osc_hi, osc_lo)))

        regime_info = REGIMES.get(system)
        mode_options = ["Fixed parameters"]
        if regime_info is not None:
            mode_options.append("Sample by regime")
        sim_mode = st.radio(
            "Parameter mode", mode_options, horizontal=True,
            key=f"simmode_{system}")

        if sim_mode == "Fixed parameters":
            f1, f2 = st.columns(2)
            n_signals = f1.number_input("Number of signals", 1, 500, 5)
            label = f2.text_input("Label for this batch", value=system)
            if st.button("Add simulated batch"):
                rng = np.random.default_rng(seed + len(ss.signals))
                sim = RQA2_simulators(seed=seed + len(ss.signals))
                for i in range(int(n_signals)):
                    p = dict(params)
                    if osc_range is not None:
                        p['n_osc'] = int(rng.integers(
                            osc_range[0], osc_range[1] + 1))
                    sig = simulate_signal(
                        sim, system, int(length), noise_sd, rng, **p)
                    ss.signals.append(sig)
                    ss.ids.append(f"{label}_{len(ss.signals) - 1}")
                    ss.labels.append(label)
                osc_code = ""
                if osc_range is not None:
                    osc_code = (f"\n    p['n_osc'] = int(rng.integers("
                                f"{osc_range[0]}, {osc_range[1]} + 1))")
                ss.recorder.record(
                    f'''rng = np.random.default_rng(SEED)
sim = RQA2_simulators(seed=SEED)
for i in range({int(n_signals)}):
    p = dict({params!r}){osc_code}
    sig = simulate_signal(sim, "{system}", {int(length)}, {noise_sd}, rng, **p)
    signals.append(sig)
    ids.append("{label}_" + str(len(signals) - 1))
    labels.append("{label}")''',
                    comment=f"Simulate {int(n_signals)} x {system} "
                            f"(fixed parameters)")
                st.success(f"Added {int(n_signals)} {system} signals")

        else:  # Sample by regime
            bif_param = regime_info['param']
            suggested = regime_threshold(system, params)
            st.info(
                f"**Bifurcation parameter: `{bif_param}`** — "
                f"suggested threshold ≈ **{suggested:.3f}**. "
                f"{regime_info['note']}")
            threshold = st.number_input(
                f"Regime threshold for {bif_param}",
                value=float(suggested), format="%.4f",
                key=f"thr_{system}")

            side_cfg = {}
            col_b, col_a = st.columns(2)
            for side, col, default_label in (
                    ('below', col_b, regime_info['below_label']),
                    ('above', col_a, regime_info['above_label'])):
                with col:
                    st.markdown(
                        f"**{side.capitalize()} threshold** "
                        f"({bif_param} {'<' if side == 'below' else '>'}"
                        f" {threshold:.3f})")
                    lab = st.text_input(
                        "Label", value=default_label,
                        key=f"lab_{system}_{side}")
                    n_sims = st.number_input(
                        "Simulations", 0, 500, 10,
                        key=f"n_{system}_{side}")
                    dist, dist_params = _distribution_inputs(
                        side, threshold, f"{system}_{side}")
                    side_cfg[side] = (lab, int(n_sims), dist,
                                      dist_params)

            if st.button("Generate regime-labelled batches"):
                rng = np.random.default_rng(seed + len(ss.signals))
                sim = RQA2_simulators(seed=seed + len(ss.signals))
                code_blocks = []
                total = 0
                progress = st.progress(0.0, text="Simulating…")
                n_total = sum(cfg[1] for cfg in side_cfg.values())
                for side, (lab, n_sims, dist, dist_params) in \
                        side_cfg.items():
                    if n_sims == 0:
                        continue
                    values = sample_regime_values(
                        dist, dist_params, n_sims, side, threshold,
                        rng)
                    for value in values:
                        p = dict(params)
                        p[bif_param] = float(value)
                        if osc_range is not None:
                            p['n_osc'] = int(rng.integers(
                                osc_range[0], osc_range[1] + 1))
                        sig = simulate_signal(
                            sim, system, int(length), noise_sd, rng,
                            **p)
                        ss.signals.append(sig)
                        ss.ids.append(
                            f"{lab}_{bif_param}={value:.3f}"
                            f"_{len(ss.signals) - 1}")
                        ss.labels.append(lab)
                        total += 1
                        progress.progress(
                            total / max(n_total, 1),
                            text=f"Simulating {lab} ({total}/{n_total})")
                    osc_code = ""
                    if osc_range is not None:
                        osc_code = (
                            f"\n    p['n_osc'] = int(rng.integers("
                            f"{osc_range[0]}, {osc_range[1]} + 1))")
                    code_blocks.append(f'''values = sample_regime_values(
    "{dist}", {dist_params!r}, {n_sims}, "{side}", {threshold}, rng)
for value in values:
    p = dict({params!r})
    p["{bif_param}"] = float(value){osc_code}
    sig = simulate_signal(sim, "{system}", {int(length)}, {noise_sd}, rng, **p)
    signals.append(sig)
    ids.append("{lab}_{bif_param}=" + f"{{value:.3f}}" + "_" + str(len(signals) - 1))
    labels.append("{lab}")''')
                progress.progress(1.0, text="Done")
                ss.recorder.record(
                    "rng = np.random.default_rng(SEED)\n"
                    "sim = RQA2_simulators(seed=SEED)\n"
                    + "\n\n".join(code_blocks),
                    comment=f"Regime-sampled {system} batches "
                            f"({bif_param} threshold {threshold})")
                st.success(
                    f"Added {total} {system} signals across "
                    f"{sum(1 for c in side_cfg.values() if c[1])} "
                    f"regimes")

    if ss.signals:
        st.markdown(
            f"**{len(ss.signals)} signals loaded.** Labels: "
            f"{sorted(set(str(x) for x in ss.labels))}")
        idx = st.selectbox(
            "Preview signal", range(len(ss.signals)),
            format_func=lambda i: ss.ids[i])
        sig = np.asarray(ss.signals[idx], dtype=float)
        n_dims = sig.shape[1] if sig.ndim > 1 else 1

        view_options = ["Time series"]
        if n_dims >= 3:
            view_options.insert(0, "3D phase portrait")
        elif n_dims == 2:
            view_options.insert(0, "2D phase portrait")
        view = st.radio("Preview type", view_options, horizontal=True,
                        key="preview_view")

        if view == "3D phase portrait":
            fig = go.Figure(go.Scatter3d(
                x=sig[:, 0], y=sig[:, 1], z=sig[:, 2], mode='lines',
                line=dict(width=2, color=np.arange(len(sig)),
                          colorscale='Viridis')))
            fig.update_layout(
                height=500, margin=dict(l=0, r=0, t=20, b=0),
                scene=dict(xaxis_title='dim 0', yaxis_title='dim 1',
                           zaxis_title='dim 2'))
        elif view == "2D phase portrait":
            fig = go.Figure(go.Scatter(
                x=sig[:, 0], y=sig[:, 1], mode='lines'))
            fig.update_layout(
                height=400, margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title='dim 0', yaxis_title='dim 1')
        else:
            fig = go.Figure()
            for d in range(min(n_dims, 5)):
                ys = sig[:, d] if sig.ndim > 1 else sig
                fig.add_trace(go.Scatter(y=ys, name=f"dim {d}",
                                         mode="lines"))
            fig.update_layout(height=300,
                              margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Clear all signals"):
            ss.signals, ss.ids, ss.labels = [], [], []
            ss.recorder.record(
                "signals, ids, labels = [], [], []",
                comment="Clear signals")
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — RQA analysis
# ---------------------------------------------------------------------------

def tab_rqa(seed):
    ss = st.session_state
    if not ss.signals:
        st.info("Load or simulate signals in the Data tab first.")
        return

    st.subheader("RQA parameters")
    c1, c2, c3, c4 = st.columns(4)
    normalize = c1.checkbox("Z-score normalise", value=True)
    reqrr = c2.number_input("Target recurrence rate", 0.01, 0.5, 0.10)
    lmin = c3.number_input("Minimum line length (lmin)", 1, 20, 2)
    manual = c4.checkbox("Manual tau/m/eps", value=False)
    manual_params = {}
    if manual:
        m1, m2, m3 = st.columns(3)
        manual_params['tau'] = m1.number_input("tau", 1, 100, 1)
        manual_params['m'] = m2.number_input("m", 1, 20, 2)
        manual_params['eps'] = m3.number_input(
            "eps", 0.001, 100.0, 0.3, format="%.3f")

    mode = st.radio(
        "Analysis mode", ("Single window (whole signal)",
                          "Sliding window"), horizontal=True)
    sliding = mode.startswith("Sliding")
    if sliding:
        w1, w2, w3 = st.columns(3)
        window_size = w1.number_input("Window size", 10, 5000, 100)
        window_step = w2.number_input("Window step", 1, 1000, 10)
        stats = w3.multiselect(
            "Central tendency", ('mean', 'median', 'mode'),
            default=['mean'])

    if st.button("Run RQA"):
        rqa_kwargs = {'normalize': normalize, 'reqrr': reqrr,
                      'lmin': int(lmin)}
        rows = []
        windowed_all = {}
        progress = st.progress(0.0, text="Computing RQA…")
        for i, sig in enumerate(ss.signals):
            progress.progress(
                i / len(ss.signals),
                text=f"Computing RQA: {ss.ids[i]}")
            rqa = RQA2(sig, normalize=normalize, reqrr=reqrr,
                       lmin=int(lmin))
            if manual:
                rqa._tau = int(manual_params['tau'])
                rqa._m = int(manual_params['m'])
                rqa._eps = float(manual_params['eps'])
            row = {'id': ss.ids[i], 'label': ss.labels[i]}
            row.update(rqa.compute_rqa_measures())
            row.update({'tau': rqa.tau, 'm': rqa.m, 'eps': rqa.eps})
            if sliding:
                wdf = rqa.compute_windowed_rqa_measures(
                    int(window_size), window_step=int(window_step))
                windowed_all[ss.ids[i]] = wdf
                summary = rqa.summarize_windowed_measures(
                    wdf, stats=tuple(stats))
                row.update({f"win_{k}": v for k, v in summary.items()})
            rows.append(row)
        progress.progress(1.0, text="Done")
        ss.rqa_measures = pd.DataFrame(rows)
        ss.rqa_windowed = windowed_all if sliding else None

        manual_code = ""
        if manual:
            manual_code = (
                f"\n    rqa._tau = {int(manual_params['tau'])}"
                f"\n    rqa._m = {int(manual_params['m'])}"
                f"\n    rqa._eps = {float(manual_params['eps'])}")
        sliding_code = ""
        if sliding:
            sliding_code = f'''
    windowed = rqa.compute_windowed_rqa_measures(
        {int(window_size)}, window_step={int(window_step)})
    summary = rqa.summarize_windowed_measures(
        windowed, stats={tuple(stats)})
    row.update({{f"win_{{k}}": v for k, v in summary.items()}})'''
        ss.recorder.record(
            f'''rows = []
for i, sig in enumerate(signals):
    rqa = RQA2(sig, normalize={normalize}, reqrr={reqrr}, lmin={int(lmin)}){manual_code}
    row = {{"id": ids[i], "label": labels[i]}}
    row.update(rqa.compute_rqa_measures())
    row.update({{"tau": rqa.tau, "m": rqa.m, "eps": rqa.eps}}){sliding_code}
    rows.append(row)
rqa_measures = pd.DataFrame(rows)''',
            comment=("Sliding-window RQA" if sliding
                     else "Whole-signal RQA"))

    if ss.rqa_measures is not None:
        st.subheader("Results")
        st.dataframe(ss.rqa_measures, use_container_width=True)
        _df_download(ss.rqa_measures, "Download measures CSV",
                     "rqa_measures.csv", "dl_meas")

        idx = st.selectbox(
            "Visualise signal", range(len(ss.signals)),
            format_func=lambda i: ss.ids[i], key="rqa_viz_sel")
        if st.button("Show recurrence plot"):
            rqa = RQA2(ss.signals[idx])
            rp = rqa.recurrence_plot
            fig = px.imshow(rp, color_continuous_scale='Greys',
                            origin='lower',
                            title=f"Recurrence plot — {ss.ids[idx]}")
            fig.update_layout(coloraxis_showscale=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            _fig_download(fig, "Download RP (interactive HTML)",
                          "recurrence_plot.html", "dl_rp")

        if ss.rqa_windowed and ss.ids[idx] in ss.rqa_windowed:
            wdf = ss.rqa_windowed[ss.ids[idx]]
            cols = st.multiselect(
                "Windowed measures to plot", list(wdf.columns),
                default=['determinism', 'laminarity'])
            if cols:
                fig = px.line(
                    wdf.reset_index(), x='window_start', y=cols,
                    title=f"Windowed RQA — {ss.ids[idx]}")
                st.plotly_chart(fig, use_container_width=True)
                _fig_download(fig, "Download windowed plot (HTML)",
                              "windowed_measures.html", "dl_win")
                _df_download(wdf, "Download windowed CSV",
                             "windowed_measures.csv", "dl_windf")


# ---------------------------------------------------------------------------
# Tab 3 — Window-size sensitivity
# ---------------------------------------------------------------------------

def tab_sensitivity(seed):
    ss = st.session_state
    if not ss.signals:
        st.info("Load or simulate signals in the Data tab first.")
        return

    st.subheader("Window-size sensitivity (bootstrap CI width)")
    st.caption(
        "Smaller CI width = more stable estimate at that window size. "
        "Bootstrap re-implementation of SMdRQA.window_size, seeded for "
        "reproducibility.")
    c1, c2, c3 = st.columns(3)
    idx = c1.selectbox(
        "Signal", range(len(ss.signals)),
        format_func=lambda i: ss.ids[i], key="sens_sel")
    measure = c2.selectbox("Measure", sorted(MEASURES))
    n_boot = c3.number_input("Bootstrap samples", 100, 10000, 1000)
    c4, c5, c6 = st.columns(3)
    min_size = c4.number_input("Min window size", 10, 1000, 20)
    max_size = c5.number_input(
        "Max window size (0 = RP size)", 0, 5000, 0)
    step = c6.number_input("Step", 1, 500, 10)

    if st.button("Run sensitivity analysis"):
        rqa = RQA2(ss.signals[idx])
        rp = rqa.recurrence_plot
        progress = st.progress(0.0)

        def cb(i, total, winsize):
            progress.progress(
                i / max(total, 1),
                text=f"Window size {winsize}")

        df = window_size_sensitivity(
            rp, measure,
            min_size=int(min_size),
            max_size=(None if max_size == 0 else int(max_size)),
            step=int(step), n_boot=int(n_boot), seed=seed,
            progress_callback=cb)
        progress.progress(1.0, text="Done")
        ss.sensitivity = (ss.ids[idx], measure, df)
        max_code = 'None' if max_size == 0 else int(max_size)
        ss.recorder.record(
            f'''from SMdRQA.ui.sensitivity import window_size_sensitivity
rqa = RQA2(signals[{idx}])
sensitivity = window_size_sensitivity(
    rqa.recurrence_plot, "{measure}",
    min_size={int(min_size)}, max_size={max_code},
    step={int(step)}, n_boot={int(n_boot)}, seed=SEED)''',
            comment=f"Window-size sensitivity ({measure})")

    if ss.sensitivity is not None:
        sig_id, measure, df = ss.sensitivity
        fig = px.line(
            df, x='window_size', y='ci_width', markers=True,
            title=f"Sensitivity of {measure} — {sig_id}",
            labels={'ci_width': '95% − 5% quantile width'})
        st.plotly_chart(fig, use_container_width=True)
        _fig_download(fig, "Download plot (HTML)",
                      "sensitivity.html", "dl_sens_fig")
        _df_download(df, "Download CSV", "sensitivity.csv", "dl_sens")


# ---------------------------------------------------------------------------
# Tab 4 — Machine learning
# ---------------------------------------------------------------------------

def tab_ml(seed):
    ss = st.session_state
    st.subheader("Statistical / machine-learning analysis")

    source = st.radio(
        "Feature source",
        ("Build from loaded signals", "Upload feature CSV"),
        horizontal=True)
    if source == "Upload feature CSV":
        up = st.file_uploader(
            "Feature table (needs a 'label' column)", type="csv")
        if up is not None:
            ss.features = pd.read_csv(up)
            ss.recorder.record(
                f'features = pd.read_csv(r"{up.name}")',
                comment="Load precomputed feature table")
            st.dataframe(ss.features.head(), use_container_width=True)
    else:
        if not ss.signals:
            st.info("Load or simulate signals in the Data tab first.")
            return
        if any(lab is None for lab in ss.labels):
            st.warning(
                "Signals have no labels — supervised analysis needs "
                "labels (set them in the Data tab).")
            return
        f1, f2 = st.columns(2)
        window_size = f1.number_input(
            "Feature window size", 10, 5000, 100, key="ml_ws")
        window_step = f2.number_input(
            "Feature window step", 1, 1000, 10, key="ml_step")

    st.markdown("**Benchmark settings**")
    b1, b2, b3 = st.columns(3)
    models = b1.multiselect(
        "Models", sorted(RQA2_ml._MODEL_REGISTRY),
        default=['knn', 'svm', 'rf', 'logreg'])
    tune = b2.checkbox("Tune hyperparameters (nested)", value=True)
    fsel = b3.selectbox(
        "Feature selection", ('auto', 'forward', 'exhaustive', 'None'))
    b4, b5, b6 = st.columns(3)
    outer_iterations = b4.number_input("Outer iterations", 5, 500, 50)
    inner_splits = b5.number_input("Inner splits", 2, 10, 2)
    inner_iterations = b6.number_input("Inner repeats", 1, 50, 5)

    if st.button("Run benchmark") and models:
        ml = RQA2_ml()
        progress = st.progress(0.0, text="Starting…")

        def cb(i, total, name):
            progress.progress(i / total, text=f"Nested CV: {name}")

        kwargs = dict(
            models=tuple(models),
            tune=tune,
            feature_selection=(None if fsel == 'None' else fsel),
            outer_iterations=int(outer_iterations),
            inner_splits=int(inner_splits),
            inner_iterations=int(inner_iterations),
            random_state=seed, verbose=False,
            progress_callback=cb)
        if source == "Upload feature CSV":
            if ss.features is None:
                st.error("Upload a feature CSV first.")
                return
            out = ml.integrated_benchmark(features=ss.features, **kwargs)
            feat_code = "features=features,"
            build_comment = ""
        else:
            out = ml.integrated_benchmark(
                ss.signals, list(ss.labels),
                window_size=int(window_size),
                window_step=int(window_step), **kwargs)
            feat_code = (f"signals, labels,\n    "
                         f"window_size={int(window_size)}, "
                         f"window_step={int(window_step)},")
            build_comment = " (features built from signals)"
        progress.progress(1.0, text="Done")
        ss.ml_out = out
        fsel_code = 'None' if fsel == 'None' else f'"{fsel}"'
        ss.recorder.record(
            f'''ml = RQA2_ml()
ml_out = ml.integrated_benchmark(
    {feat_code}
    models={tuple(models)},
    tune={tune}, feature_selection={fsel_code},
    outer_iterations={int(outer_iterations)},
    inner_splits={int(inner_splits)},
    inner_iterations={int(inner_iterations)},
    random_state=SEED)
print(ml_out["comparison"])
print(ml_out["pairwise_tests"])''',
            comment=f"Integrated ML benchmark{build_comment}")

    out = ss.ml_out
    if out is not None:
        st.subheader(f"Best model: `{out['best_model_name']}`")
        st.dataframe(out['comparison'], use_container_width=True)
        _df_download(out['comparison'], "Download comparison CSV",
                     "ml_comparison.csv", "dl_cmp")

        long_rows = []
        for name, res in out['results'].items():
            for v in res['accuracy']:
                long_rows.append({'model': name, 'accuracy': v})
        fig = px.box(
            pd.DataFrame(long_rows), x='model', y='accuracy',
            points='all', title="Outer-fold accuracy distributions")
        st.plotly_chart(fig, use_container_width=True)
        _fig_download(fig, "Download box plot (HTML)",
                      "ml_accuracy.html", "dl_mlfig")

        freq = out['results'][out['best_model_name']][
            'feature_frequency']
        fig2 = px.bar(
            freq.reset_index().rename(
                columns={'index': 'feature', 0: 'count'}),
            x='feature', y='count',
            title=f"Feature selection frequency "
                  f"({out['best_model_name']})")
        st.plotly_chart(fig2, use_container_width=True)

        if len(out['pairwise_tests']):
            st.markdown("**Pairwise Wilcoxon tests (BH-corrected)**")
            st.dataframe(out['pairwise_tests'],
                         use_container_width=True)
            _df_download(out['pairwise_tests'],
                         "Download pairwise tests CSV",
                         "ml_pairwise.csv", "dl_pw")
        _df_download(out['features'], "Download feature table CSV",
                     "ml_features.csv", "dl_feat")


# ---------------------------------------------------------------------------
# Tab 5 — Script
# ---------------------------------------------------------------------------

def tab_script():
    ss = st.session_state
    st.subheader("Reproducibility script")
    st.caption(
        "Every UI action is mirrored here as plain Python. Download and "
        "rerun to reproduce the whole session (seed included).")
    script = ss.recorder.script()
    st.code(script, language='python')
    st.download_button(
        "Download script (.py)", script.encode(),
        file_name="smdrqa_session.py", mime="text/x-python")
    if st.button("Clear recorded steps"):
        ss.recorder.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()
    st.title("SMdRQA — Recurrence Quantification Analysis")
    seed = _seed_widget()
    sidebar_status = st.sidebar.empty()
    tabs = st.tabs(
        ["1 · Data", "2 · RQA", "3 · Window-size sensitivity",
         "4 · Machine learning", "5 · Script"])
    with tabs[0]:
        tab_data(seed)
    with tabs[1]:
        tab_rqa(seed)
    with tabs[2]:
        tab_sensitivity(seed)
    with tabs[3]:
        tab_ml(seed)
    with tabs[4]:
        tab_script()
    # Filled last so the count reflects actions taken this rerun
    sidebar_status.markdown(
        f"**Signals loaded:** {len(st.session_state.signals)}")


if __name__ == "__main__" or st.runtime.exists():
    main()
