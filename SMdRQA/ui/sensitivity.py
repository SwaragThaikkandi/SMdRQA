"""
Window-size sensitivity analysis for the SMdRQA UI.

Seeded, vectorised re-implementation of the bootstrap procedure in
``SMdRQA.window_size``: for each candidate window size, line-length
distributions are pooled over all diagonal sub-windows of the recurrence
plot, resampled ``n_boot`` times, and the width of the 5–95 % quantile
interval of the chosen RQA measure is reported.  Narrow intervals mean
the measure is stable at that window size.
"""

import numpy as np
import pandas as pd

from SMdRQA.RQA2 import RQA2

#: measure name -> (histogram kind, statistic)
MEASURES = {
    'percent_det': ('diag', 'percent'),
    'percent_lam': ('vert', 'percent'),
    'avg_diag': ('diag', 'average'),
    'avg_vert': ('vert', 'average'),
}


def _pooled_line_distribution(rp, winsize, kind):
    """Pool the line-length histogram over all diagonal sub-windows."""
    helper = RQA2.__new__(RQA2)
    n = rp.shape[0]
    num_windows = n - winsize + 1
    pooled = np.zeros(winsize + 1)
    total = 0.0
    for i in range(num_windows):
        sub = rp[i:i + winsize, i:i + winsize]
        if kind == 'diag':
            hist = helper._diaghist(sub, winsize)
        else:
            hist = helper._vert_hist(sub, winsize)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            pooled += hist / hist_sum
        total += hist_sum
    n_bar = int(total / num_windows)
    pooled_sum = np.sum(pooled)
    if pooled_sum > 0:
        pooled = pooled / pooled_sum
    return pooled, max(n_bar, 1)


def _bootstrap_ci(pooled, n_bar, statistic, n_boot, rng):
    """5–95 % quantile width of the bootstrapped measure."""
    midpoints = np.arange(1, len(pooled) + 1)
    cdf = np.cumsum(pooled)
    cdf = cdf / cdf[-1]

    values = []
    for _ in range(n_boot):
        draws = midpoints[np.searchsorted(cdf, rng.random(n_bar))]
        hist, _ = np.histogram(
            draws, bins=np.concatenate([midpoints - 0.5,
                                        [midpoints[-1] + 0.5]]))
        hist = hist / max(np.sum(hist), 1)
        if statistic == 'percent':
            values.append(np.sum(hist[1:]) / max(np.sum(hist), 1e-12))
        else:
            values.append(np.mean(draws[draws > 1])
                          if np.any(draws > 1) else 0.0)
    return float(np.quantile(values, 0.95) - np.quantile(values, 0.05))


def window_size_sensitivity(rp, measure, *, min_size=20, max_size=None,
                            step=10, n_boot=1000, seed=42,
                            progress_callback=None):
    """Bootstrap CI width of an RQA measure across window sizes.

    Parameters
    ----------
    rp : ndarray
        Recurrence plot (square 0/1 matrix).
    measure : {'percent_det', 'percent_lam', 'avg_diag', 'avg_vert'}
        RQA measure to analyse.
    min_size : int, default 20
        Smallest window size tested.
    max_size : int, optional
        Largest window size tested (exclusive).  Defaults to RP size.
    step : int, default 10
        Window-size increment.
    n_boot : int, default 1000
        Number of bootstrap samples per window size.
    seed : int, default 42
        Seed for the bootstrap RNG (the legacy implementation in
        ``SMdRQA.window_size`` is unseeded).
    progress_callback : callable, optional
        Called as ``progress_callback(index, total, window_size)``.

    Returns
    -------
    pandas.DataFrame
        Columns ``window_size`` and ``ci_width``
        (95 % quantile − 5 % quantile).
    """
    if measure not in MEASURES:
        raise ValueError(
            f"Unknown measure '{measure}'. "
            f"Available: {sorted(MEASURES)}")
    kind, statistic = MEASURES[measure]

    rp = np.asarray(rp)
    if max_size is None:
        max_size = rp.shape[0]

    rng = np.random.default_rng(seed)
    sizes = list(range(min_size, max_size, step))
    rows = []
    for idx, winsize in enumerate(sizes):
        if progress_callback is not None:
            progress_callback(idx, len(sizes), winsize)
        pooled, n_bar = _pooled_line_distribution(rp, winsize, kind)
        ci = _bootstrap_ci(pooled, n_bar, statistic, n_boot, rng)
        rows.append({'window_size': winsize, 'ci_width': ci})

    return pd.DataFrame(rows)
