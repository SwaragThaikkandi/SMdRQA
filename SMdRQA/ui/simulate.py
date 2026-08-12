"""
Simulation helpers for the SMdRQA UI.

Provides editable per-system parameter defaults (matching the
``RQA2_simulators`` signatures), a registry of bifurcation-parameter
regimes with suggested chaos thresholds, distribution-based parameter
sampling for regime-labelled batch generation, and a standalone
``simulate_signal`` used both by the UI and by the exported
reproducibility scripts.
"""

import numpy as np

from SMdRQA.RQA2 import RQA2_simulators

#: Editable parameters per system, mirroring RQA2_simulators defaults.
SYSTEM_PARAM_DEFAULTS = {
    'rossler': {'a': 0.2, 'b': 0.2, 'c': 5.7},
    'lorenz': {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0},
    'henon': {'a': 1.4, 'b': 0.3},
    'chua': {'alpha': 15.6, 'beta': 28.0, 'm0': -1.143, 'm1': -0.714},
    'kuramoto': {'K': 1.0, 'omega_sd': 1.0, 'n_osc': 10},
    'sine': {},
    'white_noise': {},
}

#: Bifurcation-parameter metadata per system.  ``threshold`` is the
#: suggested value separating the two dynamical regimes (holding the
#: other parameters at their defaults); thresholds are approximate
#: guides taken from the standard literature, not exact bifurcation
#: points.
REGIMES = {
    'rossler': {
        'param': 'c',
        'threshold': 4.2,
        'below_label': 'periodic',
        'above_label': 'chaotic',
        'note': ("With a=b=0.2 the Rössler system period-doubles into "
                 "chaos as c grows: c≈3.5 gives a simple limit cycle, "
                 "chaos onsets near c≈4.2, and c=5.7 is the canonical "
                 "chaotic attractor."),
    },
    'lorenz': {
        'param': 'rho',
        'threshold': 24.74,
        'below_label': 'fixed_point',
        'above_label': 'chaotic',
        'note': ("With sigma=10, beta=8/3 the Lorenz fixed points lose "
                 "stability at rho≈24.74; rho=28 is the canonical "
                 "chaotic attractor."),
    },
    'henon': {
        'param': 'a',
        'threshold': 1.06,
        'below_label': 'periodic',
        'above_label': 'chaotic',
        'note': ("With b=0.3 the Hénon map is largely periodic below "
                 "a≈1.06 and chaotic (with periodic windows) up to the "
                 "canonical a=1.4."),
    },
    'chua': {
        'param': 'alpha',
        'threshold': 8.8,
        'below_label': 'periodic',
        'above_label': 'chaotic',
        'note': ("With the default beta/m0/m1, Chua's circuit shows "
                 "limit cycles at low alpha and double-scroll chaos "
                 "around the canonical alpha=15.6; alpha≈8.8 is an "
                 "approximate transition guide."),
    },
    'kuramoto': {
        'param': 'K',
        'threshold': None,  # depends on omega_sd; use kuramoto_kc()
        'below_label': 'incoherent',
        'above_label': 'synchronized',
        'note': ("For Gaussian natural frequencies the critical "
                 "coupling is K_c = omega_sd·sqrt(8/pi) ≈ "
                 "1.596·omega_sd: incoherent below, synchronised "
                 "above."),
    },
}

DISTRIBUTIONS = ('uniform', 'normal', 'fixed')


def kuramoto_kc(omega_sd):
    """Critical Kuramoto coupling for Gaussian frequencies."""
    return float(omega_sd) * np.sqrt(8.0 / np.pi)


def regime_threshold(system, params=None):
    """Suggested regime threshold for *system* (None if no regime)."""
    info = REGIMES.get(system)
    if info is None:
        return None
    if system == 'kuramoto':
        omega_sd = (params or {}).get(
            'omega_sd', SYSTEM_PARAM_DEFAULTS['kuramoto']['omega_sd'])
        return kuramoto_kc(omega_sd)
    return info['threshold']


def sample_regime_values(distribution, dist_params, n, side, threshold,
                         rng):
    """Draw *n* bifurcation-parameter values on one side of *threshold*.

    Parameters
    ----------
    distribution : {'uniform', 'normal', 'fixed'}
    dist_params : dict
        ``{'low', 'high'}`` for uniform, ``{'mean', 'sd'}`` for normal,
        ``{'value'}`` for fixed.
    n : int
        Number of samples.
    side : {'below', 'above'}
        Which side of *threshold* the samples must fall on; draws are
        clipped to that side so regime labels stay truthful.
    threshold : float
        Regime boundary.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of shape (n,)
    """
    if distribution == 'uniform':
        values = rng.uniform(dist_params['low'], dist_params['high'], n)
    elif distribution == 'normal':
        values = rng.normal(dist_params['mean'], dist_params['sd'], n)
    elif distribution == 'fixed':
        values = np.full(n, float(dist_params['value']))
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {DISTRIBUTIONS}")

    margin = max(abs(threshold) * 1e-3, 1e-6)
    if side == 'below':
        return np.minimum(values, threshold - margin)
    if side == 'above':
        return np.maximum(values, threshold + margin)
    raise ValueError("side must be 'below' or 'above'.")


def simulate_signal(sim, system, length, noise_sd, rng, **params):
    """Simulate one signal from *system* with explicit parameters.

    Parameters
    ----------
    sim : RQA2_simulators
        Seeded simulator instance.
    system : str
        One of ``SYSTEM_PARAM_DEFAULTS``.
    length : int
        Number of samples.
    noise_sd : float
        SD of additive Gaussian observation noise (0 disables).
    rng : numpy.random.Generator
        RNG for the observation noise.
    **params
        System parameters overriding the defaults (e.g. ``c=4.0`` for
        Rössler, ``K=2.0, n_osc=15`` for Kuramoto).

    Returns
    -------
    ndarray
        1-D for sine/white noise, else (length, n_dims).
    """
    merged = dict(SYSTEM_PARAM_DEFAULTS.get(system, {}))
    merged.update(params)

    if system == 'rossler':
        x, y, z = sim.rossler(n=length, **merged)
        sig = np.column_stack([x, y, z])
    elif system == 'lorenz':
        x, y, z = sim.lorenz(n=length, **merged)
        sig = np.column_stack([x, y, z])
    elif system == 'henon':
        x, y = sim.henon(n=length, **merged)
        sig = np.column_stack([x, y])
    elif system == 'chua':
        x, y, z = sim.chua(n=length, **merged)
        sig = np.column_stack([x, y, z])
    elif system == 'kuramoto':
        sig = sim.kuramoto(n=length, **merged)
    elif system == 'sine':
        t = np.linspace(0, 8 * np.pi, length)
        sig = np.sin(t)
    elif system == 'white_noise':
        sig = rng.standard_normal(length)
    else:
        raise ValueError(f"Unknown system '{system}'.")

    if noise_sd > 0:
        sig = sig + noise_sd * rng.standard_normal(sig.shape)
    return sig
