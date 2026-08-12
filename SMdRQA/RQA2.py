"""
RQA2 – Object-oriented Recurrence Quantification Analysis for SMdRQA.

This module provides four classes:

``RQA2``
    End-to-end RQA: data loading, parameter estimation (τ, m, ε), recurrence-plot
    generation, measure computation, visualisation, and batch processing.

``RQA2_simulators``
    Generators for well-known chaotic dynamical systems (Rössler, Lorenz, Hénon,
    Chua) used for testing and benchmarking.

``RQA2_tests``
    Surrogate-data generation (FT, AAFT, IAAFT, IDFS, WIAAFT, PPS) and
    statistical validation of nonlinear dynamics metrics.

``RQA2_ml``
    Machine learning utilities for building RQA feature tables and benchmarking
    supervised classifiers and unsupervised clustering methods.
"""

from __future__ import annotations

# Standard library
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

# Third-party – numerics
import numpy as np
import pandas as pd
import pywt
import scipy.fft as _fft
import scipy.stats as stats
from scipy.integrate import solve_ivp
from scipy.spatial import distance
from scipy.special import digamma
from sklearn.metrics import (
    mean_squared_error, accuracy_score, f1_score, silhouette_score,
    roc_auc_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, confusion_matrix, balanced_accuracy_score,
)
from sklearn.model_selection import (
    RepeatedKFold, StratifiedKFold, StratifiedShuffleSplit,
    RepeatedStratifiedKFold, StratifiedGroupKFold, ParameterGrid,
)
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from itertools import combinations

# Third-party – visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Third-party – progress
from tqdm import tqdm


class RQA2:
    """
    Comprehensive Recurrence Quantification Analysis class that handles all RQA computations,
    visualizations, and batch processing in an object-oriented manner.

    Fixed version with proper 0-based indexing throughout.
    """

    def __init__(self, data=None, normalize=True, **kwargs):
        """
        Initialize an RQA2 analysis object.

        Parameters
        ----------
        data : array_like, optional
            Time-series data of shape ``(N,)`` or ``(N, D)``.  When omitted,
            call :meth:`load_data` before accessing any computed property.
        normalize : bool, default ``True``
            Apply z-score (zero-mean, unit-variance) normalisation column-wise.
        **kwargs
            Override any default configuration parameter.  Recognised keys:

            rdiv : int, default 451
                Number of candidate *r* values tested in the FNN search.
            Rmin, Rmax : float, default 1 and 10
                Search range for the FNN ratio threshold.
            delta : float, default 0.001
                Convergence tolerance for the FNN ratio.
            bound : float, default 0.2
                Minimum drop in FNN ratio required to select an embedding
                dimension.
            reqrr : float, default 0.10
                Target recurrence rate (0 < reqrr < 1).
            rr_delta : float, default 0.005
                Tolerance around *reqrr* when searching for ε.
            epsmin, epsmax : float, default 0 and 10
                Search range for the neighbourhood radius ε.
            epsdiv : int, default 1001
                Number of candidate ε values.
            mi_method : {'histdd', 'avg'}, default 'histdd'
                Mutual-information estimator used for τ selection.
            tau_method : {'default', 'polynomial'}, default 'default'
                Strategy for choosing the optimal time delay.
            lmin : int, default 2
                Minimum line length for DET and LAM computation.
        """
        # Data properties
        self.data = None
        self.original_data = None
        self.n_samples = 0
        self.n_dimensions = 0

        # Computed parameters
        self._tau = None
        self._m = None
        self._eps = None
        self._recurrence_plot = None
        self._embedded_signal = None
        self._rqa_measures = {}

        # Configuration parameters with defaults
        self.config = {
            'rdiv': kwargs.get('rdiv', 451),
            'Rmin': kwargs.get('Rmin', 1),
            'Rmax': kwargs.get('Rmax', 10),
            'delta': kwargs.get('delta', 0.001),
            'bound': kwargs.get('bound', 0.2),
            'reqrr': kwargs.get('reqrr', 0.1),
            'rr_delta': kwargs.get('rr_delta', 0.005),
            'epsmin': kwargs.get('epsmin', 0),
            'epsmax': kwargs.get('epsmax', 10),
            'epsdiv': kwargs.get('epsdiv', 1001),
            'mi_method': kwargs.get('mi_method', 'histdd'),
            'tau_method': kwargs.get('tau_method', 'default'),
            'lmin': kwargs.get('lmin', 2)
        }

        if data is not None:
            self.load_data(data, normalize)

    # Data handling methods
    def load_data(self, data, normalize=True):
        """
        Load (and optionally normalise) time-series data, resetting all caches.

        Parameters
        ----------
        data : array_like
            Time-series array of shape ``(N,)`` or ``(N, D)``.
        normalize : bool, default ``True``
            Apply z-score normalisation column-wise.

        Returns
        -------
        None
        """
        self.original_data = np.array(data)
        if self.original_data.ndim == 1:
            self.original_data = self.original_data.reshape(-1, 1)

        self.n_samples, self.n_dimensions = self.original_data.shape

        if normalize:
            self.data = (self.original_data - np.mean(self.original_data,
                                                      axis=0,
                                                      keepdims=True)) / (np.std(self.original_data,
                                                                                axis=0,
                                                                                keepdims=True) + 1e-10)
        else:
            self.data = self.original_data.copy()

        # Reset computed values
        self._reset_computed_values()

    def _reset_computed_values(self):
        """Reset all computed values when new data is loaded."""
        self._tau = None
        self._m = None
        self._eps = None
        self._recurrence_plot = None
        self._embedded_signal = None
        self._rqa_measures = {}

    def _embedded_length(self, m, tau):
        """Number of valid delay-vectors for (m, τ) - FIXED indexing."""
        return max(0, self.n_samples - (m - 1) * tau)

    # Properties for computed values
    @property
    def tau(self):
        """Time delay parameter."""
        if self._tau is None:
            self._tau = self.compute_time_delay()
        return int(self._tau)

    @property
    def m(self):
        """Embedding dimension."""
        if self._m is None:
            self._m = self.compute_embedding_dimension()
        return int(self._m)

    @property
    def eps(self):
        """Neighborhood radius."""
        if self._eps is None:
            self._eps = self.compute_neighborhood_radius()
        return float(self._eps)

    @property
    def recurrence_plot(self):
        """Recurrence plot matrix."""
        if self._recurrence_plot is None:
            self._recurrence_plot = self.compute_recurrence_plot()
        return self._recurrence_plot

    @property
    def embedded_signal(self):
        """Time-delayed embedded signal."""
        if self._embedded_signal is None:
            self._embedded_signal = self.compute_embedded_signal()
        return self._embedded_signal

    @property
    def recurrence_rate(self):
        """Recurrence rate of the RP."""
        if self._recurrence_plot is not None:
            n = self._recurrence_plot.shape[0]
            return float(np.sum(self._recurrence_plot)) / (n * n)
        return None

    # Core computation methods
    def compute_time_delay(self, method=None, mi_method=None):
        """
        Estimate the optimal time delay τ from the mutual-information curve.

        Parameters
        ----------
        method : {'default', 'polynomial'}, optional
            Algorithm used to locate the minimum of the MI curve.
            ``'default'`` selects the first local minimum; ``'polynomial'``
            fits a cross-validated polynomial and uses its first minimum.
            Defaults to ``self.config['tau_method']``.
        mi_method : {'histdd', 'avg'}, optional
            Mutual-information estimator.  Defaults to
            ``self.config['mi_method']``.

        Returns
        -------
        int
            Optimal time delay τ ≥ 1.

        Raises
        ------
        ValueError
            If no data have been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        method = method or self.config['tau_method']
        mi_method = mi_method or self.config['mi_method']

        if method == 'default':
            tau = self._findtau_default(mi_method)
        elif method == 'polynomial':
            tau = self._findtau_polynomial(mi_method)
        else:
            raise ValueError("Method must be 'default' or 'polynomial'")

        self._tau = int(tau)
        return self._tau

    def compute_embedding_dimension(self):
        """
        Estimate the optimal embedding dimension m via False Nearest Neighbours.

        Uses the FNN criterion: the embedding dimension is the smallest m for
        which the fraction of false nearest neighbours drops below
        ``self.config['bound']``.

        Returns
        -------
        int
            Embedding dimension m ≥ 1.

        Raises
        ------
        ValueError
            If no data have been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        tau = self.tau
        sd = 3 * np.std(self.data)

        m = self._findm(tau, sd)
        self._m = int(m)
        return self._m

    def compute_neighborhood_radius(self, reqrr=None):
        """
        Find the neighbourhood radius ε that achieves a target recurrence rate.

        The search scans ``self.config['epsdiv']`` evenly-spaced candidate
        values between ``epsmin`` and ``epsmax`` and returns the first ε for
        which |RR − reqrr| < ``rr_delta``.

        Parameters
        ----------
        reqrr : float, optional
            Target recurrence rate in (0, 1).  Clamped to [0.01, 0.99].
            Defaults to ``self.config['reqrr']``.

        Returns
        -------
        float
            Neighbourhood radius ε > 0.

        Raises
        ------
        ValueError
            If no data have been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        reqrr = reqrr or self.config['reqrr']
        reqrr = max(0.01, min(0.99, reqrr))
        tau = self.tau
        m = self.m

        eps = self._findeps(tau, m, reqrr)
        self._eps = float(eps)
        return self._eps

    def compute_recurrence_plot(self):
        """
        Build the binary recurrence plot matrix from the current parameters.

        Uses the delay-embedded signal with the stored (or lazily computed)
        τ, m, and ε.  Two delay vectors are considered recurrent when their
        Euclidean distance is less than ε.

        Returns
        -------
        ndarray of int, shape (N_embedded, N_embedded)
            Symmetric binary matrix; 1 indicates recurrence, 0 otherwise.

        Raises
        ------
        ValueError
            If no data have been loaded or if the embedding parameters yield
            an empty embedded signal.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        tau = self.tau
        m = self.m
        eps = self.eps

        rplot = self._reccplot(tau, m, eps)
        self._recurrence_plot = rplot
        return rplot

    def compute_embedded_signal(self):
        """
        Build the time-delay embedding tensor.

        Constructs delay vectors of the form
        ``[x(t), x(t+τ), …, x(t+(m-1)τ)]`` for each valid time index t.

        Returns
        -------
        ndarray, shape (N_embedded, m, D)
            Delay-embedded signal where ``N_embedded = N - (m-1)*τ``,
            m is the embedding dimension, and D is the number of original
            signal dimensions.

        Raises
        ------
        ValueError
            If no data have been loaded or if the data are too short for the
            chosen (τ, m) combination.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        tau = self.tau
        m = self.m

        embedded = self._delayseries(tau, m)
        self._embedded_signal = embedded
        return embedded

    # RQA measures computation
    def compute_rqa_measures(self, lmin=None):
        """
        Compute the full set of RQA measures from the recurrence plot.

        Parameters
        ----------
        lmin : int, optional
            Minimum line length for DET and LAM computation.
            Defaults to ``self.config['lmin']`` (typically 2).

        Returns
        -------
        dict
            Dictionary with the following keys:

            ``recurrence_rate`` : float
                Fraction of recurrent points (RR).
            ``determinism`` : float
                Fraction of recurrent points on diagonal lines ≥ lmin (DET).
            ``laminarity`` : float
                Fraction of recurrent points on vertical lines ≥ lmin (LAM).
            ``diagonal_entropy`` : float
                Shannon entropy of diagonal line-length distribution (L_entr).
            ``vertical_entropy`` : float
                Shannon entropy of vertical line-length distribution (V_entr).
            ``average_diagonal_length`` : float
                Mean length of diagonal lines ≥ lmin (L).
            ``average_vertical_length`` : float
                Mean length of vertical lines ≥ lmin (TT – trapping time).
            ``max_diagonal_length`` : float
                Maximum diagonal line length (L_max).
            ``max_vertical_length`` : float
                Maximum vertical line length (V_max).
            ``diagonal_mode`` : float
                Mode of the diagonal line-length distribution.
            ``vertical_mode`` : float
                Mode of the vertical line-length distribution.
        """
        lmin = lmin or self.config['lmin']
        rp = self.recurrence_plot
        n = rp.shape[0]

        # Compute line distributions
        diag_hist = self._diaghist(rp, n)
        vert_hist = self._vert_hist(rp, n)

        measures = {
            'recurrence_rate': self.recurrence_rate,
            'determinism': self._percentmorethan(diag_hist, lmin, n),
            'laminarity': self._percentmorethan(vert_hist, lmin, n),
            'diagonal_entropy': self._entropy(diag_hist, lmin, n),
            'vertical_entropy': self._entropy(vert_hist, lmin, n),
            'average_diagonal_length': self._average(diag_hist, lmin, n),
            'average_vertical_length': self._average(vert_hist, lmin, n),
            'max_diagonal_length': self._maxi(diag_hist, lmin, n),
            'max_vertical_length': self._maxi(vert_hist, lmin, n),
            'diagonal_mode': self._mode(diag_hist, lmin, n),
            'vertical_mode': self._mode(vert_hist, lmin, n)
        }

        self._rqa_measures = measures
        return measures

    def compute_windowed_rqa_measures(
            self, window_size, window_step=1, lmin=None):
        """
        Compute RQA measures over sliding diagonal windows of the recurrence plot.

        Parameters
        ----------
        window_size : int
            Size of the square window to slide along the RP diagonal.
        window_step : int, default 1
            Step size for the sliding window.
        lmin : int, optional
            Minimum line length for DET and LAM computation.

        Returns
        -------
        pandas.DataFrame
            Per-window RQA measures indexed by window start position.
        """
        if window_size is None:
            raise ValueError(
                "window_size is required for windowed RQA measures.")
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if window_step <= 0:
            raise ValueError("window_step must be a positive integer.")

        rp = self.recurrence_plot
        if rp.size == 0:
            raise ValueError(
                "Recurrence plot is empty; cannot compute windowed measures.")

        n = rp.shape[0]
        if window_size > n:
            raise ValueError(
                f"window_size ({window_size}) exceeds RP size ({n}).")

        lmin = lmin or self.config['lmin']

        rows = []
        indices = []
        for start in range(0, n - window_size + 1, window_step):
            sub = rp[start:start + window_size, start:start + window_size]
            rr = float(np.sum(sub)) / (window_size * window_size)
            diag_hist = self._diaghist(sub, window_size)
            vert_hist = self._vert_hist(sub, window_size)

            measures = {
                'recurrence_rate': rr,
                'determinism': self._percentmorethan(diag_hist, lmin, window_size),
                'laminarity': self._percentmorethan(vert_hist, lmin, window_size),
                'diagonal_entropy': self._entropy(diag_hist, lmin, window_size),
                'vertical_entropy': self._entropy(vert_hist, lmin, window_size),
                'average_diagonal_length': self._average(diag_hist, lmin, window_size),
                'average_vertical_length': self._average(vert_hist, lmin, window_size),
                'max_diagonal_length': self._maxi(diag_hist, lmin, window_size),
                'max_vertical_length': self._maxi(vert_hist, lmin, window_size),
                'diagonal_mode': self._mode(diag_hist, lmin, window_size),
                'vertical_mode': self._mode(vert_hist, lmin, window_size),
            }

            rows.append(measures)
            indices.append(start)

        df = pd.DataFrame(rows, index=indices)
        df.index.name = "window_start"
        return df

    def summarize_windowed_measures(
        self, windowed_df, stats=(
            'mean', 'median', 'mode')):
        """
        Summarize windowed RQA measures with aggregate statistics.

        Parameters
        ----------
        windowed_df : pandas.DataFrame
            Output from :meth:`compute_windowed_rqa_measures`.
        stats : tuple of {'mean', 'median', 'mode'}, default ('mean','median','mode')
            Summary statistics to compute.

        Returns
        -------
        dict
            Flat dictionary of summary features.
        """
        if windowed_df is None or windowed_df.empty:
            raise ValueError("windowed_df is empty; cannot summarize.")

        if not stats:
            stats = ('mean', 'median', 'mode')
        allowed = {'mean', 'median', 'mode'}
        for stat in stats:
            if stat not in allowed:
                raise ValueError(
                    f"Unsupported stat '{stat}'. Allowed: {sorted(allowed)}")

        features = {}
        for col in windowed_df.columns:
            if not pd.api.types.is_numeric_dtype(windowed_df[col]):
                continue
            series = windowed_df[col].dropna()
            for stat in stats:
                if stat == 'mean':
                    val = float(series.mean()) if len(series) else np.nan
                elif stat == 'median':
                    val = float(series.median()) if len(series) else np.nan
                else:  # mode
                    if len(series) == 0:
                        val = np.nan
                    else:
                        mode_vals = series.mode()
                        val = float(
                            mode_vals.iloc[0]) if not mode_vals.empty else np.nan
                features[f"{col}__{stat}"] = val

        return features

    def determinism(self, lmin=None):
        """
        Return the determinism (DET) measure.

        Parameters
        ----------
        lmin : int, optional
            Minimum diagonal line length threshold.

        Returns
        -------
        float
            DET in [0, 1].
        """
        if 'determinism' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['determinism']

    def laminarity(self, lmin=None):
        """
        Return the laminarity (LAM) measure.

        Parameters
        ----------
        lmin : int, optional
            Minimum vertical line length threshold.

        Returns
        -------
        float
            LAM in [0, 1].
        """
        if 'laminarity' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['laminarity']

    def trapping_time(self, lmin=None):
        """
        Return the trapping time (TT) – mean vertical line length.

        Parameters
        ----------
        lmin : int, optional
            Minimum vertical line length threshold.

        Returns
        -------
        float
            Average vertical line length TT ≥ 0.
        """
        if 'average_vertical_length' not in self._rqa_measures:
            self.compute_rqa_measures(lmin)
        return self._rqa_measures['average_vertical_length']

    # Plotting methods
    def plot_recurrence_plot(self, figsize=(8, 8), title=None, save_path=None):
        """
        Display the recurrence plot.

        Parameters
        ----------
        figsize : tuple of float, default (8, 8)
            Width and height of the figure in inches.
        title : str, optional
            Figure title.  Defaults to ``'Recurrence Plot (RR=<value>)'``.
        save_path : str or path-like, optional
            If given, the figure is saved to this path at 300 dpi before
            being displayed.

        Returns
        -------
        None
        """
        rp = self.recurrence_plot

        plt.figure(figsize=figsize)
        plt.imshow(rp, cmap='binary', origin='lower')
        plt.title(title or f'Recurrence Plot (RR={self.recurrence_rate:.3f})')
        plt.xlabel('Time Index')
        plt.ylabel('Time Index')
        plt.colorbar(label='Recurrence')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tau_mi_curve(self, max_tau=None, figsize=(10, 6), save_path=None):
        """
        Plot the mutual information as a function of time delay τ.

        A vertical dashed line marks the automatically selected τ value.

        Parameters
        ----------
        max_tau : int, optional
            Maximum τ to evaluate.  Defaults to ``min(100, N/4)``.
        figsize : tuple of float, default (10, 6)
            Figure dimensions in inches.
        save_path : str or path-like, optional
            Destination path for saving the figure at 300 dpi.

        Returns
        -------
        None
        """
        max_tau = max_tau or min(100, self.n_samples // 4)

        tau_values = []
        mi_values = []

        for tau in range(1, max_tau + 1):
            mi = self._timedelayMI(tau)
            tau_values.append(tau)
            mi_values.append(mi)

        plt.figure(figsize=figsize)
        plt.plot(tau_values, mi_values, 'b-o', markersize=4)
        plt.axvline(x=self.tau, color='r', linestyle='--',
                    label=f'Optimal τ = {self.tau}')
        plt.xlabel('Time Delay (τ)')
        plt.ylabel('Mutual Information')
        plt.title('Time Delay vs Mutual Information')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fnn_curve(self, max_m=None, figsize=(10, 6), save_path=None):
        """
        Plot the false nearest-neighbours ratio as a function of embedding
        dimension m.

        A vertical dashed line marks the automatically selected m value.

        Parameters
        ----------
        max_m : int, optional
            Maximum embedding dimension to evaluate.
        figsize : tuple of float, default (10, 6)
            Figure dimensions in inches.
        save_path : str or path-like, optional
            Destination path for saving the figure at 300 dpi.

        Returns
        -------
        None
        """
        max_m = max_m or min(15, (3 * self.n_dimensions + 11) // 2)
        tau = self.tau
        sd = 3 * np.std(self.data)

        m_values = []
        fnn_values = []

        for m in range(1, max_m + 1):
            fnn = self._fnnratio(m, tau, 10, sd)
            m_values.append(m)
            fnn_values.append(fnn)

        plt.figure(figsize=figsize)
        plt.plot(m_values, fnn_values, 'g-o', markersize=4)
        plt.axvline(x=self.m, color='r', linestyle='--',
                    label=f'Optimal m = {self.m}')
        plt.xlabel('Embedding Dimension (m)')
        plt.ylabel('False Nearest Neighbors Ratio')
        plt.title('Embedding Dimension vs False Nearest Neighbors')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_rqa_measures_summary(self, figsize=(12, 8), save_path=None):
        """
        Display a 2×3 panel summarising the main RQA measures and parameters.

        Panels: main measures (RR, DET, LAM), entropy measures, average line
        lengths, maximum line lengths, mode line lengths, and RQA parameters
        (τ, m, ε).

        Parameters
        ----------
        figsize : tuple of float, default (12, 8)
            Figure dimensions in inches.
        save_path : str or path-like, optional
            Destination path for saving the figure at 300 dpi.

        Returns
        -------
        None
        """
        measures = self.compute_rqa_measures()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('RQA Measures Summary', fontsize=16)

        main_measures = ['recurrence_rate', 'determinism', 'laminarity']
        axes[0, 0].bar(main_measures, [measures[m] for m in main_measures])
        axes[0, 0].set_title('Main RQA Measures')
        axes[0, 0].tick_params(axis='x', rotation=45)

        entropy_measures = ['diagonal_entropy', 'vertical_entropy']
        axes[0, 1].bar(entropy_measures, [measures[m]
                       for m in entropy_measures])
        axes[0, 1].set_title('Entropy Measures')
        axes[0, 1].tick_params(axis='x', rotation=45)

        avg_measures = ['average_diagonal_length', 'average_vertical_length']
        axes[0, 2].bar(avg_measures, [measures[m] for m in avg_measures])
        axes[0, 2].set_title('Average Line Lengths')
        axes[0, 2].tick_params(axis='x', rotation=45)

        max_measures = ['max_diagonal_length', 'max_vertical_length']
        axes[1, 0].bar(max_measures, [measures[m] for m in max_measures])
        axes[1, 0].set_title('Maximum Line Lengths')
        axes[1, 0].tick_params(axis='x', rotation=45)

        mode_measures = ['diagonal_mode', 'vertical_mode']
        axes[1, 1].bar(mode_measures, [measures[m] for m in mode_measures])
        axes[1, 1].set_title('Mode Line Lengths')
        axes[1, 1].tick_params(axis='x', rotation=45)

        params = ['tau', 'm', 'eps']
        param_values = [self.tau, self.m, self.eps]
        axes[1, 2].bar(params, param_values)
        axes[1, 2].set_title('RQA Parameters')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_series(self, figsize=(12, 6), save_path=None):
        """
        Plot the original (unnormalised) time-series data.

        For univariate data a single line plot is drawn; for multivariate
        data each of the first five dimensions is shown in a stacked subplot.

        Parameters
        ----------
        figsize : tuple of float, default (12, 6)
            Figure dimensions in inches.
        save_path : str or path-like, optional
            Destination path for saving the figure at 300 dpi.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no data have been loaded.
        """
        if self.original_data is None:
            raise ValueError("No data to plot.")

        plt.figure(figsize=figsize)

        if self.n_dimensions == 1:
            plt.plot(self.original_data.flatten())
            plt.title('Time Series')
            plt.xlabel('Time Index')
            plt.ylabel('Value')
        else:
            for i in range(min(self.n_dimensions, 5)):
                plt.subplot(min(self.n_dimensions, 5), 1, i + 1)
                plt.plot(self.original_data[:, i])
                plt.title(f'Dimension {i+1}')
                plt.xlabel('Time Index')
                plt.ylabel('Value')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Batch processing methods
    @classmethod
    def batch_process(cls, input_path, output_path, group_level=False,
                      group_level_estimates=None, **kwargs):
        """
        Analyse all ``.npy`` files in *input_path* and write results to
        *output_path*.

        Two CSV files are written: ``rqa_results.csv`` (one row per file with
        all RQA measures) and ``error_report.csv`` (files that could not be
        processed).  Recurrence plots are saved as ``.npy`` arrays alongside
        the CSVs.

        Parameters
        ----------
        input_path : str or path-like
            Directory containing ``.npy`` time-series files.
        output_path : str or path-like
            Directory where outputs are written (created if absent).
        group_level : bool, default ``False``
            If ``True``, replace per-file parameters with group averages for
            the parameters listed in *group_level_estimates*.
        group_level_estimates : list of {'tau', 'm', 'eps'}, optional
            Which parameters to estimate at the group level.
        **kwargs
            Passed to ``RQA2.__init__`` for each file.

        Returns
        -------
        results : list of dict
            Per-file dictionaries containing file name, RQA parameters, and
            measure values.
        error_files : list of dict
            Per-file dictionaries for failed files, containing ``'file'`` and
            ``'error'`` keys.
        """
        os.makedirs(output_path, exist_ok=True)

        files = [f for f in os.listdir(input_path) if f.endswith('.npy')]

        results = []
        error_files = []

        # First pass: compute individual parameters
        rqa_objects = []
        for file in tqdm(files, desc="Processing files"):
            try:
                file_path = os.path.join(input_path, file)
                data = np.load(file_path)

                # Ensure 2D data
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                elif data.ndim == 0:
                    raise ValueError("Scalar data not supported")

                # Check minimum samples
                if data.shape[0] < 10:
                    raise ValueError(f"Too few samples: {data.shape[0]}")

                rqa = cls(data, **kwargs)

                # Compute basic parameters
                tau = rqa.compute_time_delay()
                m = rqa.compute_embedding_dimension()
                eps = rqa.compute_neighborhood_radius()

                rqa_objects.append(rqa)
                results.append({
                    'file': file,
                    'tau': tau,
                    'm': m,
                    'eps': eps
                })

            except Exception as e:
                error_files.append({'file': file, 'error': str(e)})
                print(f"Error processing {file}: {e}")
                continue

        # Group-level parameter estimation if requested
        if group_level and group_level_estimates and results:
            group_params = {}

            if 'tau' in group_level_estimates:
                group_params['tau'] = int(np.mean([r['tau'] for r in results]))
            if 'm' in group_level_estimates:
                group_params['m'] = int(np.mean([r['m'] for r in results]))
            if 'eps' in group_level_estimates:
                group_params['eps'] = cls._compute_group_epsilon(
                    rqa_objects, **kwargs)

        # Second pass: compute RPs and RQA measures
        for i, rqa in enumerate(tqdm(rqa_objects, desc="Computing RPs")):
            try:
                file = results[i]['file']

                # Use group parameters if specified
                if group_level and group_level_estimates:
                    if 'tau' in group_level_estimates:
                        rqa._tau = group_params['tau']
                    if 'm' in group_level_estimates:
                        rqa._m = group_params['m']
                    if 'eps' in group_level_estimates:
                        rqa._eps = group_params['eps']

                # Compute RP and measures
                rp = rqa.compute_recurrence_plot()
                measures = rqa.compute_rqa_measures()

                # Save results
                np.save(os.path.join(output_path, file), rp)

                # Update results with measures
                results[i].update(measures)

            except Exception as e:
                error_files.append(
                    {'file': results[i]['file'], 'error': str(e)})
                print(f"Error computing RP for {results[i]['file']}: {e}")

        # Save summary files
        if results:
            pd.DataFrame(results).to_csv(
                os.path.join(
                    output_path,
                    'rqa_results.csv'),
                index=False)
        if error_files:
            pd.DataFrame(error_files).to_csv(
                os.path.join(
                    output_path,
                    'error_report.csv'),
                index=False)

        return results, error_files

    # Utility methods
    def save_results(self, filepath):
        """
        Serialise all computed RQA state to a pickle file.

        Saved fields: ``data``, ``tau``, ``m``, ``eps``,
        ``recurrence_plot``, ``rqa_measures``, ``config``.

        Parameters
        ----------
        filepath : str or path-like
            Destination file path (e.g. ``'analysis.pkl'``).

        Returns
        -------
        None
        """
        results = {
            'data': self.data,
            'tau': self._tau,
            'm': self._m,
            'eps': self._eps,
            'recurrence_plot': self._recurrence_plot,
            'rqa_measures': self._rqa_measures,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    def load_results(self, filepath):
        """
        Restore RQA state from a pickle file previously written by
        :meth:`save_results`.

        Parameters
        ----------
        filepath : str or path-like
            Path to a ``.pkl`` file created by :meth:`save_results`.

        Returns
        -------
        None
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)

        self.data = results['data']
        self._tau = results['tau']
        self._m = results['m']
        self._eps = results['eps']
        self._recurrence_plot = results['recurrence_plot']
        self._rqa_measures = results['rqa_measures']
        self.config.update(results.get('config', {}))

        if self.data is not None:
            self.n_samples, self.n_dimensions = self.data.shape

    def get_summary(self):
        """
        Return a nested dictionary summarising data info, parameters, and
        (if already computed) RQA measures.

        Returns
        -------
        dict
            Top-level keys: ``'Data Info'``, ``'Parameters'``, and
            optionally ``'RQA Measures'``.
        """
        summary = {
            'Data Info': {
                'Samples': self.n_samples,
                'Dimensions': self.n_dimensions
            },
            'Parameters': {
                'Time Delay (τ)': self.tau,
                'Embedding Dimension (m)': self.m,
                'Neighborhood Radius (ε)': self.eps,
                'Recurrence Rate': self.recurrence_rate
            }
        }

        if self._rqa_measures:
            summary['RQA Measures'] = self._rqa_measures

        return summary

    # Internal computation methods - ALL FIXED FOR 0-BASED INDEXING
    def _findtau_default(self, mi_method):
        """Find optimal time delay using first minima of MI curve."""
        max_tau = min(100, self.n_samples // 4)
        if max_tau < 2:
            return 1

        min_mi = self._timedelayMI(1, mi_method)

        for tau in range(2, max_tau):
            next_mi = self._timedelayMI(tau, mi_method)
            if next_mi > min_mi:
                return tau - 1
            min_mi = next_mi

        return max_tau - 1

    def _findtau_polynomial(self, mi_method):
        """Find optimal time delay using polynomial fit to MI curve."""
        max_tau = min(100, self.n_samples // 4)
        if max_tau < 3:
            return self._findtau_default(mi_method)

        tau_values = []
        mi_values = []

        for tau in range(1, max_tau):  # Fixed: 1 to max_tau-1
            mi = self._timedelayMI(tau, mi_method)
            tau_values.append(tau)
            mi_values.append(mi)

        if len(tau_values) < 3:
            return self._findtau_default(mi_method)

        tau_values = np.array(tau_values)
        mi_values = np.array(mi_values)

        degree = self._find_poly_degree(tau_values, mi_values)

        coefficients = np.polyfit(tau_values, mi_values, degree)
        polynomial = np.poly1d(coefficients)
        y_pred = polynomial(tau_values)

        tau_index = self._find_first_minima_or_global_minima_index(y_pred)
        return int(tau_values[tau_index])

    def _timedelayMI(self, tau, method='histdd'):
        """Compute time-delayed mutual information."""
        if tau >= self.n_samples:
            return 0.0

        # Fixed indexing: ensure we don't exceed bounds
        max_idx = self.n_samples - tau
        if max_idx <= 0:
            return 0.0

        X = self.data[:max_idx, :]  # 0 to max_idx-1
        Y = self.data[tau:tau + max_idx, :]  # tau to tau+max_idx-1

        return self._mutualinfo(X, Y, method)

    def _mutualinfo(self, X, Y, method='histdd'):
        """Compute mutual information between two time series."""
        n, d = X.shape

        if method == "histdd":
            return self._mutualinfo_histdd(X, Y, n, d)
        elif method == "avg":
            return self._mutualinfo_avg(X, Y, n, d)
        else:
            return self._mutualinfo_histdd(X, Y, n, d)

    def _mutualinfo_histdd(self, X, Y, n, d):
        """Mutual information using multidimensional histogram."""
        if n == 0:
            return 0.0

        points = np.concatenate((X, Y), axis=1)
        bins = min(15, max(3, int(np.cbrt(n))))

        try:
            p_xy = np.histogramdd(points, bins=bins)[0] + 1e-12
            p_x = np.histogramdd(X, bins=bins)[0] + 1e-12
            p_y = np.histogramdd(Y, bins=bins)[0] + 1e-12

            p_xy /= np.sum(p_xy)
            p_x /= np.sum(p_x)
            p_y /= np.sum(p_y)

            return np.sum(p_xy * np.log2(p_xy)) - np.sum(p_x *
                                                         np.log2(p_x)) - np.sum(p_y * np.log2(p_y))
        except BaseException:
            return 0.0

    def _mutualinfo_avg(self, X, Y, n, d):
        """Average mutual information across dimensions."""
        mi = 0
        for i in range(d):
            X_i = X[:, i].reshape(-1, 1)
            Y_i = Y[:, i].reshape(-1, 1)
            mi += self._mutualinfo_histdd(X_i, Y_i, n, 1)
        return mi / d

    def _findm(self, tau, sd):
        """Find optimal embedding dimension using FNN method."""
        mmax = min(int((3 * self.n_dimensions + 11) / 2), 10)

        # Check if we have enough samples
        min_samples_needed = (mmax + 1) * tau  # Fixed: need m+1 for FNN
        if self.n_samples <= min_samples_needed:
            mmax = max(1, (self.n_samples - tau) // tau)

        if mmax < 1:
            return 1

        rm = self._fnnhitszero(mmax, tau, sd)
        if mmax > 1:
            rmp = self._fnnhitszero(mmax + 1, tau, sd)

            if rm != -1 and rmp != -1 and rm - rmp > self.config['bound']:
                return mmax + 1

        for m in range(1, mmax):
            rmp = rm
            rm = self._fnnhitszero(mmax - m, tau, sd)
            if rm != -1 and rmp != -1 and rm - rmp > self.config['bound']:
                return mmax + 1 - m

        return max(1, mmax)

    def _fnnhitszero(self, m, tau, sd):
        """Find r value where FNN ratio hits zero."""
        # Fixed: Check proper bounds for embedding
        min_samples_needed = (m + 1) * tau
        if self.n_samples <= min_samples_needed:
            return -1

        r_values = np.linspace(
            self.config['Rmin'],
            self.config['Rmax'],
            self.config['rdiv'])

        # The embedding and nearest-neighbour search do not depend on r, so
        # compute them once and sweep every candidate r vectorized instead of
        # calling _fnnratio (which would redo the O(n^2) search per r).
        if self.n_samples <= m * tau + 1:
            return -1

        try:
            s1 = self._delayseries(tau, m)
            s2 = self._delayseries(tau, m + 1)
        except BaseException:
            return -1

        nn = self._nearest(s1)
        max_valid = min(s1.shape[0], s2.shape[0], len(nn))
        if max_valid == 0:
            return -1

        idx = np.arange(max_valid)
        valid = nn[:max_valid] < max_valid
        idx = idx[valid]
        neigh_idx = nn[idx]

        s1_flat = s1.reshape(s1.shape[0], -1)
        s2_flat = s2.reshape(s2.shape[0], -1)
        disto = np.linalg.norm(
            s1_flat[idx] - s1_flat[neigh_idx], axis=1) + 1e-12
        distp = np.linalg.norm(s2_flat[idx] - s2_flat[neigh_idx], axis=1)

        # ratios[k, i]: neighbour/false-neighbour status of point i at
        # r_values[k]
        isneigh = disto[None, :] < (sd / r_values)[:, None]
        isfalse = isneigh & ((distp / disto)[None, :] > r_values[:, None])
        ratios = isfalse.sum(axis=1) / (isneigh.sum(axis=1) + 1e-12)

        hits = np.nonzero(ratios < self.config['delta'])[0]
        if len(hits) > 0:
            return r_values[hits[0]]
        return -1

    def _fnnratio(self, m, tau, r, sd):
        """Compute false nearest neighbors ratio - FIXED INDEXING."""
        # Fixed: Check bounds for both m and m+1 embeddings
        min_samples_m = (m - 1) * tau + 1
        min_samples_mp1 = m * tau + 1

        if self.n_samples <= min_samples_mp1:
            return 1.0

        try:
            s1 = self._delayseries(tau, m)
            s2 = self._delayseries(tau, m + 1)
        except BaseException:
            return 1.0

        nn = self._nearest(s1)

        n_embedded = s1.shape[0]
        n_embedded_mp1 = s2.shape[0]

        # Fixed: Use minimum length to avoid indexing errors
        max_valid = min(n_embedded, n_embedded_mp1, len(nn))

        isneigh = np.zeros(max_valid)
        isfalse = np.zeros(max_valid)

        for i in range(max_valid):
            if nn[i] < max_valid:  # Fixed: ensure valid index
                disto = np.linalg.norm(s1[i] - s1[nn[i]]) + 1e-12
                distp = np.linalg.norm(s2[i] - s2[nn[i]])

                if disto < sd / r:
                    isneigh[i] = 1
                    if distp / disto > r:
                        isfalse[i] = 1

        return np.sum(isneigh * isfalse) / (np.sum(isneigh) + 1e-12)

    def _delayseries(self, tau, m):
        """Create time-delayed embedding - COMPLETELY FIXED INDEXING."""
        n_embedded = self._embedded_length(m, tau)

        if n_embedded <= 0:
            raise ValueError(
                f"Insufficient data for embedding: need {(m-1)*tau + 1} samples, have {self.n_samples}")

        s = np.zeros((n_embedded, m, self.n_dimensions))

        # Fixed: Proper 0-based indexing
        for j in range(m):
            start_idx = j * tau
            end_idx = start_idx + n_embedded
            if end_idx <= self.n_samples:  # Fixed: ensure we don't exceed bounds
                s[:, j, :] = self.data[start_idx:end_idx, :]
            else:
                raise ValueError(
                    f"Index out of bounds in embedding: trying to access {end_idx} with array size {self.n_samples}")

        return s

    def _nearest(self, s):
        """Find nearest neighbors - FIXED INDEXING."""
        n_embedded = s.shape[0]
        if n_embedded == 0:
            return np.array([])

        s_flat = s.reshape(n_embedded, -1)
        nn = np.zeros(n_embedded, dtype=int)

        # Chunked cdist keeps memory bounded (~32 MB of float64 per block)
        chunk = max(1, int(4_000_000 // max(1, n_embedded)))
        for start in range(0, n_embedded, chunk):
            stop = min(start + chunk, n_embedded)
            distances = distance.cdist(s_flat[start:stop], s_flat)
            distances[np.arange(stop - start),
                      np.arange(start, stop)] = np.inf  # Exclude self-match
            nn[start:stop] = np.argmin(distances, axis=1)

        return nn

    def _findeps(self, tau, m, reqrr):
        """Find neighborhood radius - FIXED INDEXING."""
        eps_values = np.linspace(
            self.config['epsmin'],
            self.config['epsmax'],
            self.config['epsdiv'])

        if np.all(eps_values == 0):
            eps_values = np.linspace(0.001, 1.0, self.config['epsdiv'])

        n_embedded = self._embedded_length(m, tau)
        if n_embedded <= 0:
            return 0.1

        try:
            s = self._delayseries(tau, m)
        except BaseException:
            return 0.1

        s_flat = s.reshape(n_embedded, -1)

        # The distance matrix does not depend on eps: compute it once, then
        # count recurrences for every candidate eps via a sorted search.
        try:
            D_sorted = np.sort(distance.cdist(s_flat, s_flat), axis=None)
        except BaseException:
            return (self.config['epsmin'] + self.config['epsmax']) / 2

        for eps in eps_values:
            if eps <= 0:
                continue

            # Number of pairs with distance < eps, i.e. np.sum(D < eps)
            n_recurrent = np.searchsorted(D_sorted, eps, side='left')
            rr = float(n_recurrent) / (n_embedded * n_embedded)

            if abs(rr - reqrr) < self.config['rr_delta']:
                return eps

        return (self.config['epsmin'] + self.config['epsmax']) / 2

    def _reccplot(self, tau, m, eps):
        """Compute recurrence plot - FIXED INDEXING."""
        try:
            s = self._delayseries(tau, m)
        except ValueError as e:
            raise ValueError(f"Cannot compute recurrence plot: {e}")

        n_embedded = s.shape[0]

        if n_embedded == 0:
            return np.array([[]])

        # Fixed: Proper shape handling
        s_flat = s.reshape(n_embedded, -1)
        D = distance.cdist(s_flat, s_flat)

        rplot = (D < eps).astype(int)
        return rplot

    # RQA measure computation methods - FIXED INDEXING
    @staticmethod
    def _run_lengths(rows):
        """Lengths of maximal runs of 1s along each row of a 0/1 matrix."""
        n_rows, n_cols = rows.shape
        padded = np.zeros((n_rows, n_cols + 2), dtype=np.int8)
        padded[:, 1:-1] = rows
        flat_diff = np.diff(padded.ravel())
        starts = np.nonzero(flat_diff == 1)[0]
        ends = np.nonzero(flat_diff == -1)[0]
        return ends - starts

    def _vert_hist(self, rplot, n):
        """Compute vertical line distribution."""
        if n == 0:
            return np.array([0])

        rows = min(n, rplot.shape[0])
        cols = min(n, rplot.shape[1])
        rp = (np.asarray(rplot)[:rows, :cols] == 1)

        nvert = np.zeros(n + 1)
        lengths = self._run_lengths(rp.T.astype(np.int8))
        counts = np.bincount(lengths, minlength=n + 1)
        nvert[1:] = counts[1:n + 1]
        # Zero-length records emitted by the scanning formulation: one per
        # zero cell not terminating a run, plus one per column ending in zero.
        nvert[0] = (rp.size - int(rp.sum())) - len(lengths) + n

        return nvert

    def _diaghist(self, rplot, n):
        """Compute diagonal line distribution."""
        if n == 0:
            return np.array([0])

        rp = np.asarray(rplot)

        # Row i of M holds the diagonal starting at rplot[i, 0] (zero-padded),
        # so all lower-triangle diagonals can be run-length encoded in one
        # pass.
        M = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            diag = np.diagonal(rp, offset=-i)
            k = min(len(diag), n - i)
            if k > 0:
                M[i, :k] = (diag[:k] == 1)

        dghist = np.zeros(n + 1)
        lengths = self._run_lengths(M)
        counts = np.bincount(lengths, minlength=n + 1)
        dghist[1:] = counts[1:n + 1]
        # Zero-length records from the per-diagonal scan (diagonal i has
        # n - i cells, n*(n+1)/2 in total).
        total_cells = n * (n + 1) // 2
        dghist[0] = (total_cells - int(M.sum())) - len(lengths) + n

        dghist *= 2
        if len(dghist) > n:
            dghist[n] /= 2

        return dghist

    def _onedhist(self, arr, n):
        """Compute 1D histogram of line lengths."""
        if n == 0 or len(arr) == 0:
            return np.array([1])

        hst = np.zeros(n + 1)
        counter = 0

        for i in range(len(arr)):  # Fixed: 0 to len(arr)-1
            if arr[i] == 1:
                counter += 1
            else:
                if counter < len(hst):
                    hst[counter] += 1
                counter = 0

        if counter < len(hst):
            hst[counter] += 1

        return hst

    def _percentmorethan(self, hst, mini, n):
        """Compute percentage of recurrent points in lines longer than mini."""
        if len(hst) == 0 or n == 0:
            return 0.0

        max_idx = min(len(hst), n + 1)
        numer = sum(i * hst[i] for i in range(mini, max_idx))
        denom = sum(i * hst[i] for i in range(1, max_idx)) + 1e-12
        return numer / denom

    def _average(self, hst, mini, n):
        """Compute average line length."""
        if len(hst) == 0 or n == 0:
            return 0.0

        max_idx = min(len(hst), n + 1)
        numer = sum(i * hst[i] for i in range(mini, max_idx))
        denom = sum(hst[i] for i in range(mini, max_idx)) + 1e-12
        return numer / denom

    def _entropy(self, hst, mini, n):
        """Compute entropy of line length distribution."""
        if len(hst) == 0 or n == 0:
            return 0.0

        max_idx = min(len(hst), n + 1)
        total = sum(hst[i] for i in range(mini, max_idx))
        if total == 0:
            return 0

        entropy = 0
        for i in range(mini, max_idx):
            if hst[i] > 0:
                p = hst[i] / total
                entropy -= p * np.log(p)

        return entropy

    def _mode(self, hst, mini, n):
        """Find mode of line length distribution."""
        if len(hst) == 0 or n == 0:
            return mini

        max_idx = min(len(hst), n + 1)
        mode_val = mini
        for i in range(mini + 1, max_idx):
            if hst[i] > hst[mode_val]:
                mode_val = i
        return mode_val

    def _maxi(self, hst, mini, n):
        """Find maximum line length."""
        if len(hst) == 0 or n == 0:
            return 1

        max_idx = min(len(hst), n + 1)
        for i in range(max_idx - 1, 0, -1):  # Fixed: max_idx-1 to 1
            if hst[i] > 0:
                return i
        return 1

    # Helper methods - FIXED INDEXING
    def _find_first_minima_or_global_minima_index(self, arr):
        """Find first local minimum or global minimum."""
        if len(arr) == 0:
            return None

        n = len(arr)
        if n == 1:
            return 0
        if n == 2:
            return 0 if arr[0] <= arr[1] else 1

        # Check first element
        if arr[0] < arr[1]:
            return 0

        # Check middle elements
        for i in range(1, n - 1):  # Fixed: 1 to n-2
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                return int(i)

        # Check last element
        if arr[n - 1] < arr[n - 2]:
            return n - 1

        # Fallback to global minimum
        return int(np.argmin(arr))

    def _find_poly_degree(self, x, y):
        """Find optimal polynomial degree using cross-validation."""
        if len(x) < 3:
            return 1

        max_deg = min(len(x) - 1, 10)
        best_rmse = float('inf')
        best_degree = 1

        # Convert to pandas Series if needed
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        for deg in range(1, max_deg + 1):
            try:
                n_splits = min(5, len(x))
                if n_splits < 2:
                    break

                cv = RepeatedKFold(
                    n_splits=n_splits, n_repeats=3, random_state=1)
                mse_scores = []

                x_vals = x.values
                y_vals = y.values

                for train_idx, test_idx in cv.split(x_vals, y_vals):
                    # Fixed: ensure indices are within bounds
                    train_idx = train_idx[train_idx < len(x_vals)]
                    test_idx = test_idx[test_idx < len(x_vals)]

                    if len(train_idx) == 0 or len(test_idx) == 0:
                        continue

                    x_train, x_test = x_vals[train_idx], x_vals[test_idx]
                    y_train, y_test = y_vals[train_idx], y_vals[test_idx]

                    coefficients = np.polyfit(x_train, y_train, deg)
                    polynomial = np.poly1d(coefficients)
                    y_pred = polynomial(x_test)

                    mse = mean_squared_error(y_test, y_pred)
                    mse_scores.append(mse)

                if mse_scores:
                    rmse = np.sqrt(np.mean(mse_scores))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_degree = deg

            except BaseException:
                continue

        return best_degree

    @staticmethod
    def _compute_group_epsilon(rqa_objects, **kwargs):
        """Compute group-level epsilon for multiple time series."""
        eps_values = [
            rqa.eps for rqa in rqa_objects if rqa._eps is not None and rqa._eps > 0]
        return float(np.mean(eps_values)) if eps_values else 0.1


@dataclass
class RQA2_simulators:
    """
    Generators for chaotic dynamical systems used to test RQA2 and surrogate
    methods.

    All ODE systems are integrated with ``scipy.integrate.solve_ivp`` using
    the RK45 solver at tight tolerances (``rtol=1e-9``, ``atol=1e-12``).
    Discrete maps are iterated directly.

    Parameters
    ----------
    seed : int or None, optional
        Seed for the internal :class:`numpy.random.Generator`.  Use an
        integer for reproducible results.

    Examples
    --------
    >>> sim = RQA2_simulators(seed=42)
    >>> x, y, z = sim.rossler(n=1000)
    >>> battery = sim.generate_test_battery()
    """

    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def rossler(self,
                tmax: int = 10000,
                n: int = 2000,
                Xi: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                a: float = 0.2,
                b: float = 0.2,
                c: float = 5.7,
                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the Rössler attractor.

        The system is defined by::

            dx/dt = -y - z
            dy/dt =  x + a·y
            dz/dt =  b + z·(x - c)

        With ``b=0.2, c=5.7``: chaotic for ``a ≲ 0.2``, periodic for
        ``a ≳ 0.2``.

        Parameters
        ----------
        tmax : int, default 10000
            Number of integration steps.
        n : int, default 2000
            Number of output points (sub-sampled from the solution).
        Xi : tuple of float, default (1.0, 1.0, 1.0)
            Initial conditions (x₀, y₀, z₀).
        a, b, c : float
            System parameters.  Classic chaotic: ``a=0.2, b=0.2, c=5.7``.
        dt : float, default 0.01
            Integration step size.

        Returns
        -------
        x, y, z : ndarray, shape (n,)
            State variables sampled at *n* equidistant times.
        """
        def rossler_system(t, state):
            x, y, z = state
            dxdt = -y - z
            dydt = x + a * y
            dzdt = b + z * (x - c)
            return [dxdt, dydt, dzdt]

        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)

        sol = solve_ivp(rossler_system, t_span, Xi, t_eval=t_eval,
                        method='RK45', rtol=1e-9, atol=1e-12)

        # Subsample to get n points
        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]

        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]

    def lorenz(self,
               tmax: int = 10000,
               n: int = 2000,
               Xi: Tuple[float, float, float] = (1.0, 1.0, 1.0),
               sigma: float = 10.0,
               rho: float = 28.0,
               beta: float = 8.0 / 3.0,
               dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the Lorenz attractor.

        The system is defined by::

            dx/dt = σ·(y - x)
            dy/dt = ρ·x - y - x·z
            dz/dt = x·y - β·z

        The classic chaotic butterfly attractor uses
        ``σ=10, ρ=28, β=8/3``.

        Parameters
        ----------
        tmax : int, default 10000
            Number of integration steps.
        n : int, default 2000
            Number of output points.
        Xi : tuple of float, default (1.0, 1.0, 1.0)
            Initial conditions (x₀, y₀, z₀).
        sigma, rho, beta : float
            System parameters.
        dt : float, default 0.01
            Integration step size.

        Returns
        -------
        x, y, z : ndarray, shape (n,)
            State variables at *n* equidistant times.
        """
        def lorenz_system(t, state):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = rho * x - y - x * z
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]

        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)

        sol = solve_ivp(lorenz_system, t_span, Xi, t_eval=t_eval,
                        method='RK45', rtol=1e-9, atol=1e-12)

        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]

        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]

    def henon(self,
              n: int = 2000,
              Xi: Tuple[float, float] = (0.1, 0.1),
              a: float = 1.4,
              b: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Hénon map.

        Defined by the 2-D discrete-time recurrence::

            x_{n+1} = 1 - a·x_n² + y_n
            y_{n+1} = b·x_n

        The classic chaotic attractor uses ``a=1.4, b=0.3``.

        Parameters
        ----------
        n : int, default 2000
            Number of iterations.
        Xi : tuple of float, default (0.1, 0.1)
            Initial conditions (x₀, y₀).
        a, b : float
            Map parameters.

        Returns
        -------
        X, Y : ndarray, shape (n,)
            Trajectory of the x and y coordinates.
        """
        x, y = Xi
        X, Y = [x], [y]

        for i in range(n - 1):
            x_next = 1 - a * x**2 + y
            y_next = b * x
            X.append(x_next)
            Y.append(y_next)
            x, y = x_next, y_next

        return np.array(X), np.array(Y)

    def chua(self,
             tmax: int = 10000,
             n: int = 2000,
             Xi: Tuple[float, float, float] = (0.1, 0.1, 0.1),
             alpha: float = 15.6,
             beta: float = 28.0,
             m0: float = -1.143,
             m1: float = -0.714,
             dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Chua's circuit attractor.

        Uses the piecewise-linear Chua diode characteristic h(x) and the
        three-dimensional ODE::

            dx/dt = α·(y - x - h(x))
            dy/dt = x - y + z
            dz/dt = -β·y

        Parameters
        ----------
        tmax : int, default 10000
            Number of integration steps.
        n : int, default 2000
            Number of output points.
        Xi : tuple of float, default (0.1, 0.1, 0.1)
            Initial conditions (x₀, y₀, z₀).
        alpha, beta : float
            Circuit parameters.
        m0, m1 : float
            Slopes of the piecewise-linear Chua diode.
        dt : float, default 0.01
            Integration step size.

        Returns
        -------
        x, y, z : ndarray, shape (n,)
            State variables at *n* equidistant times.
        """
        def chua_system(t, state):
            x, y, z = state
            # Piecewise-linear function
            if x >= 1:
                h = m1 * x + (m0 - m1)
            elif x >= -1:
                h = m0 * x
            else:
                h = m1 * x - (m0 - m1)

            dxdt = alpha * (y - x - h)
            dydt = x - y + z
            dzdt = -beta * y
            return [dxdt, dydt, dzdt]

        t_span = (0, tmax * dt)
        t_eval = np.linspace(0, tmax * dt, tmax)

        sol = solve_ivp(chua_system, t_span, Xi, t_eval=t_eval,
                        method='RK45', rtol=1e-9, atol=1e-12)

        step = len(sol.y[0]) // n
        indices = np.arange(0, len(sol.y[0]), step)[:n]

        return sol.y[0][indices], sol.y[1][indices], sol.y[2][indices]

    def kuramoto(self,
                 n: int = 2000,
                 n_osc: int = 10,
                 K: float = 1.0,
                 omega_sd: float = 1.0,
                 tmax: float = 100.0) -> np.ndarray:
        """
        Simulate a network of Kuramoto phase oscillators.

        The system is defined by::

            dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)

        with natural frequencies ``ω_i ~ Normal(0, omega_sd)`` and
        initial phases drawn uniformly from ``[0, 2π)`` using the
        instance RNG (seeded via the ``seed`` field).

        For Gaussian frequencies the critical coupling is
        ``K_c = 2 / (π·g(0)) = omega_sd · √(8/π) ≈ 1.596·omega_sd``:
        below ``K_c`` the oscillators stay incoherent, above it they
        synchronise.

        Parameters
        ----------
        n : int, default 2000
            Number of output samples (uniform subsample of the
            trajectory).
        n_osc : int, default 10
            Number of oscillators — the output dimensionality.
        K : float, default 1.0
            Coupling strength.
        omega_sd : float, default 1.0
            Standard deviation of the natural-frequency distribution.
        tmax : float, default 100.0
            Integration time.

        Returns
        -------
        ndarray of shape (n, n_osc)
            ``sin(θ_i(t))`` for each oscillator — a multivariate signal
            directly usable with :class:`RQA2`.
        """
        n_osc = int(n_osc)
        omega = self._rng.normal(0.0, omega_sd, n_osc)
        theta0 = self._rng.uniform(0.0, 2.0 * np.pi, n_osc)

        def kuramoto_system(t, theta):
            phase_diff = theta[None, :] - theta[:, None]
            return omega + (K / n_osc) * np.sin(phase_diff).sum(axis=1)

        t_eval = np.linspace(0.0, tmax, n)
        sol = solve_ivp(kuramoto_system, (0.0, tmax), theta0,
                        t_eval=t_eval, method='RK45',
                        rtol=1e-6, atol=1e-9)

        return np.sin(sol.y.T)

    def generate_test_battery(self) -> dict:
        """
        Generate a standard battery of chaotic and periodic test signals.

        Returns
        -------
        dict
            Keys: ``'rossler_chaotic'``, ``'rossler_sync'``, ``'lorenz'``,
            ``'henon'``, ``'chua'``.  Each value is a dict with keys
            ``'x'``, ``'y'``, (optionally ``'z'``), and ``'regime'``.
        """
        systems = {}

        # Rössler chaotic regime
        x, y, z = self.rossler(tmax=5000, n=2000, a=0.1)
        systems['rossler_chaotic'] = {
            'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}

        # Rössler synchronous regime
        x, y, z = self.rossler(tmax=5000, n=2000, a=0.3)
        systems['rossler_sync'] = {
            'x': x, 'y': y, 'z': z, 'regime': 'synchronous'}

        # Lorenz system
        x, y, z = self.lorenz(tmax=5000, n=2000)
        systems['lorenz'] = {'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}

        # Hénon map
        x, y = self.henon(n=2000)
        systems['henon'] = {'x': x, 'y': y, 'regime': 'chaotic'}

        # Chua circuit
        x, y, z = self.chua(tmax=5000, n=2000)
        systems['chua'] = {'x': x, 'y': y, 'z': z, 'regime': 'chaotic'}

        return systems


Algorithm = Literal[
    "FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"
]


@dataclass
class RQA2_tests:
    """
    Surrogate-data generation and statistical validation for nonlinear
    dynamics testing.

    Implements six surrogate algorithms and a comprehensive validation
    framework that tests each algorithm against multiple nonlinear metrics
    (Lyapunov exponent, sample entropy, correlation dimension, etc.).

    Parameters
    ----------
    signal : ndarray of float
        1-D floating-point time series to be tested.
    seed : int or None, optional
        Seed for reproducible surrogate generation.
    max_workers : int, default 1
        Number of parallel workers for surrogate generation
        (parallelism kicks in when ``n_surrogates >= 50``).

    Raises
    ------
    TypeError
        If *signal* is not a floating-point array.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> signal = rng.standard_normal(512).astype(float)
    >>> tester = RQA2_tests(signal, seed=42)
    >>> surrogates = tester.generate('IAAFT', n_surrogates=100)
    >>> surrogates.shape
    (100, 512)
    """

    signal: np.ndarray
    seed: int | None = None
    max_workers: int = 1
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        if not np.issubdtype(self.signal.dtype, np.floating):
            raise TypeError("Input signal must be floating-point")

    # ------------------------------------------------------------------
    # Public façade with enhanced validation
    # ------------------------------------------------------------------
    def generate(
        self,
        kind: Algorithm = "FT",
        *,
        n_surrogates: int = 200,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate an ensemble of surrogate time series.

        Each surrogate is produced with an independent random seed to
        ensure statistical independence.  When ``n_surrogates >= 50`` and
        ``max_workers > 1`` the ensemble is generated in parallel.

        Parameters
        ----------
        kind : {'FT', 'AAFT', 'IAAFT', 'IDFS', 'WIAAFT', 'PPS'}, default 'FT'
            Surrogate algorithm:

            * **FT** – Fourier-Transform phase randomisation.
            * **AAFT** – Amplitude-Adjusted Fourier Transform.
            * **IAAFT** – Iterative AAFT (n_iter=100 by default).
            * **IDFS** – Iterative Digitally-Filtered Shuffled.
            * **WIAAFT** – Wavelet-based IAAFT.
            * **PPS** – Pseudo-Periodic Surrogate.
        n_surrogates : int, default 200
            Number of surrogates to generate.
        **kwargs
            Passed to the underlying surrogate algorithm (e.g.
            ``n_iter=200`` for IAAFT, ``wavelet='db4'`` for WIAAFT).

        Returns
        -------
        ndarray, shape (n_surrogates, N)
            Array of surrogate time series, one per row.

        Raises
        ------
        KeyError
            If *kind* is not a recognised algorithm name.
        """
        _dispatcher = {
            "FT": self._ft,
            "AAFT": self._aaft,
            "IAAFT": self._iaaft,
            "IDFS": self._idfs,
            "WIAAFT": self._wiaaft,
            "PPS": self._pps,
        }
        if kind not in _dispatcher:
            raise KeyError(f"Unknown surrogate type {kind}")

        # Generate unique seeds for each surrogate
        seeds = self._rng.integers(0, 2**32, size=n_surrogates)

        # Parallel generation for large ensembles
        if n_surrogates >= 50 and self.max_workers > 1:
            return self._parallel_generate(kind, seeds, **kwargs)
        else:
            return np.vstack([
                self._generate_with_seed(kind, seed, **kwargs)
                for seed in seeds
            ])

    def _parallel_generate(
        self,
        kind: Algorithm,
        seeds: Sequence[int],
        **kwargs
    ) -> np.ndarray:
        """Parallel surrogate generation with unique seeds."""
        surrogates = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_with_seed,
                    kind, seed, **kwargs
                )
                for seed in seeds
            ]
            for future in as_completed(futures):
                surrogates.append(future.result())
        return np.vstack(surrogates)

    def _generate_with_seed(
        self,
        kind: Algorithm,
        seed: int,
        **kwargs
    ) -> np.ndarray:
        """Generate one surrogate with a specific seed."""
        # Create temporary RQA2_tests instance with unique seed
        temp_tests = RQA2_tests(
            signal=self.signal.copy(),
            seed=seed,
            max_workers=1  # Disable nested parallelism
        )

        # Dispatch to appropriate surrogate method
        method_name = f"_{kind.lower()}"
        method = getattr(temp_tests, method_name)
        return method(**kwargs)

    def comprehensive_validation(
        self,
        systems_data: Dict[str, Dict[str, np.ndarray]],
        n_surrogates: int = 200,
        save_path: str = "surrogate_validation_results.png"
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all six surrogate algorithms against multiple dynamical systems
        and return rank-based two-sided p-values for six nonlinear metrics.

        A heatmap of the results is saved to *save_path*.

        Parameters
        ----------
        systems_data : dict
            Mapping from system name (str) to a dict with at least an
            ``'x'`` key containing a 1-D ndarray (the x-coordinate of
            the attractor).  Typically produced by
            :meth:`RQA2_simulators.generate_test_battery`.
        n_surrogates : int, default 200
            Number of surrogates per algorithm per system.
        save_path : str, default 'surrogate_validation_results.png'
            Path where the validation heatmap is saved.

        Returns
        -------
        dict
            Nested dict: ``results[system_name][surrogate_method][metric]``
            → p-value (float, or ``nan`` if the metric could not be
            computed).
        """
        surrogate_methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
        metrics = [
            "lyapunov_exponent", "time_irreversibility", "sample_entropy",
            "correlation_dimension", "nonlinearity_index", "predictability"
        ]

        results = {}

        for system_name, system_data in systems_data.items():
            print(f"\nProcessing {system_name}...")
            system_results = {}

            # Use first component (x-coordinate) for analysis
            signal = system_data['x']
            self.signal = signal

            # Calculate metrics for original signal
            original_metrics = self._calculate_all_metrics(signal)

            for surrogate_method in surrogate_methods:
                print(f"  Testing {surrogate_method} surrogates...")
                method_results = {}

                # Generate surrogate ensemble
                surrogates = self.generate(
                    surrogate_method, n_surrogates=n_surrogates)

                # Calculate metrics for all surrogates
                surrogate_metrics = {metric: [] for metric in metrics}
                for i in range(n_surrogates):
                    surrogate_signal = surrogates[i]
                    metrics_values = self._calculate_all_metrics(
                        surrogate_signal)
                    for metric in metrics:
                        surrogate_metrics[metric].append(
                            metrics_values[metric])

                # Calculate p-values
                for metric in metrics:
                    original_value = original_metrics[metric]
                    surrogate_values = np.array(surrogate_metrics[metric])

                    # Two-sided test: probability that observed differs from
                    # surrogate
                    if not np.isnan(original_value) and not np.any(
                            np.isnan(surrogate_values)):
                        # Rank-based p-value calculation
                        n_extreme = np.sum(
                            (surrogate_values <= original_value) |
                            (surrogate_values >= original_value)
                        )
                        p_value = min(
                            n_extreme, n_surrogates - n_extreme) / n_surrogates
                        p_value = 2 * p_value  # Two-sided test
                        p_value = min(p_value, 1.0)
                    else:
                        p_value = np.nan

                    method_results[metric] = p_value

                system_results[surrogate_method] = method_results

            results[system_name] = system_results

        # Create visualization
        self._create_validation_heatmap(results, save_path)

        return results

    def _calculate_all_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of nonlinear dynamics metrics."""
        metrics = {}

        try:
            metrics["lyapunov_exponent"] = self._lyapunov_exponent(signal)
        except BaseException:
            metrics["lyapunov_exponent"] = np.nan

        try:
            metrics["time_irreversibility"] = self._time_irreversibility(
                signal)
        except BaseException:
            metrics["time_irreversibility"] = np.nan

        try:
            metrics["sample_entropy"] = self._sample_entropy(signal)
        except BaseException:
            metrics["sample_entropy"] = np.nan

        try:
            metrics["correlation_dimension"] = self._correlation_dimension(
                signal)
        except BaseException:
            metrics["correlation_dimension"] = np.nan

        try:
            metrics["nonlinearity_index"] = self._nonlinearity_index(signal)
        except BaseException:
            metrics["nonlinearity_index"] = np.nan

        try:
            metrics["predictability"] = self._predictability_index(signal)
        except BaseException:
            metrics["predictability"] = np.nan

        return metrics

    def _lyapunov_exponent(
            self,
            signal: np.ndarray,
            dim: int = 3,
            tau: int = 1) -> float:
        """Calculate largest Lyapunov exponent using Rosenstein method."""
        # Phase space reconstruction
        N = len(signal)
        M = N - (dim - 1) * tau

        if M < 50:  # Not enough points
            return np.nan

        embedded = np.zeros((M, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * tau:i * tau + M]

        # Find nearest neighbors
        lyap_values = []

        for i in range(min(M - 1, 500)):  # Limit for computational efficiency
            distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
            distances[i] = np.inf  # Exclude self

            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] > 0:
                # Track divergence
                divergences = []
                for k in range(1, min(20, M - max(i, nearest_idx))):
                    if i + k < M and nearest_idx + k < M:
                        dist_k = np.sqrt(
                            np.sum((embedded[i + k] - embedded[nearest_idx + k])**2))
                        if dist_k > 0:
                            divergences.append(
                                np.log(dist_k / distances[nearest_idx]) / k)

                if divergences:
                    lyap_values.extend(divergences)

        return np.mean(lyap_values) if lyap_values else np.nan

    def _time_irreversibility(self, signal: np.ndarray) -> float:
        """Calculate time irreversibility using third-order statistics."""
        n = len(signal)
        if n < 10:
            return np.nan

        # Centered signal
        signal_centered = signal - np.mean(signal)

        # Calculate asymmetry measure
        irreversibility = 0
        count = 0

        for lag in range(1, min(n // 10, 20)):
            if n - lag > 0:
                forward = signal_centered[lag:] * (signal_centered[:-lag]**2)
                backward = signal_centered[:-lag] * (signal_centered[lag:]**2)

                diff = np.mean(forward) - np.mean(backward)
                irreversibility += diff**2
                count += 1

        return np.sqrt(irreversibility / count) if count > 0 else np.nan

    def _sample_entropy(
            self,
            signal: np.ndarray,
            m: int = 2,
            r: float = None) -> float:
        """Calculate sample entropy."""
        N = len(signal)
        if r is None:
            r = 0.2 * np.std(signal)

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j]) <= r:
                        C[i] += 1.0

            phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
            return phi

        return _phi(m) - _phi(m + 1)

    def _correlation_dimension(
            self,
            signal: np.ndarray,
            dim: int = 5) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
        # Phase space reconstruction
        N = len(signal)
        if N < 100:
            return np.nan

        tau = self._estimate_delay(signal)
        M = N - (dim - 1) * tau

        if M < 50:
            return np.nan

        embedded = np.zeros((M, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * tau:i * tau + M]

        # Calculate correlation integral for different radii
        radii = np.logspace(-2, 0, 20) * np.std(signal)
        correlations = []

        for r in radii:
            count = 0
            total_pairs = 0

            # Sample pairs for computational efficiency
            for i in range(0, min(M, 200), 5):
                for j in range(i + 1, min(M, 200), 5):
                    dist = np.sqrt(np.sum((embedded[i] - embedded[j])**2))
                    if dist < r:
                        count += 1
                    total_pairs += 1

            if total_pairs > 0:
                correlations.append(count / total_pairs)
            else:
                correlations.append(0)

        # Fit linear region
        log_radii = np.log(radii)
        log_correlations = np.log(np.maximum(correlations, 1e-10))

        valid_idx = np.isfinite(log_correlations)
        if np.sum(valid_idx) < 5:
            return np.nan

        slope, _, _, _, _ = stats.linregress(
            log_radii[valid_idx], log_correlations[valid_idx]
        )

        return slope

    def _nonlinearity_index(self, signal: np.ndarray) -> float:
        """Calculate nonlinearity index based on phase space asymmetry."""
        # Simple nonlinearity measure based on skewness of increments
        increments = np.diff(signal)
        return abs(stats.skew(increments))

    def _predictability_index(self, signal: np.ndarray) -> float:
        """Calculate predictability index using local prediction error."""
        if len(signal) < 20:
            return np.nan

        errors = []
        for i in range(10, len(signal) - 1):
            # Simple local linear prediction
            local_data = signal[i - 10:i]
            if len(local_data) >= 2:
                slope = (local_data[-1] - local_data[-2])
                prediction = local_data[-1] + slope
                error = abs(signal[i] - prediction)
                errors.append(error)

        return np.mean(errors) / np.std(signal) if errors else np.nan

    def _estimate_delay(self, signal: np.ndarray) -> int:
        """Estimate optimal delay using first minimum of autocorrelation."""
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]

        # Find first zero crossing or minimum
        for i in range(1, min(len(autocorr), 100)):
            if autocorr[i] < 0:
                return i
            if i > 1 and autocorr[i] > autocorr[i - 1]:
                return i - 1

        return 1

    def _create_validation_heatmap(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        save_path: str
    ):
        """Create comprehensive validation heatmap."""
        surrogate_methods = ["FT", "AAFT", "IAAFT", "IDFS", "WIAAFT", "PPS"]
        metrics = [
            "lyapunov_exponent", "time_irreversibility", "sample_entropy",
            "correlation_dimension", "nonlinearity_index", "predictability"
        ]

        n_systems = len(results)
        fig, axes = plt.subplots(1, n_systems, figsize=(5 * n_systems, 6))
        if n_systems == 1:
            axes = [axes]

        for idx, (system_name, system_results) in enumerate(results.items()):
            # Create p-value matrix
            p_matrix = np.full((len(surrogate_methods), len(metrics)), np.nan)

            for i, method in enumerate(surrogate_methods):
                if method in system_results:
                    for j, metric in enumerate(metrics):
                        if metric in system_results[method]:
                            p_matrix[i, j] = system_results[method][metric]

            # Create heatmap
            im = axes[idx].imshow(
                p_matrix,
                cmap='RdYlBu_r',
                aspect='auto',
                vmin=0, vmax=1,
                interpolation='nearest'
            )

            # Add text annotations
            for i in range(len(surrogate_methods)):
                for j in range(len(metrics)):
                    if not np.isnan(p_matrix[i, j]):
                        text = axes[idx].text(
                            j, i, f'{p_matrix[i, j]:.3f}',
                            ha="center", va="center",
                            color="black" if p_matrix[i, j] > 0.5 else "white",
                            fontsize=8
                        )

            axes[idx].set_xticks(range(len(metrics)))
            axes[idx].set_xticklabels([m.replace('_', '\n')
                                      for m in metrics], rotation=45)
            axes[idx].set_yticks(range(len(surrogate_methods)))
            axes[idx].set_yticklabels(surrogate_methods)
            axes[idx].set_title(
                f'{system_name.replace("_", " ").title()}\nP-values (Observed ≠ Surrogate)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('P-value', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nValidation results saved to {save_path}")

    # ------------------------------------------------------------------
    # Individual surrogate algorithms (same as before but optimized)
    # ------------------------------------------------------------------
    def _ft(self) -> np.ndarray:
        """Phase-randomised Fourier surrogate."""
        x = self._detrend(self.signal)
        fftx = _fft.rfft(x)
        phases = self._rng.uniform(0, 2 * np.pi, len(fftx))
        fftx_randomised = np.abs(fftx) * np.exp(1j * phases)
        surrogate = _fft.irfft(fftx_randomised, n=len(x))
        return surrogate.astype(np.float64)

    def _aaft(self) -> np.ndarray:
        """Amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        ranks = x.argsort().argsort()
        gaussian = self._rng.normal(size=len(x))
        gaussian_sorted = np.sort(gaussian)
        x_gauss = gaussian_sorted[ranks]
        fftx = _fft.rfft(x_gauss)
        phases = self._rng.uniform(0, 2 * np.pi, len(fftx))
        x_phase = _fft.irfft(np.abs(fftx) * np.exp(1j * phases), n=len(x))
        surrogate = np.sort(x)[x_phase.argsort().argsort()]
        return surrogate.astype(np.float64)

    def _iaaft(self, *, n_iter: int = 100) -> np.ndarray:
        """Iterative amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        amplitude_target = np.sort(x)
        fft_target = np.abs(_fft.rfft(x))

        y = self._rng.permutation(x)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(fft_target * np.exp(1j * np.angle(yf)), n=len(x))
            y = amplitude_target[y.argsort().argsort()]
        return y.astype(np.float64)

    def _idfs(self, *, n_iter: int = 50) -> np.ndarray:
        """Iterative digitally-filtered shuffled surrogate."""
        x = self._detrend(self.signal)
        xp = self._rng.permutation(x)
        target_fft = np.abs(_fft.rfft(xp))

        y = amplitude_target = np.sort(x)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(target_fft * np.exp(1j * np.angle(yf)), n=len(x))
            y = amplitude_target[y.argsort().argsort()]
        return y.astype(np.float64)

    def _wiaaft(
        self,
        *,
        wavelet: str = "db4",
        level: int | None = None,
        n_iter: int = 20,
    ) -> np.ndarray:
        """Wavelet iterative amplitude adjusted Fourier surrogate."""
        x = self._detrend(self.signal)
        coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
        coeff_sur = []
        for c in coeffs[:-1]:
            csurr = self._iaaft_array(c, n_iter=n_iter)
            csurp = csurr[::-1]
            if np.linalg.norm(c - csurr) > np.linalg.norm(c - csurp):
                csurr = csurp
            coeff_sur.append(csurr)
        coeff_sur.append(coeffs[-1])
        y = pywt.waverec(coeff_sur, wavelet, mode="periodization")
        return y.astype(np.float64)[: len(x)]

    def _pps(
        self,
        *,
        tau: int | None = None,
        dim: int = 10,
        noise_factor: float | None = None,
    ) -> np.ndarray:
        """Pseudo-periodic surrogate."""
        x = self._detrend(self.signal)
        if tau is None:
            tau = self._first_zero_crossing_acf(x)
        embed = self._embed(x, dim, tau)
        dists = np.linalg.norm(embed - embed[-1], axis=1)
        q = np.argmin(dists[: len(embed) // 2])
        if noise_factor is None:
            noise_factor = 0.7 * np.mean(np.min(
                np.linalg.norm(embed[:-1, None] - embed[None, :-1], axis=2),
                axis=1,
            ))
        sur = [embed[self._rng.integers(len(embed))]]
        while len(sur) < len(embed):
            candidate = sur[-1] + \
                self._rng.normal(scale=noise_factor, size=dim)
            j = np.argmin(np.linalg.norm(embed - candidate, axis=1))
            if j == len(embed) - 1:
                j = q
            sur.append(embed[(j + 1) % len(embed)])
        return np.asarray(sur)[:, 0].astype(np.float64)[: len(x)]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _detrend(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean()

    def _iaaft_array(self, arr: np.ndarray, n_iter: int) -> np.ndarray:
        ranks = np.sort(arr)
        fft_target = np.abs(_fft.rfft(arr))
        y = self._rng.permutation(arr)
        for _ in range(n_iter):
            yf = _fft.rfft(y)
            y = _fft.irfft(fft_target * np.exp(1j * np.angle(yf)), n=len(arr))
            y = ranks[y.argsort().argsort()]
        return y

    def _embed(self, x: np.ndarray, dim: int, tau: int) -> np.ndarray:
        N = len(x) - (dim - 1) * tau
        return np.column_stack([x[i: i + N] for i in range(0, dim * tau, tau)])

    def _first_zero_crossing_acf(self, x: np.ndarray) -> int:
        acf = np.correlate(x, x, mode="full")[len(x) - 1:]
        acf /= acf[0]
        zero_crossings = np.where(acf < 0)[0]
        return int(zero_crossings[0]) if len(zero_crossings) > 0 else 1


_SURROGATE_NULL_DESCRIPTIONS = {
    'FT': 'Linear autocorrelation alone explains classification',
    'AAFT': 'Linear structure + marginal distribution suffice',
    'IAAFT': 'Refined linear structure + amplitude distribution suffice',
    'IDFS': 'Digitally-filtered shuffled structure suffices',
    'WIAAFT': 'Wavelet time-frequency structure suffices',
    'PPS': 'Periodic structure alone suffices',
}


class RQA2_ml:
    """
    Machine learning utilities built on top of RQA2 features.

    This class provides a complete feature-engineering and benchmarking
    pipeline for time-series classification and clustering using Recurrence
    Quantification Analysis measures.  It implements the nested
    cross-validation with best-subset feature selection procedure
    described in the SMdRQA paper, surrogate-based null baselines,
    permutation feature importance, and publication-ready visualisations.

    Parameters
    ----------
    rqa_kwargs : dict, optional
        Default keyword arguments forwarded to :class:`RQA2` when
        constructing RQA objects internally (e.g. ``normalize``,
        ``mi_method``).  Per-call overrides can be passed via
        ``rqa_kwargs`` in :meth:`build_feature_table`.

    Examples
    --------
    >>> ml = RQA2_ml()
    >>> features = ml.build_feature_table(
    ...     [signal_a, signal_b], labels=["a", "b"],
    ...     window_size=100, window_step=20,
    ...     rqa_kwargs={"tau": 2, "m": 3, "eps": 0.3},
    ... )
    >>> results = ml.nested_cv_benchmark(
    ...     features.drop(columns=["id", "label"]),
    ...     features["label"], model="knn",
    ... )
    """

    _MODEL_REGISTRY = {
        'knn': lambda rs: KNeighborsClassifier(),
        'svm': lambda rs: SVC(
            kernel='rbf', gamma='scale', probability=True,
            random_state=rs),
        'rf': lambda rs: RandomForestClassifier(
            n_estimators=200, random_state=rs),
        'logreg': lambda rs: LogisticRegression(
            max_iter=5000, random_state=rs),
        'lda': lambda rs: LinearDiscriminantAnalysis(),
        'nb': lambda rs: GaussianNB(),
        'gb': lambda rs: HistGradientBoostingClassifier(
            random_state=rs),
        'et': lambda rs: ExtraTreesClassifier(
            n_estimators=200, random_state=rs),
    }

    #: Compact hyperparameter grids searched in the inner CV loop when
    #: ``tune=True``.  Kept deliberately small so nested search stays
    #: tractable; override per call via ``param_grid``.
    _PARAM_GRIDS = {
        'knn': {'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']},
        'svm': {'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 0.01, 0.1]},
        'rf': {'max_features': ['sqrt', 'log2', None],
               'min_samples_leaf': [1, 3, 5]},
        'et': {'max_features': ['sqrt', 'log2', None],
               'min_samples_leaf': [1, 3, 5]},
        'logreg': {'C': [0.01, 0.1, 1, 10, 100]},
        'lda': [{'solver': ['svd']},
                {'solver': ['lsqr'], 'shrinkage': ['auto']}],
        'nb': {'var_smoothing': [1e-9, 1e-7, 1e-5]},
        'gb': {'learning_rate': [0.05, 0.1],
               'max_leaf_nodes': [7, 15, 31]},
    }

    def __init__(self, rqa_kwargs: Optional[Dict[str, Any]] = None):
        self.rqa_kwargs = rqa_kwargs or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_signals(self, signals_or_dir):
        """Load signals from a directory, array, or sequence.

        Parameters
        ----------
        signals_or_dir : str, PathLike, ndarray, list, or tuple
            If a directory path, all ``.npy`` files inside are loaded.
            If an ndarray, it is treated as a single signal.
            If a list/tuple, each element is treated as one signal.

        Returns
        -------
        signals : list of ndarray
        ids : list of str
            Identifier for each signal (filename or ``signal_<i>``).
        """
        if isinstance(signals_or_dir, (str, os.PathLike)):
            input_path = os.fspath(signals_or_dir)
            if not os.path.isdir(input_path):
                raise ValueError(
                    f"signals_or_dir must be a directory path; "
                    f"got {input_path}")
            files = sorted([
                f for f in os.listdir(input_path) if f.endswith('.npy')])
            if not files:
                raise ValueError(
                    f"No .npy files found in directory: {input_path}")
            signals = []
            ids = []
            for fname in files:
                data = np.load(os.path.join(input_path, fname))
                signals.append(data)
                ids.append(fname)
            return signals, ids

        if isinstance(signals_or_dir, np.ndarray):
            return [signals_or_dir], ["signal_0"]

        if isinstance(signals_or_dir, (list, tuple)):
            if len(signals_or_dir) == 0:
                raise ValueError("signals_or_dir is empty.")
            return (list(signals_or_dir),
                    [f"signal_{i}" for i in range(len(signals_or_dir))])

        raise TypeError(
            "signals_or_dir must be a directory path, numpy array, "
            "list, or tuple.")

    @classmethod
    def _make_model(cls, name, random_state=42):
        """Create a fresh unfitted estimator by name.

        Parameters
        ----------
        name : str
            One of ``'knn'``, ``'svm'``, ``'rf'``.
        random_state : int
            Seed forwarded to estimators that accept it.

        Returns
        -------
        sklearn estimator
        """
        if name not in cls._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {sorted(cls._MODEL_REGISTRY)}")
        return cls._MODEL_REGISTRY[name](random_state)

    @staticmethod
    def _compute_roc_auc(model, X, y_true):
        """Compute ROC AUC, handling binary and multiclass cases.

        Returns ``np.nan`` when probability estimates are unavailable.
        """
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X)
                else:
                    return np.nan
                return float(roc_auc_score(y_true, y_score))
            else:
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X)
                    return float(roc_auc_score(
                        y_true, y_score, multi_class='ovr'))
                return np.nan
        except Exception:
            return np.nan

    class _PrecomputedSplitter:
        """Splitter facade over a fixed list of (train, val) index pairs.

        Lets group-aware splits flow through code written against the
        sklearn splitter interface (feature selection, tuning).
        """

        def __init__(self, splits):
            self._splits = list(splits)

        def split(self, X=None, y=None, groups=None):
            for train_idx, val_idx in self._splits:
                yield train_idx, val_idx

    @staticmethod
    def _outer_splits(X, y, groups, n_iterations, test_fraction,
                      random_state):
        """Yield stratified outer train/test splits, group-aware if needed.

        Without *groups* this matches the historical behaviour
        (StratifiedShuffleSplit).  With *groups*, folds come from
        re-seeded StratifiedGroupKFold sweeps so that no group ever
        appears on both sides of a split.
        """
        if groups is None:
            splitter = StratifiedShuffleSplit(
                n_splits=n_iterations,
                test_size=test_fraction,
                random_state=random_state,
            )
            yield from splitter.split(X, y)
            return

        n_splits = max(2, int(round(1.0 / test_fraction)))
        n_splits = min(n_splits, len(np.unique(groups)))
        produced = 0
        sweep = 0
        while produced < n_iterations:
            skf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True,
                random_state=random_state + sweep)
            for train_idx, test_idx in skf.split(X, y, groups):
                yield train_idx, test_idx
                produced += 1
                if produced >= n_iterations:
                    break
            sweep += 1

    @classmethod
    def _inner_splitter(cls, X, y, groups, n_splits, n_repeats,
                        random_state):
        """Build the inner-CV splitter, group-aware when *groups* given."""
        if groups is None:
            return RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats,
                random_state=random_state)

        n_splits = min(n_splits, len(np.unique(groups)))
        splits = []
        for repeat in range(n_repeats):
            skf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True,
                random_state=random_state + repeat)
            splits.extend(skf.split(X, y, groups))
        return cls._PrecomputedSplitter(splits)

    def _tune_hyperparams(self, X, y, splitter, model_name, use_scaler,
                          param_grid, random_state):
        """Grid-search hyperparameters over the inner CV splits.

        Returns the parameter dict with the highest mean validation
        accuracy (empty dict when the grid is empty).
        """
        if param_grid is None:
            param_grid = self._PARAM_GRIDS.get(model_name, {})
        candidates = list(ParameterGrid(param_grid))
        if len(candidates) <= 1:
            return candidates[0] if candidates else {}

        splits = list(splitter.split(X, y))
        best_score = -np.inf
        best_params = {}
        for params in candidates:
            scores = []
            for train_idx, val_idx in splits:
                est = self._make_model(model_name, random_state)
                est.set_params(**params)
                if use_scaler:
                    pipe = make_pipeline(StandardScaler(), est)
                else:
                    pipe = est
                # A candidate can be infeasible for small folds (e.g.
                # knn n_neighbors > fold size); skip it like
                # GridSearchCV's error_score instead of crashing.
                try:
                    pipe.fit(X[train_idx], y[train_idx])
                    scores.append(accuracy_score(
                        y[val_idx], pipe.predict(X[val_idx])))
                except Exception:
                    scores = []
                    break
            if not scores:
                continue
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        return best_params

    def _select_features(self, X, y, splitter, model_name, use_scaler,
                         method, max_size, random_state):
        """Dispatch to exhaustive or forward feature selection."""
        n_features = X.shape[1]
        if max_size is None:
            max_size = n_features
        if method == 'auto':
            method = 'exhaustive' if n_features <= 12 else 'forward'
        if method == 'exhaustive':
            return self._select_features_exhaustive(
                X, y, splitter, model_name, use_scaler, max_size,
                random_state)
        return self._select_features_forward(
            X, y, splitter, model_name, use_scaler, max_size,
            random_state)

    def _select_features_exhaustive(self, X, y, splitter, model_name,
                                    use_scaler, max_size, random_state):
        """Exhaustive best-subset selection over inner CV folds.

        Tests all feature subsets of size 1 through *max_size* and
        returns the subset with the highest mean inner-CV accuracy.
        """
        n_features = X.shape[1]
        best_score = -np.inf
        best_subset = tuple(range(n_features))
        for size in range(1, min(max_size, n_features) + 1):
            for subset in combinations(range(n_features), size):
                X_sub = X[:, list(subset)]
                scores = []
                for train_idx, val_idx in splitter.split(X_sub, y):
                    est = self._make_model(model_name, random_state)
                    if use_scaler:
                        pipe = make_pipeline(StandardScaler(), est)
                    else:
                        pipe = est
                    try:
                        pipe.fit(X_sub[train_idx], y[train_idx])
                        scores.append(accuracy_score(
                            y[val_idx], pipe.predict(X_sub[val_idx])))
                    except Exception:
                        scores = []
                        break
                if not scores:
                    continue
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_subset = subset
        return best_subset

    def _select_features_forward(self, X, y, splitter, model_name,
                                 use_scaler, max_size, random_state):
        """Forward sequential feature selection over inner CV folds.

        Greedily adds features that maximise mean inner-CV accuracy.
        Stops when no further improvement is found.
        """
        n_features = X.shape[1]
        max_size = min(max_size, n_features)
        selected = []
        remaining = list(range(n_features))
        prev_best_score = -np.inf
        for _ in range(max_size):
            best_score = -np.inf
            best_feat = None
            for feat in remaining:
                subset = selected + [feat]
                X_sub = X[:, subset]
                scores = []
                for train_idx, val_idx in splitter.split(X_sub, y):
                    est = self._make_model(model_name, random_state)
                    if use_scaler:
                        pipe = make_pipeline(StandardScaler(), est)
                    else:
                        pipe = est
                    try:
                        pipe.fit(X_sub[train_idx], y[train_idx])
                        scores.append(accuracy_score(
                            y[val_idx], pipe.predict(X_sub[val_idx])))
                    except Exception:
                        scores = []
                        break
                if not scores:
                    continue
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_feat = feat
            if best_feat is None or best_score <= prev_best_score:
                break
            selected.append(best_feat)
            remaining.remove(best_feat)
            prev_best_score = best_score
        return tuple(selected) if selected else tuple(range(n_features))

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_feature_table(
        self,
        signals_or_dir,
        labels=None,
        *,
        window_size,
        window_step=1,
        window_stats=('mean', 'median', 'mode'),
        include_params=True,
        group_level_params=None,
        rqa_kwargs=None,
    ):
        """Build a feature table from RQA2 measures and windowed summaries.

        For each signal the method computes:

        * Whole-signal RQA measures (recurrence rate, determinism, etc.).
        * Windowed RQA measures aggregated with the requested summary
          statistics (mean, median, mode).  These columns are prefixed
          with ``win_`` to avoid collisions.
        * Optionally the embedding parameters ``tau``, ``m``, ``eps``.

        Parameters
        ----------
        signals_or_dir : str, PathLike, ndarray, list, or tuple
            Input signals — see :meth:`_load_signals`.
        labels : array-like, optional
            Class labels aligned with the signals.  Required for
            supervised benchmarking later.
        window_size : int
            Size of the square sliding window on the recurrence plot.
        window_step : int, default 1
            Step size for the sliding window.
        window_stats : tuple of str, default ``('mean', 'median', 'mode')``
            Aggregate statistics computed over windowed measures.
        include_params : bool, default True
            Whether to include ``tau``, ``m``, ``eps`` as feature columns.
        group_level_params : set or list of str, optional
            Subset of ``{'tau', 'm', 'eps'}`` to estimate once across all
            signals and then apply uniformly (useful for fair cross-system
            comparisons).
        rqa_kwargs : dict, optional
            Per-call overrides merged on top of ``self.rqa_kwargs``.
            Keys ``tau``, ``m``, ``eps`` are extracted and applied as
            manual parameter overrides rather than passed to the RQA2
            constructor.

        Returns
        -------
        pandas.DataFrame
            One row per signal with columns: ``id``, (``label``),
            whole-signal measures, ``win_*`` windowed summaries, and
            optionally ``tau``, ``m``, ``eps``.
        """
        if window_size is None:
            raise ValueError("window_size must be provided.")

        signals, ids = self._load_signals(signals_or_dir)
        if labels is not None:
            if len(labels) != len(signals):
                raise ValueError(
                    "labels length must match number of signals.")

        kwargs = dict(self.rqa_kwargs)
        if rqa_kwargs:
            kwargs.update(rqa_kwargs)

        manual_params = {}
        for key in ('tau', 'm', 'eps'):
            if key in kwargs:
                manual_params[key] = kwargs.pop(key)

        group_params = {}
        if group_level_params:
            allowed = {'tau', 'm', 'eps'}
            unknown = set(group_level_params) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown group_level_params: {sorted(unknown)}")

            rqa_objects = []
            records = []
            for signal in signals:
                rqa = RQA2(signal, **kwargs)
                record = {}
                if 'tau' in group_level_params:
                    record['tau'] = rqa.compute_time_delay()
                if 'm' in group_level_params:
                    record['m'] = rqa.compute_embedding_dimension()
                if 'eps' in group_level_params:
                    record['eps'] = rqa.compute_neighborhood_radius()
                rqa_objects.append(rqa)
                records.append(record)

            if 'tau' in group_level_params:
                group_params['tau'] = int(
                    np.mean([r['tau'] for r in records]))
            if 'm' in group_level_params:
                group_params['m'] = int(
                    np.mean([r['m'] for r in records]))
            if 'eps' in group_level_params:
                group_params['eps'] = RQA2._compute_group_epsilon(
                    rqa_objects)

        rows = []
        for idx, signal in enumerate(signals):
            rqa = RQA2(signal, **kwargs)

            if 'tau' in manual_params:
                rqa._tau = int(manual_params['tau'])
            if 'm' in manual_params:
                rqa._m = int(manual_params['m'])
            if 'eps' in manual_params:
                rqa._eps = float(manual_params['eps'])

            if group_params:
                if 'tau' in group_params:
                    rqa._tau = int(group_params['tau'])
                if 'm' in group_params:
                    rqa._m = int(group_params['m'])
                if 'eps' in group_params:
                    rqa._eps = float(group_params['eps'])

            measures = rqa.compute_rqa_measures()
            windowed = rqa.compute_windowed_rqa_measures(
                window_size, window_step=window_step)
            summary = rqa.summarize_windowed_measures(
                windowed, stats=window_stats)

            row = {'id': ids[idx], **measures}
            row.update({f"win_{k}": v for k, v in summary.items()})

            if include_params:
                row['tau'] = rqa.tau
                row['m'] = rqa.m
                row['eps'] = rqa.eps
            if labels is not None:
                row['label'] = labels[idx]
            rows.append(row)

        df = pd.DataFrame(rows)
        if 'label' in df.columns:
            cols = (['id', 'label']
                    + [c for c in df.columns if c not in ('id', 'label')])
        else:
            cols = ['id'] + [c for c in df.columns if c != 'id']
        return df[cols]

    # ------------------------------------------------------------------
    # Supervised: nested cross-validation (paper methodology)
    # ------------------------------------------------------------------

    def nested_cv_benchmark(
        self, X, y, *,
        model='knn',
        outer_iterations=100,
        test_fraction=1.0 / 3,
        inner_splits=2,
        inner_iterations=10,
        feature_selection='auto',
        max_subset_size=None,
        tune=False,
        param_grid=None,
        groups=None,
        scaler=True,
        random_state=42,
    ):
        """Nested cross-validation with feature selection and tuning.

        Implements the validation procedure described in the SMdRQA paper:
        the outer loop evaluates generalisation performance on held-out
        data, while the inner loop performs feature selection — and,
        with ``tune=True``, hyperparameter search — exclusively on the
        training fold to prevent data leakage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (e.g. from :meth:`build_feature_table`).
        y : array-like of shape (n_samples,)
            Target labels.  Must contain at least two unique classes.
        model : str, default ``'knn'``
            Any key of ``_MODEL_REGISTRY``: ``'knn'``, ``'svm'``,
            ``'rf'``, ``'logreg'``, ``'lda'``, ``'nb'``, ``'gb'``,
            ``'et'``.
        outer_iterations : int, default 100
            Number of stratified train/test splits.
        test_fraction : float, default 1/3
            Fraction of data held out as the test set in each outer
            iteration.
        inner_splits : int, default 2
            Number of folds in the inner stratified CV used for feature
            selection and tuning.
        inner_iterations : int, default 10
            Number of repeats of the inner CV.
        feature_selection : {``'auto'``, ``'exhaustive'``, ``'forward'``,
                             ``None``}, default ``'auto'``
            Feature selection strategy.  ``'auto'`` uses exhaustive search
            when the number of features is ≤ 12, otherwise forward
            sequential selection.  ``None`` disables feature selection.
        max_subset_size : int, optional
            Maximum number of features to select.  Defaults to all.
        tune : bool, default ``False``
            When ``True``, grid-search the model hyperparameters on the
            inner CV of each outer training fold (after feature
            selection) instead of using fixed defaults.
        param_grid : dict or list of dict, optional
            Hyperparameter grid to search when ``tune=True``.  Defaults
            to ``_PARAM_GRIDS[model]``.
        groups : array-like of shape (n_samples,), optional
            Group labels (e.g. subject or recording IDs).  When given,
            both outer and inner splits keep each group entirely on one
            side, preventing group-level leakage.
        scaler : bool, default True
            Whether to prepend a
            :class:`~sklearn.preprocessing.StandardScaler`.
        random_state : int, default 42
            Seed for reproducibility.

        Returns
        -------
        dict
            ``'accuracy'`` : ndarray of shape (*outer_iterations*,)
                Test-set accuracy per outer iteration.
            ``'balanced_accuracy'`` : ndarray
                Test-set balanced accuracy per outer iteration.
            ``'f1_macro'`` : ndarray
                Test-set macro-averaged F1 per outer iteration.
            ``'roc_auc'`` : ndarray of shape (*outer_iterations*,)
                Test-set ROC AUC per outer iteration.
            ``'selected_features'`` : list of tuple
                Feature indices selected in each outer iteration.
            ``'best_params'`` : list of dict
                Hyperparameters chosen in each outer iteration
                (empty dicts when ``tune=False``).
            ``'feature_names'`` : list of str or None
                Column names if *X* is a DataFrame.
            ``'feature_frequency'`` : Series
                How often each feature was selected across iterations.
            ``'model'`` : str
                The model name used.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)
        groups_arr = None if groups is None else np.asarray(groups)
        feature_names = (list(X.columns)
                         if isinstance(X, pd.DataFrame) else None)

        if len(np.unique(y_arr)) < 2:
            raise ValueError("y must contain at least two classes.")
        if groups_arr is not None and len(groups_arr) != len(y_arr):
            raise ValueError(
                "groups length must match number of samples.")

        accuracies = []
        balanced_accuracies = []
        f1_macros = []
        roc_aucs = []
        selected_features_list = []
        best_params_list = []

        outer = self._outer_splits(
            X_arr, y_arr, groups_arr, outer_iterations, test_fraction,
            random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(outer):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            groups_train = (None if groups_arr is None
                            else groups_arr[train_idx])

            inner_splitter = self._inner_splitter(
                X_train, y_train, groups_train, inner_splits,
                inner_iterations, random_state + fold_idx)

            # Inner CV for feature selection
            if feature_selection is not None:
                subset = self._select_features(
                    X_train, y_train, inner_splitter, model, scaler,
                    feature_selection, max_subset_size, random_state)
            else:
                subset = tuple(range(X_arr.shape[1]))

            selected_features_list.append(subset)

            X_train_sub = X_train[:, list(subset)]
            X_test_sub = X_test[:, list(subset)]

            # Inner CV for hyperparameter tuning on the selected subset
            if tune:
                best_params = self._tune_hyperparams(
                    X_train_sub, y_train, inner_splitter, model, scaler,
                    param_grid, random_state)
            else:
                best_params = {}
            best_params_list.append(best_params)

            # Train on full training set with selected features/params
            est = self._make_model(model, random_state)
            if best_params:
                est.set_params(**best_params)
            if scaler:
                pipe = make_pipeline(StandardScaler(), est)
            else:
                pipe = est
            pipe.fit(X_train_sub, y_train)

            preds = pipe.predict(X_test_sub)
            accuracies.append(accuracy_score(y_test, preds))
            balanced_accuracies.append(
                balanced_accuracy_score(y_test, preds))
            f1_macros.append(f1_score(y_test, preds, average='macro'))
            roc_aucs.append(
                self._compute_roc_auc(pipe, X_test_sub, y_test))

        # Feature selection frequency
        n_features = X_arr.shape[1]
        freq = np.zeros(n_features)
        for subset in selected_features_list:
            for idx in subset:
                freq[idx] += 1
        names = (feature_names if feature_names
                 else [f"feature_{i}" for i in range(n_features)])
        freq_series = pd.Series(
            freq, index=names).sort_values(ascending=False)

        return {
            'accuracy': np.array(accuracies),
            'balanced_accuracy': np.array(balanced_accuracies),
            'f1_macro': np.array(f1_macros),
            'roc_auc': np.array(roc_aucs),
            'selected_features': selected_features_list,
            'best_params': best_params_list,
            'feature_names': feature_names,
            'feature_frequency': freq_series,
            'model': model,
        }

    # ------------------------------------------------------------------
    # Supervised: quick benchmark (convenience wrapper)
    # ------------------------------------------------------------------

    def supervised_benchmark(
        self, X, y, *,
        models=('knn', 'svm', 'rf'),
        cv=5,
        scaler=True,
        random_state=42,
    ):
        """Quick supervised benchmark with stratified cross-validation.

        This is a lighter alternative to :meth:`nested_cv_benchmark`
        for exploratory analysis.  It does **not** perform feature
        selection; for paper-quality validation use
        :meth:`nested_cv_benchmark` instead.

        After cross-validation the best-performing model (by accuracy,
        then macro-F1 as tiebreaker) is refit on the **full** dataset
        and returned alongside the per-model results table.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        models : tuple of str or ``'all'``, default ``('knn', 'svm', 'rf')``
            Model names to benchmark; ``'all'`` runs every registered
            model.
        cv : int, default 5
        scaler : bool, default True
        random_state : int, default 42

        Returns
        -------
        results_df : pandas.DataFrame
            Columns: ``model``, ``accuracy_mean``, ``accuracy_std``,
            ``f1_macro_mean``, ``f1_macro_std``, ``roc_auc_mean``,
            ``roc_auc_std``.
        best_model : sklearn estimator
            Best classifier refit on all of *X* and *y*.
        """
        if models == 'all':
            models = tuple(sorted(self._MODEL_REGISTRY))
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)

        if len(np.unique(y_arr)) < 2:
            raise ValueError("y must contain at least two classes.")

        splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state)
        results = []
        best_name = None
        best_score = (-np.inf, -np.inf)

        for name in models:
            base_est = self._make_model(name, random_state)
            acc_scores, f1_scores, auc_scores = [], [], []
            for train_idx, test_idx in splitter.split(X_arr, y_arr):
                est = clone(base_est)
                if scaler:
                    pipe = make_pipeline(StandardScaler(), est)
                else:
                    pipe = est
                pipe.fit(X_arr[train_idx], y_arr[train_idx])
                preds = pipe.predict(X_arr[test_idx])
                acc_scores.append(
                    accuracy_score(y_arr[test_idx], preds))
                f1_scores.append(
                    f1_score(y_arr[test_idx], preds, average='macro'))
                auc_scores.append(
                    self._compute_roc_auc(
                        pipe, X_arr[test_idx], y_arr[test_idx]))

            acc_mean = float(np.mean(acc_scores))
            f1_mean = float(np.mean(f1_scores))
            results.append({
                'model': name,
                'accuracy_mean': acc_mean,
                'accuracy_std': float(np.std(acc_scores)),
                'f1_macro_mean': f1_mean,
                'f1_macro_std': float(np.std(f1_scores)),
                'roc_auc_mean': float(np.nanmean(auc_scores)),
                'roc_auc_std': float(np.nanstd(auc_scores)),
            })
            if (acc_mean, f1_mean) > best_score:
                best_score = (acc_mean, f1_mean)
                best_name = name

        results_df = pd.DataFrame(results)

        best_est = self._make_model(best_name, random_state)
        if scaler:
            best_model = make_pipeline(StandardScaler(), best_est)
        else:
            best_model = best_est
        best_model.fit(X_arr, y_arr)

        return results_df, best_model

    # ------------------------------------------------------------------
    # Integrated end-to-end pipeline
    # ------------------------------------------------------------------

    def integrated_benchmark(
        self,
        signals_or_dir=None,
        labels=None,
        *,
        features=None,
        window_size=None,
        window_step=1,
        models='all',
        tune=True,
        feature_selection='auto',
        outer_iterations=50,
        test_fraction=1.0 / 3,
        inner_splits=2,
        inner_iterations=5,
        groups=None,
        scaler=True,
        alpha=0.05,
        random_state=42,
        rqa_kwargs=None,
        verbose=True,
        progress_callback=None,
    ):
        """End-to-end pipeline: signals → features → tuned nested CV →
        statistical model comparison.

        Runs :meth:`build_feature_table` (unless a precomputed
        *features* table is supplied), benchmarks every requested model
        with :meth:`nested_cv_benchmark` (hyperparameter tuning on by
        default), compares all model pairs with Wilcoxon signed-rank
        tests under Benjamini–Hochberg correction, and refits the best
        model on the full dataset.

        Parameters
        ----------
        signals_or_dir : str, PathLike, ndarray, list, or tuple, optional
            Input signals — see :meth:`_load_signals`.  Not needed when
            *features* is given.
        labels : array-like, optional
            Class labels aligned with the signals.  Required unless
            *features* contains a ``label`` column.
        features : pandas.DataFrame, optional
            Precomputed feature table (e.g. from
            :meth:`build_feature_table`).  Columns ``id`` and ``label``
            are treated as metadata, everything else as features.
        window_size : int, optional
            Sliding-window size forwarded to
            :meth:`build_feature_table`.  Required when building
            features from signals.
        window_step : int, default 1
            Sliding-window step forwarded to
            :meth:`build_feature_table`.
        models : tuple of str or ``'all'``, default ``'all'``
            Model names to benchmark.
        tune : bool, default ``True``
            Grid-search hyperparameters in the inner CV loop.
        feature_selection : {``'auto'``, ``'exhaustive'``, ``'forward'``,
                             ``None``}, default ``'auto'``
        outer_iterations : int, default 50
        test_fraction : float, default 1/3
        inner_splits : int, default 2
        inner_iterations : int, default 5
        groups : array-like, optional
            Group labels (one per sample) for leakage-free splitting —
            see :meth:`nested_cv_benchmark`.
        scaler : bool, default True
        alpha : float, default 0.05
            FDR level for the Benjamini–Hochberg correction of the
            pairwise model comparisons.
        random_state : int, default 42
        rqa_kwargs : dict, optional
            Forwarded to :meth:`build_feature_table`.
        verbose : bool, default True
            Print progress messages.
        progress_callback : callable, optional
            Called as ``progress_callback(index, total, model_name)``
            before each model's nested CV run (e.g. to drive a UI
            progress bar).

        Returns
        -------
        dict
            ``'features'`` : DataFrame — the feature table used.
            ``'results'`` : dict — model name → full
            :meth:`nested_cv_benchmark` result dict.
            ``'comparison'`` : DataFrame — one row per model with mean and
            std of every metric, sorted by mean accuracy.
            ``'pairwise_tests'`` : DataFrame — Wilcoxon test per model
            pair with BH-corrected significance.
            ``'best_model_name'`` : str
            ``'best_model'`` : fitted sklearn estimator — best model
            refit on the full dataset.
        """
        if features is None:
            if signals_or_dir is None:
                raise ValueError(
                    "Provide either signals_or_dir or features.")
            if window_size is None:
                raise ValueError(
                    "window_size is required when building features "
                    "from signals.")
            features = self.build_feature_table(
                signals_or_dir, labels,
                window_size=window_size, window_step=window_step,
                rqa_kwargs=rqa_kwargs)

        meta_cols = [c for c in ('id', 'label') if c in features.columns]
        X = features.drop(columns=meta_cols)
        if labels is not None and len(labels) == len(features):
            y = np.asarray(labels)
        elif 'label' in features.columns:
            y = features['label'].to_numpy()
        else:
            raise ValueError(
                "labels must be given or features must contain a "
                "'label' column.")

        if models == 'all':
            models = tuple(sorted(self._MODEL_REGISTRY))

        results = {}
        comparison_rows = []
        for model_idx, name in enumerate(models):
            if progress_callback is not None:
                progress_callback(model_idx, len(models), name)
            if verbose:
                print(f"[integrated_benchmark] nested CV: {name}")
            res = self.nested_cv_benchmark(
                X, y,
                model=name,
                outer_iterations=outer_iterations,
                test_fraction=test_fraction,
                inner_splits=inner_splits,
                inner_iterations=inner_iterations,
                feature_selection=feature_selection,
                tune=tune,
                groups=groups,
                scaler=scaler,
                random_state=random_state,
            )
            results[name] = res
            comparison_rows.append({
                'model': name,
                'accuracy_mean': float(np.mean(res['accuracy'])),
                'accuracy_std': float(np.std(res['accuracy'])),
                'balanced_accuracy_mean': float(
                    np.mean(res['balanced_accuracy'])),
                'balanced_accuracy_std': float(
                    np.std(res['balanced_accuracy'])),
                'f1_macro_mean': float(np.mean(res['f1_macro'])),
                'f1_macro_std': float(np.std(res['f1_macro'])),
                'roc_auc_mean': float(np.nanmean(res['roc_auc'])),
                'roc_auc_std': float(np.nanstd(res['roc_auc'])),
            })

        comparison = pd.DataFrame(comparison_rows).sort_values(
            'accuracy_mean', ascending=False).reset_index(drop=True)

        # Pairwise Wilcoxon tests with BH correction
        pairs = list(combinations(models, 2))
        pair_rows = []
        p_values = []
        for name_a, name_b in pairs:
            a = results[name_a]['accuracy']
            b = results[name_b]['accuracy']
            if np.allclose(a, b):
                test = {'statistic': np.nan, 'p_value': 1.0,
                        'effect_size': 0.0, 'n': len(a)}
            else:
                test = self.compare_scores(a, b)
            pair_rows.append({
                'model_a': name_a,
                'model_b': name_b,
                'statistic': test['statistic'],
                'p_value': test['p_value'],
                'effect_size': test['effect_size'],
            })
            p_values.append(test['p_value'])

        if pair_rows:
            p_dict = {f"{a}|{b}": p
                      for (a, b), p in zip(pairs, p_values)}
            adjusted, rejected = self._benjamini_hochberg(
                p_dict, alpha=alpha)
            for row, (name_a, name_b) in zip(pair_rows, pairs):
                key = f"{name_a}|{name_b}"
                row['p_adjusted'] = adjusted[key]
                row['significant'] = rejected[key]
        pairwise_tests = pd.DataFrame(pair_rows)

        # Refit the best model (by mean accuracy) on the full dataset
        best_name = comparison.iloc[0]['model']
        best_est = self._make_model(best_name, random_state)
        if tune:
            full_splitter = self._inner_splitter(
                np.asarray(X, dtype=float), y,
                None if groups is None else np.asarray(groups),
                inner_splits, inner_iterations, random_state)
            best_params = self._tune_hyperparams(
                np.asarray(X, dtype=float), y, full_splitter,
                best_name, scaler, None, random_state)
            if best_params:
                best_est.set_params(**best_params)
        if scaler:
            best_model = make_pipeline(StandardScaler(), best_est)
        else:
            best_model = best_est
        best_model.fit(X, y)

        return {
            'features': features,
            'results': results,
            'comparison': comparison,
            'pairwise_tests': pairwise_tests,
            'best_model_name': best_name,
            'best_model': best_model,
        }

    # ------------------------------------------------------------------
    # Supervised: surrogate null baseline
    # ------------------------------------------------------------------

    def surrogate_baseline(
        self, X, y, *,
        n_permutations=100,
        model='knn',
        cv=5,
        scaler=True,
        random_state=42,
    ):
        """Null performance distribution by permuting labels.

        Shuffles *y* ``n_permutations`` times and evaluates each with
        stratified *k*-fold CV, yielding a null distribution against
        which real performance can be compared via
        :meth:`compare_scores`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_permutations : int, default 100
        model : str, default ``'knn'``
        cv : int, default 5
        scaler : bool, default True
        random_state : int, default 42

        Returns
        -------
        dict
            ``'null_accuracy'`` : ndarray of shape (*n_permutations*,)
            ``'null_roc_auc'`` : ndarray of shape (*n_permutations*,)
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)
        rng = np.random.default_rng(random_state)

        null_accs, null_aucs = [], []
        for i in range(n_permutations):
            y_perm = rng.permutation(y_arr)
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True,
                random_state=random_state + i)
            fold_accs, fold_aucs = [], []
            for train_idx, test_idx in splitter.split(X_arr, y_perm):
                est = self._make_model(model, random_state)
                if scaler:
                    pipe = make_pipeline(StandardScaler(), est)
                else:
                    pipe = est
                pipe.fit(X_arr[train_idx], y_perm[train_idx])
                preds = pipe.predict(X_arr[test_idx])
                fold_accs.append(
                    accuracy_score(y_perm[test_idx], preds))
                fold_aucs.append(
                    self._compute_roc_auc(
                        pipe, X_arr[test_idx], y_perm[test_idx]))
            null_accs.append(float(np.mean(fold_accs)))
            null_aucs.append(float(np.nanmean(fold_aucs)))

        return {
            'null_accuracy': np.array(null_accs),
            'null_roc_auc': np.array(null_aucs),
        }

    # ------------------------------------------------------------------
    # Internal: surrogate signal generation & statistical helpers
    # ------------------------------------------------------------------

    def _generate_surrogate_signals(self, signals, kind, *,
                                    surrogate_kwargs=None,
                                    random_state=42):
        """Generate one surrogate time series per input signal.

        Parameters
        ----------
        signals : list of ndarray
            Original time-series signals.
        kind : str
            Surrogate algorithm forwarded to
            :meth:`RQA2_tests.generate`.
        surrogate_kwargs : dict, optional
            Extra keyword arguments for the surrogate algorithm
            (e.g. ``n_iter`` for IAAFT).
        random_state : int, default 42

        Returns
        -------
        list of ndarray
            One surrogate signal per input signal.
        """
        surr_signals = []
        rng = np.random.default_rng(random_state)
        kw = dict(surrogate_kwargs or {})
        for sig in signals:
            seed = int(rng.integers(1_000_000_000))
            tester = RQA2_tests(
                signal=np.asarray(sig, dtype=float),
                seed=seed, max_workers=1)
            surrogates = tester.generate(kind=kind, n_surrogates=1,
                                         **kw)
            surr_signals.append(surrogates[0])
        return surr_signals

    @staticmethod
    def _rank_p_value(real_score, null_scores, alternative='greater'):
        """Monte Carlo rank-based p-value.

        Parameters
        ----------
        real_score : float
            Observed test statistic.
        null_scores : array-like
            Null distribution values.
        alternative : str, default ``'greater'``
            ``'greater'``: fraction of null values ≥ real_score.
            ``'less'``: fraction of null values ≤ real_score.

        Returns
        -------
        float
            p-value in ``[1/(B+1), 1]`` where *B* = len(null_scores).
        """
        null = np.asarray(null_scores, dtype=float)
        B = len(null)
        if alternative == 'greater':
            count = np.sum(null >= real_score)
        else:
            count = np.sum(null <= real_score)
        return (count + 1) / (B + 1)

    @staticmethod
    def _benjamini_hochberg(p_values, alpha=0.05):
        """Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        p_values : dict
            Mapping of label → raw p-value.
        alpha : float, default 0.05
            Target false-discovery rate.

        Returns
        -------
        adjusted : dict
            Mapping of label → adjusted p-value.
        rejected : dict
            Mapping of label → bool (significant after correction).
        """
        keys = list(p_values.keys())
        pvals = np.array([p_values[k] for k in keys])
        m = len(pvals)
        if m == 0:
            return {}, {}

        order = np.argsort(pvals)
        ranks = np.empty(m, dtype=float)
        ranks[order] = np.arange(1, m + 1)
        adjusted = np.minimum(1.0, pvals * m / ranks)

        # Enforce monotonicity (step-down from largest)
        sorted_idx = np.argsort(pvals)[::-1]
        for i in range(1, m):
            adjusted[sorted_idx[i]] = min(
                adjusted[sorted_idx[i]],
                adjusted[sorted_idx[i - 1]])

        rejected = adjusted <= alpha
        return (
            {k: float(adjusted[i]) for i, k in enumerate(keys)},
            {k: bool(rejected[i]) for i, k in enumerate(keys)},
        )

    # ------------------------------------------------------------------
    # Supervised: surrogate-based null hypothesis testing
    # ------------------------------------------------------------------

    def surrogate_null_benchmark(
        self, signals, labels, *,
        window_size,
        window_step=1,
        window_stats=('mean', 'median', 'mode'),
        include_params=True,
        group_level_params=None,
        rqa_kwargs=None,
        surrogate_kinds=('FT', 'AAFT', 'IAAFT'),
        n_surrogate_iterations=20,
        surrogate_kwargs=None,
        model='knn',
        outer_iterations=100,
        surrogate_outer_iterations=30,
        test_fraction=1.0 / 3,
        inner_splits=2,
        inner_iterations=10,
        feature_selection='auto',
        max_subset_size=None,
        scaler=True,
        random_state=42,
        alpha=0.05,
        correction='fdr_bh',
        include_permutation=True,
        n_permutations=100,
        verbose=True,
    ):
        """Surrogate-based null hypothesis testing for classification.

        For each surrogate algorithm the method replaces every original
        time series with a surrogate, re-extracts RQA features, and runs
        the full nested cross-validation procedure.  Repeating this
        ``n_surrogate_iterations`` times yields a null distribution of
        classification accuracy that would be obtained from data whose
        nonlinear structure has been destroyed according to the specific
        null hypothesis encoded by the surrogate type:

        * **FT** — preserves power spectrum (null: linear autocorrelation
          alone explains classification).
        * **AAFT** — preserves amplitude distribution + spectrum (null:
          linear structure + marginal distribution suffice).
        * **IAAFT** — tighter amplitude + spectrum match.
        * **WIAAFT** — preserves time-frequency structure.
        * **PPS** — preserves periodic structure (null: periodicity alone
          suffices).

        Optionally a label-permutation baseline (the classical approach)
        is computed alongside for comparison.  All p-values are corrected
        for multiple comparisons using Benjamini-Hochberg FDR.

        Parameters
        ----------
        signals : list of ndarray
            Raw time-series signals (same order as *labels*).
        labels : array-like
            Class labels aligned with *signals*.
        window_size : int
            Forwarded to :meth:`build_feature_table`.
        window_step : int, default 1
        window_stats : tuple of str, default ``('mean', 'median', 'mode')``
        include_params : bool, default True
        group_level_params : set, optional
        rqa_kwargs : dict, optional
        surrogate_kinds : tuple of str, default ``('FT', 'AAFT', 'IAAFT')``
            Surrogate algorithms to test.
        n_surrogate_iterations : int, default 20
            How many surrogate datasets to generate per algorithm.
        surrogate_kwargs : dict, optional
            Extra keyword arguments for surrogate generation.
        model : str, default ``'knn'``
        outer_iterations : int, default 100
            Outer CV iterations for the **real** data.
        surrogate_outer_iterations : int, default 30
            Outer CV iterations for each surrogate dataset (lower for
            computational tractability).
        test_fraction : float, default 1/3
        inner_splits : int, default 2
        inner_iterations : int, default 10
        feature_selection : str or None, default ``'auto'``
        max_subset_size : int, optional
        scaler : bool, default True
        random_state : int, default 42
        alpha : float, default 0.05
            Significance level for FDR correction.
        correction : str, default ``'fdr_bh'``
            ``'fdr_bh'`` for Benjamini-Hochberg; ``'bonferroni'`` for
            Bonferroni correction.
        include_permutation : bool, default True
            Whether to also compute a label-permutation null.
        n_permutations : int, default 100
        verbose : bool, default True
            Show progress bars.

        Returns
        -------
        dict
            ``'real'`` : dict — output of :meth:`nested_cv_benchmark`
            on the real data.

            ``'surrogates'`` : dict of {str: dict} — per-surrogate-kind
            results containing ``'null_accuracies'``,
            ``'null_roc_aucs'``, ``'p_value_accuracy'``,
            ``'p_value_roc_auc'``, ``'effect_size'``.

            ``'permutation'`` : dict or ``None`` — label-permutation
            null (if *include_permutation* is True).

            ``'corrected_p_values'`` : dict — FDR-corrected p-values
            per surrogate kind and metric.

            ``'summary'`` : DataFrame — one-row-per-null summary table.
        """
        _valid_surrogates = ('FT', 'AAFT', 'IAAFT', 'IDFS',
                             'WIAAFT', 'PPS')
        for kind in surrogate_kinds:
            if kind not in _valid_surrogates:
                raise ValueError(
                    f"Unknown surrogate kind '{kind}'. "
                    f"Available: {_valid_surrogates}")

        labels_arr = np.asarray(labels)

        # Common feature-table kwargs
        ft_kwargs = dict(
            window_size=window_size,
            window_step=window_step,
            window_stats=window_stats,
            include_params=include_params,
            group_level_params=group_level_params,
            rqa_kwargs=rqa_kwargs,
        )

        # Common nested-CV kwargs (for surrogates, use lighter settings)
        cv_base = dict(
            model=model,
            test_fraction=test_fraction,
            inner_splits=inner_splits,
            inner_iterations=inner_iterations,
            feature_selection=feature_selection,
            max_subset_size=max_subset_size,
            scaler=scaler,
            random_state=random_state,
        )

        # --- Step 1: Real data benchmark ---
        if verbose:
            print("Building feature table from real signals...")
        real_features = self.build_feature_table(
            signals, labels=labels, **ft_kwargs)

        feature_cols = [c for c in real_features.columns
                        if c not in ('id', 'label')]
        X_real = real_features[feature_cols].values
        y_real = real_features['label'].values

        if verbose:
            print("Running nested CV on real data...")
        real_results = self.nested_cv_benchmark(
            X_real, y_real,
            outer_iterations=outer_iterations, **cv_base)

        real_mean_acc = float(np.mean(real_results['accuracy']))
        real_mean_auc = float(np.nanmean(real_results['roc_auc']))

        # --- Step 2: Surrogate null distributions ---
        rng = np.random.default_rng(random_state)
        surrogate_results = {}

        for kind in surrogate_kinds:
            null_accs = []
            null_aucs = []
            iterator = range(n_surrogate_iterations)
            if verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"Surrogate null ({kind})",
                    leave=True)

            for it in iterator:
                seed = int(rng.integers(1_000_000_000))

                # Generate surrogates
                surr_signals = self._generate_surrogate_signals(
                    signals, kind,
                    surrogate_kwargs=surrogate_kwargs,
                    random_state=seed)

                # Build features from surrogates
                surr_features = self.build_feature_table(
                    surr_signals, labels=labels, **ft_kwargs)
                X_surr = surr_features[feature_cols].values

                # Run nested CV on surrogate features
                surr_cv = self.nested_cv_benchmark(
                    X_surr, y_real,
                    outer_iterations=surrogate_outer_iterations,
                    **cv_base)
                null_accs.append(float(np.mean(surr_cv['accuracy'])))
                null_aucs.append(float(np.nanmean(
                    surr_cv['roc_auc'])))

            null_accs = np.array(null_accs)
            null_aucs = np.array(null_aucs)

            # Rank-based p-values
            p_acc = self._rank_p_value(real_mean_acc, null_accs)
            p_auc = self._rank_p_value(real_mean_auc, null_aucs)

            # Effect size (Cohen's d)
            d_acc = ((real_mean_acc - np.mean(null_accs))
                     / max(np.std(null_accs), 1e-12))
            d_auc = ((real_mean_auc - np.nanmean(null_aucs))
                     / max(np.nanstd(null_aucs), 1e-12))

            surrogate_results[kind] = {
                'null_accuracies': null_accs,
                'null_roc_aucs': null_aucs,
                'null_mean_accuracy': float(np.mean(null_accs)),
                'null_std_accuracy': float(np.std(null_accs)),
                'null_mean_roc_auc': float(np.nanmean(null_aucs)),
                'null_std_roc_auc': float(np.nanstd(null_aucs)),
                'p_value_accuracy': p_acc,
                'p_value_roc_auc': p_auc,
                'effect_size_accuracy': d_acc,
                'effect_size_roc_auc': d_auc,
            }

        # --- Step 3: Optional label-permutation null ---
        permutation_results = None
        if include_permutation:
            if verbose:
                print("Running label-permutation null...")
            min_class = min(np.bincount(
                np.unique(y_real, return_inverse=True)[1]))
            perm_cv = min(5, min_class)
            permutation_results = self.surrogate_baseline(
                X_real, y_real,
                n_permutations=n_permutations,
                model=model, cv=max(2, perm_cv), scaler=scaler,
                random_state=random_state)
            perm_mean_acc = float(np.mean(
                permutation_results['null_accuracy']))
            perm_mean_auc = float(np.nanmean(
                permutation_results['null_roc_auc']))
            permutation_results['p_value_accuracy'] = (
                self._rank_p_value(
                    real_mean_acc,
                    permutation_results['null_accuracy']))
            permutation_results['p_value_roc_auc'] = (
                self._rank_p_value(
                    real_mean_auc,
                    permutation_results['null_roc_auc']))
            d_acc_perm = (
                (real_mean_acc - perm_mean_acc)
                / max(np.std(
                    permutation_results['null_accuracy']), 1e-12))
            d_auc_perm = (
                (real_mean_auc - perm_mean_auc)
                / max(np.nanstd(
                    permutation_results['null_roc_auc']), 1e-12))
            permutation_results['effect_size_accuracy'] = d_acc_perm
            permutation_results['effect_size_roc_auc'] = d_auc_perm

        # --- Step 4: Multiple comparison correction ---
        all_p_acc = {k: v['p_value_accuracy']
                     for k, v in surrogate_results.items()}
        all_p_auc = {k: v['p_value_roc_auc']
                     for k, v in surrogate_results.items()}
        if include_permutation:
            all_p_acc['permutation'] = (
                permutation_results['p_value_accuracy'])
            all_p_auc['permutation'] = (
                permutation_results['p_value_roc_auc'])

        if correction == 'fdr_bh':
            adj_acc, rej_acc = self._benjamini_hochberg(
                all_p_acc, alpha)
            adj_auc, rej_auc = self._benjamini_hochberg(
                all_p_auc, alpha)
        elif correction == 'bonferroni':
            m = len(all_p_acc)
            adj_acc = {k: min(1.0, v * m)
                       for k, v in all_p_acc.items()}
            rej_acc = {k: v <= alpha for k, v in adj_acc.items()}
            adj_auc = {k: min(1.0, v * m)
                       for k, v in all_p_auc.items()}
            rej_auc = {k: v <= alpha for k, v in adj_auc.items()}
        else:
            adj_acc, rej_acc = all_p_acc, {
                k: v <= alpha for k, v in all_p_acc.items()}
            adj_auc, rej_auc = all_p_auc, {
                k: v <= alpha for k, v in all_p_auc.items()}

        corrected = {
            'accuracy': {
                'adjusted_p': adj_acc,
                'rejected': rej_acc,
            },
            'roc_auc': {
                'adjusted_p': adj_auc,
                'rejected': rej_auc,
            },
        }

        # --- Step 5: Summary table ---
        rows = []
        for kind in surrogate_kinds:
            sr = surrogate_results[kind]
            rows.append({
                'null_type': kind,
                'null_hypothesis': _SURROGATE_NULL_DESCRIPTIONS.get(
                    kind, kind),
                'real_accuracy': real_mean_acc,
                'null_mean_accuracy': sr['null_mean_accuracy'],
                'null_std_accuracy': sr['null_std_accuracy'],
                'p_value_accuracy': sr['p_value_accuracy'],
                'adjusted_p_accuracy': adj_acc[kind],
                'significant_accuracy': rej_acc[kind],
                'effect_size_accuracy': sr['effect_size_accuracy'],
                'real_roc_auc': real_mean_auc,
                'null_mean_roc_auc': sr['null_mean_roc_auc'],
                'null_std_roc_auc': sr['null_std_roc_auc'],
                'p_value_roc_auc': sr['p_value_roc_auc'],
                'adjusted_p_roc_auc': adj_auc[kind],
                'significant_roc_auc': rej_auc[kind],
                'effect_size_roc_auc': sr['effect_size_roc_auc'],
            })
        if include_permutation:
            rows.append({
                'null_type': 'permutation',
                'null_hypothesis': 'Labels unrelated to features',
                'real_accuracy': real_mean_acc,
                'null_mean_accuracy': float(np.mean(
                    permutation_results['null_accuracy'])),
                'null_std_accuracy': float(np.std(
                    permutation_results['null_accuracy'])),
                'p_value_accuracy': (
                    permutation_results['p_value_accuracy']),
                'adjusted_p_accuracy': adj_acc.get(
                    'permutation', np.nan),
                'significant_accuracy': rej_acc.get(
                    'permutation', False),
                'effect_size_accuracy': (
                    permutation_results['effect_size_accuracy']),
                'real_roc_auc': real_mean_auc,
                'null_mean_roc_auc': float(np.nanmean(
                    permutation_results['null_roc_auc'])),
                'null_std_roc_auc': float(np.nanstd(
                    permutation_results['null_roc_auc'])),
                'p_value_roc_auc': (
                    permutation_results['p_value_roc_auc']),
                'adjusted_p_roc_auc': adj_auc.get(
                    'permutation', np.nan),
                'significant_roc_auc': rej_auc.get(
                    'permutation', False),
                'effect_size_roc_auc': (
                    permutation_results['effect_size_roc_auc']),
            })

        summary = pd.DataFrame(rows)

        return {
            'real': real_results,
            'surrogates': surrogate_results,
            'permutation': permutation_results,
            'corrected_p_values': corrected,
            'summary': summary,
        }

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_scores(scores_a, scores_b, *,
                       alternative='two-sided'):
        """Wilcoxon signed-rank test with rank-biserial effect size.

        Parameters
        ----------
        scores_a, scores_b : array-like
            Paired score arrays of equal length (e.g. from
            :meth:`nested_cv_benchmark` and :meth:`surrogate_baseline`).
        alternative : str, default ``'two-sided'``
            ``'two-sided'``, ``'greater'``, or ``'less'``.

        Returns
        -------
        dict
            ``'statistic'`` : float — Wilcoxon W statistic.
            ``'p_value'`` : float
            ``'effect_size'`` : float — rank-biserial *r*.
            ``'n'`` : int — number of paired observations.
        """
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        if len(a) != len(b):
            raise ValueError("Score arrays must have equal length.")
        n = len(a)
        stat_val, p_val = stats.wilcoxon(
            a, b, alternative=alternative)
        effect = 1.0 - (2.0 * stat_val) / (n * (n + 1) / 2)
        return {
            'statistic': float(stat_val),
            'p_value': float(p_val),
            'effect_size': float(effect),
            'n': n,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @staticmethod
    def feature_importance(model, X, y, *,
                           n_repeats=10, random_state=42):
        """Permutation feature importance on a fitted model.

        Parameters
        ----------
        model : fitted sklearn estimator
            E.g. the ``best_model`` returned by
            :meth:`supervised_benchmark`.
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_repeats : int, default 10
        random_state : int, default 42

        Returns
        -------
        pandas.DataFrame
            Columns: ``feature``, ``importance_mean``,
            ``importance_std``, sorted by descending importance.
        """
        from sklearn.inspection import permutation_importance

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        result = permutation_importance(
            model, X_arr, y_arr,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='accuracy',
        )
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
        else:
            names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return pd.DataFrame({
            'feature': names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values(
            'importance_mean', ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Unsupervised: enhanced benchmarking
    # ------------------------------------------------------------------

    def unsupervised_benchmark(
        self, X, *,
        y_true=None,
        methods=('kmeans', 'gmm', 'agglo'),
        n_clusters=None,
        k_range=(2, 6),
        scaler=True,
        random_state=42,
    ):
        """Evaluate clustering with multiple validity indices.

        Reports silhouette, Calinski–Harabasz, and Davies–Bouldin
        scores for each method × *k* combination.  When ground-truth
        labels *y_true* are provided, the adjusted Rand index is also
        computed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_true : array-like, optional
            Ground-truth labels for computing adjusted Rand index.
        methods : tuple of str, default ``('kmeans', 'gmm', 'agglo')``
        n_clusters : int, optional
            Fixed cluster count.  Overrides *k_range*.
        k_range : tuple of (int, int), default ``(2, 6)``
        scaler : bool, default True
        random_state : int, default 42

        Returns
        -------
        results_df : pandas.DataFrame
            One row per method × *k* with columns: ``method``,
            ``n_clusters``, ``silhouette``, ``calinski_harabasz``,
            ``davies_bouldin``, and optionally ``adjusted_rand``.
        labels_out : dict of {str: ndarray}
            Best cluster labels (by silhouette) for each method.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if scaler:
            X_use = StandardScaler().fit_transform(X_arr)
        else:
            X_use = X_arr

        k_list = ([int(n_clusters)] if n_clusters is not None
                  else list(range(int(k_range[0]), int(k_range[1]) + 1)))

        results = []
        labels_out = {}

        for method in methods:
            if method not in ('kmeans', 'gmm', 'agglo'):
                raise ValueError(
                    f"Unknown method '{method}'. "
                    "Use 'kmeans', 'gmm', or 'agglo'.")

            best_sil = -np.inf
            best_labels = None

            for k in k_list:
                if k < 2 or k >= len(X_use):
                    continue

                if method == 'kmeans':
                    mdl = KMeans(
                        n_clusters=k, random_state=random_state,
                        n_init=10)
                elif method == 'gmm':
                    mdl = GaussianMixture(
                        n_components=k, random_state=random_state)
                else:
                    mdl = AgglomerativeClustering(n_clusters=k)

                lbls = mdl.fit_predict(X_use)
                n_unique = len(np.unique(lbls))

                sil = ch = db = np.nan
                if n_unique >= 2:
                    try:
                        sil = float(silhouette_score(X_use, lbls))
                    except Exception:
                        pass
                    try:
                        ch = float(
                            calinski_harabasz_score(X_use, lbls))
                    except Exception:
                        pass
                    try:
                        db = float(davies_bouldin_score(X_use, lbls))
                    except Exception:
                        pass

                row = {
                    'method': method,
                    'n_clusters': k,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                }
                if y_true is not None:
                    try:
                        row['adjusted_rand'] = float(
                            adjusted_rand_score(y_true, lbls))
                    except Exception:
                        row['adjusted_rand'] = np.nan
                results.append(row)

                if not np.isnan(sil) and sil > best_sil:
                    best_sil = sil
                    best_labels = lbls

            if best_labels is not None:
                labels_out[method] = best_labels
            else:
                # Fallback: first feasible k
                for k in k_list:
                    if k < 2 or k >= len(X_use):
                        continue
                    if method == 'kmeans':
                        mdl = KMeans(
                            n_clusters=k, random_state=random_state,
                            n_init=10)
                    elif method == 'gmm':
                        mdl = GaussianMixture(
                            n_components=k, random_state=random_state)
                    else:
                        mdl = AgglomerativeClustering(n_clusters=k)
                    labels_out[method] = mdl.fit_predict(X_use)
                    break

        return pd.DataFrame(results), labels_out

    # ------------------------------------------------------------------
    # Unsupervised: cluster stability
    # ------------------------------------------------------------------

    def cluster_stability(
        self, X, *,
        method='kmeans',
        n_clusters=2,
        n_bootstrap=100,
        subsample_fraction=0.8,
        scaler=True,
        random_state=42,
    ):
        """Assess cluster stability via bootstrap resampling.

        For each bootstrap iteration a random subsample is drawn, the
        clustering model is fit on the subsample, and labels are
        predicted for the **full** dataset.  Pairwise adjusted Rand
        indices between bootstrap label arrays measure how stable the
        clustering is.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        method : str, default ``'kmeans'``
            ``'kmeans'`` or ``'gmm'``.  Agglomerative clustering has no
            ``predict`` method and is not supported here.
        n_clusters : int, default 2
        n_bootstrap : int, default 100
        subsample_fraction : float, default 0.8
        scaler : bool, default True
        random_state : int, default 42

        Returns
        -------
        dict
            ``'mean_ari'`` : float
            ``'std_ari'`` : float
            ``'ari_scores'`` : ndarray — pairwise ARI values.
            ``'n_bootstrap'`` : int
            ``'method'`` : str
            ``'n_clusters'`` : int
        """
        if method not in ('kmeans', 'gmm'):
            raise ValueError(
                "cluster_stability requires 'kmeans' or 'gmm' "
                "(models with a predict method).")

        X_arr = np.asarray(X, dtype=float)
        if scaler:
            X_use = StandardScaler().fit_transform(X_arr)
        else:
            X_use = X_arr.copy()

        rng = np.random.default_rng(random_state)
        n = len(X_use)
        sub_n = max(int(n * subsample_fraction), n_clusters + 1)

        full_labels = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=sub_n, replace=False)
            seed = int(rng.integers(1_000_000))
            if method == 'kmeans':
                mdl = KMeans(
                    n_clusters=n_clusters, random_state=seed,
                    n_init=10)
            else:
                mdl = GaussianMixture(
                    n_components=n_clusters, random_state=seed)
            mdl.fit(X_use[idx])
            full_labels.append(mdl.predict(X_use))

        # Pairwise ARI (cap at 500 pairs for efficiency)
        pairs = list(combinations(range(n_bootstrap), 2))
        if len(pairs) > 500:
            sel = rng.choice(len(pairs), size=500, replace=False)
            pairs = [pairs[i] for i in sel]
        ari_scores = np.array([
            adjusted_rand_score(full_labels[i], full_labels[j])
            for i, j in pairs
        ])

        return {
            'mean_ari': float(np.mean(ari_scores)),
            'std_ari': float(np.std(ari_scores)),
            'ari_scores': ari_scores,
            'n_bootstrap': n_bootstrap,
            'method': method,
            'n_clusters': n_clusters,
        }

    # ------------------------------------------------------------------
    # Unsupervised: surrogate-based null hypothesis testing
    # ------------------------------------------------------------------

    def surrogate_cluster_validation(
        self, signals, *,
        window_size,
        window_step=1,
        window_stats=('mean', 'median', 'mode'),
        include_params=True,
        group_level_params=None,
        rqa_kwargs=None,
        surrogate_kinds=('FT', 'AAFT', 'IAAFT'),
        n_surrogate_iterations=20,
        surrogate_kwargs=None,
        methods=('kmeans', 'gmm', 'agglo'),
        n_clusters=None,
        k_range=(2, 6),
        scaler=True,
        random_state=42,
        alpha=0.05,
        correction='fdr_bh',
        verbose=True,
    ):
        """Surrogate-based null hypothesis testing for clustering.

        Evaluates whether the clustering structure found in RQA features
        exceeds what would be expected from data whose nonlinear dynamics
        have been destroyed.  For each surrogate algorithm, surrogate
        time series are generated, RQA features are extracted, and
        clustering validity indices are computed.  The resulting null
        distributions are compared against the real-data validity indices
        via rank-based p-values with multiple comparison correction.

        Parameters
        ----------
        signals : list of ndarray
            Raw time-series signals.
        window_size : int
            Forwarded to :meth:`build_feature_table`.
        window_step : int, default 1
        window_stats : tuple of str, default ``('mean', 'median', 'mode')``
        include_params : bool, default True
        group_level_params : set, optional
        rqa_kwargs : dict, optional
        surrogate_kinds : tuple of str, default ``('FT', 'AAFT', 'IAAFT')``
        n_surrogate_iterations : int, default 20
        surrogate_kwargs : dict, optional
        methods : tuple of str, default ``('kmeans', 'gmm', 'agglo')``
        n_clusters : int, optional
        k_range : tuple, default ``(2, 6)``
        scaler : bool, default True
        random_state : int, default 42
        alpha : float, default 0.05
        correction : str, default ``'fdr_bh'``
        verbose : bool, default True

        Returns
        -------
        dict
            ``'real'`` : dict — validity indices from real data
            (best silhouette per method).

            ``'surrogates'`` : dict of {str: dict} — per-surrogate-kind
            null distributions and p-values for each validity index.

            ``'corrected_p_values'`` : dict — FDR-corrected p-values.

            ``'summary'`` : DataFrame — one-row-per-surrogate summary.
        """
        _valid_surrogates = ('FT', 'AAFT', 'IAAFT', 'IDFS',
                             'WIAAFT', 'PPS')
        for kind in surrogate_kinds:
            if kind not in _valid_surrogates:
                raise ValueError(
                    f"Unknown surrogate kind '{kind}'. "
                    f"Available: {_valid_surrogates}")

        ft_kwargs = dict(
            window_size=window_size,
            window_step=window_step,
            window_stats=window_stats,
            include_params=include_params,
            group_level_params=group_level_params,
            rqa_kwargs=rqa_kwargs,
        )
        cluster_kwargs = dict(
            methods=methods,
            n_clusters=n_clusters,
            k_range=k_range,
            scaler=scaler,
            random_state=random_state,
        )

        # --- Step 1: Real data clustering ---
        if verbose:
            print("Building feature table from real signals...")
        real_features = self.build_feature_table(signals, **ft_kwargs)
        feature_cols = [c for c in real_features.columns
                        if c not in ('id', 'label')]
        X_real = real_features[feature_cols].values

        if verbose:
            print("Clustering real data...")
        real_df, real_labels = self.unsupervised_benchmark(
            X_real, **cluster_kwargs)

        # Extract best scores per method
        _validity_metrics = ['silhouette', 'calinski_harabasz',
                             'davies_bouldin']
        real_best = {}
        for method in methods:
            sub = real_df[real_df['method'] == method]
            if sub.empty:
                continue
            best_row = sub.loc[sub['silhouette'].idxmax()]
            real_best[method] = {
                m: float(best_row[m]) for m in _validity_metrics
                if m in best_row and not np.isnan(best_row[m])
            }

        # --- Step 2: Surrogate null distributions ---
        rng = np.random.default_rng(random_state)
        surrogate_results = {}

        for kind in surrogate_kinds:
            null_scores = {method: {m: [] for m in _validity_metrics}
                           for method in methods}
            iterator = range(n_surrogate_iterations)
            if verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"Surrogate cluster ({kind})",
                    leave=True)

            for it in iterator:
                seed = int(rng.integers(1_000_000_000))
                surr_signals = self._generate_surrogate_signals(
                    signals, kind,
                    surrogate_kwargs=surrogate_kwargs,
                    random_state=seed)
                surr_features = self.build_feature_table(
                    surr_signals, **ft_kwargs)
                X_surr = surr_features[feature_cols].values

                surr_df, _ = self.unsupervised_benchmark(
                    X_surr, **cluster_kwargs)

                for method in methods:
                    sub = surr_df[surr_df['method'] == method]
                    if sub.empty:
                        for m in _validity_metrics:
                            null_scores[method][m].append(np.nan)
                        continue
                    best_row = sub.loc[sub['silhouette'].idxmax()]
                    for m in _validity_metrics:
                        val = (float(best_row[m])
                               if m in best_row
                               and not np.isnan(best_row[m])
                               else np.nan)
                        null_scores[method][m].append(val)

            # Compute p-values per method × metric
            kind_results = {}
            for method in methods:
                method_results = {}
                for m in _validity_metrics:
                    null_arr = np.array(null_scores[method][m])
                    real_val = real_best.get(method, {}).get(m, np.nan)
                    if np.isnan(real_val) or np.all(np.isnan(null_arr)):
                        method_results[f'null_{m}'] = null_arr
                        method_results[f'p_value_{m}'] = np.nan
                        method_results[f'effect_size_{m}'] = np.nan
                        continue
                    # For silhouette & CH: higher is better → greater
                    # For DB: lower is better → less
                    alt = 'less' if m == 'davies_bouldin' else 'greater'
                    valid = null_arr[~np.isnan(null_arr)]
                    p_val = (self._rank_p_value(real_val, valid, alt)
                             if len(valid) > 0 else np.nan)
                    d = ((real_val - np.nanmean(null_arr))
                         / max(np.nanstd(null_arr), 1e-12))
                    method_results[f'null_{m}'] = null_arr
                    method_results[f'p_value_{m}'] = p_val
                    method_results[f'effect_size_{m}'] = d
                kind_results[method] = method_results
            surrogate_results[kind] = kind_results

        # --- Step 3: Multiple comparison correction ---
        all_raw_p = {}
        for kind in surrogate_kinds:
            for method in methods:
                for m in _validity_metrics:
                    key = f"{kind}|{method}|{m}"
                    p = surrogate_results[kind].get(
                        method, {}).get(f'p_value_{m}', np.nan)
                    if not np.isnan(p):
                        all_raw_p[key] = p

        if correction == 'fdr_bh':
            adj_p, rej_p = self._benjamini_hochberg(all_raw_p, alpha)
        elif correction == 'bonferroni':
            n_tests = len(all_raw_p)
            adj_p = {k: min(1.0, v * n_tests)
                     for k, v in all_raw_p.items()}
            rej_p = {k: v <= alpha for k, v in adj_p.items()}
        else:
            adj_p = all_raw_p
            rej_p = {k: v <= alpha for k, v in all_raw_p.items()}

        # --- Step 4: Summary table ---
        rows = []
        for kind in surrogate_kinds:
            for method in methods:
                mr = surrogate_results[kind].get(method, {})
                rb = real_best.get(method, {})
                row = {
                    'surrogate': kind,
                    'null_hypothesis': (
                        _SURROGATE_NULL_DESCRIPTIONS.get(kind, kind)),
                    'cluster_method': method,
                }
                for m in _validity_metrics:
                    key = f"{kind}|{method}|{m}"
                    row[f'real_{m}'] = rb.get(m, np.nan)
                    null_arr = mr.get(f'null_{m}', np.array([]))
                    row[f'null_mean_{m}'] = float(np.nanmean(null_arr))
                    row[f'null_std_{m}'] = float(np.nanstd(null_arr))
                    row[f'p_value_{m}'] = mr.get(
                        f'p_value_{m}', np.nan)
                    row[f'adjusted_p_{m}'] = adj_p.get(key, np.nan)
                    row[f'significant_{m}'] = rej_p.get(key, False)
                    row[f'effect_size_{m}'] = mr.get(
                        f'effect_size_{m}', np.nan)
                rows.append(row)

        summary = pd.DataFrame(rows)

        return {
            'real': {'validity': real_df, 'labels': real_labels,
                     'best_per_method': real_best},
            'surrogates': surrogate_results,
            'corrected_p_values': {'adjusted_p': adj_p,
                                   'rejected': rej_p},
            'summary': summary,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_benchmark_results(
            results, *, baseline=None, save_path=None, title=None):
        """Box plots of nested CV score distributions.

        Parameters
        ----------
        results : dict
            Output of :meth:`nested_cv_benchmark`.
        baseline : dict, optional
            Output of :meth:`surrogate_baseline`.  If given, null
            distributions are overlaid as grey box plots.
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, metric, label in [
            (axes[0], 'accuracy', 'Accuracy'),
            (axes[1], 'roc_auc', 'ROC AUC'),
        ]:
            data = [results[metric]]
            labels = [results.get('model', 'model')]
            if baseline is not None:
                null_key = f'null_{metric}'
                if null_key in baseline:
                    data.append(baseline[null_key])
                    labels.append('null')
            bp = ax.boxplot(data, tick_labels=labels,
                            patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(
                    '#4C72B0' if i == 0 else '#CCCCCC')
            ax.set_ylabel(label)
            ax.axhline(0.5, color='grey', linestyle='--',
                       linewidth=0.8, label='chance')
            ax.legend(fontsize=8)

        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_confusion_matrix(
            y_true, y_pred, *, labels=None,
            save_path=None, title=None):
        """Annotated confusion-matrix heatmap.

        Parameters
        ----------
        y_true, y_pred : array-like
        labels : list of str, optional
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(cm.shape[0])
        if labels is not None:
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh
                        else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title or 'Confusion Matrix')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_feature_importance(
            importance_df, *, top_n=15,
            save_path=None, title=None):
        """Horizontal bar chart of feature importance.

        Parameters
        ----------
        importance_df : pandas.DataFrame
            Output of :meth:`feature_importance`.
        top_n : int, default 15
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = importance_df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(
            figsize=(6, max(3, 0.35 * len(df))))
        ax.barh(df['feature'], df['importance_mean'],
                xerr=df['importance_std'], color='#4C72B0',
                edgecolor='white')
        ax.set_xlabel('Permutation importance (accuracy drop)')
        ax.set_title(title or 'Feature Importance')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_cluster_validity(
            results_df, *, save_path=None, title=None):
        """Line plots of clustering validity indices vs *k*.

        Parameters
        ----------
        results_df : pandas.DataFrame
            Output of :meth:`unsupervised_benchmark`.
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        metrics = [c for c in results_df.columns
                   if c not in ('method', 'n_clusters')]
        methods = results_df['method'].unique()
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics, figsize=(4 * n_metrics, 3.5),
            squeeze=False)
        axes = axes[0]
        for ax, metric in zip(axes, metrics):
            for method in methods:
                sub = results_df[results_df['method'] == method]
                ax.plot(sub['n_clusters'], sub[metric],
                        marker='o', label=method)
            ax.set_xlabel('k')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend(fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_cluster_scatter(
            X, labels, *, method='pca',
            save_path=None, title=None):
        """2-D scatter plot coloured by cluster/class labels.

        Dimensionality is reduced to two components via PCA before
        plotting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        labels : array-like of shape (n_samples,)
        method : str, default ``'pca'``
            Currently only ``'pca'`` is supported.
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        X_arr = np.asarray(X, dtype=float)
        labels_arr = np.asarray(labels)
        if X_arr.shape[1] > 2:
            X_2d = PCA(n_components=2).fit_transform(X_arr)
        else:
            X_2d = X_arr[:, :2]

        fig, ax = plt.subplots(figsize=(6, 5))
        for lab in np.unique(labels_arr):
            mask = labels_arr == lab
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       label=str(lab), alpha=0.7, s=30,
                       edgecolors='white', linewidth=0.3)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(fontsize=8)
        ax.set_title(title or 'Cluster Scatter (PCA)')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_surrogate_null_results(
            results, *, save_path=None, title=None):
        """Multi-panel visualisation of surrogate null benchmarking.

        Displays the real classification performance against the null
        distribution from each surrogate type (and optionally the
        label-permutation null).  Each panel shows violin plots of the
        null accuracy distributions with the real mean accuracy overlaid
        as a horizontal line, annotated with the FDR-corrected p-value.

        Parameters
        ----------
        results : dict
            Output of :meth:`surrogate_null_benchmark`.
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        real_results = results['real']
        surr_results = results['surrogates']
        perm_results = results.get('permutation')
        corrected = results['corrected_p_values']

        real_mean_acc = float(np.mean(real_results['accuracy']))
        real_mean_auc = float(np.nanmean(real_results['roc_auc']))

        # Collect null distributions
        null_labels = []
        null_acc_data = []
        null_auc_data = []
        p_acc_labels = []
        p_auc_labels = []

        for kind, sr in surr_results.items():
            null_labels.append(kind)
            null_acc_data.append(sr['null_accuracies'])
            null_auc_data.append(sr['null_roc_aucs'])
            adj_a = corrected['accuracy']['adjusted_p'].get(
                kind, sr['p_value_accuracy'])
            adj_u = corrected['roc_auc']['adjusted_p'].get(
                kind, sr['p_value_roc_auc'])
            p_acc_labels.append(adj_a)
            p_auc_labels.append(adj_u)

        if perm_results is not None:
            null_labels.append('Perm.')
            null_acc_data.append(perm_results['null_accuracy'])
            null_auc_data.append(perm_results['null_roc_auc'])
            p_acc_labels.append(
                corrected['accuracy']['adjusted_p'].get(
                    'permutation', perm_results['p_value_accuracy']))
            p_auc_labels.append(
                corrected['roc_auc']['adjusted_p'].get(
                    'permutation', perm_results['p_value_roc_auc']))

        n_nulls = len(null_labels)
        fig, axes = plt.subplots(1, 2, figsize=(5 + 1.5 * n_nulls, 5))

        for ax, data_list, real_val, p_list, ylabel in [
            (axes[0], null_acc_data, real_mean_acc,
             p_acc_labels, 'Accuracy'),
            (axes[1], null_auc_data, real_mean_auc,
             p_auc_labels, 'ROC AUC'),
        ]:
            parts = ax.violinplot(
                data_list, positions=range(n_nulls),
                showmeans=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#CCCCCC')
                pc.set_alpha(0.7)

            ax.axhline(real_val, color='#C44E52', linewidth=2,
                       linestyle='--', label=f'Real ({real_val:.3f})')
            ax.axhline(0.5, color='grey', linewidth=0.8,
                       linestyle=':', label='Chance')

            ax.set_xticks(range(n_nulls))
            ax.set_xticklabels(null_labels, fontsize=8)
            ax.set_ylabel(ylabel)

            # Annotate p-values
            for i, p in enumerate(p_list):
                sig = '***' if p < 0.001 else (
                    '**' if p < 0.01 else (
                        '*' if p < 0.05 else 'n.s.'))
                ax.text(i, ax.get_ylim()[1] * 0.98,
                        f'p={p:.3f}\n{sig}',
                        ha='center', va='top', fontsize=7)

            ax.legend(fontsize=7, loc='lower left')

        fig.suptitle(
            title or 'Surrogate Null Hypothesis Testing (Supervised)',
            fontsize=11)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_surrogate_cluster_validation(
            results, *, metric='silhouette',
            save_path=None, title=None):
        """Visualise surrogate null distributions for clustering validity.

        For each surrogate type and clustering method, shows the null
        distribution of the selected validity metric as a violin plot
        with the real-data value overlaid.

        Parameters
        ----------
        results : dict
            Output of :meth:`surrogate_cluster_validation`.
        metric : str, default ``'silhouette'``
            Validity metric to plot (``'silhouette'``,
            ``'calinski_harabasz'``, or ``'davies_bouldin'``).
        save_path : str, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        summary = results['summary']
        surr_data = results['surrogates']
        real_best = results['real']['best_per_method']

        surrogate_kinds = summary['surrogate'].unique()
        cluster_methods = summary['cluster_method'].unique()
        n_kinds = len(surrogate_kinds)
        n_methods = len(cluster_methods)

        fig, axes = plt.subplots(
            1, n_methods,
            figsize=(4 * n_methods, 5),
            squeeze=False)
        axes = axes[0]

        for ax_idx, method in enumerate(cluster_methods):
            ax = axes[ax_idx]
            data_list = []
            labels_list = []
            p_values = []

            for kind in surrogate_kinds:
                mr = surr_data.get(kind, {}).get(method, {})
                null_arr = mr.get(f'null_{metric}', np.array([]))
                valid = null_arr[~np.isnan(null_arr)]
                data_list.append(
                    valid if len(valid) > 0 else np.array([0.0]))
                labels_list.append(kind)
                p_val = mr.get(f'p_value_{metric}', np.nan)
                p_values.append(p_val)

            n = len(labels_list)
            if n > 0:
                parts = ax.violinplot(
                    data_list, positions=range(n),
                    showmeans=True, showextrema=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('#CCCCCC')
                    pc.set_alpha(0.7)

            real_val = real_best.get(method, {}).get(metric, np.nan)
            if not np.isnan(real_val):
                ax.axhline(real_val, color='#C44E52', linewidth=2,
                           linestyle='--',
                           label=f'Real ({real_val:.3f})')

            ax.set_xticks(range(n))
            ax.set_xticklabels(labels_list, fontsize=8)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(method, fontsize=10)

            for i, p in enumerate(p_values):
                if np.isnan(p):
                    continue
                sig = '***' if p < 0.001 else (
                    '**' if p < 0.01 else (
                        '*' if p < 0.05 else 'n.s.'))
                ax.text(i, ax.get_ylim()[1] * 0.98,
                        f'p={p:.3f}\n{sig}',
                        ha='center', va='top', fontsize=7)

            ax.legend(fontsize=7, loc='lower left')

        fig.suptitle(
            title or f'Surrogate Null Testing: {metric.replace("_", " ").title()}',
            fontsize=11)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
