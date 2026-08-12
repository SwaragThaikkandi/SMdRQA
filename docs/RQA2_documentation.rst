============================================
RQA2 – Comprehensive Reference Guide
============================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The **RQA2** module is the modern, object-oriented core of the *SMdRQA* package
(version 2025.7.27).  It supersedes the legacy ``RQA_functions`` procedural
interface and bundles four cooperating classes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - Purpose
   * - :class:`~SMdRQA.RQA2.RQA2`
     - End-to-end RQA pipeline: data loading, parameter estimation, recurrence-plot
       construction, measure computation, visualisation, and batch processing.
   * - :class:`~SMdRQA.RQA2.RQA2_simulators`
     - Reproducible generators for well-known chaotic dynamical systems
       (Rössler, Lorenz, Hénon, Chua) used for benchmarking.
   * - :class:`~SMdRQA.RQA2.RQA2_tests`
     - Surrogate-data generation (FT, AAFT, IAAFT, IDFS, WIAAFT, PPS) and
       comprehensive statistical validation of nonlinear dynamics metrics.
   * - :class:`~SMdRQA.RQA2.RQA2_ml`
     - Machine learning benchmarking: feature engineering from RQA measures,
       nested cross-validation with best-subset feature selection, surrogate
       null baselines, statistical comparison (Wilcoxon signed-rank), feature
       importance, clustering validity, and publication-ready visualisations.

Key Design Principles
---------------------

* **Lazy evaluation** – Parameters τ, m, and ε are computed on first access
  and cached; re-loading data resets all caches automatically.
* **Unified API** – One class instance drives the entire analysis pipeline.
* **Reproducibility** – Full state (data, parameters, measures, config) is
  serialisable via ``save_results`` / ``load_results``.
* **Separation of concerns** – Simulation, surrogate testing, and measurement
  live in dedicated classes that are each independently unit-testable.

Quick Start
===========

.. code-block:: python

   from SMdRQA.RQA2 import RQA2
   import numpy as np

   # ── 1. Load data (shape (N,) or (N, D)) ──────────────────────────────
   data = np.load('mytimeseries.npy')

   # ── 2. Create analysis object (z-scores the data by default) ─────────
   rqa = RQA2(data, reqrr=0.10)   # target 10 % recurrence rate

   # ── 3. Inspect automatically selected parameters ──────────────────────
   print(f"tau = {rqa.tau},  m = {rqa.m},  eps = {rqa.eps:.4f}")

   # ── 4. Compute all RQA measures in one call ──────────────────────────
   measures = rqa.compute_rqa_measures()
   for k, v in measures.items():
       print(f"  {k:30s}: {v:.4f}")

   # ── 5. Visualise ─────────────────────────────────────────────────────
   rqa.plot_recurrence_plot(save_path='rp.png')
   rqa.plot_rqa_measures_summary()

   # ── 6. Save for auditable workflows ──────────────────────────────────
   rqa.save_results('analysis.pkl')


Mathematical Background
=======================

Takens' Delay Embedding
-----------------------

Given a scalar time series :math:`\{x(t)\}_{t=1}^{N}` sampled from an unknown
dynamical system, the **delay-embedding theorem** (Takens, 1981) guarantees that
the map

.. math::

   \mathbf{X}(t) =
   \bigl[x(t),\; x(t+\tau),\; x(t+2\tau),\; \ldots,\; x(t+(m-1)\tau)\bigr]

is a *diffeomorphism* (smooth bijection) between the original attractor and the
reconstructed manifold in :math:`\mathbb{R}^m`, provided:

* the embedding dimension satisfies :math:`m \geq 2 d_f + 1` (where :math:`d_f`
  is the fractal dimension of the attractor), and
* the delay :math:`\tau` avoids both redundancy (too small) and independence
  (too large).

The result extends straightforwardly to multivariate data: for a
:math:`D`-dimensional input the delay vector has shape
:math:`(N_\text{emb},\, m,\, D)` with
:math:`N_\text{emb} = N - (m-1)\tau`.

Recurrence Matrix
-----------------

Given the embedded signal, the **recurrence matrix** is

.. math::

   R_{ij} = \Theta\!\bigl(\varepsilon - \|\mathbf{X}(t_i) - \mathbf{X}(t_j)\|\bigr),
   \qquad i,j = 1,\ldots,N_\text{emb}

where :math:`\Theta` is the Heaviside step function and :math:`\varepsilon`
is the neighbourhood radius.  :math:`R_{ij}=1` iff the two state vectors are
within :math:`\varepsilon` of each other in Euclidean distance.

RQA Measures – Definitions
---------------------------

All measures are derived from the lengths and densities of diagonal and vertical
line structures in :math:`R`.  Let :math:`\ell_\text{min}` denote the minimum
accepted line length (``lmin`` in the config).

**Diagonal-line measures** (predictability, determinism)

.. math::

   RR  &= \frac{1}{N^2}\sum_{i,j}R_{ij} \\[4pt]
   DET &= \frac{\displaystyle\sum_{\ell \geq \ell_\min} \ell\,P(\ell)}
               {\displaystyle\sum_{\ell \geq 1} \ell\,P(\ell)} \\[4pt]
   L   &= \frac{\displaystyle\sum_{\ell \geq \ell_\min} \ell\,P(\ell)}
               {\displaystyle\sum_{\ell \geq \ell_\min} P(\ell)} \\[4pt]
   L_\text{entr} &= -\sum_{\ell \geq \ell_\min} p(\ell)\,\ln p(\ell)

where :math:`P(\ell)` is the diagonal line-length histogram and
:math:`p(\ell) = P(\ell)/\sum P(\ell)`.

**Vertical-line measures** (laminarity, trapping time)

.. math::

   LAM &= \frac{\displaystyle\sum_{v \geq \ell_\min} v\,V(v)}
               {\displaystyle\sum_{v \geq 1} v\,V(v)} \\[4pt]
   TT  &= \frac{\displaystyle\sum_{v \geq \ell_\min} v\,V(v)}
               {\displaystyle\sum_{v \geq \ell_\min} V(v)} \\[4pt]
   V_\text{entr} &= -\sum_{v \geq \ell_\min} p(v)\,\ln p(v)

where :math:`V(v)` is the vertical line-length histogram.

Full measure table:

.. list-table::
   :header-rows: 1
   :widths: 8 20 72

   * - Key
     - Name
     - Interpretation
   * - ``recurrence_rate``
     - Recurrence Rate (RR)
     - Fraction of recurrent points. Fix RR constant when comparing conditions.
   * - ``determinism``
     - Determinism (DET)
     - Share of recurrent points on diagonal lines >= lmin. High DET signals rule-based dynamics.
   * - ``average_diagonal_length``
     - Average diagonal length (L)
     - Mean predictability horizon; inversely related to Lyapunov exponent.
   * - ``max_diagonal_length``
     - Max diagonal length (Lmax)
     - 1/Lmax ~ largest Lyapunov exponent for chaotic flows.
   * - ``diagonal_entropy``
     - Diagonal entropy (Lentr)
     - Complexity of the diagonal line-length distribution.
   * - ``diagonal_mode``
     - Diagonal mode
     - Most frequent diagonal line length.
   * - ``laminarity``
     - Laminarity (LAM)
     - Share of recurrent points on vertical lines. Signals laminar phases.
   * - ``average_vertical_length``
     - Trapping Time (TT)
     - Mean duration of laminar (slowly-changing) states.
   * - ``max_vertical_length``
     - Max vertical length (Vmax)
     - Longest laminar episode.
   * - ``vertical_entropy``
     - Vertical entropy (Ventr)
     - Complexity of the vertical line-length distribution.
   * - ``vertical_mode``
     - Vertical mode
     - Most frequent vertical line length.


Parameter Estimation Algorithms
================================

Time Delay tau
--------------

**Default method – first MI minimum**

Computes the time-delayed mutual information

.. math::

   I[\tau] = \sum_{i,j} p_{ij}(\tau)\,\log\!\left(\frac{p_{ij}(\tau)}{p_i\,p_j}\right)

using a multidimensional histogram estimator (``mi_method='histdd'``) or by
averaging 1-D MI across dimensions (``mi_method='avg'``).  The optimal delay
:math:`\tau^*` is the first local minimum of :math:`I[\tau]`.

**Polynomial method**

Fits a cross-validated polynomial to the MI curve and returns the first minimum
of the fitted function.  More robust for noisy or short series where the discrete
minimum is ambiguous.

.. code-block:: python

   rqa = RQA2(data, tau_method='polynomial', mi_method='avg')
   # or override per call:
   tau = rqa.compute_time_delay(method='polynomial', mi_method='histdd')

Inspect the MI curve to verify the automatic choice:

.. code-block:: python

   rqa.plot_tau_mi_curve(max_tau=80)


Embedding Dimension m
---------------------

Uses the **False Nearest Neighbours** (FNN) algorithm (Kennel *et al.*, 1992):

1. For each candidate :math:`m`, find the nearest neighbour of every embedded
   point in :math:`\mathbb{R}^m`.
2. Lift both points to :math:`\mathbb{R}^{m+1}` by appending the next delayed
   coordinate.
3. A neighbour is *false* if the ratio of distances after/before the lift
   exceeds a threshold :math:`r`.
4. The FNN fraction is computed across a range of :math:`r` values
   (``Rmin`` to ``Rmax``, with ``rdiv`` steps).
5. The optimal :math:`m^*` is the smallest dimension for which the FNN fraction
   drops by at least ``bound`` relative to the previous dimension.

.. code-block:: python

   rqa = RQA2(data, Rmin=1, Rmax=10, rdiv=451, bound=0.2)
   m   = rqa.compute_embedding_dimension()
   rqa.plot_fnn_curve(max_m=10)


Neighbourhood Radius epsilon
-----------------------------

A linear scan over ``epsdiv`` candidate values in [``epsmin``, ``epsmax``]
finds the first epsilon for which

.. math::

   \bigl| RR(\varepsilon) - RR_\text{target} \bigr| < \delta_{RR}

If no candidate satisfies the tolerance the midpoint
``(epsmin + epsmax) / 2`` is returned as a fallback.

.. code-block:: python

   rqa = RQA2(data, reqrr=0.05, rr_delta=0.002,
              epsmin=0, epsmax=5, epsdiv=2001)
   eps = rqa.compute_neighborhood_radius()
   print(f"Achieved RR = {rqa.recurrence_rate:.4f}")


Configuration Reference
=======================

.. _configuration-reference:

All configuration keys are passed as ``**kwargs`` to the constructor and stored
in ``rqa.config``.  They can also be inspected or updated at runtime:

.. code-block:: python

   rqa = RQA2(data, reqrr=0.05, lmin=3)
   print(rqa.config)
   rqa.config['lmin'] = 5   # update before recomputing

.. list-table::
   :header-rows: 1
   :widths: 15 12 73

   * - Key
     - Default
     - Description
   * - ``reqrr``
     - 0.10
     - Target recurrence rate (0 < reqrr < 1).  Clamped to [0.01, 0.99].
   * - ``rr_delta``
     - 0.005
     - Tolerance |RR - reqrr| for accepting an epsilon candidate.
   * - ``epsmin``
     - 0
     - Lower bound of epsilon search range.
   * - ``epsmax``
     - 10
     - Upper bound of epsilon search range.
   * - ``epsdiv``
     - 1001
     - Resolution of the linear epsilon scan.
   * - ``lmin``
     - 2
     - Minimum line length for DET, LAM, entropy, average, and mode measures.
   * - ``tau_method``
     - ``'default'``
     - ``'default'`` (first MI minimum) or ``'polynomial'`` (poly-fit minimum).
   * - ``mi_method``
     - ``'histdd'``
     - ``'histdd'`` (multidimensional histogram MI) or ``'avg'`` (per-dimension average).
   * - ``Rmin``
     - 1
     - Lower bound of FNN threshold ratio search.
   * - ``Rmax``
     - 10
     - Upper bound of FNN threshold ratio search.
   * - ``rdiv``
     - 451
     - Number of candidate FNN threshold values.
   * - ``delta``
     - 0.001
     - FNN convergence tolerance (FNN ratio < delta -> accept dimension).
   * - ``bound``
     - 0.2
     - Minimum fractional drop in FNN ratio required to select a dimension.

.. tip::

   For long, high-dimensional signals reduce ``epsdiv`` and ``rdiv`` (e.g. to 501
   and 201) to keep runtimes reasonable.  For very short signals (N < 200)
   increase ``reqrr`` slightly (e.g. 0.15) to guarantee a non-trivial recurrence
   plot.


API Reference
=============

Constructor
-----------

.. code-block:: python

   rqa = RQA2(data=None, normalize=True, **kwargs)

``data`` may be omitted; call :meth:`load_data` before accessing any computed
property.  ``normalize=True`` applies z-score normalisation column-wise.

Lazy Properties
---------------

Accessing any property triggers computation on first call and caches the result.
Loading new data with ``load_data()`` resets all caches.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``rqa.tau``
     - Optimal time delay (int >= 1).
   * - ``rqa.m``
     - Optimal embedding dimension (int >= 1).
   * - ``rqa.eps``
     - Neighbourhood radius (float > 0).
   * - ``rqa.recurrence_rate``
     - Fraction of recurrent points RR in [0, 1].
   * - ``rqa.recurrence_plot``
     - Binary recurrence matrix, shape (N_emb, N_emb).
   * - ``rqa.embedded_signal``
     - Delay-embedded tensor, shape (N_emb, m, D).

Core Computation Methods
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method call
     - Description
   * - ``compute_time_delay(method, mi_method)``
     - Recompute tau with an explicit algorithm choice.
   * - ``compute_embedding_dimension()``
     - Recompute m via FNN.
   * - ``compute_neighborhood_radius(reqrr)``
     - Recompute epsilon for a given target recurrence rate.
   * - ``compute_recurrence_plot()``
     - Build the recurrence matrix from current tau, m, epsilon.
   * - ``compute_embedded_signal()``
     - Build the delay-embedding tensor.
   * - ``compute_rqa_measures(lmin)``
     - Compute all 11 RQA measures; returns a dict.
   * - ``determinism(lmin)``
     - Return DET as a float in [0, 1].
   * - ``laminarity(lmin)``
     - Return LAM as a float in [0, 1].
   * - ``trapping_time(lmin)``
     - Return TT (mean vertical line length).
   * - ``get_summary()``
     - Return a nested dict of data info, parameters, and measures.

Visualisation Methods
---------------------

All plot methods accept ``figsize=(w, h)`` and ``save_path=None``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method call
     - Description
   * - ``plot_recurrence_plot(figsize, title, save_path)``
     - Display the binary recurrence matrix.
   * - ``plot_tau_mi_curve(max_tau, figsize, save_path)``
     - Plot MI vs tau with optimal tau marked.
   * - ``plot_fnn_curve(max_m, figsize, save_path)``
     - Plot FNN ratio vs m with optimal m marked.
   * - ``plot_rqa_measures_summary(figsize, save_path)``
     - 2x3 panel: main measures, entropy, avg/max/mode lengths, parameters.
   * - ``plot_time_series(figsize, save_path)``
     - Plot original (unnormalised) signal; stacked for multivariate data.

Persistence and Batch Processing
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method call
     - Description
   * - ``save_results(filepath)``
     - Pickle the full analysis state.
   * - ``load_results(filepath)``
     - Restore a previously saved analysis state.
   * - ``batch_process(input_path, output_path, group_level, group_level_estimates, **kwargs)``
     - Process all ``.npy`` files in *input_path*; write CSVs and RPs to *output_path*.


Common Workflows
================

Workflow 1 – Single Univariate Signal
--------------------------------------

.. code-block:: python

   import numpy as np
   from SMdRQA.RQA2 import RQA2

   t    = np.linspace(0, 20 * np.pi, 2000)
   data = np.sin(t) + 0.1 * np.random.randn(2000)

   rqa      = RQA2(data, reqrr=0.10, lmin=2)
   measures = rqa.compute_rqa_measures()

   print(f"DET = {measures['determinism']:.3f}")
   print(f"LAM = {measures['laminarity']:.3f}")
   print(f"TT  = {measures['average_vertical_length']:.2f}")
   rqa.plot_recurrence_plot()

Workflow 2 – Multivariate (MdRQA)
----------------------------------

.. code-block:: python

   import numpy as np
   from SMdRQA.RQA2 import RQA2, RQA2_simulators

   sim      = RQA2_simulators(seed=0)
   x, y, z  = sim.lorenz(n=2000)
   data     = np.column_stack([x, y, z])   # shape (2000, 3)

   rqa      = RQA2(data, reqrr=0.08)
   measures = rqa.compute_rqa_measures()
   rqa.plot_rqa_measures_summary(save_path='lorenz_summary.png')

Workflow 3 – Surrogate Validation
-----------------------------------

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_simulators, RQA2_tests

   sim      = RQA2_simulators(seed=42)
   x, _, _  = sim.rossler(n=1000, a=0.1)
   signal   = x.astype(float)

   tester   = RQA2_tests(signal, seed=0, max_workers=4)
   surr     = tester.generate('IAAFT', n_surrogates=200)

   systems  = sim.generate_test_battery()
   results  = tester.comprehensive_validation(systems, n_surrogates=100)
   # results[system_name][surrogate_method][metric] -> p-value

Workflow 4 – Batch Processing
------------------------------

.. code-block:: python

   from SMdRQA.RQA2 import RQA2

   results, errors = RQA2.batch_process(
       input_path='./data/raw/',
       output_path='./data/processed/',
       group_level=True,
       group_level_estimates=['tau', 'm'],
       reqrr=0.10,
   )

   import pandas as pd
   df = pd.DataFrame(results)
   print(df[['file', 'determinism', 'laminarity']].head())

Workflow 5 – Save and Reload
-----------------------------

.. code-block:: python

   rqa = RQA2(data)
   _   = rqa.compute_rqa_measures()
   rqa.save_results('my_analysis.pkl')

   # Later / different session
   rqa2 = RQA2()
   rqa2.load_results('my_analysis.pkl')
   print(rqa2.get_summary())

Workflow 6 – Inspecting Parameter Curves
-----------------------------------------

Always visually verify automatic parameter choices before trusting them:

.. code-block:: python

   rqa = RQA2(data)

   # Visualise MI curve – confirm tau is at the first minimum
   rqa.plot_tau_mi_curve(max_tau=100)

   # Visualise FNN curve – confirm m is where FNN reaches zero
   rqa.plot_fnn_curve(max_m=12)

   # Verify the achieved recurrence rate
   print(f"Target RR = {rqa.config['reqrr']:.3f}")
   print(f"Actual RR = {rqa.recurrence_rate:.3f}")

Workflow 7 – ML Pipelines (Supervised + Unsupervised)
------------------------------------------------------

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_ml, RQA2_simulators
   import numpy as np

   sim = RQA2_simulators(seed=42)
   battery = sim.generate_test_battery()

   # Build a small labeled dataset by slicing trajectories
   signals, labels = [], []
   for name, data in battery.items():
       x = data['x']
       seg_len = len(x) // 4
       for i in range(4):
           signals.append(x[i * seg_len:(i + 1) * seg_len])
           labels.append(name)

   ml = RQA2_ml()
   features = ml.build_feature_table(
       signals,
       labels=labels,
       window_size=100,
       window_step=20,
       window_stats=('mean', 'median', 'mode'),
   )

   X = features.drop(columns=['id', 'label'])
   y = features['label']

   supervised, best_model = ml.supervised_benchmark(X, y, cv=3)
   print(supervised)

   unsupervised, cluster_labels = ml.unsupervised_benchmark(
       X, n_clusters=len(set(labels)))
   print(unsupervised)


Machine Learning Pipelines (RQA2_ml)
====================================

The **RQA2_ml** class provides a full feature-to-model workflow built on top
of the RQA2 measures. It is designed for quick hypothesis testing in both
supervised and unsupervised settings, with reproducible defaults and minimal
boilerplate.

Beginner-Friendly Overview
--------------------------

If you are new to RQA or machine learning, here is the simplest way to think
about this pipeline:

1. **Start with a time series** (a list of numbers that changes over time).
2. **RQA turns it into numbers** that describe repeating patterns.
3. **Windowed RQA** repeats this in small chunks to capture local changes.
4. **Summaries (mean/median/mode)** turn those chunks into a few stable features.
5. **Machine learning** uses those features to classify or cluster systems.

Minimal “Hello World” Example
-----------------------------

.. code-block:: python

   import numpy as np
   from SMdRQA.RQA2 import RQA2_ml

   # Two very simple signals
   s1 = np.sin(np.linspace(0, 4 * np.pi, 200))
   s2 = np.random.default_rng(0).standard_normal(200)

   ml = RQA2_ml()
   features = ml.build_feature_table(
       [s1, s2],
       labels=["sine", "noise"],
       window_size=60,
       window_step=10,
       window_stats=("mean", "median", "mode"),
   )

   X = features.drop(columns=["id", "label"])
   y = features["label"]

   results, model = ml.supervised_benchmark(X, y, models=("svm",), cv=2)
   print(results)

What You Need to Provide
------------------------

* **Signals**: either a list of arrays or a folder of ``.npy`` files.
* **Labels (optional)**: required only for supervised learning.
* **window_size**: must be provided; it controls how long each window is.

Rule of thumb: choose a ``window_size`` that is large enough to see
repeating structure, but small enough to detect local changes.

Windowed RQA Features
---------------------

Windowed RQA is computed by sliding a square window along the *diagonal* of
the recurrence plot (the same strategy used in the legacy sliding-window
utilities). For each window, RQA measures are computed, then aggregated.

.. code-block:: python

   rqa = RQA2(data)
   windows = rqa.compute_windowed_rqa_measures(window_size=100, window_step=10)
   summary = rqa.summarize_windowed_measures(
       windows, stats=('mean', 'median', 'mode'))

The summary features are flattened with names like:

* ``recurrence_rate__mean``
* ``determinism__median``
* ``laminarity__mode``

Feature Table Builder
---------------------

``build_feature_table`` returns a pandas DataFrame containing:

* Whole-signal RQA measures
* Windowed summary features (mean, median, mode)
* Optional parameters ``tau``, ``m``, ``eps`` (when ``include_params=True``)

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_ml

   ml = RQA2_ml()
   features = ml.build_feature_table(
       signals_or_dir="./data/npy_files/",
       labels=None,
       window_size=120,
       window_step=20,
       window_stats=('mean', 'median', 'mode'),
       include_params=True,
   )

Windowed features are prefixed with ``win_`` in the output DataFrame to
avoid collisions with whole-signal measures.

Available Models
----------------

Eight classifiers are registered (all scikit-learn, no extra
dependencies), chosen for small-sample tabular RQA feature tables:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Key
     - Estimator
     - Why it is included
   * - ``knn``
     - KNeighborsClassifier
     - Non-parametric local baseline.
   * - ``svm``
     - SVC (RBF, probability)
     - Strong non-linear margin classifier.
   * - ``rf``
     - RandomForestClassifier
     - Robust tree ensemble.
   * - ``logreg``
     - LogisticRegression
     - Essential linear baseline with calibrated probabilities.
   * - ``lda``
     - LinearDiscriminantAnalysis
     - Classic small-sample linear model (closed form).
   * - ``nb``
     - GaussianNB
     - Fast high-bias baseline.
   * - ``gb``
     - HistGradientBoostingClassifier
     - Modern gradient boosting, strongest tabular family.
   * - ``et``
     - ExtraTreesClassifier
     - Extra-randomised trees, good on small noisy data.

Each model has a compact hyperparameter grid in
``RQA2_ml._PARAM_GRIDS`` used by the nested tuning described below.
Pass ``models='all'`` anywhere a model tuple is accepted to run the
whole registry.

Supervised Benchmark (Quick)
----------------------------

``supervised_benchmark`` evaluates multiple classifiers with stratified
cross-validation and reports accuracy, macro-F1, and ROC AUC:

.. code-block:: python

   X = features.drop(columns=['id', 'label'])
   y = features['label']

   results, best_model = ml.supervised_benchmark(
       X, y, models=('knn', 'svm', 'rf'), cv=5)   # or models='all'

The method returns:

* A results DataFrame with mean/std scores for each model
* The best-performing fitted model (ready to ``predict``)

This is a lighter alternative for exploratory analysis. For paper-quality
validation, use ``nested_cv_benchmark`` instead.

Nested Cross-Validation with Feature Selection
-----------------------------------------------

``nested_cv_benchmark`` implements the validation procedure described in
the SMdRQA paper: the outer loop evaluates generalisation on held-out data,
while the inner loop performs best-subset feature selection — and, with
``tune=True``, hyperparameter grid search — exclusively on the training
fold to prevent data leakage.

.. code-block:: python

   results = ml.nested_cv_benchmark(
       X, y, model='knn',
       outer_iterations=100,
       test_fraction=1/3,
       feature_selection='auto',   # exhaustive if <= 12 features, else forward
       tune=True,                  # grid-search hyperparameters per outer fold
   )
   print(f"Accuracy: {results['accuracy'].mean():.3f} +/- {results['accuracy'].std():.3f}")
   print(f"ROC AUC:  {results['roc_auc'].mean():.3f} +/- {results['roc_auc'].std():.3f}")
   print("Most selected features:")
   print(results['feature_frequency'].head(5))
   print("Hyperparameters chosen per fold:", results['best_params'][:3])

Returns a dict with:

* ``accuracy`` / ``balanced_accuracy`` / ``f1_macro`` / ``roc_auc``:
  score arrays (one per outer iteration)
* ``selected_features``: list of feature-index tuples per iteration
* ``best_params``: hyperparameters chosen in each outer iteration
  (empty dicts when ``tune=False``, the default)
* ``feature_frequency``: Series counting how often each feature was selected

Hyperparameter Tuning (``tune=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the default ``tune=False`` every model runs with fixed default
hyperparameters.  Setting ``tune=True`` grid-searches the model's
``_PARAM_GRIDS`` entry (or a custom ``param_grid=``) on the inner CV of
each outer training fold, *after* feature selection and on the selected
subset — so the procedure is truly nested and model comparisons are not
biased by arbitrary defaults.

Group-Aware Splitting (``groups=``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When several samples come from the same subject or recording, ordinary
stratified splits leak information: windows of one subject end up in
both train and test.  Pass ``groups=`` (one label per sample) and both
the outer and inner splits use ``StratifiedGroupKFold`` so each group
stays entirely on one side of every split:

.. code-block:: python

   results = ml.nested_cv_benchmark(
       X, y, model='rf', tune=True,
       groups=features_meta['subject_id'],
   )

Integrated Benchmark (One Call)
-------------------------------

``integrated_benchmark`` chains the whole pipeline: signals (or a
precomputed feature table) → ``build_feature_table`` → tuned nested CV
for every requested model → per-model comparison table → pairwise
Wilcoxon signed-rank tests with Benjamini–Hochberg correction → the best
model refit on the full dataset.

.. code-block:: python

   out = ml.integrated_benchmark(
       signals, labels,
       window_size=100, window_step=20,
       models='all',          # or a tuple such as ('svm', 'rf', 'logreg')
       tune=True,
       outer_iterations=50,
   )

   print(out['comparison'])        # mean/std of every metric per model
   print(out['pairwise_tests'])    # Wilcoxon + BH-corrected significance
   print(out['best_model_name'])
   predictions = out['best_model'].predict(new_features)

An optional ``progress_callback(index, total, model_name)`` is invoked
before each model's nested CV run (used by the UI progress bar), and
``groups=`` is forwarded to every nested CV call.

Surrogate Null Baseline
-----------------------

``surrogate_baseline`` shuffles the labels ``n_permutations`` times and
evaluates each with stratified k-fold CV, yielding a null distribution:

.. code-block:: python

   null = ml.surrogate_baseline(X, y, model='knn', n_permutations=100)

Statistical Comparison
----------------------

``compare_scores`` implements the Wilcoxon signed-rank test with
rank-biserial effect size:

.. code-block:: python

   stat = RQA2_ml.compare_scores(
       results['accuracy'], null['null_accuracy'],
       alternative='greater')
   print(f"p = {stat['p_value']:.4f}, effect size r = {stat['effect_size']:.2f}")

Feature Importance
------------------

``feature_importance`` computes permutation importance on a fitted model:

.. code-block:: python

   imp = RQA2_ml.feature_importance(best_model, X, y, n_repeats=10)
   print(imp.head(10))
   RQA2_ml.plot_feature_importance(imp, save_path='importance.png')

Unsupervised Benchmark
----------------------

``unsupervised_benchmark`` evaluates clustering methods and reports multiple
validity indices: silhouette, Calinski-Harabasz, and Davies-Bouldin. When
ground-truth labels are available, the adjusted Rand index is also computed.

.. code-block:: python

   results, labels = ml.unsupervised_benchmark(
       X, y_true=y, methods=('kmeans', 'gmm', 'agglo'), k_range=(2, 6))
   ml.plot_cluster_validity(results, save_path='validity.png')

Cluster Stability
-----------------

``cluster_stability`` assesses reproducibility via bootstrap resampling:

.. code-block:: python

   stab = ml.cluster_stability(X, method='kmeans', n_clusters=2, n_bootstrap=100)
   print(f"Stability ARI: {stab['mean_ari']:.3f} +/- {stab['std_ari']:.3f}")

Visualisation Methods
---------------------

All plotting methods accept an optional ``save_path`` for direct export and
return the matplotlib Figure object:

* ``plot_benchmark_results(results, baseline=None)`` — box plots of score
  distributions, optionally overlaying null distributions.
* ``plot_confusion_matrix(y_true, y_pred, labels=None)`` — annotated heatmap.
* ``plot_feature_importance(importance_df, top_n=15)`` — horizontal bar chart.
* ``plot_cluster_validity(results_df)`` — line plots of validity indices vs *k*.
* ``plot_cluster_scatter(X, labels, method='pca')`` — PCA-projected 2-D scatter.

Notes and Best Practices
------------------------

* Choose a **window_size** that preserves local dynamics but still provides
  enough points for reliable RQA measures.
* For fair cross-system comparisons, fix parameters across groups using
  ``group_level_params`` in ``build_feature_table``.
* Use ``nested_cv_benchmark`` for paper-quality validation and
  ``supervised_benchmark`` for quick exploration.
* Always compare against a surrogate baseline to ensure classification
  performance exceeds chance.
* Use ``models=('svm',)`` or ``methods=('kmeans',)`` if you want a single
  baseline rather than full benchmarking.

Glossary (Plain Language)
-------------------------

* **Recurrence plot (RP)**: a square image that marks when the system revisits
  similar states.
* **RQA measures**: numeric summaries extracted from the RP (e.g., recurrence
  rate, determinism).
* **Windowed RQA**: computing RQA on smaller slices of the RP to capture
  local changes over time.
* **Feature table**: a spreadsheet‑like table where each row is one signal and
  each column is a numeric feature.
* **Supervised learning**: you already know the label (e.g., “chaotic” vs
  “periodic”) and train a classifier.
* **Unsupervised learning**: you do not supply labels; clustering groups
  similar signals together.

Common Pitfalls
---------------

* **“window_size exceeds RP size”**: your signal is too short or the window is
  too large. Reduce the window size or use longer signals.
* **Only one class in y**: supervised learning needs at least two labels.
* **Very small windows**: can produce unstable measures; increase window size
  or step size.


RQA2_simulators – Chaotic System Generators
============================================

``RQA2_simulators`` integrates four continuous-time attractors (using
``scipy.integrate.solve_ivp``, RK45, ``rtol=1e-9``, ``atol=1e-12``), one
discrete-time map, and a Kuramoto phase-oscillator network.

Available systems:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Method
     - System
     - Default chaotic parameters
   * - ``rossler()``
     - Rössler attractor
     - a=0.2, b=0.2, c=5.7; chaotic band attractor.  Use a=0.1 for periodic.
   * - ``lorenz()``
     - Lorenz attractor
     - sigma=10, rho=28, beta=8/3; classic butterfly attractor.
   * - ``henon()``
     - Hénon map
     - a=1.4, b=0.3; discrete 2-D map on a fractal attractor.
   * - ``chua()``
     - Chua's circuit
     - alpha=15.6, beta=28, m0=-1.143, m1=-0.714; double-scroll attractor.
   * - ``kuramoto()``
     - Kuramoto oscillator network
     - K=1.0, omega_sd=1.0, n_osc=10; returns sin(theta) as an
       (n, n_osc) multivariate signal.  Critical coupling
       K_c = omega_sd·sqrt(8/pi) ≈ 1.596·omega_sd separates the
       incoherent (below) and synchronised (above) regimes.
   * - ``generate_test_battery()``
     - All chaotic systems
     - Returns a dict with keys: ``'rossler_chaotic'``, ``'rossler_sync'``, ``'lorenz'``, ``'henon'``, ``'chua'``.

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_simulators

   sim = RQA2_simulators(seed=42)

   x, y, z = sim.rossler(tmax=5000, n=2000, a=0.1)   # limit cycle
   x, y, z = sim.lorenz(n=2000)                        # butterfly
   x, y    = sim.henon(n=2000)                         # Hénon map
   theta   = sim.kuramoto(n=2000, n_osc=15, K=2.5)     # (2000, 15) signal
   systems = sim.generate_test_battery()               # full battery

Regime thresholds (useful for building labelled chaotic-vs-periodic
datasets; see :mod:`SMdRQA.ui.simulate` and the UI's regime-sampling
mode):

* Rössler ``c`` ≈ 4.2 (with a=b=0.2): periodic below, chaotic above.
* Lorenz ``rho`` ≈ 24.74 (with sigma=10, beta=8/3): stable fixed points
  below, chaotic above.
* Hénon ``a`` ≈ 1.06 (with b=0.3): largely periodic below, chaotic above.
* Chua ``alpha`` ≈ 8.8 (approximate): limit cycles below, double-scroll
  chaos towards alpha=15.6.
* Kuramoto ``K_c = omega_sd·sqrt(8/pi)``: incoherent below, synchronised
  above.


RQA2_tests – Surrogate Data and Validation
==========================================

Surrogate data testing determines whether a measured signal exhibits genuine
nonlinear structure by comparing statistics to ensembles of null surrogates.

Surrogate Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 10 22 68

   * - Key
     - Algorithm
     - Null hypothesis and notes
   * - ``'FT'``
     - Fourier Transform
     - Signal is a stationary linear Gaussian process.  Fastest; randomises Fourier phases only.
   * - ``'AAFT'``
     - Amplitude-Adjusted FT
     - Same as FT but also matches the marginal amplitude distribution.
   * - ``'IAAFT'``
     - Iterative AAFT
     - Iteratively matches spectrum *and* amplitude; best accuracy/speed trade-off.  ``n_iter`` controls convergence.
   * - ``'IDFS'``
     - Iterative Digitally-Filtered Shuffled
     - Targets higher-order cumulants; starts from a shuffled realisation.
   * - ``'WIAAFT'``
     - Wavelet-based IAAFT
     - Applies IAAFT per wavelet level; preserves multiscale structure.  ``wavelet`` and ``level`` parameters.
   * - ``'PPS'``
     - Pseudo-Periodic Surrogate
     - Preserves return-map geometry; best for near-periodic / weakly chaotic signals.  ``tau``, ``dim``, ``noise_factor`` parameters.

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_tests
   import numpy as np

   signal = np.random.randn(512).astype(float)
   tester = RQA2_tests(signal, seed=42, max_workers=4)

   surr_iaaft = tester.generate('IAAFT', n_surrogates=200, n_iter=200)
   surr_wave  = tester.generate('WIAAFT', n_surrogates=100, wavelet='db8', level=4)
   surr_pps   = tester.generate('PPS',   n_surrogates=50,  dim=5, noise_factor=0.1)

Nonlinear Validation Metrics
-----------------------------

Six metrics are evaluated in :meth:`comprehensive_validation`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric key
     - Description
   * - ``lyapunov_exponent``
     - Largest Lyapunov exponent (Rosenstein nearest-neighbour divergence).
       Positive value -> chaotic dynamics.
   * - ``time_irreversibility``
     - Third-order temporal asymmetry (Ramsey & Rothman 1996).
       Non-zero -> time-irreversible, hence nonlinear, process.
   * - ``sample_entropy``
     - Approximate entropy of length-m patterns; lower -> more regular.
   * - ``correlation_dimension``
     - Grassberger-Procaccia fractal dimension estimate.
       Finite, low value -> low-dimensional deterministic attractor.
   * - ``nonlinearity_index``
     - Absolute skewness of first differences; asymmetric amplitude fluctuations.
   * - ``predictability``
     - Normalised local linear prediction error; lower -> more predictable.

Comprehensive Validation
-------------------------

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_simulators, RQA2_tests

   sim     = RQA2_simulators(seed=0)
   systems = sim.generate_test_battery()
   signal  = systems['rossler_chaotic']['x'].astype(float)
   tester  = RQA2_tests(signal, seed=42)

   results = tester.comprehensive_validation(
       systems_data=systems,
       n_surrogates=200,
       save_path='validation_heatmap.png',
   )
   # results[system_name][surrogate_method][metric] -> p-value

A p-value near 0 means the original signal is *significantly different* from the
surrogate ensemble for that metric, providing evidence of the type of structure
the surrogate destroys.

Parallel Generation
-------------------

Parallelism activates automatically when ``n_surrogates >= 50`` and
``max_workers > 1``:

.. code-block:: python

   tester = RQA2_tests(signal, seed=0, max_workers=8)
   surr   = tester.generate('IAAFT', n_surrogates=500)


Implementation Notes
====================

Indexing
--------

All internal arrays use **0-based** indexing.  The embedded signal has shape
``(N_emb, m, D)`` where ``N_emb = N - (m-1)*tau``.  Line-length histogram
arrays have length ``N_emb + 1`` so that index ``i`` directly counts lines of
length ``i``.

Distance Computation
--------------------

Both the recurrence matrix and the epsilon search use
``scipy.spatial.distance.cdist`` for vectorised Euclidean distances.  Memory
scales as :math:`O(N_\text{emb}^2)`; for ``N_emb > 5000`` consider windowed
analysis via ``RP_maker`` or down-sampling.

Cross-Validated Polynomial Degree
----------------------------------

The polynomial tau method uses ``sklearn.model_selection.RepeatedKFold``
(5-fold, 3 repeats) to select the degree that minimises RMSE on held-out data,
then finds the first minimum of the fitted polynomial.

Surrogate Seeding
-----------------

Each surrogate receives a unique seed derived from the parent RNG, guaranteeing
statistical independence and reproducibility given the same ``seed`` argument.


Reproducibility Checklist
=========================

1. **Normalise** multivariate data (``normalize=True``, the default).
2. **Verify parameters** visually before analysis:

   .. code-block:: python

      rqa.plot_tau_mi_curve()
      rqa.plot_fnn_curve()

3. **Check the achieved RR**:

   .. code-block:: python

      print(rqa.recurrence_rate)

4. **Use consistent lmin** across all signals in a study.
5. **Fix parameters across groups** with ``batch_process(..., group_level=True)``.
6. **Save state** after each expensive computation:

   .. code-block:: python

      rqa.save_results('checkpoint.pkl')


Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Likely cause and fix
   * - ``ValueError: Insufficient data for embedding``
     - N is too small for the chosen (m, tau).  Reduce m or tau, or increase N.
   * - RR is 0 or 1 regardless of epsilon
     - ``epsmin``/``epsmax`` bracket the wrong scale.  Set ``epsmax`` to a
       multiple of the signal's standard deviation, e.g.
       ``RQA2(data, epsmax=3*np.std(data))``.
   * - tau = 1 for all signals
     - MI curve has no local minimum.  Try ``tau_method='polynomial'``.
   * - m is very large (> 10)
     - FNN never drops below ``bound``.  Increase ``bound`` (e.g. 0.4) or use
       more data (N > 500 recommended).
   * - Surrogates take very long
     - Reduce ``n_surrogates`` or use ``max_workers > 1``.  For WIAAFT use a
       lower decomposition ``level``.
   * - ``TypeError: Input signal must be floating-point``
     - Cast your array: ``signal = signal.astype(float)``.
   * - Sphinx ``autoclass`` renders nothing
     - Ensure SMdRQA is importable from the docs build environment and all
       dependencies (PyWavelets, scikit-learn, seaborn) are installed.
   * - Recurrence plot is all black or all white
     - ``reqrr`` is out of range for this signal.  Try ``reqrr=0.05``
       (sparser) or ``reqrr=0.20`` (denser) and re-run
       ``compute_neighborhood_radius()``.


Performance Notes
=================

The parameter-estimation and measure kernels are fully vectorised
(numpy/scipy, no extra dependencies):

* ``_fnnhitszero`` computes the embedding and nearest-neighbour search
  once and sweeps all candidate *r* values vectorised (previously the
  O(N²) search ran once per candidate, ``rdiv=451`` times).
* ``_findeps`` computes the distance matrix once and counts recurrences
  for every candidate epsilon via a sorted search (previously ``cdist``
  ran up to ``epsdiv=1001`` times).
* ``_nearest`` uses chunked ``cdist`` + ``argmin``; ``_vert_hist`` and
  ``_diaghist`` use vectorised run-length encoding.

End-to-end parameter estimation (tau → m → epsilon → RP → measures) is
roughly **100× faster** than the loop-based implementation, with
bit-identical outputs.


Interactive UI (Streamlit)
==========================

An optional browser UI covers the full workflow — data import or
simulation, RQA analysis, window-size sensitivity, machine-learning
benchmarking, and reproducibility — without writing code.

Installation and Launch
-----------------------

.. code-block:: console

   pip install SMdRQA[ui]     # installs streamlit + plotly
   smdrqa-ui                  # or: python -m SMdRQA.ui

Tabs
----

1. **Data** — load signals from a folder (``.npy``/``.csv``; labels can
   be derived from filename prefixes) or simulate labelled batches from
   the built-in systems.  All system parameters are editable, and
   signals can be previewed as time series or 2-D/3-D phase portraits.
2. **RQA** — automatic or manual tau/m/epsilon, target recurrence rate
   and ``lmin`` (defaults match the script defaults), single-window or
   sliding-window analysis with window size, step and central-tendency
   choice; recurrence-plot heatmap, windowed-measure curves, CSV/HTML
   export.
3. **Window-size sensitivity** — a seeded, vectorised re-implementation
   of the :mod:`SMdRQA.window_size` bootstrap
   (:func:`SMdRQA.ui.sensitivity.window_size_sensitivity`): for each
   window size the pooled line-length distribution is resampled and the
   5–95 % quantile width of the chosen measure is plotted; narrow is
   stable.
4. **Machine learning** — runs :meth:`RQA2_ml.integrated_benchmark`
   over the selected models with a per-model progress bar, then shows
   the comparison table, accuracy box plots, feature-selection
   frequency, and BH-corrected pairwise tests, all exportable.
5. **Script** — see *Reproducibility* below.

Regime-Based Simulation
-----------------------

For building labelled chaotic-vs-periodic datasets, the Data tab offers
a *Sample by regime* mode.  The UI names the system's bifurcation
parameter and suggests a literature-based threshold (see the simulator
regime table above).  For each side of the threshold you choose:

* a class label (e.g. ``periodic`` / ``chaotic``),
* a sampling distribution for the parameter (uniform, normal, or fixed),
* the number of simulations.

Draws are clipped to their side of the threshold so the labels stay
truthful, and the labelled signals feed directly into the ML tab.  For
the Kuramoto system the oscillator count (the signal dimensionality)
can be fixed or sampled per-simulation from a range.

Reproducibility
---------------

The sidebar requires a random seed, and every action taken in the UI is
mirrored into an equivalent block of plain Python
(:class:`SMdRQA.ui.recorder.ScriptRecorder`).  The *Script* tab shows
the accumulated code and offers it as a standalone ``.py`` download:
rerunning that file reproduces the entire session, including every
seed-dependent step.  Seed changes made mid-session are themselves
recorded.


Extending RQA2
==============

Subclass :class:`~SMdRQA.RQA2.RQA2` and override one or more private methods
to integrate custom algorithms without changing the public API:

.. code-block:: python

   from SMdRQA.RQA2 import RQA2

   class MyRQA(RQA2):

       def _findtau_default(self, mi_method):
           """Custom tau estimator (e.g. autocorrelation zero crossing)."""
           acf = np.correlate(self.data[:, 0], self.data[:, 0], mode='full')
           acf = acf[len(acf) // 2:]
           zeros = np.where(acf < 0)[0]
           return int(zeros[0]) if len(zeros) > 0 else 1

       def _findm(self, tau, sd):
           """Override with a global false-strand or singular-value method."""
           ...

Key extension points:

* ``_findtau_default``, ``_findtau_polynomial`` – time-delay strategies
* ``_findm`` – embedding dimension selection
* ``_findeps`` – neighbourhood radius selection
* ``_diaghist``, ``_vert_hist`` – line structure extraction
* ``_percentmorethan``, ``_entropy``, ``_average``, ``_maxi``, ``_mode`` – measure computation


Citing This Work
================

Please cite the following when using any component of RQA2 or SMdRQA in
academic publications:

.. code-block:: text

   Thaikkandi, S., Sharika, K. M., & Nivedita. (2025). SMdRQA: Sliding Window
   Multidimensional Recurrence Quantification Analysis (Version 2025.7.27)
   [Software]. Zenodo. https://doi.org/10.5281/zenodo.10854678

BibTeX:

.. code-block:: bibtex

   @software{smdrqa2025,
     author    = {Thaikkandi, Swarag and Sharika, K. M. and Nivedita},
     title     = {{SMdRQA}: Sliding Window Multidimensional Recurrence
                  Quantification Analysis},
     year      = {2025},
     version   = {2025.7.27},
     publisher = {Zenodo},
     doi       = {10.5281/zenodo.10854678},
     url       = {https://doi.org/10.5281/zenodo.10854678}
   }

Key references for the underlying algorithms:

* **Recurrence plots** – Eckmann *et al.*, *Europhys. Lett.* 4 (1987)
* **RQA measures** – Marwan *et al.*, *Phys. Rep.* 438 (2007)
* **Takens embedding** – Takens, *Lecture Notes in Mathematics* 898 (1981)
* **FNN algorithm** – Kennel *et al.*, *Phys. Rev. A* 45 (1992)
* **MI for tau** – Fraser & Swinney, *Phys. Rev. A* 33 (1986)
* **IAAFT** – Schreiber & Schmitz, *Phys. Rev. Lett.* 77 (1996)
* **PPS** – Small *et al.*, *Phys. Rev. Lett.* 87 (2001)
