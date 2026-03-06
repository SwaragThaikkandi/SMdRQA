<<<<<<< HEAD
========================
RQA2.rst – Documentation
========================

.. contents:: Table of Contents
   :depth: 2
   :local:

Introduction
============
The **RQA2** class provides a modern, object-oriented interface to Recurrence Quantification
Analysis (RQA) tailored for scientific research.  It unifies data handling,
parameter estimation, recurrence-plot generation, quantitative measure
extraction, and visualization in a single coherent workflow.  RQA2 is part of
the *SMdRQA* package (version 2.0.0) and is intended for researchers working in
nonlinear dynamics, neuroscience, climate science, physiology, finance, and
other fields that require robust analysis of complex time series.

Key Capabilities
================

• **Unified API** – All functionality is accessed through one class instance.  
• **Lazy Evaluation** – Parameters τ (delay), m (embedding dimension), and ε
  (radius) are computed only when first requested and cached thereafter.  
• **Advanced Parameter Selection** – Mutual-information and polynomial
  minimisation algorithms for τ; False Nearest Neighbours for m; adaptive
  recurrence-rate targeting for ε.  
• **Full RQA Measure Suite** – Determinism, Laminarity, Trapping Time, line-length
  statistics, and entropic measures.  
• **High-Quality Visualisation** – Publication-ready plots for recurrence
  matrices, MI curves, FNN curves, measure summaries, and raw time series.  
• **Batch Processing** – Automated analysis of entire directories with optional
  group-level parameter estimation.  
• **Reproducibility** – Complete state serialisation (`save_results`/`load_results`).
=======
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
interface and bundles three cooperating classes:

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

Key Design Principles
---------------------

* **Lazy evaluation** – Parameters τ, m, and ε are computed on first access
  and cached; re-loading data resets all caches automatically.
* **Unified API** – One class instance drives the entire analysis pipeline.
* **Reproducibility** – Full state (data, parameters, measures, config) is
  serialisable via ``save_results`` / ``load_results``.
* **Separation of concerns** – Simulation, surrogate testing, and measurement
  live in dedicated classes that are each independently unit-testable.
>>>>>>> origin/master

Quick Start
===========

.. code-block:: python

<<<<<<< HEAD
   from SMdRQA.RQA_functions import RQA2
   import numpy as np

   # Load or create time-series data
   data = np.load('mytimeseries.npy')      # shape (N,) or (N, D)

   # Initialise analysis object (normalises by default)
   rqa = RQA2(data, reqrr=0.10)            # target 10 % recurrence-rate

   # Access core parameters (computed lazily)
   print("τ =", rqa.tau)
   print("m =", rqa.m)
   print("ε =", rqa.eps)

   # Generate recurrence plot and compute measures
   rp      = rqa.recurrence_plot           # binary matrix
   measures = rqa.compute_rqa_measures()

   print("Determinism:", measures['determinism'])

   # Visualise
   rqa.plot_recurrence_plot()
   rqa.plot_rqa_measures_summary()

   # Persist results for full reproducibility
   rqa.save_results('analysis.pkl')

=======
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


>>>>>>> origin/master
API Reference
=============

Constructor
-----------
<<<<<<< HEAD
.. pyclass:: RQA2(data=None, normalize=True, **kwargs)

   :param ndarray data: Time-series array of shape *(N,)* or *(N, D)*.
   :param bool normalize: Apply z-score normalisation (default *True*).
   :param float reqrr: Target recurrence-rate (default 0.10).
   :param int lmin: Minimum line length for DET/LAM (default 2).
   :param str tau_method: ``'default'`` | ``'polynomial'`` (default ``'default'``).
   :param str mi_method:  ``'histdd'`` | ``'avg'`` (default ``'histdd'``).

Lazy Properties
---------------
.. autosummary::
   :nosignatures:

   RQA2.tau
   RQA2.m
   RQA2.eps
   RQA2.recurrence_rate
   RQA2.recurrence_plot
   RQA2.embedded_signal

Core Methods
------------
.. autosummary::
   :nosignatures:

   RQA2.compute_rqa_measures
   RQA2.determinism
   RQA2.laminarity
   RQA2.trapping_time
   RQA2.get_summary
   RQA2.save_results
   RQA2.load_results
   RQA2.plot_recurrence_plot
   RQA2.plot_tau_mi_curve
   RQA2.plot_fnn_curve
   RQA2.plot_rqa_measures_summary
   RQA2.plot_time_series

Batch Processing
----------------
.. autosummary::
   :nosignatures:

   RQA2.batch_process

Implementation Highlights
=========================

Algorithmic Choices
-------------------
* **Time-Delay (τ)** – The default method selects the first local minimum of
  the time-delay mutual information curve; the polynomial variant fits a
  cross-validated polynomial and chooses its first minimum.
* **Embedding Dimension (m)** – Determined using the false-nearest-neighbour
  criterion with adaptive search for the *r* threshold where the FNN ratio
  first reaches zero.
* **Neighbourhood Radius (ε)** – Binary search over a user-defined interval
  to achieve the desired recurrence-rate within tolerance *rr_delta*.

Data Structures
---------------
* **Delay-Embedded Tensor:** ``shape = (N_embedded, m, D)``  
* **Recurrence Plot:** ``shape = (N_embedded, N_embedded)``, dtype ``uint8``.

Performance Considerations
--------------------------
Vectorised distance computations from *SciPy*’s :pyfunc:`scipy.spatial.distance.cdist`
are used to minimise Python-level loops.  Memory usage scales quadratically
with *N_embedded*; users analysing very large datasets should apply windowed
or down-sampled approaches.

Reproducibility & Best Practices
===============================
1. **Normalisation** – Always normalise multivariate data to unit variance.
2. **Validation** – Inspect MI and FNN curves to confirm automatic parameter
   choices.
3. **Line Threshold `lmin`** – Adapt `lmin` to the sampling frequency and
   scientific question (e.g., heartbeat intervals vs. daily climate data).
4. **Saving State** – Persist full analysis objects with
   :pyfunc:`RQA2.save_results` to guarantee auditable workflows.

Extending RQA2
==============
Researchers can subclass *RQA2* to integrate custom parameter heuristics or
novel RQA measures.  Key extension points are methods:

* ``_findtau_*`` – time-delay strategies
* ``_findm`` – embedding selection
* ``_findeps`` – radius selection
* Measure helpers (e.g., ``_diaghist``) for new statistics

Citing RQA2
===========
Please cite the following when using RQA2 in academic publications:

.. code-block:: text

   SMdRQA Development Team. (2025). SMdRQA



=======

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


RQA2_simulators – Chaotic System Generators
============================================

``RQA2_simulators`` integrates four continuous-time attractors (using
``scipy.integrate.solve_ivp``, RK45, ``rtol=1e-9``, ``atol=1e-12``) and one
discrete-time map.

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
   * - ``generate_test_battery()``
     - All of the above
     - Returns a dict with keys: ``'rossler_chaotic'``, ``'rossler_sync'``, ``'lorenz'``, ``'henon'``, ``'chua'``.

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_simulators

   sim = RQA2_simulators(seed=42)

   x, y, z = sim.rossler(tmax=5000, n=2000, a=0.1)   # limit cycle
   x, y, z = sim.lorenz(n=2000)                        # butterfly
   x, y    = sim.henon(n=2000)                         # Hénon map
   systems = sim.generate_test_battery()               # full battery


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
>>>>>>> origin/master
