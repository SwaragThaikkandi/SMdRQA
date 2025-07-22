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

Quick Start
===========

.. code-block:: python

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

API Reference
=============

Constructor
-----------
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



