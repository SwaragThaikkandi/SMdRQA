RQA2 ML Pipeline Example
========================

This example shows how to use the **RQA2_ml** pipeline to:

* build a feature table from simulated dynamical systems,
* benchmark supervised classifiers, and
* benchmark unsupervised clustering methods.

The full runnable script is available in:

``docs/examples/rqa2_ml_pipeline_example.py``

Quick Walkthrough
-----------------

If you are brand new, think of this example as:

1. Generate a few example time series.
2. Turn each series into a set of numbers (RQA features).
3. Train a classifier and try clustering.

.. code-block:: python

   from SMdRQA.RQA2 import RQA2_ml, RQA2_simulators
   import numpy as np

   sim = RQA2_simulators(seed=42)
   battery = sim.generate_test_battery()

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

Expected Output
---------------

You should see:

* A table of supervised performance metrics (accuracy and macro-F1)
* A table of unsupervised silhouette scores
* Cluster label arrays for each method

Adjust the `window_size` and `window_step` to trade off temporal resolution
and computational cost.
