import numpy as np
import pandas as pd

from SMdRQA.RQA2 import RQA2, RQA2_ml


def test_windowed_measures_summary():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    rqa = RQA2(data)
    rqa._tau = 1
    rqa._m = 2
    rqa._eps = 0.5

    windowed = rqa.compute_windowed_rqa_measures(
        window_size=20, window_step=10)
    assert len(windowed) == 5
    assert 'recurrence_rate' in windowed.columns

    summary = rqa.summarize_windowed_measures(windowed)
    assert 'recurrence_rate__mean' in summary
    assert 'recurrence_rate__median' in summary
    assert 'recurrence_rate__mode' in summary


def test_build_feature_table_includes_window_stats():
    signals = [
        np.sin(np.linspace(0, 2 * np.pi, 80)),
        np.cos(np.linspace(0, 2 * np.pi, 80)),
    ]
    labels = ['a', 'b']
    pipeline = RQA2_ml()
    features = pipeline.build_feature_table(
        signals,
        labels=labels,
        window_size=20,
        window_step=10,
        rqa_kwargs={'tau': 1, 'm': 2, 'eps': 0.5},
    )

    assert features.shape[0] == 2
    assert 'label' in features.columns
    assert 'win_recurrence_rate__mean' in features.columns
    assert 'win_recurrence_rate__median' in features.columns
    assert 'win_recurrence_rate__mode' in features.columns


def test_supervised_benchmark_returns_results():
    X = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    pipeline = RQA2_ml()
    results, model = pipeline.supervised_benchmark(
        X, y, cv=2, models=('knn',))

    assert 'accuracy_mean' in results.columns
    assert hasattr(model, 'predict')


def test_unsupervised_benchmark_silhouette():
    X = np.array([[0, 0], [0, 1], [3, 3], [3, 4]])
    pipeline = RQA2_ml()
    results, labels = pipeline.unsupervised_benchmark(
        X, n_clusters=2, methods=('kmeans',))

    score = float(
        results.loc[results['method'] == 'kmeans', 'silhouette'].iloc[0])
    assert -1.0 <= score <= 1.0
    assert len(labels['kmeans']) == len(X)
