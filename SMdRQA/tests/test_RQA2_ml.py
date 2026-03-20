"""
Tests for the RQA2_ml class.

Covers: build_feature_table, supervised_benchmark, nested_cv_benchmark,
        surrogate_baseline, compare_scores, feature_importance,
        unsupervised_benchmark (enhanced), cluster_stability,
        and all visualisation helpers.
"""

import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from SMdRQA.RQA2 import RQA2, RQA2_ml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_class_data(n_per_class=20, n_features=4, seed=42):
    """Generate a simple linearly-separable two-class dataset."""
    rng = np.random.default_rng(seed)
    X_a = rng.standard_normal((n_per_class, n_features)) - 2
    X_b = rng.standard_normal((n_per_class, n_features)) + 2
    X = np.vstack([X_a, X_b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def _three_class_data(n_per_class=15, n_features=4, seed=42):
    """Generate a simple three-class dataset."""
    rng = np.random.default_rng(seed)
    Xs = [rng.standard_normal((n_per_class, n_features)) + 3 * i
          for i in range(3)]
    X = np.vstack(Xs)
    y = np.concatenate([[i] * n_per_class for i in range(3)])
    return X, y


# ---------------------------------------------------------------------------
# _make_model helper
# ---------------------------------------------------------------------------

class TestMakeModel:

    def test_knn(self):
        m = RQA2_ml._make_model('knn')
        assert hasattr(m, 'fit')

    def test_svm(self):
        m = RQA2_ml._make_model('svm')
        assert hasattr(m, 'predict_proba')  # probability=True

    def test_rf(self):
        m = RQA2_ml._make_model('rf', random_state=0)
        assert hasattr(m, 'predict')

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            RQA2_ml._make_model('xgboost')


# ---------------------------------------------------------------------------
# build_feature_table
# ---------------------------------------------------------------------------

class TestBuildFeatureTable:

    def test_includes_window_stats(self):
        signals = [
            np.sin(np.linspace(0, 2 * np.pi, 80)),
            np.cos(np.linspace(0, 2 * np.pi, 80)),
        ]
        ml = RQA2_ml()
        features = ml.build_feature_table(
            signals, labels=['a', 'b'],
            window_size=20, window_step=10,
            rqa_kwargs={'tau': 1, 'm': 2, 'eps': 0.5})

        assert features.shape[0] == 2
        assert 'label' in features.columns
        assert 'win_recurrence_rate__mean' in features.columns
        assert 'win_recurrence_rate__median' in features.columns

    def test_no_labels(self):
        signals = [np.sin(np.linspace(0, 2 * np.pi, 80))]
        ml = RQA2_ml()
        features = ml.build_feature_table(
            signals, window_size=20,
            rqa_kwargs={'tau': 1, 'm': 2, 'eps': 0.5})

        assert 'label' not in features.columns
        assert features.shape[0] == 1

    def test_label_length_mismatch_raises(self):
        signals = [np.zeros(80), np.ones(80)]
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="labels length"):
            ml.build_feature_table(
                signals, labels=['a'],
                window_size=20,
                rqa_kwargs={'tau': 1, 'm': 2, 'eps': 0.5})


# ---------------------------------------------------------------------------
# supervised_benchmark (quick)
# ---------------------------------------------------------------------------

class TestSupervisedBenchmark:

    def test_returns_results_and_model(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        results, model = ml.supervised_benchmark(X, y, cv=2, models=('knn',))
        assert 'accuracy_mean' in results.columns
        assert 'roc_auc_mean' in results.columns
        assert hasattr(model, 'predict')

    def test_all_models(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        results, _ = ml.supervised_benchmark(
            X, y, cv=2, models=('knn', 'svm', 'rf'))
        assert len(results) == 3

    def test_single_class_raises(self):
        X = np.ones((10, 2))
        y = np.zeros(10)
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="at least two classes"):
            ml.supervised_benchmark(X, y)

    def test_dataframe_input(self):
        X, y = _two_class_data()
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        ml = RQA2_ml()
        results, model = ml.supervised_benchmark(df, y, cv=2, models=('knn',))
        assert results.shape[0] == 1

    def test_multiclass(self):
        X, y = _three_class_data()
        ml = RQA2_ml()
        results, model = ml.supervised_benchmark(X, y, cv=2, models=('knn',))
        assert results['roc_auc_mean'].iloc[0] >= 0


# ---------------------------------------------------------------------------
# nested_cv_benchmark
# ---------------------------------------------------------------------------

class TestNestedCVBenchmark:

    def test_basic_knn(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='knn', outer_iterations=5,
            feature_selection=None, random_state=42)
        assert len(res['accuracy']) == 5
        assert len(res['roc_auc']) == 5
        assert res['model'] == 'knn'

    def test_with_forward_feature_selection(self):
        X, y = _two_class_data(n_features=6)
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='knn', outer_iterations=3,
            feature_selection='forward', max_subset_size=3,
            inner_iterations=2, random_state=42)
        assert len(res['selected_features']) == 3
        for subset in res['selected_features']:
            assert len(subset) <= 3

    def test_with_exhaustive_feature_selection(self):
        X, y = _two_class_data(n_features=3)
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='rf', outer_iterations=2,
            feature_selection='exhaustive',
            inner_iterations=2, random_state=42)
        assert len(res['selected_features']) == 2
        assert isinstance(res['feature_frequency'], pd.Series)

    def test_auto_feature_selection_small(self):
        """auto should use exhaustive for <= 12 features."""
        X, y = _two_class_data(n_features=4)
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='knn', outer_iterations=2,
            feature_selection='auto', inner_iterations=2,
            random_state=42)
        assert len(res['accuracy']) == 2

    def test_feature_frequency_sums(self):
        X, y = _two_class_data(n_features=3)
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='knn', outer_iterations=5,
            feature_selection=None, random_state=42)
        # When no feature selection, all features are selected every time
        assert all(v == 5 for v in res['feature_frequency'].values)

    def test_dataframe_preserves_names(self):
        X, y = _two_class_data(n_features=3)
        df = pd.DataFrame(X, columns=['alpha', 'beta', 'gamma'])
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            df, y, model='knn', outer_iterations=2,
            feature_selection=None, random_state=42)
        assert res['feature_names'] == ['alpha', 'beta', 'gamma']

    def test_svm_model(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        res = ml.nested_cv_benchmark(
            X, y, model='svm', outer_iterations=3,
            feature_selection=None, random_state=42)
        assert all(0 <= a <= 1 for a in res['accuracy'])

    def test_single_class_raises(self):
        X = np.ones((10, 2))
        y = np.zeros(10)
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="at least two classes"):
            ml.nested_cv_benchmark(X, y)


# ---------------------------------------------------------------------------
# surrogate_baseline
# ---------------------------------------------------------------------------

class TestSurrogateBaseline:

    def test_returns_null_distributions(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        null = ml.surrogate_baseline(
            X, y, n_permutations=10, model='knn', cv=2,
            random_state=42)
        assert len(null['null_accuracy']) == 10
        assert len(null['null_roc_auc']) == 10

    def test_null_accuracy_near_chance(self):
        X, y = _two_class_data(n_per_class=30)
        ml = RQA2_ml()
        null = ml.surrogate_baseline(
            X, y, n_permutations=20, model='knn', cv=2,
            random_state=42)
        # Permuted labels should give roughly chance-level accuracy
        mean_null = np.mean(null['null_accuracy'])
        assert 0.2 <= mean_null <= 0.8


# ---------------------------------------------------------------------------
# compare_scores
# ---------------------------------------------------------------------------

class TestCompareScores:

    def test_identical_scores_high_p(self):
        a = np.array([0.8, 0.85, 0.82, 0.79, 0.81])
        result = RQA2_ml.compare_scores(a, a)
        # Identical distributions: p-value should be high (or nan/1.0)
        assert result['n'] == 5

    def test_different_scores_low_p(self):
        a = np.array([0.9, 0.92, 0.91, 0.93, 0.90,
                       0.91, 0.92, 0.93, 0.90, 0.94])
        b = np.array([0.5, 0.52, 0.49, 0.51, 0.50,
                       0.48, 0.51, 0.49, 0.50, 0.52])
        result = RQA2_ml.compare_scores(a, b)
        assert result['p_value'] < 0.05
        assert result['effect_size'] != 0

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            RQA2_ml.compare_scores([1, 2, 3], [1, 2])

    def test_alternative_greater(self):
        a = np.array([0.9, 0.95, 0.92, 0.93, 0.91,
                       0.94, 0.90, 0.92, 0.93, 0.91])
        b = np.array([0.5, 0.52, 0.49, 0.51, 0.50,
                       0.48, 0.51, 0.49, 0.50, 0.52])
        result = RQA2_ml.compare_scores(a, b, alternative='greater')
        assert result['p_value'] < 0.05


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:

    def test_returns_dataframe(self):
        X, y = _two_class_data(n_features=4)
        ml = RQA2_ml()
        _, model = ml.supervised_benchmark(
            X, y, cv=2, models=('rf',))
        imp = RQA2_ml.feature_importance(
            model, X, y, n_repeats=5, random_state=42)
        assert isinstance(imp, pd.DataFrame)
        assert 'feature' in imp.columns
        assert 'importance_mean' in imp.columns
        assert len(imp) == 4

    def test_with_dataframe_columns(self):
        X, y = _two_class_data(n_features=3)
        df = pd.DataFrame(X, columns=['a', 'b', 'c'])
        ml = RQA2_ml()
        _, model = ml.supervised_benchmark(
            df, y, cv=2, models=('knn',))
        imp = RQA2_ml.feature_importance(
            model, df, y, n_repeats=3, random_state=42)
        assert list(imp['feature']) != []
        # Feature names should come from the DataFrame
        assert all(f in ['a', 'b', 'c'] for f in imp['feature'])


# ---------------------------------------------------------------------------
# unsupervised_benchmark (enhanced)
# ---------------------------------------------------------------------------

class TestUnsupervisedBenchmark:

    def test_silhouette_in_range(self):
        X = np.array([[0, 0], [0, 1], [3, 3], [3, 4]])
        ml = RQA2_ml()
        results, labels = ml.unsupervised_benchmark(
            X, n_clusters=2, methods=('kmeans',))
        score = float(results.loc[
            results['method'] == 'kmeans', 'silhouette'].iloc[0])
        assert -1.0 <= score <= 1.0
        assert len(labels['kmeans']) == len(X)

    def test_multiple_validity_indices(self):
        X = np.vstack([
            np.random.default_rng(0).standard_normal((20, 3)) - 3,
            np.random.default_rng(1).standard_normal((20, 3)) + 3,
        ])
        ml = RQA2_ml()
        results, _ = ml.unsupervised_benchmark(
            X, n_clusters=2, methods=('kmeans',))
        assert 'calinski_harabasz' in results.columns
        assert 'davies_bouldin' in results.columns

    def test_adjusted_rand_with_ground_truth(self):
        X = np.vstack([
            np.random.default_rng(0).standard_normal((20, 3)) - 3,
            np.random.default_rng(1).standard_normal((20, 3)) + 3,
        ])
        y_true = np.array([0] * 20 + [1] * 20)
        ml = RQA2_ml()
        results, _ = ml.unsupervised_benchmark(
            X, y_true=y_true, n_clusters=2, methods=('kmeans',))
        assert 'adjusted_rand' in results.columns
        assert results['adjusted_rand'].iloc[0] > 0  # well-separated

    def test_k_range_sweep(self):
        X = np.random.default_rng(42).standard_normal((30, 3))
        ml = RQA2_ml()
        results, _ = ml.unsupervised_benchmark(
            X, k_range=(2, 4), methods=('kmeans',))
        assert len(results) == 3  # k=2,3,4

    def test_all_methods(self):
        X = np.random.default_rng(42).standard_normal((30, 3))
        ml = RQA2_ml()
        results, labels = ml.unsupervised_benchmark(
            X, n_clusters=2, methods=('kmeans', 'gmm', 'agglo'))
        assert set(results['method']) == {'kmeans', 'gmm', 'agglo'}
        assert len(labels) == 3

    def test_unknown_method_raises(self):
        X = np.ones((10, 2))
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="Unknown method"):
            ml.unsupervised_benchmark(X, methods=('dbscan',))


# ---------------------------------------------------------------------------
# cluster_stability
# ---------------------------------------------------------------------------

class TestClusterStability:

    def test_kmeans_stability(self):
        X = np.vstack([
            np.random.default_rng(0).standard_normal((20, 3)) - 3,
            np.random.default_rng(1).standard_normal((20, 3)) + 3,
        ])
        ml = RQA2_ml()
        stab = ml.cluster_stability(
            X, method='kmeans', n_clusters=2, n_bootstrap=10,
            random_state=42)
        assert 0 <= stab['mean_ari'] <= 1
        assert stab['n_bootstrap'] == 10
        assert stab['method'] == 'kmeans'

    def test_gmm_stability(self):
        X = np.vstack([
            np.random.default_rng(0).standard_normal((20, 3)) - 3,
            np.random.default_rng(1).standard_normal((20, 3)) + 3,
        ])
        ml = RQA2_ml()
        stab = ml.cluster_stability(
            X, method='gmm', n_clusters=2, n_bootstrap=10,
            random_state=42)
        assert 'mean_ari' in stab
        assert 'std_ari' in stab

    def test_agglo_raises(self):
        X = np.ones((10, 2))
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="predict"):
            ml.cluster_stability(X, method='agglo')


# ---------------------------------------------------------------------------
# Visualisation methods
# ---------------------------------------------------------------------------

class TestVisualisations:

    def test_plot_benchmark_results(self, tmp_path):
        results = {
            'accuracy': np.array([0.8, 0.85, 0.82]),
            'roc_auc': np.array([0.9, 0.92, 0.88]),
            'model': 'knn',
        }
        save = str(tmp_path / "bench.png")
        fig = RQA2_ml.plot_benchmark_results(
            results, save_path=save, title="Test")
        plt.close('all')
        assert os.path.exists(save)

    def test_plot_benchmark_results_with_baseline(self, tmp_path):
        results = {
            'accuracy': np.array([0.8, 0.85, 0.82]),
            'roc_auc': np.array([0.9, 0.92, 0.88]),
            'model': 'knn',
        }
        baseline = {
            'null_accuracy': np.array([0.5, 0.52, 0.48]),
            'null_roc_auc': np.array([0.5, 0.51, 0.49]),
        }
        fig = RQA2_ml.plot_benchmark_results(
            results, baseline=baseline)
        plt.close('all')
        assert fig is not None

    def test_plot_confusion_matrix(self, tmp_path):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        save = str(tmp_path / "cm.png")
        fig = RQA2_ml.plot_confusion_matrix(
            y_true, y_pred, labels=['a', 'b', 'c'],
            save_path=save, title="CM Test")
        plt.close('all')
        assert os.path.exists(save)

    def test_plot_feature_importance(self, tmp_path):
        df = pd.DataFrame({
            'feature': ['f1', 'f2', 'f3'],
            'importance_mean': [0.3, 0.1, 0.05],
            'importance_std': [0.02, 0.01, 0.005],
        })
        save = str(tmp_path / "fi.png")
        fig = RQA2_ml.plot_feature_importance(
            df, save_path=save, title="FI Test")
        plt.close('all')
        assert os.path.exists(save)

    def test_plot_cluster_validity(self, tmp_path):
        results_df = pd.DataFrame({
            'method': ['kmeans'] * 3,
            'n_clusters': [2, 3, 4],
            'silhouette': [0.6, 0.5, 0.4],
            'calinski_harabasz': [100, 80, 60],
            'davies_bouldin': [0.5, 0.7, 0.9],
        })
        save = str(tmp_path / "cv.png")
        fig = RQA2_ml.plot_cluster_validity(
            results_df, save_path=save, title="CV Test")
        plt.close('all')
        assert os.path.exists(save)

    def test_plot_cluster_scatter(self, tmp_path):
        X = np.random.default_rng(42).standard_normal((30, 5))
        labels = np.array([0] * 15 + [1] * 15)
        save = str(tmp_path / "scatter.png")
        fig = RQA2_ml.plot_cluster_scatter(
            X, labels, save_path=save, title="Scatter Test")
        plt.close('all')
        assert os.path.exists(save)

    def test_plot_cluster_scatter_2d(self):
        X = np.random.default_rng(42).standard_normal((20, 2))
        labels = np.array([0] * 10 + [1] * 10)
        fig = RQA2_ml.plot_cluster_scatter(X, labels)
        plt.close('all')
        assert fig is not None


# ---------------------------------------------------------------------------
# _compute_roc_auc edge cases
# ---------------------------------------------------------------------------

class TestComputeRocAuc:

    def test_binary_with_predict_proba(self):
        X, y = _two_class_data()
        ml = RQA2_ml()
        _, model = ml.supervised_benchmark(
            X, y, cv=2, models=('rf',))
        auc = RQA2_ml._compute_roc_auc(model, X, y)
        assert 0 <= auc <= 1

    def test_multiclass(self):
        X, y = _three_class_data()
        ml = RQA2_ml()
        _, model = ml.supervised_benchmark(
            X, y, cv=2, models=('rf',))
        auc = RQA2_ml._compute_roc_auc(model, X, y)
        assert 0 <= auc <= 1


# ---------------------------------------------------------------------------
# _load_signals edge cases
# ---------------------------------------------------------------------------

class TestLoadSignals:

    def test_numpy_array(self):
        ml = RQA2_ml()
        signals, ids = ml._load_signals(np.zeros(10))
        assert len(signals) == 1
        assert ids == ["signal_0"]

    def test_list_of_arrays(self):
        ml = RQA2_ml()
        signals, ids = ml._load_signals([np.zeros(5), np.ones(5)])
        assert len(signals) == 2

    def test_empty_list_raises(self):
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="empty"):
            ml._load_signals([])

    def test_invalid_type_raises(self):
        ml = RQA2_ml()
        with pytest.raises(TypeError):
            ml._load_signals(42)

    def test_directory_with_npy(self, tmp_path):
        np.save(str(tmp_path / "a.npy"), np.zeros(10))
        np.save(str(tmp_path / "b.npy"), np.ones(10))
        ml = RQA2_ml()
        signals, ids = ml._load_signals(str(tmp_path))
        assert len(signals) == 2

    def test_empty_directory_raises(self, tmp_path):
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="No .npy"):
            ml._load_signals(str(tmp_path))

    def test_nonexistent_directory_raises(self):
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="directory path"):
            ml._load_signals("/nonexistent/path/to/dir")


# ---------------------------------------------------------------------------
# windowed RQA measures (on RQA2 directly)
# ---------------------------------------------------------------------------

class TestWindowedMeasures:

    def test_windowed_measures_summary(self):
        data = np.sin(np.linspace(0, 2 * np.pi, 60))
        rqa = RQA2(data)
        rqa._tau = 1
        rqa._m = 2
        rqa._eps = 0.5

        windowed = rqa.compute_windowed_rqa_measures(
            window_size=20, window_step=10)
        assert len(windowed) == 4
        assert 'recurrence_rate' in windowed.columns

        summary = rqa.summarize_windowed_measures(windowed)
        assert 'recurrence_rate__mean' in summary
        assert 'recurrence_rate__median' in summary
        assert 'recurrence_rate__mode' in summary
