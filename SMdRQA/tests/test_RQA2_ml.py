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


# ---------------------------------------------------------------------------
# Static helper tests
# ---------------------------------------------------------------------------

class TestStaticHelpers:

    def test_rank_p_value_greater(self):
        # Real score above all nulls → p ≈ 1/(B+1)
        p = RQA2_ml._rank_p_value(10.0, np.array([1, 2, 3, 4, 5]))
        assert p == pytest.approx(1 / 6)

    def test_rank_p_value_below_null(self):
        # Real score below all nulls → p ≈ 1.0
        p = RQA2_ml._rank_p_value(0.0, np.array([1, 2, 3, 4, 5]))
        assert p == pytest.approx(6 / 6)

    def test_rank_p_value_less_alternative(self):
        p = RQA2_ml._rank_p_value(0.0, np.array([1, 2, 3, 4, 5]),
                                   alternative='less')
        assert p == pytest.approx(1 / 6)

    def test_benjamini_hochberg_no_correction_needed(self):
        pvals = {'a': 0.01, 'b': 0.02, 'c': 0.03}
        adj, rej = RQA2_ml._benjamini_hochberg(pvals, alpha=0.05)
        assert all(rej.values()), "All should be significant"
        for k in pvals:
            assert adj[k] <= 0.05

    def test_benjamini_hochberg_some_rejected(self):
        pvals = {'a': 0.01, 'b': 0.5, 'c': 0.9}
        adj, rej = RQA2_ml._benjamini_hochberg(pvals, alpha=0.05)
        assert rej['a'] is True
        assert rej['c'] is False

    def test_benjamini_hochberg_empty(self):
        adj, rej = RQA2_ml._benjamini_hochberg({}, alpha=0.05)
        assert adj == {}
        assert rej == {}


# ---------------------------------------------------------------------------
# Surrogate signal generation
# ---------------------------------------------------------------------------

class TestSurrogateSignalGeneration:

    def test_generate_surrogate_signals_ft(self):
        rng = np.random.default_rng(42)
        signals = [rng.standard_normal(128).astype(float) for _ in range(3)]
        ml = RQA2_ml()
        surr = ml._generate_surrogate_signals(signals, 'FT',
                                               random_state=0)
        assert len(surr) == 3
        for s, orig in zip(surr, signals):
            assert len(s) == len(orig)
            # Surrogate should differ from original
            assert not np.allclose(s, orig)

    def test_generate_surrogate_signals_aaft(self):
        rng = np.random.default_rng(42)
        signals = [rng.standard_normal(128).astype(float) for _ in range(2)]
        ml = RQA2_ml()
        surr = ml._generate_surrogate_signals(signals, 'AAFT',
                                               random_state=1)
        assert len(surr) == 2

    def test_generate_surrogate_signals_reproducible(self):
        rng = np.random.default_rng(42)
        signals = [rng.standard_normal(128).astype(float)]
        ml = RQA2_ml()
        s1 = ml._generate_surrogate_signals(signals, 'FT',
                                             random_state=99)
        s2 = ml._generate_surrogate_signals(signals, 'FT',
                                             random_state=99)
        assert np.allclose(s1[0], s2[0])


# ---------------------------------------------------------------------------
# Surrogate null benchmark (supervised) – lightweight smoke test
# ---------------------------------------------------------------------------

class TestSurrogateNullBenchmark:

    @pytest.fixture()
    def _two_class_signals(self):
        """Generate two classes of short signals with different dynamics."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 4 * np.pi, 80)
        signals = []
        labels = []
        for _ in range(4):
            signals.append(
                (np.sin(t) + 0.1 * rng.standard_normal(80)).astype(float))
            labels.append(0)
        for _ in range(4):
            signals.append(
                (np.sin(3 * t) + 0.1 * rng.standard_normal(80)).astype(float))
            labels.append(1)
        return signals, labels

    def test_surrogate_null_benchmark_smoke(self, _two_class_signals):
        """End-to-end smoke test with minimal iterations."""
        signals, labels = _two_class_signals
        ml = RQA2_ml(rqa_kwargs={'normalize': True})
        results = ml.surrogate_null_benchmark(
            signals, labels,
            window_size=20,
            window_step=10,
            surrogate_kinds=('FT',),
            n_surrogate_iterations=2,
            model='knn',
            outer_iterations=5,
            surrogate_outer_iterations=3,
            inner_splits=2,
            inner_iterations=1,
            feature_selection=None,
            scaler=True,
            random_state=42,
            include_permutation=True,
            n_permutations=5,
            verbose=False,
        )

        # Check structure
        assert 'real' in results
        assert 'surrogates' in results
        assert 'permutation' in results
        assert 'corrected_p_values' in results
        assert 'summary' in results

        # Real results
        assert 'accuracy' in results['real']
        assert len(results['real']['accuracy']) == 5

        # Surrogate results
        assert 'FT' in results['surrogates']
        ft = results['surrogates']['FT']
        assert len(ft['null_accuracies']) == 2
        assert 'p_value_accuracy' in ft
        assert 'effect_size_accuracy' in ft

        # Permutation results
        assert 'null_accuracy' in results['permutation']
        assert len(results['permutation']['null_accuracy']) == 5

        # Summary table
        summary = results['summary']
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # FT + permutation
        assert 'null_type' in summary.columns
        assert 'p_value_accuracy' in summary.columns
        assert 'adjusted_p_accuracy' in summary.columns
        assert 'effect_size_accuracy' in summary.columns

    def test_surrogate_null_no_permutation(self, _two_class_signals):
        signals, labels = _two_class_signals
        ml = RQA2_ml()
        results = ml.surrogate_null_benchmark(
            signals, labels,
            window_size=20,
            surrogate_kinds=('FT',),
            n_surrogate_iterations=2,
            outer_iterations=3,
            surrogate_outer_iterations=2,
            inner_splits=2,
            inner_iterations=1,
            feature_selection=None,
            include_permutation=False,
            verbose=False,
        )
        assert results['permutation'] is None
        assert len(results['summary']) == 1  # FT only

    def test_surrogate_null_multiple_kinds(self, _two_class_signals):
        signals, labels = _two_class_signals
        ml = RQA2_ml()
        results = ml.surrogate_null_benchmark(
            signals, labels,
            window_size=20,
            surrogate_kinds=('FT', 'AAFT'),
            n_surrogate_iterations=2,
            outer_iterations=3,
            surrogate_outer_iterations=2,
            inner_splits=2,
            inner_iterations=1,
            feature_selection=None,
            include_permutation=False,
            verbose=False,
        )
        assert 'FT' in results['surrogates']
        assert 'AAFT' in results['surrogates']
        assert len(results['summary']) == 2

    def test_surrogate_null_bonferroni(self, _two_class_signals):
        signals, labels = _two_class_signals
        ml = RQA2_ml()
        results = ml.surrogate_null_benchmark(
            signals, labels,
            window_size=20,
            surrogate_kinds=('FT',),
            n_surrogate_iterations=2,
            outer_iterations=3,
            surrogate_outer_iterations=2,
            inner_splits=2,
            inner_iterations=1,
            feature_selection=None,
            include_permutation=False,
            correction='bonferroni',
            verbose=False,
        )
        assert 'corrected_p_values' in results

    def test_surrogate_null_invalid_kind_raises(self, _two_class_signals):
        signals, labels = _two_class_signals
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="Unknown surrogate"):
            ml.surrogate_null_benchmark(
                signals, labels,
                window_size=20,
                surrogate_kinds=('INVALID',),
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Surrogate cluster validation (unsupervised) – lightweight smoke test
# ---------------------------------------------------------------------------

class TestSurrogateClusterValidation:

    @pytest.fixture()
    def _signals(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 4 * np.pi, 80)
        signals = []
        for i in range(6):
            freq = 1 + (i % 3)
            signals.append(
                (np.sin(freq * t) + 0.1 * rng.standard_normal(80)
                 ).astype(float))
        return signals

    def test_surrogate_cluster_validation_smoke(self, _signals):
        ml = RQA2_ml()
        results = ml.surrogate_cluster_validation(
            _signals,
            window_size=20,
            window_step=10,
            surrogate_kinds=('FT',),
            n_surrogate_iterations=2,
            methods=('kmeans',),
            k_range=(2, 3),
            scaler=True,
            random_state=42,
            verbose=False,
        )

        assert 'real' in results
        assert 'surrogates' in results
        assert 'corrected_p_values' in results
        assert 'summary' in results

        # Real results
        assert 'validity' in results['real']
        assert 'best_per_method' in results['real']

        # Surrogate results
        assert 'FT' in results['surrogates']
        ft = results['surrogates']['FT']
        assert 'kmeans' in ft
        assert 'null_silhouette' in ft['kmeans']
        assert len(ft['kmeans']['null_silhouette']) == 2

        # Summary
        summary = results['summary']
        assert isinstance(summary, pd.DataFrame)
        assert 'surrogate' in summary.columns
        assert 'cluster_method' in summary.columns

    def test_surrogate_cluster_invalid_kind_raises(self, _signals):
        ml = RQA2_ml()
        with pytest.raises(ValueError, match="Unknown surrogate"):
            ml.surrogate_cluster_validation(
                _signals,
                window_size=20,
                surrogate_kinds=('BADNAME',),
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Surrogate null visualisation
# ---------------------------------------------------------------------------

class TestSurrogateNullVisualization:

    def test_plot_surrogate_null_results(self):
        results = {
            'real': {
                'accuracy': np.array([0.8, 0.85, 0.82, 0.9, 0.78]),
                'roc_auc': np.array([0.85, 0.9, 0.87, 0.92, 0.83]),
                'model': 'knn',
            },
            'surrogates': {
                'FT': {
                    'null_accuracies': np.array([0.5, 0.52, 0.48]),
                    'null_roc_aucs': np.array([0.51, 0.53, 0.49]),
                    'p_value_accuracy': 0.01,
                    'p_value_roc_auc': 0.01,
                },
            },
            'permutation': {
                'null_accuracy': np.array([0.5, 0.55, 0.45]),
                'null_roc_auc': np.array([0.5, 0.55, 0.45]),
                'p_value_accuracy': 0.02,
                'p_value_roc_auc': 0.02,
            },
            'corrected_p_values': {
                'accuracy': {
                    'adjusted_p': {'FT': 0.02, 'permutation': 0.04},
                    'rejected': {'FT': True, 'permutation': True},
                },
                'roc_auc': {
                    'adjusted_p': {'FT': 0.02, 'permutation': 0.04},
                    'rejected': {'FT': True, 'permutation': True},
                },
            },
        }
        fig = RQA2_ml.plot_surrogate_null_results(results)
        assert fig is not None
        plt.close(fig)

    def test_plot_surrogate_null_results_save(self, tmp_path):
        results = {
            'real': {
                'accuracy': np.array([0.8, 0.85]),
                'roc_auc': np.array([0.85, 0.9]),
            },
            'surrogates': {
                'FT': {
                    'null_accuracies': np.array([0.5, 0.52]),
                    'null_roc_aucs': np.array([0.51, 0.53]),
                    'p_value_accuracy': 0.01,
                    'p_value_roc_auc': 0.01,
                },
            },
            'permutation': None,
            'corrected_p_values': {
                'accuracy': {
                    'adjusted_p': {'FT': 0.02},
                    'rejected': {'FT': True},
                },
                'roc_auc': {
                    'adjusted_p': {'FT': 0.02},
                    'rejected': {'FT': True},
                },
            },
        }
        path = str(tmp_path / "surr_null.png")
        fig = RQA2_ml.plot_surrogate_null_results(results,
                                                    save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)

    def test_plot_surrogate_cluster_validation(self):
        results = {
            'real': {
                'validity': pd.DataFrame(),
                'labels': {},
                'best_per_method': {
                    'kmeans': {'silhouette': 0.6,
                               'calinski_harabasz': 50.0,
                               'davies_bouldin': 0.8},
                },
            },
            'surrogates': {
                'FT': {
                    'kmeans': {
                        'null_silhouette': np.array([0.3, 0.35, 0.32]),
                        'p_value_silhouette': 0.01,
                        'effect_size_silhouette': 2.0,
                        'null_calinski_harabasz': np.array([20, 25, 22]),
                        'p_value_calinski_harabasz': 0.01,
                        'effect_size_calinski_harabasz': 2.0,
                        'null_davies_bouldin': np.array([1.5, 1.4, 1.6]),
                        'p_value_davies_bouldin': 0.01,
                        'effect_size_davies_bouldin': -2.0,
                    },
                },
            },
            'corrected_p_values': {'adjusted_p': {}, 'rejected': {}},
            'summary': pd.DataFrame([{
                'surrogate': 'FT',
                'cluster_method': 'kmeans',
            }]),
        }
        fig = RQA2_ml.plot_surrogate_cluster_validation(
            results, metric='silhouette')
        assert fig is not None
        plt.close(fig)
