#!/usr/bin/env python3

from SMdRQA.RQA2 import RQA2_ml, RQA2_simulators
import numpy as np


def main():
    print("Generating simulator battery...")
    sim = RQA2_simulators(seed=42)
    battery = sim.generate_test_battery()

    signals = []
    labels = []

    # Create multiple samples per system by slicing the trajectory
    segments = 4
    for name, data in battery.items():
        x = data['x']
        seg_len = len(x) // segments
        for i in range(segments):
            seg = x[i * seg_len:(i + 1) * seg_len]
            signals.append(seg)
            labels.append(name)

    print(f"Total samples: {len(signals)} across {len(set(labels))} classes")

    pipeline = RQA2_ml()
    features = pipeline.build_feature_table(
        signals,
        labels=labels,
        window_size=100,
        window_step=20,
        window_stats=('mean', 'median', 'mode'),
        include_params=True,
    )

    X = features.drop(columns=['id', 'label'])
    y = features['label']

    print("\nSupervised benchmark...")
    sup_results, best_model = pipeline.supervised_benchmark(X, y, cv=3)
    print(sup_results)

    print("\nUnsupervised benchmark...")
    n_clusters = len(set(labels))
    unsup_results, unsup_labels = pipeline.unsupervised_benchmark(
        X, n_clusters=n_clusters)
    print(unsup_results)

    for method, lbls in unsup_labels.items():
        print(f"{method} labels: {np.asarray(lbls)}")


if __name__ == "__main__":
    main()
