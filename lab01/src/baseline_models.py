
import csv
import gc
import random
import statistics
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )
    }


def save_csv(filename, headers, rows):
    path = RESULTS_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved: {path}")


def measure_peak_memory(action, interval=0.005):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def monitor():
        nonlocal peak

        while not stop.is_set():
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.Error:
                break

            stop.wait(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    try:
        action()
    finally:
        peak = max(peak, process.memory_info().rss)
        stop.set()
        thread.join()

    increase_bytes = max(0, peak - baseline)

    return {
        "bytes": increase_bytes,
        "mb": increase_bytes / (1024 ** 2)
    }


def measure_inference_latency(model, sample, repeats=100):
    model.predict(sample)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return statistics.median(times) * 1000


def main():
    print("Loading Breast Cancer Wisconsin dataset...")

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    print("\nDataset information")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    accuracy_results = []
    training_results = []
    latency_results = []
    size_results = []
    training_memory_results = []
    inference_memory_results = []

    trained_models = {}

    for name in create_models():
        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)

        # One warm-up training run
        warmup_model = create_models()[name]
        warmup_model.fit(X_train, y_train)
        del warmup_model
        gc.collect()

        # Five measured training runs
        training_times = []

        for _ in range(5):
            model = create_models()[name]

            start = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            training_times.append(elapsed)

            del model

        median_training_time = statistics.median(training_times)

        # Measure observed process memory increase during training
        memory_model = create_models()[name]

        training_memory = measure_peak_memory(
            lambda: memory_model.fit(X_train, y_train)
        )

        del memory_model
        gc.collect()

        # Fit the final model for evaluation and saving
        model = create_models()[name]
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_name = name.lower().replace(" ", "_")
        model_path = RESULTS_DIR / f"{safe_name}.joblib"

        joblib.dump(model, model_path)

        model_size_bytes = model_path.stat().st_size
        model_size_kb = model_size_bytes / 1024
        model_size_mb = model_size_bytes / (1024 ** 2)

        # Median single-sample inference latency over 100 repetitions
        sample = X_test[0:1]

        latency_ms = measure_inference_latency(
            model,
            sample,
            repeats=100
        )

        # Measure observed process memory increase during inference
        inference_memory = measure_peak_memory(
            lambda: [
                model.predict(sample)
                for _ in range(100)
            ]
        )

        trained_models[name] = model

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Median training time: {median_training_time:.6f} seconds")
        print(f"Median inference latency: {latency_ms:.4f} ms")
        print(f"Model size: {model_size_bytes} bytes")
        print(f"Model size: {model_size_kb:.4f} KB")
        print(f"Model size: {model_size_mb:.6f} MB")
        print(
            "Observed training RSS increase: "
            f"{training_memory['mb']:.4f} MB"
        )
        print(
            "Observed inference RSS increase: "
            f"{inference_memory['mb']:.4f} MB"
        )

        accuracy_results.append([name, f"{accuracy:.4f}"])

        training_results.append([
            name,
            f"{median_training_time:.6f}"
        ])

        latency_results.append([
            name,
            f"{latency_ms:.4f}"
        ])

        size_results.append([
            name,
            model_size_bytes,
            f"{model_size_kb:.4f}",
            f"{model_size_mb:.6f}"
        ])

        training_memory_results.append([
            name,
            training_memory["bytes"],
            f"{training_memory['mb']:.4f}"
        ])

        inference_memory_results.append([
            name,
            inference_memory["bytes"],
            f"{inference_memory['mb']:.4f}"
        ])

        del model
        gc.collect()

    # Save the two trained models together as well
    combined_model_path = RESULTS_DIR / "model.joblib"
    joblib.dump(trained_models, combined_model_path)
    print(f"\nSaved both models: {combined_model_path}")

    # Save benchmark results
    save_csv(
        "baseline_accuracy.csv",
        ["model", "test_accuracy"],
        accuracy_results
    )

    save_csv(
        "training_time.csv",
        ["model", "median_training_time_seconds"],
        training_results
    )

    save_csv(
        "inference_latency.csv",
        ["model", "median_inference_latency_ms"],
        latency_results
    )

    save_csv(
        "model_size.csv",
        ["model", "model_size_bytes", "model_size_kb", "model_size_mb"],
        size_results
    )

    save_csv(
        "training_memory.csv",
        ["model", "observed_rss_increase_bytes", "observed_rss_increase_mb"],
        training_memory_results
    )

    save_csv(
        "inference_memory.csv",
        ["model", "observed_rss_increase_bytes", "observed_rss_increase_mb"],
        inference_memory_results
    )

    print("\nLab 1 benchmark completed successfully.")
    print("The report.md file was not modified by this script.")


if __name__ == "__main__":
    main()
