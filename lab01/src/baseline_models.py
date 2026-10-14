
import csv
import gc
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Reproducibility and paths
# --------------------------------------------------

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)


# --------------------------------------------------
# 2. Load and split the dataset
# --------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print("\nDataset information")
print("Total samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 3. Define the baseline models
# --------------------------------------------------

def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
    }


# --------------------------------------------------
# 4. Measure inference latency
# --------------------------------------------------

def measure_inference_latency(model, sample, repeats=100):
    # Warm up inference once
    model.predict(sample)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times) * 1000


# --------------------------------------------------
# 5. Measure peak process memory during inference
# --------------------------------------------------

def measure_inference_memory(model, sample, interval=0.005):
    process = psutil.Process()
    baseline_memory = process.memory_info().rss
    peak_memory = baseline_memory
    stop_monitoring = threading.Event()

    def monitor_memory():
        nonlocal peak_memory

        while not stop_monitoring.is_set():
            try:
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)
            except psutil.Error:
                break

            stop_monitoring.wait(interval)

    # Warm-up before measuring
    model.predict(sample)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    try:
        for _ in range(100):
            model.predict(sample)
    finally:
        stop_monitoring.set()
        monitor.join()

    # Include a final process-memory reading
    peak_memory = max(peak_memory, process.memory_info().rss)

    # RSS is process-wide, not memory used exclusively by the model.
    return max(0, peak_memory - baseline_memory) / (1024 ** 2)


# --------------------------------------------------
# 6. Train, evaluate, time, and save each model
# --------------------------------------------------

accuracy_results = []
training_results = []
latency_results = []
size_results = []
memory_results = []

for name in create_models():
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    # Create a fresh model for each training run
    warmup_model = create_models()[name]
    warmup_model.fit(X_train, y_train)

    training_times = []

    for _ in range(5):
        model = create_models()[name]

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        training_times.append(end - start)

    median_training_time = statistics.median(training_times)

    # Train the final model used for evaluation and deployment
    model = create_models()[name]
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save the trained model
    safe_name = name.lower().replace(" ", "_")
    model_path = RESULTS_DIR / f"{safe_name}.joblib"
    joblib.dump(model, model_path)

    model_size_bytes = model_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 ** 2)

    # Measure single-sample inference latency
    sample = X_test[0:1]
    latency_ms = measure_inference_latency(model, sample)

    # Approximate process RSS increase during inference
    memory_mb = measure_inference_memory(model, sample)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Median training time: {median_training_time:.6f} seconds")
    print(f"Median inference latency: {latency_ms:.4f} ms")
    print(f"Saved model size: {model_size_mb:.4f} MB")
    print(f"Observed process RSS increase: {memory_mb:.4f} MB")

    accuracy_results.append([name, f"{accuracy:.4f}"])
    training_results.append([name, f"{median_training_time:.6f}"])
    latency_results.append([name, f"{latency_ms:.4f}"])
    size_results.append([name, model_size_bytes, f"{model_size_mb:.6f}"])
    memory_results.append([name, f"{memory_mb:.4f}"])

    del model, warmup_model
    gc.collect()


# --------------------------------------------------
# 7. Save CSV results
# --------------------------------------------------

def save_csv(filename, headers, rows):
    path = RESULTS_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved: {path}")


save_csv(
    "baseline_accuracy.csv",
    ["model", "test_accuracy"],
    accuracy_results,
)

save_csv(
    "training_time.csv",
    ["model", "median_training_time_seconds"],
    training_results,
)

save_csv(
    "inference_latency.csv",
    ["model", "median_inference_latency_ms"],
    latency_results,
)

save_csv(
    "model_size.csv",
    ["model", "model_size_bytes", "model_size_mb"],
    size_results,
)

save_csv(
    "inference_memory.csv",
    ["model", "observed_process_rss_increase_mb"],
    memory_results,
)

print("\nLab 1 baseline experiment completed.")
