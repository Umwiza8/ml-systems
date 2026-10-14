# Lab 1: Baseline Model Benchmarking

# Goal

The goal of this experiment was to compare Logistic Regression and Random Forest on the Breast Cancer Wisconsin Diagnostic dataset. The comparison focused on test accuracy, training time, inference latency, saved model size, and observed process memory usage.

# Method

The dataset contained 569 samples and 30 features. The data was divided into training and testing sets using a 70/30 split with `random_state=42` and stratification.

The models used were:
- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with 100 trees and `random_state=42`.

Training time was measured over five runs after a warm-up, and the median was reported. Inference latency was measured for a single sample over 100 repetitions, using the median latency.

# Results

| Measurement | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.9415 | 0.9357 |
| Median training time (s) | 0.381733 | 0.133023 |
| Median inference latency (ms) | 0.0471 | 1.8988 |
| Saved model size (MB) | 0.0010 | 0.2774 |
| Observed process RSS increase (MB) | 0.0898 | 0.1523 |

The Logistic Regression run produced a convergence warning because it reached the maximum number of iterations. This should be investigated in a follow-up experiment using feature scaling.

The memory figures represent observed increases in process resident memory during the measurement, not the total memory required to run each model.

# Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest on this test split.
2. Random Forest trained faster in this experiment, but Logistic Regression had much lower single-sample inference latency.
3. Logistic Regression produced a substantially smaller saved model file. However, deployment suitability also depends on runtime memory requirements and other device constraints.

# Limitations and Next Steps

The results come from one train/test split. The Logistic Regression convergence warning indicates that feature scaling should be investigated. Further work should validate the results, measure memory usage more rigorously, and assess suitability against the deployment budgets for cloud, edge, mobile, and TinyML environments.