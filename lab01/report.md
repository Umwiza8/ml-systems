# Lab 1: Environment and First System Measurements

## Goal

The goal of this experiment was to compare Logistic Regression and Random Forest on the Breast Cancer Wisconsin Diagnostic dataset. The comparison considered test accuracy, training time, single-sample inference latency, saved model size, and observed process memory increase. The results were also compared with the resource budgets for Cloud, Edge, Mobile, and TinyML deployment.

## Method

The dataset was loaded using `load_breast_cancer(return_X_y=True)`. It contained 569 samples and 30 features. The data was split into training and testing sets using a stratified 70/30 split with `random_state=42`, resulting in 398 training samples and 171 test samples.

The models were:

- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with `n_estimators=100` and `random_state=42`.

Training time was measured over five runs after one warm-up run, and the median was reported. Single-sample inference latency was measured over 100 repetitions, with the median reported. The trained models were saved using `joblib`, and their file sizes were recorded in bytes, KB, and MB.

Process memory was measured by sampling resident set size (RSS) during training and inference. These measurements represent observed increases in process memory, not the total memory required by each model.

## Results

| Measurement | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.9415 | 0.9357 |
| Median training time (s) | 0.461468 | 0.172186 |
| Median inference latency (ms) | 0.0570 | 2.6693 |
| Saved model size (bytes) | 1,055 | 290,889 |
| Saved model size (KB) | 1.0303 | 284.0713 |
| Saved model size (MB) | 0.001006 | 0.277413 |
| Observed training RSS increase (MB) | 0.1211 | 0.0234 |
| Observed inference RSS increase (MB) | 0.0195 | 0.0195 |

Logistic Regression produced a convergence warning because the optimizer reached the maximum number of iterations. Its result should therefore be interpreted with this limitation in mind. A separate experiment with feature scaling could investigate whether convergence improves.

## Conclusions

1. Logistic Regression achieved slightly higher test accuracy than Random Forest on this split: 94.15% compared with 93.57%. This result applies to the current test set and does not guarantee performance on new data.

2. Random Forest trained faster in the measured runs, with a median of 0.172186 seconds compared with 0.461468 seconds for Logistic Regression. Logistic Regression had lower single-sample inference latency.

3. Logistic Regression produced a much smaller saved model file. However, deployment decisions must also consider actual runtime memory, device compatibility, and the convergence warning.

## Deployment Suitability

The following resource budgets are specified in the laboratory manual.

| Environment | Memory budget | Latency limit | Model size limit |
|---|---:|---:|---:|
| Cloud | ≥ 1 GB | ≤ 100 ms | ≤ 500 MB |
| Edge | 256–1024 MB | ≤ 50 ms | ≤ 50 MB |
| Mobile | 64–256 MB | ≤ 20 ms | ≤ 10 MB |
| TinyML | ≤ 256 KB | ≤ 10 ms | ≤ 100 KB |

### Cloud

Both models meet the specified latency and model-size limits. Memory suitability remains unverified because the measured RSS increases do not represent total runtime memory.

### Edge

Both models meet the specified latency and model-size limits. Actual runtime memory must be measured before confirming suitability for an Edge device.

### Mobile

Both models meet the specified latency and model-size limits. Further testing is needed to establish whether their total runtime memory fits the Mobile budget.

### TinyML

Logistic Regression meets the measured latency and saved-model-size limits. Random Forest exceeds the 100 KB saved-model-size limit because its model file is approximately 284.07 KB. The total memory requirement and compatibility with a specific TinyML device have not been verified.

These comparisons are preliminary. Measurements were collected on the current computer rather than on the target deployment devices.

## Limitations and Next Steps

The experiment used one train/test split and measurements from one computer. The memory-monitoring interval may miss short-lived peaks. Logistic Regression also produced a convergence warning.

Future work should test feature scaling separately, verify convergence, and measure total runtime memory on representative deployment hardware.
