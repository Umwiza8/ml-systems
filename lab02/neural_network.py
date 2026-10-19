
"""
Lab 2: Neural Network from Scratch
Dataset: sklearn load_digits
Architecture: 64 -> W -> W -> 10
Uses NumPy for the neural network calculations.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]

rng = np.random.default_rng(SEED)

digits = load_digits()
X = digits.data.astype(np.float64) / 16.0
y = digits.target

# One-hot encoding: digit 7 becomes a vector with
# a 1 at index 7 and zeros everywhere else.
Y = np.eye(10, dtype=np.float64)[y]

X_train, X_test, Y_train, Y_test, y_train, y_test = (
    train_test_split(
        X,
        Y,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)


# --------------------------------------------------
# 2. ACTIVATION FUNCTIONS
# --------------------------------------------------

def relu(z):
    """Keep positive values; replace negative values with 0."""
    return np.maximum(0.0, z)


def softmax(z):
    """Convert output scores into probabilities."""
    # Subtract the maximum for numerical stability.
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


# --------------------------------------------------
# 3. INITIALISE THE NETWORK PARAMETERS
# --------------------------------------------------

def initialize_parameters(width, seed=42):
    """
    W1: (64, width)   b1: (width,)
    W2: (width, width) b2: (width,)
    W3: (width, 10)   b3: (10,)

    Weight entries follow N(0, 0.01), interpreted
    as mean=0 and standard deviation=0.01.
    Biases start at zero.
    """
    random = np.random.default_rng(seed)

    return {
        "W1": random.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": random.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": random.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10),
    }


# --------------------------------------------------
# 4. FORWARD PROPAGATION
# --------------------------------------------------

def forward(X_batch, params):
    """
    Calculate predictions layer by layer.

    z1 = X @ W1 + b1
    a1 = ReLU(z1)
    z2 = a1 @ W2 + b2
    a2 = ReLU(z2)
    logits = a2 @ W3 + b3
    probabilities = softmax(logits)
    """
    z1 = X_batch @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    logits = a2 @ params["W3"] + params["b3"]
    probabilities = softmax(logits)

    cache = {
        "X": X_batch,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "probabilities": probabilities,
    }

    return probabilities, cache


# --------------------------------------------------
# 5. CROSS-ENTROPY LOSS
# --------------------------------------------------

def cross_entropy(probabilities, targets):
    """
    Average cross-entropy:
    L = -mean(sum(y_true * log(y_pred), axis=1))
    """
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)

    return -np.mean(
        np.sum(targets * np.log(probabilities), axis=1)
    )


def loss_function(X_batch, targets, params):
    probabilities, _ = forward(X_batch, params)
    return cross_entropy(probabilities, targets)


# --------------------------------------------------
# 6. BACKPROPAGATION
# --------------------------------------------------

def backward(targets, params, cache):
    """
    Compute gradients using the chain rule.

    For softmax + cross-entropy:
        d_logits = (probabilities - targets) / batch_size

    Then propagate the gradients backwards through
    the output layer and the two ReLU hidden layers.
    """
    X_batch = cache["X"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
    probabilities = cache["probabilities"]

    batch_size = X_batch.shape[0]

    # Output layer gradient
    d_logits = (probabilities - targets) / batch_size

    dW3 = a2.T @ d_logits
    db3 = np.sum(d_logits, axis=0)

    # Hidden layer 2 gradient
    da2 = d_logits @ params["W3"].T
    dz2 = da2 * (z2 > 0)

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    # Hidden layer 1 gradient
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)

    dW1 = X_batch.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }


# --------------------------------------------------
# 7. CHECK THE FORWARD PASS AND LOSS
# --------------------------------------------------

print("\n--- Forward pass and loss checks ---")

for width in WIDTHS:
    params = initialize_parameters(width, seed=SEED)
    probs, _ = forward(X_train[:32], params)

    initial_loss = loss_function(X_train, Y_train, params)

    print(f"\nWidth: {width}")
    print("Parameter shapes:")
    for name, value in params.items():
        print(f"  {name}: {value.shape}")

    print("Output shape:", probs.shape)
    print("First probability row sum:", probs[0].sum())
    print("Initial loss:", initial_loss)

    assert probs.shape == (32, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert abs(initial_loss - np.log(10)) < 0.01


# --------------------------------------------------
# 8. NUMERICAL GRADIENT CHECKING
# --------------------------------------------------



def gradient_check(X_batch, targets, params, num_checks=10):
    _, base_cache = forward(X_batch, params)
    analytical_grads = backward(targets, params, base_cache)

    rng = np.random.default_rng(123)
    weight_names = ["W1", "W2", "W3"]
    h = 1e-5

    base_mask1 = base_cache["z1"] > 0
    base_mask2 = base_cache["z2"] > 0

    passed = 0
    checked = 0
    attempts = 0
    max_attempts = 5000
    used_indices = set()

    while checked < num_checks and attempts < max_attempts:
        attempts += 1

        name = rng.choice(weight_names)
        index = tuple(
            rng.integers(size) for size in params[name].shape
        )

        # Avoid checking the same weight twice.
        key = (name, index)
        if key in used_indices:
            continue

        original = params[name][index]

        # Calculate loss with the weight increased by h.
        params[name][index] = original + h
        _, plus_cache = forward(X_batch, params)
        loss_plus = cross_entropy(
            plus_cache["probabilities"], targets
        )

        # Calculate loss with the weight decreased by h.
        params[name][index] = original - h
        _, minus_cache = forward(X_batch, params)
        loss_minus = cross_entropy(
            minus_cache["probabilities"], targets
        )

        # Always restore the original weight.
        params[name][index] = original

        # ReLU must behave the same at both perturbations.
        stable_relu = (
            np.array_equal(plus_cache["z1"] > 0, base_mask1)
            and np.array_equal(minus_cache["z1"] > 0, base_mask1)
            and np.array_equal(plus_cache["z2"] > 0, base_mask2)
            and np.array_equal(minus_cache["z2"] > 0, base_mask2)
        )

        if not stable_relu:
            continue

        analytical = analytical_grads[name][index]

        # Very small gradients make relative-error comparisons
        # unreliable, so skip them rather than misclassifying them.
        if abs(analytical) < 1e-5:
            continue

        numerical = (loss_plus - loss_minus) / (2 * h)

        denominator = max(
            1e-12, abs(numerical) + abs(analytical)
        )
        relative_error = (
            abs(numerical - analytical) / denominator
        )

        is_pass = relative_error < 1e-6

        used_indices.add(key)
        checked += 1
        passed += int(is_pass)

        print(
            f"{name}{index}: "
            f"analytical={analytical:.10e}, "
            f"numerical={numerical:.10e}, "
            f"relative error={relative_error:.3e}, "
            f"{'PASS' if is_pass else 'CHECK'}"
        )

    print(f"Gradient check: {passed}/{checked} passed")

    if checked < num_checks:
        print(
            "Warning: only found "
            f"{checked} informative gradients after {attempts} attempts."
        )

    return passed


print("\n--- Numerical gradient check ---")
gradient_params = initialize_parameters(16, seed=42)

gradient_check(
    X_train[:8],
    Y_train[:8],
    gradient_params,
    num_checks=10,
)


# --------------------------------------------------
# 9. MINI-BATCH TRAINING
# --------------------------------------------------

def train_network(width, learning_rate, seed=42):
    params = initialize_parameters(width, seed=seed)
    random = np.random.default_rng(seed)

    history = []

    for epoch in range(EPOCHS):
        # Shuffle the training examples every epoch.
        order = random.permutation(len(X_train))

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]

            xb = X_train[batch_indices]
            yb = Y_train[batch_indices]

            probabilities, cache = forward(xb, params)
            gradients = backward(yb, params, cache)

            # Gradient descent update
            for name in params:
                params[name] -= learning_rate * gradients[name]

        # Measure loss on the full training set.
        epoch_loss = loss_function(X_train, Y_train, params)
        history.append(epoch_loss)

        print(
            f"Width={width:2d}, LR={learning_rate:.3f}, "
            f"Epoch={epoch + 1:02d}/{EPOCHS}, "
            f"Loss={epoch_loss:.6f}"
        )

    test_probs, _ = forward(X_test, params)
    predictions = np.argmax(test_probs, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)

    return params, history, accuracy


# --------------------------------------------------
# 10. RUN ALL SIX EXPERIMENTS
# --------------------------------------------------

results = {}

for width in WIDTHS:
    for learning_rate in LEARNING_RATES:
        key = (width, learning_rate)

        print("\n" + "=" * 60)
        print(f"Training width={width}, learning rate={learning_rate}")
        print("=" * 60)

        params, history, accuracy = train_network(
            width,
            learning_rate,
            seed=SEED,
        )

        results[key] = {
            "params": params,
            "loss": history,
            "accuracy": accuracy,
        }

        print(f"Final test accuracy: {accuracy:.4%}")


# --------------------------------------------------
# 11. PLOT AND SAVE LOSS CURVES
# --------------------------------------------------

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, width in zip(axes, WIDTHS):
    for learning_rate in LEARNING_RATES:
        history = results[(width, learning_rate)]["loss"]

        ax.plot(
            range(1, EPOCHS + 1),
            history,
            label=f"LR={learning_rate}",
        )

    ax.set_title(f"Hidden width W={width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cross-entropy loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / "loss_curves.png", dpi=150)
plt.close(fig)

print("\nSaved loss curves to results/loss_curves.png")


# --------------------------------------------------
# 12. FORWARD AND BACKWARD TIMING
# --------------------------------------------------

def benchmark(width, runs=100):
    params = initialize_parameters(width, seed=42)

    # A full mini-batch of 32 images
    xb = X_train[:BATCH_SIZE]
    yb = Y_train[:BATCH_SIZE]

    # Warm-up runs are not included in the measurements.
    for _ in range(10):
        _, cache = forward(xb, params)
        backward(yb, params, cache)

    forward_times = []
    backward_times = []

    for _ in range(runs):
        start = time.perf_counter()
        _, cache = forward(xb, params)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward(yb, params, cache)
        backward_times.append(time.perf_counter() - start)

    forward_ms = np.median(forward_times) * 1000
    backward_ms = np.median(backward_times) * 1000

    return forward_ms, backward_ms


print("\n--- Median timing results ---")

for width in WIDTHS:
    forward_ms, backward_ms = benchmark(width, runs=100)

    print(f"\nWidth={width}")
    print(f"Forward median:  {forward_ms:.4f} ms")
    print(f"Backward median: {backward_ms:.4f} ms")

    if backward_ms > forward_ms:
        print("Backward pass was slower in this measurement.")
    else:
        print("Forward pass was slower in this measurement.")


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n--- FINAL TEST ACCURACY SUMMARY ---")
print("Width | Learning rate | Test accuracy")
print("-" * 42)

for (width, learning_rate), result in results.items():
    print(
        f"{width:5d} | {learning_rate:13.3f} | "
        f"{result['accuracy']:.4%}"
    )

best_key = max(
    results,
    key=lambda key: results[key]["accuracy"],
)

print(
    f"\nBest observed setting: width={best_key[0]}, "
    f"learning rate={best_key[1]}, "
    f"test accuracy={results[best_key]['accuracy']:.4%}"
)
