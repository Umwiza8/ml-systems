
import csv
import json
import random
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

from models import MLP, CNN, AttentionModel, count_parameters

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_DIR = ROOT.parent / "data"

RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    train_subset, validation_subset = random_split(
        train_dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training images: {len(train_subset):,}")
    print(f"Validation images: {len(validation_subset):,}")
    print(f"Test images: {len(test_dataset):,}")

    return train_loader, validation_loader, test_loader

def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(name, model, train_loader, validation_loader, test_loader):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy = evaluate(
            model, validation_loader, criterion
        )

        history.append({
            "model": name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| train accuracy {train_accuracy:.2f}% "
            f"| validation accuracy {val_accuracy:.2f}%"
        )

    training_seconds = time.perf_counter() - start_time
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion
    )

    print(
        f"{name} | Test accuracy: {test_accuracy:.2f}% "
        f"| Training time: {training_seconds:.2f}s"
    )

    return model, {
        "model": name,
        "parameters": count_parameters(model),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_seconds": training_seconds,
    }, history

def profile_model(name, model):
    model.eval()
    dummy = torch.randn(1, 1, 28, 28, device=DEVICE)

    print(f"\nModel summary: {name}")
    summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=1,
    )

    measured_macs = None
    measured_flops = None
    profiling_error = ""

    try:
        macs, profiled_params = profile(
            model, inputs=(dummy,), verbose=False
        )
        measured_macs = float(macs)
        
        measured_flops = 2.0 * measured_macs

        print(f"THOP MACs: {measured_macs:,.0f}")
        print(f"Approximate FLOPs (2 x MACs): {measured_flops:,.0f}")
        print(f"THOP parameters: {profiled_params:,.0f}")

    except Exception as error:
        profiling_error = str(error)
        print(f"THOP profiling failed: {profiling_error}")

    return {
        "model": name,
        "parameters": count_parameters(model),
        "thop_macs": measured_macs,
        "thop_approx_flops": measured_flops,
        "profiling_error": profiling_error,
    }


def hand_calculated_flops():
    
    mlp = (
        2 * 784 * 256
        + 2 * 256 * 128
        + 2 * 128 * 10
    )

    cnn = (
        2 * 1 * 16 * 3 * 3 * 28 * 28
        + 2 * 16 * 32 * 3 * 3 * 14 * 14
        + 2 * (32 * 7 * 7) * 10
    )

    attention = 2 * 16 * 49 * 64

    
    seq_len = 16
    d_model = 64
    ff_dim = 128
    layers = 2

    for _ in range(layers):
        
        attention += 3 * 2 * seq_len * d_model * d_model

        
        attention += 2 * seq_len * seq_len * d_model
        attention += 2 * seq_len * seq_len * d_model
        attention += 2 * seq_len * d_model * d_model
        attention += 2 * seq_len * d_model * ff_dim
        attention += 2 * seq_len * ff_dim * d_model

    attention += 2 * 64 * 10

    return {
        "MLP": mlp,
        "CNN": cnn,
        "Attention": attention,
    }

def measure_latency(model, batch_size, warmup=10, runs=None):
    if runs is None:
        runs = 100 if batch_size == 1 else 20

    model.eval()
    images = torch.randn(batch_size, 1, 28, 28, device=DEVICE)

    with torch.inference_mode():
        for _ in range(warmup):
            model(images)

        measurements_ms = []

        for _ in range(runs):
            start = time.perf_counter()
            model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000
            measurements_ms.append(elapsed_ms)

    return statistics.median(measurements_ms)

def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_plots(results):
    names = [row["model"] for row in results]
    accuracies = [row["test_accuracy"] for row in results]
    parameters = [row["parameters"] for row in results]
    latencies = [row["latency_batch1_ms"] for row in results]

    fig, ax = plt.subplots()
    ax.scatter(parameters, accuracies)

    for name, x, y in zip(names, parameters, accuracies):
        ax.annotate(name, (x, y))

    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fashion-MNIST: Accuracy vs Parameters")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_parameters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(latencies, accuracies)

    for name, x, y in zip(names, latencies, accuracies):
        ax.annotate(name, (x, y))

    ax.set_xlabel("Median batch-1 inference latency (ms)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fashion-MNIST: Accuracy vs Latency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_latency.png", dpi=150)
    plt.close(fig)

def main():
    train_loader, validation_loader, test_loader = load_data()

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    training_results = []
    histories = []
    profiling_results = []
    latency_results = []
    hand_flops = hand_calculated_flops()

    for name, model in models.items():
        trained_model, result, history = train_model(
            name,
            model,
            train_loader,
            validation_loader,
            test_loader,
        )

        training_results.append(result)
        histories.extend(history)

        profiling = profile_model(name, trained_model)
        profiling["hand_calculated_flops"] = hand_flops[name]
        profiling_results.append(profiling)

        latency_batch1 = measure_latency(trained_model, 1)
        latency_batch128 = measure_latency(trained_model, 128)

        latency_results.append({
            "model": name,
            "latency_batch1_ms": latency_batch1,
            "latency_batch128_ms": latency_batch128,
            "batch128_per_image_ms": latency_batch128 / 128,
            "batch1_runs": 100,
            "batch128_runs": 20,
            "warmup_runs": 10,
            "device": str(DEVICE),
        })

        print(
            f"{name} | Median latency batch 1: "
            f"{latency_batch1:.4f} ms"
        )
        print(
            f"{name} | Median latency batch 128: "
            f"{latency_batch128:.4f} ms"
        )

    latency_by_model = {row["model"]: row for row in latency_results}

    combined_results = []
    for result in training_results:
        combined_results.append({
            **result,
            **latency_by_model[result["model"]],
        })

    write_csv(RESULTS / "training_history.csv", histories)
    write_csv(RESULTS / "training_results.csv", training_results)
    write_csv(RESULTS / "flops_comparison.csv", profiling_results)
    write_csv(RESULTS / "inference_latency.csv", latency_results)
    write_csv(RESULTS / "model_comparison.csv", combined_results)

    create_plots(combined_results)

    with (RESULTS / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE),
            "train_size": 55000,
            "validation_size": 5000,
            "test_size": 10000,
            "normalization_mean": 0.2860,
            "normalization_std": 0.3530,
        }, file, indent=2)

    print("\nAll experiments finished.")
    print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()
