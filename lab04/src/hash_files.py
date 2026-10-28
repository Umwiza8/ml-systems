
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "adult_raw.csv",
    ROOT / "lab04" / "config.yaml",
    ROOT / "lab04" / "data_card.md",
    ROOT / "lab04" / "results" / "X_train_processed.npy",
    ROOT / "lab04" / "results" / "X_test_processed.npy",
    ROOT / "lab04" / "results" / "y_train_clean.csv",
    ROOT / "lab04" / "results" / "model_comparison.csv",
    ROOT / "lab04" / "results" / "label_quality_report.txt",
]

OUTPUT = ROOT / "lab04" / "results" / "sha256_hashes.txt"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    lines = []

    for path in FILES:
        if path.exists():
            digest = sha256_file(path)
            lines.append(f"{digest}  {path.relative_to(ROOT)}")
            print(f"{digest}  {path.relative_to(ROOT)}")
        else:
            print(f"Missing file: {path.relative_to(ROOT)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nHashes saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
