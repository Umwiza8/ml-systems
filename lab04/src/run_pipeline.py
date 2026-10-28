
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "run_lab.py",
    "prepare_data.py",
    "preprocess_data.py",
    "compare_models.py",
    "label_quality_audit.py",
    "hash_files.py",
]


def main():
    for script in SCRIPTS:
        path = ROOT / "lab04" / "src" / script

        if not path.exists():
            raise FileNotFoundError(f"Required script not found: {path}")

        print(f"\n{'=' * 60}")
        print(f"Running {script}")
        print("=" * 60)

        subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            check=True,
        )

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
