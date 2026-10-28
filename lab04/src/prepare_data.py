
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_STATE = 42
TEST_SIZE = 0.30


def main():
    print("Lab 4: Train/Test Split")
    print("=" * 45)

    raw_path = DATA_DIR / "adult_raw.csv"
    df = pd.read_csv(raw_path)

    # Separate features and target
    X = df.drop(columns=["class"])
    y = df["class"].astype(str).str.strip()

    # Stratified split: preserve income-class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save the split for reproducible later steps
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(RESULTS_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(RESULTS_DIR / "X_test_raw.csv", index=False)
    y_train.to_csv(RESULTS_DIR / "y_train.csv", index=False)
    y_test.to_csv(RESULTS_DIR / "y_test.csv", index=False)

    print(f"Total rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    print("\nTraining target proportions:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nTesting target proportions:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\nMissing values in training features:")
    print(X_train.isna().sum()[X_train.isna().sum() > 0])

    print("\nMissing values in testing features:")
    print(X_test.isna().sum()[X_test.isna().sum() > 0])

    print("\nExact duplicate feature rows:")
    print(f"Training: {X_train.duplicated().sum():,}")
    print(f"Testing: {X_test.duplicated().sum():,}")

    print("\nSaved the train/test split to lab04/results/")


if __name__ == "__main__":
    main()
