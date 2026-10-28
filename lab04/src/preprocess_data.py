
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "lab04" / "results"

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

TARGET_COLUMN = "class"
ENGINEERED_COLUMNS = ["capital_net", "hours_per_age"]


class TrainingPercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(
            X, self.lower_percentile, axis=0
        )
        self.upper_bounds_ = np.nanquantile(
            X, self.upper_percentile, axis=0
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def add_features(data):
    data = data.copy()
    data["capital_net"] = data["capital-gain"] - data["capital-loss"]
    data["hours_per_age"] = data["hours-per-week"] / (data["age"] - 16)
    return data


def main():
    X_train = pd.read_csv(RESULTS_DIR / "X_train_raw.csv")
    X_test = pd.read_csv(RESULTS_DIR / "X_test_raw.csv")
    y_train = pd.read_csv(RESULTS_DIR / "y_train.csv").iloc[:, 0]

    training_data = X_train.copy()
    training_data[TARGET_COLUMN] = y_train.to_numpy()

    duplicate_mask = training_data.duplicated(
        subset=X_train.columns, keep=False
    )
    duplicate_groups = training_data.loc[duplicate_mask]

    conflicting_groups = 0
    if not duplicate_groups.empty:
        conflicting_groups = (
            duplicate_groups.groupby(
                list(X_train.columns), dropna=False
            )[TARGET_COLUMN].nunique().gt(1).sum()
        )

    print("Original training rows:", len(X_train))
    print("Original testing rows:", len(X_test))
    print("Rows in duplicate-feature groups:", int(duplicate_mask.sum()))
    print("Conflicting duplicate-feature groups:", int(conflicting_groups))

    training_data = training_data.drop_duplicates(
        subset=X_train.columns, keep="first"
    )

    y_train_clean = training_data[TARGET_COLUMN].reset_index(drop=True)
    X_train = training_data.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train.to_csv(RESULTS_DIR / "X_train_clean.csv", index=False)
    y_train_clean.to_frame(name=TARGET_COLUMN).to_csv(
        RESULTS_DIR / "y_train_clean.csv", index=False
    )

    X_train = add_features(X_train)
    X_test = add_features(X_test)

    numeric_features = NUMERIC_COLUMNS + ENGINEERED_COLUMNS

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", TrainingPercentileClipper()),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    np.save(RESULTS_DIR / "X_train_processed.npy", X_train_processed)
    np.save(RESULTS_DIR / "X_test_processed.npy", X_test_processed)
    joblib.dump(preprocessor, RESULTS_DIR / "preprocessor.joblib")

    print("Training rows after duplicate removal:", len(X_train))
    print("Testing rows retained:", len(X_test))
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
