
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab04" / "results"

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


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


def build_preprocessor(use_engineered_features):
    numeric_features = NUMERIC_COLUMNS.copy()

    if use_engineered_features:
        numeric_features += ["capital_net", "hours_per_age"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", TrainingPercentileClipper()),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def main():
    X_train = pd.read_csv(RESULTS / "X_train_clean.csv")
    X_test = pd.read_csv(RESULTS / "X_test_raw.csv")
    y_train = pd.read_csv(RESULTS / "y_train_clean.csv").iloc[:, 0]
    y_test = pd.read_csv(RESULTS / "y_test.csv").iloc[:, 0]

    if not (len(X_train) == len(y_train) and len(X_test) == len(y_test)):
        raise ValueError("Feature and label counts do not match.")

    results = []

    for use_features in [False, True]:
        train_data = X_train.copy()
        test_data = X_test.copy()

        if use_features:
            train_data = add_features(train_data)
            test_data = add_features(test_data)

        model = Pipeline([
            ("preprocessor", build_preprocessor(use_features)),
            ("classifier", LogisticRegression(
                max_iter=1000, random_state=42
            )),
        ])

        model.fit(train_data, y_train)
        predictions = model.predict(test_data)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, pos_label=">50K")

        results.append({
            "model": (
                "With engineered features"
                if use_features
                else "Without engineered features"
            ),
            "accuracy": accuracy,
            "f1_score": f1,
        })

        print(results[-1])

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / "model_comparison.csv", index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
