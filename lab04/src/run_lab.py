
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab04" / "results"
DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "class",
]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


def load_data():
    print("Downloading Adult dataset from OpenML...")

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    df = adult.frame.copy()

    # Preserve the original downloaded data
    raw_path = DATA_DIR / "adult_raw.csv"
    df.to_csv(raw_path, index=False)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Raw dataset saved to: {raw_path}")

    return df


def validate_data(df):
    report = []

    def record(check, passed, details):
        status = "PASS" if passed else "FAIL"
        report.append(
            f"[{status}] {check}: {details}"
        )

    # 1. Dataset dimensions
    record(
        "Dataset dimensions",
        df.shape == (48842, 15),
        f"Found {df.shape[0]} rows and {df.shape[1]} columns",
    )

    # 2. Column names
    record(
        "Schema columns",
        list(df.columns) == EXPECTED_COLUMNS,
        f"Columns match expected schema: "
        f"{list(df.columns) == EXPECTED_COLUMNS}",
    )

    # 3. Data types
    for column in NUMERIC_COLUMNS:
        valid = pd.api.types.is_numeric_dtype(df[column])
        record(
            f"Numeric type: {column}",
            valid,
            f"Observed dtype: {df[column].dtype}",
        )

    for column in CATEGORICAL_COLUMNS:
        valid = (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[column])
        )
        record(
            f"Categorical type: {column}",
            valid,
            f"Observed dtype: {df[column].dtype}",
        )

    # 4. Allowed numeric ranges
    for column, (minimum, maximum) in RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = (
            values.notna()
            & ~values.between(minimum, maximum)
        )
        count = int(invalid.sum())

        record(
            f"Range: {column}",
            count == 0,
            f"{count} violations; allowed range "
            f"is {minimum} to {maximum}",
        )

    # 5. Missing values and Adult's missing-value marker
    missing_report = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        question_mark_count = 0
        if (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[column])
        ):
            question_mark_count = int(
                df[column].astype("string").eq("?").sum()
            )

        if missing_count > 0 or question_mark_count > 0:
            missing_report.append(
                f"{column}: {missing_count} null values, "
                f"{question_mark_count} '?' markers"
            )

    record(
        "Missing-value inspection",
        True,
        "; ".join(missing_report)
        if missing_report
        else "No missing values detected",
    )

    # 6. Unexpected target labels
    target_values = set(
        df["class"].dropna().astype(str).unique()
    )
    allowed_targets = {"<=50K", ">50K"}

    unexpected_targets = target_values - allowed_targets

    record(
        "Target categories",
        len(unexpected_targets) == 0,
        f"Observed: {sorted(target_values)}; "
        f"unexpected: {sorted(unexpected_targets)}",
    )

    # 7. Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    record(
        "Duplicate inspection",
        True,
        f"{duplicate_count} exact duplicate rows found; "
        "no rows removed during validation",
    )

    # Save the report
    report_path = RESULTS_DIR / "validation_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("LAB 4 - DATA VALIDATION REPORT\n")
        file.write("=" * 50 + "\n\n")
        file.write("\n".join(report))
        file.write("\n")

    print("\nVALIDATION RESULTS")
    print("=" * 50)

    for line in report:
        print(line)

    print(f"\nValidation report saved to: {report_path}")


def main():
    print("Lab 4: Data Pipeline and Quality")
    print("=" * 50)

    df = load_data()
    validate_data(df)


if __name__ == "__main__":
    main()
