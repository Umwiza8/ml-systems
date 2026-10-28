
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab04" / "results"
RAW_DATA = ROOT / "data" / "adult_raw.csv"

NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
    "education-num": (1, 16),
}

CATEGORICAL_VALUES = {
    "class": {"<=50K", ">50K"},
    "sex": {"Female", "Male"},
    "race": {
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
    },
    "relationship": {
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    },
}


def main():
    data = pd.read_csv(RAW_DATA)

    sample = data.sample(n=50, random_state=42)
    sample.to_csv(RESULTS / "label_audit_sample_50.csv", index=False)

    report = []
    report.append("LAB 4 LABEL QUALITY AUDIT")
    report.append("=" * 50)
    report.append(f"Dataset rows: {len(data)}")
    report.append(f"Dataset columns: {len(data.columns)}")
    report.append("Random sample size: 50")
    report.append("Sampling random state: 42")
    report.append("")

    report.append("1. MISSING VALUES")
    report.append("-" * 50)

    missing = data.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        report.append("No missing values found.")
    else:
        for column, count in missing.items():
            report.append(f"{column}: {count}")

    report.append("")
    report.append("2. NUMERIC RANGE VALIDATION")
    report.append("-" * 50)

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = values.notna() & (
            (values < minimum) | (values > maximum)
        )
        report.append(
            f"{column}: {int(invalid.sum())} values outside "
            f"{minimum}-{maximum}"
        )

    fnlwgt = pd.to_numeric(data["fnlwgt"], errors="coerce")
    invalid_fnlwgt = fnlwgt.notna() & (fnlwgt <= 0)
    report.append(
        f"fnlwgt: {int(invalid_fnlwgt.sum())} non-positive values"
    )

    report.append("")
    report.append("3. CATEGORICAL VALUE VALIDATION")
    report.append("-" * 50)

    for column, allowed_values in CATEGORICAL_VALUES.items():
        observed = set(data[column].dropna().unique())
        unexpected = sorted(observed - allowed_values)

        report.append(
            f"{column}: {len(unexpected)} unexpected categories"
        )

        if unexpected:
            report.append(f"Unexpected values: {unexpected}")

    report.append("")
    report.append("4. CONFLICTING INCOME LABELS")
    report.append("-" * 50)

    feature_columns = [
        column for column in data.columns if column != "class"
    ]

    grouped = data.groupby(feature_columns, dropna=False)["class"].agg(
        ["nunique", "size"]
    )

    conflicts = grouped[grouped["nunique"] > 1]

    report.append(
        f"Duplicate feature groups with conflicting labels: {len(conflicts)}"
    )
    report.append(
        f"Rows belonging to conflicting groups: "
        f"{int(conflicts['size'].sum()) if not conflicts.empty else 0}"
    )

    report.append("")
    report.append("5. RANDOM SAMPLE FOR MANUAL REVIEW")
    report.append("-" * 50)
    report.append(
        "Inspect label_audit_sample_50.csv against the Adult dataset "
        "data dictionary."
    )
    report.append(
        "Automated range checks cannot establish whether an income "
        "label is factually correct."
    )

    report.append("")
    report.append("6. OUTPUT FILES")
    report.append("-" * 50)
    report.append("label_audit_sample_50.csv")
    report.append("label_quality_report.txt")

    report_path = RESULTS / "label_quality_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved to: {report_path}")
    print(f"Sample saved to: {RESULTS / 'label_audit_sample_50.csv'}")


if __name__ == "__main__":
    main()
