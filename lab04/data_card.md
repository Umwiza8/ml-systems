# Adult Census Income Dataset — Data Card

## 1. Dataset Overview

- **Name:** Adult Census Income Dataset
- **Source:** UCI Adult dataset accessed through OpenML.
- **Task:** Binary classification to predict whether an individual's annual income exceeds $50,000.
- **Target column:** `class`
- **Raw dataset size:** 48,842 rows and 15 columns.
- **Feature count:** 14 input features.
- **Target classes:** `<=50K` and `>50K`.

## 2. Intended Use

The dataset is intended for educational machine learning experiments, classification model development, preprocessing practice, and data-quality analysis.

Predictions should not be used to make real-world decisions about individuals, employment, credit, or access to services.

## 3. Data Processing

The dataset is split into training and testing sets using a 70/30 stratified split with random state 42.

Missing numeric values are imputed using the training median. Missing categorical values are imputed using the training mode. Numeric values are clipped using the training 1st and 99th percentiles and standardized. Categorical features are one-hot encoded.

Two engineered features are evaluated: `capital_net` and `hours_per_age`.

Duplicate feature rows are removed from the training set. The test set is retained unchanged. Preprocessing parameters are learned from training data only.

## 4. Data Quality

The raw dataset contains missing values in `workclass`, `occupation`, and `native-country`.

The numeric range checks and selected categorical checks passed. The label-quality audit identified five groups of identical feature rows with conflicting income labels in the full raw dataset.

A reproducible random sample of 50 rows was reviewed against the documented feature definitions and specified validation ranges. No violations of the specified numeric ranges were identified in the reviewed sample. The review does not independently verify the factual correctness of individual income labels.

## 5. Limitations and Biases

The data represents a historical US Census-derived population and may not reflect current demographics, employment patterns, or income distributions.

The dataset contains demographic attributes, including age, sex, and race. These attributes may reflect historical inequalities and should be handled carefully.

The dataset is not necessarily representative of populations outside the United States or of present-day circumstances.

## 6. Privacy and Ethics

The dataset contains sensitive demographic and socioeconomic information. Although it is commonly used for public research and education, its use should respect applicable privacy, fairness, and data-protection requirements.

Model predictions may reproduce or amplify biases in the source data.

## 7. License and Access

The dataset was obtained through OpenML using the dataset name `adult` and version `2`.

License: Creative Commons Attribution 4.0 International (CC BY 4.0). Dataset citation: Becker, B. and Kohavi, R. (1996), Adult, UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20.

## 8. Reproducibility

- Random state: 42
- Training fraction: 70%
- Testing fraction: 30%
- Baseline model: Logistic Regression
- Evaluation metrics: Accuracy and F1 score for the `>50K` class
- Experiment configuration: `config.yaml`
- Model comparison results: `results/model_comparison.csv`
- Label audit report: `results/label_quality_report.txt`
