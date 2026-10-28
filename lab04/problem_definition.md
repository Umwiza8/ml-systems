# Lab 4: Problem Definition

## Adult Census Income Classification

The objective of this project is to develop a reproducible machine learning pipeline that predicts whether an individual's annual income exceeds $50,000 using the UCI Adult dataset accessed through OpenML. The dataset contains 48,842 records and 15 columns, comprising 14 input features and one target column, `class`.

The input features include numerical variables, such as age, education number, hours worked per week, capital gain, and capital loss, as well as categorical variables, such as workclass, education, occupation, marital status, race, and sex. The target is a binary income classification: `>50K` or `<=50K`.

Logistic Regression will serve as the baseline model. Its performance will be evaluated using accuracy and F1-score, with `>50K` treated as the positive class. Accuracy measures the proportion of correct predictions, while the F1-score balances precision and recall. Both metrics are important because the two income classes are not equally represented in the dataset.

The pipeline must be reproducible and prevent data leakage. The dataset will be divided into stratified training and testing sets using a 70/30 split and random state 42 before cleaning and transformation. Imputation values, numerical clipping thresholds, scaling parameters, and categorical encoding will be learned from the training data only. Exact duplicate rows will be removed from the training data, while the test set will remain unchanged.

Two derived features, `capital_net` and `hours_per_age`, will be evaluated to determine whether they improve baseline performance. The project will also include schema validation, range and missing-value checks, duplicate and label-quality audits, SHA-256 hashing, a YAML configuration, and a data card. The final pipeline must be executable with one command and produce documented, reproducible results.