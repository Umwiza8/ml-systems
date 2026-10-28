
# Lab 2: Dataset preparation

# 1. Import the tools we need
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 2. Load the handwritten digits dataset
digits = load_digits()

# X contains the images as numerical features
# y contains the correct digit labels
X = digits.data
y = digits.target

# 3. Scale pixel values from 0-16 to 0-1
X = X / 16.0

# 4. Convert each label into a vector of 10 values
y_one_hot = np.eye(10)[y]

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_one_hot,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 6. Print information to check our work
print("Original dataset shape:", digits.data.shape)
print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

print("Minimum scaled pixel:", X.min())
print("Maximum scaled pixel:", X.max())

print("First training label (one-hot):", y_train[0])
print("Sum of its label values:", y_train[0].sum())
