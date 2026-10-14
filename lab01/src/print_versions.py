
import sys
import numpy
import pandas
import sklearn
import scipy
import matplotlib
import seaborn
import joblib
import psutil

print("Python:", sys.version.split()[0])

packages = [
    numpy,
    pandas,
    sklearn,
    scipy,
    matplotlib,
    seaborn,
    joblib,
    psutil,
]

for package in packages:
    print(f"{package.__name__}: {package.__version__}")
