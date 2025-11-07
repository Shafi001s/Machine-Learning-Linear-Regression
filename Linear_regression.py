import math, copy
from stringprep import b3_exceptions # For mathematical operations and deep copying
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting

x_train = np.array([1.0, 2.0]) # Training input data
y_train = np.array([300.0, 500.0]) # Training output data

# Function to compute the cost for linear regression
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = cost / (2*m)

    return total_cost