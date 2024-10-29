"""
Curve Fitting Analysis Tool
==========================

Overview:
---------
This Python script performs curve fitting analysis on time series data using a combination 
of linear and sinusoidal functions. It's designed to model data that exhibits both a 
linear trend and periodic oscillations.

Dependencies:
------------
- pandas
- numpy
- matplotlib
- scipy
- sklearn

Data Requirements:
----------------
The script expects a CSV file named 'train_data.csv' with columns 'x' and 'y'
representing points that follow a pattern combining linear trend and periodic behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Load and prepare data
data = pd.read_csv(r"train_data.csv")
x = data['x'].values
y = data['y'].values

# Step 1: Linear Fitting
# ---------------------
# Define linear function y = kx + D
def linear_func(x, k, D):
    """
    Linear function for fitting
    Parameters:
        x: input values
        k: slope
        D: y-intercept
    Returns:
        Linear function values
    """
    return k * x + D

# Fit linear part
k_opt, _ = curve_fit(linear_func, x, y)
y_linear_fit = linear_func(x, *k_opt)

print("Linear Fitting Results:")
print(f"Fitted linear slope: k = {k_opt[0]}, D = {k_opt[1]}")

# Plot linear fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, y_linear_fit, color='red', label='Linear fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Fit Component')
plt.show()

# Step 2: Sinusoidal Fitting
# -------------------------
# Define sinusoidal function y = A*sin(Bx + C)
def sin_func(x, A, B, C):
    """
    Sinusoidal function for fitting
    Parameters:
        x: input values
        A: amplitude
        B: frequency
        C: phase shift
    Returns:
        Sinusoidal function values
    """
    return A * np.sin(B * x + C)

# Fit sinusoidal part
initial_guess_sin = [10, 0.1, 0]  # Initial guess for A, B, C
params_sin, _ = curve_fit(sin_func, x, y, p0=initial_guess_sin)
y_sin_fit = sin_func(x, *params_sin)

print("\nSinusoidal Fitting Results:")
print(f"Fitted parameters: A = {params_sin[0]}, B = {params_sin[1]}, C = {params_sin[2]}")

# Plot sinusoidal fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, y_sin_fit, color='red', label='Sinusoidal fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sinusoidal Fit Component')
plt.show()

# Step 3: Combined Fitting Results
# ------------------------------
# Combine both components
y_final_fit = y_linear_fit + y_sin_fit

# Plot final fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, y_final_fit, color='red', label='Combined fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Combined Fit Result')
plt.show()

# Calculate and print loss
loss = mean_squared_error(y, y_final_fit)
print(f"\nMean Squared Error: {loss}")

# Step 4: Predictions
# -----------------
# Generate random test points and predict
x2 = np.sort(np.random.uniform(70, 100, 50))
y_pred_l = linear_func(x2, *k_opt)
y_pred_s = sin_func(x2, *params_sin)
y_pred = y_pred_l + y_pred_s

print("\nPrediction points generated:", x2)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Training Data')
plt.plot(x, y_final_fit, color='red', label='Fitted curve')
plt.plot(x2, y_pred, color='blue', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Model Predictions')
plt.show()

"""
Usage Notes:
-----------
1. Ensure all dependencies are installed
2. Place your train_data.csv in the same directory
3. Run the entire script
4. Check the console for fitting parameters and MSE
5. Examine the four plots for visual analysis

Customization:
-------------
- Adjust initial_guess_sin parameters if fitting fails
- Modify prediction range (currently 70-100) as needed
- Change number of prediction points (currently 50)
- Customize plot styles and colors as desired
"""
