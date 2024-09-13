# Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


base_path = "/Users/lucas1/Desktop/Uni/Physiklabor B/Physiklabor für Fortgeschrittene/Versuch 4 - Supraleiter/"


def get_path(index: int):
    return base_path + f"Messung_{index}.csv"


df = pd.read_csv(get_path(10))


columns = ['Zeit t / s', 'Spannung U_A1 / V']
df = df[columns]

x = df[columns[0]].to_numpy()
y = df[columns[1]].to_numpy()

# 19.26 -> 19.4
remove_indices = [int((50 * val) / 100) for val in range(1926, 1940) if val % 2 == 0]
remove_indices += [i for i in range(0, 12 * 50)]

print(remove_indices)


# Define the piecewise function with a Heaviside step function
# def model_func(t, A, D, E, t0):
#     linear_part = A
#     exp_part = A * np.exp(-D * (t - t0)) + E
#     heaviside = np.heaviside(t - t0, 1)  # 1 for t >= t0, 0 for t < t0
#     return linear_part * (1 - heaviside) + exp_part * heaviside


# The following part still has to be fixed / automated

def model_func(t, A, D, E):
    linear_part = A
    exp_part = A * np.exp(-D * (t - 12)) + E
    heaviside = np.heaviside(t - 12, 1)  # 1 for t >= t0, 0 for t < t0
    return linear_part * (1 - heaviside) + exp_part * heaviside


# Example data (replace this with your actual data)
t_data = np.delete(x, remove_indices)
y_data = np.delete(y, remove_indices)

# Initial guess for parameters [A, B, C, D, E, t0]
initial_guess = [1, 0.01, 0]

# Perform curve fitting
popt, pcov = curve_fit(model_func, t_data, y_data, p0=initial_guess)

# Extract fitted parameters
A_fit, D_fit, E_fit = popt

# Plot the data and the fit
plt.plot(t_data, y_data, label="Data")
plt.plot(t_data, model_func(t_data, *popt), label="Fitted Curve")
# plt.axvline(x=t0_fit, color='r', linestyle='--', label=f'Transition Time t0 = {t0_fit:.2f} s')
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
plt.show()

# plt.figure(figsize=(12, 5))
#
# plt.scatter(x, y)
# plt.show()
