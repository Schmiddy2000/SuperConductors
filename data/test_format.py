# Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


base_path = "/Users/lucas1/Desktop/Uni/Physiklabor B/Physiklabor für Fortgeschrittene/Versuch 4 - Supraleiter/"


def get_path(index: int):
    return base_path + f"Messung_{index}.csv"


df = pd.read_csv(get_path(10))

print(df.columns)


def wire_voltage_to_temperature(voltages: np.array) -> np.array:
    voltage_at_R_min = 191.7e-3
    voltage_at_0 = 0.45e-3

    return -(voltages - voltage_at_0) * 196 / (voltage_at_R_min - voltage_at_0)


columns = ['Zeit t / s', 'Spannung U_A1 / V']
df = df[columns]

x = df[columns[0]].to_numpy()
y = df[columns[1]].to_numpy()

y = wire_voltage_to_temperature(y)

x_lin = np.linspace(-0.1 * max(x), 1.1 * max(x))

# Perform linear regression using scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")


# Predict y values using the regression model
def lin_func(x, a, b):
    return a * x + b


# Plot the original data and the regression line
plt.figure(figsize=(12, 5))
plt.scatter(x, y, color='green', s=5, label='Data Points')
plt.plot(x_lin, lin_func(x_lin, slope, intercept), color='black', label='Regression Line')
plt.plot(x_lin, lin_func(x_lin, slope, intercept + 1.25), color='red', ls='--', label='Offset of +1.25mV')
plt.plot(x_lin, lin_func(x_lin, slope, intercept - 2.25), color='orange', ls='--',  label='Offset of -2.25mV')
plt.xlabel('Time in [s]', fontsize=13)
plt.ylabel('Voltage in [mV]', fontsize=13)
plt.title('Linear regression over voltage fluctuations around room temperature', fontsize=16)
plt.legend()
plt.xlim(-25, 325)
plt.tight_layout()
plt.savefig('current_fluctuation_visualization.png', dpi=200)
plt.show()


# 19.26 -> 19.4
# remove_indices = [int((50 * val) / 100) for val in range(1926, 1940) if val % 2 == 0]
# remove_indices += [i for i in range(0, 12 * 50)]
#
# print(remove_indices)


# Define the piecewise function with a Heaviside step function
# def model_func(t, A, D, E, t0):
#     linear_part = A
#     exp_part = A * np.exp(-D * (t - t0)) + E
#     heaviside = np.heaviside(t - t0, 1)  # 1 for t >= t0, 0 for t < t0
#     return linear_part * (1 - heaviside) + exp_part * heaviside


# The following part still has to be fixed / automated

# def model_func(t, A, D, E):
#     linear_part = A
#     exp_part = A * np.exp(-D * (t - 12)) + E
#     heaviside = np.heaviside(t - 12, 1)  # 1 for t >= t0, 0 for t < t0
#     return linear_part * (1 - heaviside) + exp_part * heaviside
#
#
# # Example data (replace this with your actual data)
# t_data = np.delete(x, remove_indices)
# y_data = np.delete(y, remove_indices)
#
# # Initial guess for parameters [A, B, C, D, E, t0]
# initial_guess = [1, 0.01, 0]
#
# # Perform curve fitting
# popt, pcov = curve_fit(model_func, t_data, y_data, p0=initial_guess)
#
# # Extract fitted parameters
# A_fit, D_fit, E_fit = popt

# Plot the data and the fit
plt.scatter(x, y, label="Data", s=5)
# plt.plot(t_data, model_func(t_data, *popt), label="Fitted Curve")
# plt.axvline(x=t0_fit, color='r', linestyle='--', label=f'Transition Time t0 = {t0_fit:.2f} s')
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
# plt.show()

# plt.figure(figsize=(12, 5))
#
# plt.scatter(x, y)
# plt.show()
