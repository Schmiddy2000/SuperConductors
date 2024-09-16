import numpy as np
from matplotlib import pyplot as plt


A = 3.9083e-3
B = -5.775e-7
C = -4.183e-12

# U = RI


def wire_voltage_to_temperature(voltages: np.array) -> np.array:
    voltage_at_R_min = 191.7e-3
    voltage_at_0 = 0.45e-3

    return -(voltages - voltage_at_0) * 196 / (voltage_at_R_min - voltage_at_0)


def R_0_coefficient(temperature: int):
    return 1 + A * temperature + B * temperature ** 2 + C * (temperature - 100) * temperature ** 3


lin_temp = np.linspace(-200, 0, 250)

plt.plot(lin_temp, 100 * R_0_coefficient(lin_temp))
plt.plot()
plt.show()
