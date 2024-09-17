# Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tools.transformers import slice_measurement_data


base_path = "/Users/lucas1/Desktop/Uni/Physiklabor B/Physiklabor für Fortgeschrittene/Versuch 4 - Supraleiter/"


def get_path(index: int):
    return base_path + f"Messung_{index}.csv"


df = pd.read_csv(get_path(20))

print(df.columns)


def conductor_voltage_to_temperature(voltages: np.array) -> np.array:
    voltage_at_R_min = 191.1e-3
    voltage_at_0 = 0.2e-3

    return -(voltages - voltage_at_0) * 196 / (voltage_at_R_min - voltage_at_0)


columns = ['Zeit t / s', 'Spannung U_A1 / V', 'Spannung U_B1 / V']
df = df[columns]

t = df[columns[0]].to_numpy()
x = df[columns[2]].to_numpy()
y = df[columns[1]].to_numpy()

t, x, y = slice_measurement_data(t, [x, y], 1250, 1500)

y = conductor_voltage_to_temperature(y)

# remove_indices = [int((50 * val) / 100) for val in range(1926, 1940) if val % 2 == 0]
# remove_indices_2 = [i for i in range(0, 12 * 50)]
#
# x = np.delete(x, remove_indices)
# y = np.delete(y, remove_indices)


plt.figure(figsize=(12, 5))
plt.title('Zoom-in on negative current before leaving superconductive temperature range', fontsize=16)
plt.xlabel('Time in [s]', fontsize=13)
plt.ylabel('Voltage in [V]', fontsize=13)
plt.scatter(t, x, color='green', s=5, label='Data Points')
plt.hlines(-0.0012, 1435, 1485, color='k', ls='--', lw=1.25, label='Line of linear plateau')
plt.xlim(1435, 1485)
plt.ylim(-0.05, 0.25)
plt.legend()
plt.savefig('zoom_in_negative_current_dip.png', dpi=200)
plt.show()
