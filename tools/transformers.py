# Imports
from gc import freeze

import numpy as np

from typing import List, Optional, Tuple, Any


def slice_measurement_data(time_array: np.array,
                           data: List[np.array],
                           start_time: Optional[float],
                           stop_time: Optional[float]
                           ) -> Tuple[np.array, ...]:
    """

    """
    start_index = 0
    stop_index = len(time_array) - 1

    while time_array[start_index] < start_time:
        start_index += 1

    while time_array[stop_index] > stop_time:
        stop_index -= 1

    return tuple([time_array[start_index:stop_index]] + [data_set[start_index:stop_index] for data_set in data])



def voltage_to_temperature(voltages: np.array, freeze_voltage: float, nitrogen_voltage: float) -> np.array:
    """
    Converts an array of voltage values to an array of temperature
    values, given the voltages at 0 and -196 degrees.
    """
    pass


def advanced_voltage_to_temperature(voltages: np.array,
                                    temperature_points: List[float],
                                    voltage_points: List[float],
                                    non_linear_transformation: bool
                                    ) -> np.array:
    """
    Converts an array of voltage values to an array of temperature
    values, given a set of reference temperatures and voltages.
    """
    pass


def voltage_to_resistance(voltages: np.array, current: float, freeze_voltage: float, nitrogen_voltage: float):
    freeze_resistance = freeze_voltage / current
    nitrogen_resistance = nitrogen_voltage / current

    freeze_temperature = 0
    nitrogen_temperature = -196

    # Parameters for the function R(T)
    A = 3.9083e-3
    B = -5.775e-7
    C = -4.183e-12

    def R_0_coefficient(temperature: int):
        return 1 + A * temperature + B * temperature ** 2 + C * (temperature - 100) * temperature ** 3

    R_0_freeze = freeze_resistance / R_0_coefficient(freeze_temperature)
    R_0_nitrogen = nitrogen_resistance / R_0_coefficient(nitrogen_temperature)

    print(R_0_coefficient(freeze_temperature))
    print(R_0_coefficient(nitrogen_temperature))

    print(R_0_freeze)
    print(R_0_nitrogen)

    return None


voltage_to_resistance(np.array([0]), 51.2e-6, -0.001, 0.1925)
