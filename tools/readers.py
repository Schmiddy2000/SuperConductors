# Imports
import numpy as np
import pandas as pd

from typing import Tuple

# Base path to the csv files
data_base_path = "data/csv_files/"


def get_path(index: int) -> str:
    """
    Get the path to a specific measurement.
    """
    return data_base_path + f"Messung_{index}.csv"


def get_voltage_A_data(index: int) -> Tuple[np.array, np.array]:
    """
    Returns numpy arrays for time and measured voltage for a given
    measurement series in this order.
    """

    file_path = get_path(index)
    dataframe = pd.read_csv(file_path)
    columns = ['Zeit t / s', 'Spannung U_A1 / V']

    time = dataframe[columns[0]].to_numpy()
    voltage_A = dataframe[columns[1]].to_numpy()

    return time, voltage_A


def get_voltage_B_data(index: int) -> Tuple[np.array, np.array]:
    """
    Returns numpy arrays for time and measured voltage for a given
    measurement series in this order.
    """

    file_path = get_path(index)
    dataframe = pd.read_csv(file_path)
    columns = ['Zeit t / s', 'Spannung U_B1 / V']

    time = dataframe[columns[0]].to_numpy()
    voltage_B = dataframe[columns[1]].to_numpy()

    return time, voltage_B


def get_double_voltage_data(index: int) -> Tuple[np.array, np.array, np.array]:
    """
    Returns numpy arrays for time and both measured voltages (A, B) for
    a given measurement series in this order.
    """

    file_path = get_path(index)
    dataframe = pd.read_csv(file_path)
    columns = ['Zeit t / s', 'Spannung U_A1 / V', 'Spannung U_B1 / V']

    time = dataframe[columns[0]].to_numpy()
    voltage_A = dataframe[columns[1]].to_numpy()
    voltage_B = dataframe[columns[2]].to_numpy()

    return time, voltage_A, voltage_B
