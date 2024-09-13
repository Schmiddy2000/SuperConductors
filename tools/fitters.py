# Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from typing import Optional, List, Tuple


def linear_fit(x_data: np.array,
               y_data: np.array,
               initial_guesses: Optional[List[float]] = None,
               print_params: bool = False
               ) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    def linear_function(x, a, b):
        return a * x + b

    if initial_guesses is not None:
        popt, pcov = curve_fit(linear_function, x_data, y_data, initial_guesses)
    else:
        popt, pcov = curve_fit(linear_function, x_data, y_data)

    perr = np.sqrt(np.diag(pcov))

    if print_params:
        print(f'a = {round(popt[0], 4)} ± {round(perr[0], 4)}')
        print(f'b = {round(popt[1], 4)} ± {round(perr[1], 4)}')

    return popt, perr


def exponential_fit(x_data: np.array,
                    y_data: np.array,
                    initial_guesses: Optional[List[float]] = None,
                    print_params: bool = False
                    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    def exponential_function(x, a, b, c):
        return a * np.exp(-b * x) + c

    if initial_guesses is not None:
        popt, pcov = curve_fit(exponential_function, x_data, y_data, initial_guesses)
    else:
        popt, pcov = curve_fit(exponential_function, x_data, y_data)

    perr = np.sqrt(np.diag(pcov))

    if print_params:
        print(f'a = {round(popt[0], 4)} ± {round(perr[0], 4)}')
        print(f'b = {round(popt[1], 4)} ± {round(perr[1], 4)}')
        print(f'c = {round(popt[2], 4)} ± {round(perr[2], 4)}')

    return popt, perr
