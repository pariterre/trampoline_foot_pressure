from enum import Enum

import numpy as np


class PhaseType(Enum):
    AERIAL = "aerial"
    MAT = "mat"


class JumpDirection(Enum):
    FRONTWARD = "frontward"
    BACKWARD = "backward"


class JumpType(Enum):
    BACK_SOMERSAULT_STRAIGHT = JumpDirection.BACKWARD
    BACK_SOMERSAULT_TUCK = JumpDirection.BACKWARD
    BARANI_STRAIGHT = JumpDirection.FRONTWARD
    BARANI_TUCK = JumpDirection.FRONTWARD


def concatenate_data(all_data: list):
    concatenated_data = all_data[0]
    for data in all_data[1:]:
        concatenated_data = concatenated_data.concatenate(data)
    return concatenated_data


def derivative(t: np.ndarray, data: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Compute the non-sliding derivative of current data

    Parameters
    ----------
    t
        The time vector
    data
        The data to compute the derivative from
    window
        The sliding window to perform on

    Returns
    -------
    A Data structure with the value differentiated
    """

    two_windows = window * 2
    padding = np.nan * np.zeros((window, data.shape[1]))

    return np.concatenate(
        (
            padding,
            (data[:-two_windows, :] - data[two_windows:]) / (t[:-two_windows] - t[two_windows:])[:, np.newaxis],
            padding,
        )
    )


def integral(t: np.ndarray, data: np.ndarray) -> float:
    """
    Compute the integral of the data using trapezoid

    Parameters
    ----------
    t
        The time vector
    data
        The data to compute the integral from

    Returns
    -------
    The integral of the data
    """
    return np.nansum((t[1:] - t[:-1]) * ((data[1:, :] + data[:-1, :]) / 2).T)
