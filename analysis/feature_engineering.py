"""Feature transformations shared by model training and inference."""

from __future__ import annotations

import numpy as np


ENERGY_FEATURE_NAMES = (
    "Temperatura",
    "Hora_sin",
    "Hora_cos",
    "Dia_sin",
    "Dia_cos",
)


def energy_feature_matrix(values: np.ndarray) -> np.ndarray:
    """Encode hour and weekday as cyclic sine/cosine pairs.

    Input columns must be ordered as temperature, hour (1–24), weekday (1–7).
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(
            "Energy inputs must have three columns: Temperatura, Hora, Dia_Semana."
        )

    temperature, hour, weekday = matrix.T
    hour_angle = 2 * np.pi * (hour - 1) / 24
    weekday_angle = 2 * np.pi * (weekday - 1) / 7
    return np.column_stack(
        (
            temperature,
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
        )
    )