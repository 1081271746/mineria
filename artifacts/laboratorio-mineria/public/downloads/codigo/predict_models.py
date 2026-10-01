"""Load one exported joblib model and predict from raw CSV-style inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from analysis.feature_engineering import energy_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/laboratorio-mineria/public/downloads/modelos"

MODEL_INPUTS = {
    "dolar": ["Dia", "Inflacion", "Tasa_interes"],
    "glucosa": ["Edad", "IMC", "Actividad_Fisica"],
    "energia": ["Temperatura", "Hora", "Dia_Semana"],
}


def predict(model_name: str, values: dict[str, float]) -> float:
    """Predict one target from a mapping of raw input names to numeric values."""
    if model_name not in MODEL_INPUTS:
        raise ValueError(f"Unknown model '{model_name}'. Choose: {', '.join(MODEL_INPUTS)}")

    required = MODEL_INPUTS[model_name]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing inputs: {', '.join(missing)}")

    row = pd.DataFrame([{name: float(values[name]) for name in required}], columns=required)
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("All inputs must be finite numbers.")

    if model_name == "energia":
        hour = row["Hora"].iloc[0]
        weekday = row["Dia_Semana"].iloc[0]
        if (
            not math.isclose(hour, round(hour))
            or not math.isclose(weekday, round(weekday))
            or not 1 <= hour <= 24
            or not 1 <= weekday <= 7
        ):
            raise ValueError("Hora must be 1–24 and Dia_Semana must be 1–7.")

    estimator = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    matrix = (
        energy_feature_matrix(row.to_numpy(dtype=float))
        if model_name == "energia"
        else row.to_numpy(dtype=float)
    )
    return float(estimator.predict(matrix)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_INPUTS)
    parser.add_argument(
        "inputs",
        help='JSON object, e.g. \'{"Dia": 12, "Inflacion": 0.02, "Tasa_interes": 5}\'',
    )
    args = parser.parse_args()

    try:
        values: Any = json.loads(args.inputs)
        if not isinstance(values, dict):
            raise ValueError("Inputs must be a JSON object.")
        result = predict(args.model, values)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps({"model": args.model, "prediction": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()