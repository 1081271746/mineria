"""Train, evaluate, export, and document the three CRISP-DM lab models."""

from __future__ import annotations

import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from analysis.feature_engineering import energy_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis/data"
ARTIFACT_DIR = ROOT / "artifacts/laboratorio-mineria"
PUBLIC_DIR = ARTIFACT_DIR / "public"
DOWNLOAD_DIR = PUBLIC_DIR / "downloads"
MODEL_DIR = DOWNLOAD_DIR / "modelos"
DATA_DOWNLOAD_DIR = DOWNLOAD_DIR / "datasets"
CODE_DOWNLOAD_DIR = DOWNLOAD_DIR / "codigo"
WEB_DATA_DIR = ARTIFACT_DIR / "src/data"
RESULTS_DIR = ROOT / "analysis/results"

MODEL_SPECS = {
    "dolar": {
        "title": "Precio del dólar",
        "subtitle": "Predicción del precio con variables económicas y día observado",
        "csv": "dolar_data.csv",
        "target": "Precio_Dolar",
        "target_label": "Precio del dólar",
        "target_unit": "unidad monetaria del dataset (no especificada)",
        "predictors": ["Dia", "Inflacion", "Tasa_interes"],
        "labels": {
            "Dia": ("Día", "día"),
            "Inflacion": ("Inflación", "tasa decimal"),
            "Tasa_interes": ("Tasa de interés", "unidad original"),
        },
        "split": "Temporal: primeras 400 filas para entrenamiento y últimas 100 para prueba.",
        "holdout": "chronological",
    },
    "glucosa": {
        "title": "Nivel de glucosa",
        "subtitle": "Estimación de glucosa a partir de datos demográficos y actividad física",
        "csv": "glucosa_data.csv",
        "target": "Nivel_Glucosa",
        "target_label": "Nivel de glucosa",
        "target_unit": "mg/dL",
        "predictors": ["Edad", "IMC", "Actividad_Fisica"],
        "labels": {
            "Edad": ("Edad", "años"),
            "IMC": ("IMC", "kg/m²"),
            "Actividad_Fisica": ("Actividad física", "horas/semana"),
        },
        "split": "Aleatorio reproducible: 80 % entrenamiento, 20 % prueba, random_state=42.",
        "holdout": "random",
    },
    "energia": {
        "title": "Consumo de energía",
        "subtitle": "Estimación del consumo con temperatura y patrones cíclicos de calendario",
        "csv": "energia_data.csv",
        "target": "Consumo_Energia",
        "target_label": "Consumo de energía",
        "target_unit": "kWh",
        "predictors": ["Temperatura", "Hora", "Dia_Semana"],
        "labels": {
            "Temperatura": ("Temperatura", "°C"),
            "Hora": ("Hora del día", "hora"),
            "Dia_Semana": ("Día de la semana", "día"),
        },
        "split": "Aleatorio reproducible: 80 % entrenamiento, 20 % prueba, random_state=42.",
        "holdout": "random",
    },
}

EQUATION_FEATURES = {
    "dolar": [
        ("Dia", "raw", "Día"),
        ("Inflacion", "raw", "Inflación"),
        ("Tasa_interes", "raw", "Tasa de interés"),
    ],
    "glucosa": [
        ("Edad", "raw", "Edad"),
        ("IMC", "raw", "IMC"),
        ("Actividad_Fisica", "raw", "Actividad física"),
    ],
    "energia": [
        ("Temperatura", "raw", "Temperatura"),
        ("Hora_sin", "hourSin", "seno de la hora"),
        ("Hora_cos", "hourCos", "coseno de la hora"),
        ("Dia_sin", "weekdaySin", "seno del día"),
        ("Dia_cos", "weekdayCos", "coseno del día"),
    ],
}

PREDICTOR_TERMS = {
    "dolar": {
        "Dia": [0],
        "Inflacion": [1],
        "Tasa_interes": [2],
    },
    "glucosa": {
        "Edad": [0],
        "IMC": [1],
        "Actividad_Fisica": [2],
    },
    "energia": {
        "Temperatura": [0],
        "Hora": [1, 2],
        "Dia_Semana": [3, 4],
    },
}

MODEL_INPUTS = {
    model_id: spec["predictors"] for model_id, spec in MODEL_SPECS.items()
}


def make_estimator(model_id: str) -> Any:
    del model_id
    return LinearRegression()


def design_matrix(model_id: str, values: pd.DataFrame | np.ndarray) -> np.ndarray:
    matrix = (
        values.to_numpy(dtype=float)
        if isinstance(values, pd.DataFrame)
        else np.asarray(values, dtype=float)
    )
    return energy_feature_matrix(matrix) if model_id == "energia" else matrix


def estimator_coefficients(estimator: Any, model_id: str) -> tuple[float, np.ndarray]:
    del model_id
    return float(estimator.intercept_), np.asarray(estimator.coef_, dtype=float)


def round_number(value: float, digits: int = 8) -> float:
    result = float(value)
    return 0.0 if abs(result) < 10 ** (-digits) else round(result, digits)


def finite_range(values: pd.Series) -> tuple[float, float]:
    return float(values.min()), float(values.max())


def make_prediction_line(
    model_id: str,
    estimator: Any,
    frame: pd.DataFrame,
    predictor_id: str,
) -> list[list[float]]:
    spec = MODEL_SPECS[model_id]
    baseline = {
        name: float(frame[name].median()) for name in spec["predictors"]
    }

    if model_id == "energia" and predictor_id == "Hora":
        values = np.arange(1, 25, dtype=int)
    elif model_id == "energia" and predictor_id == "Dia_Semana":
        values = np.arange(1, 8, dtype=int)
    else:
        minimum, maximum = finite_range(frame[predictor_id])
        values = np.linspace(minimum, maximum, 70)

    scenarios = pd.DataFrame(
        [{**baseline, predictor_id: float(value)} for value in values],
        columns=spec["predictors"],
    )
    predictions = estimator.predict(design_matrix(model_id, scenarios))
    return [
        [round_number(float(value), 6), round_number(float(prediction), 6)]
        for value, prediction in zip(values, predictions, strict=True)
    ]


def correlation(frame: pd.DataFrame, predictor: str, target: str) -> float | None:
    result = frame[predictor].corr(frame[target])
    return None if pd.isna(result) else round_number(float(result))


def evaluate_split(
    model_id: str,
    frame: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    spec = MODEL_SPECS[model_id]
    features = spec["predictors"]
    X = frame[features]
    y = frame[spec["target"]]

    if spec["holdout"] == "chronological":
        if "Dia" not in frame or not frame["Dia"].is_monotonic_increasing:
            raise ValueError("The dollar dataset must be ordered by ascending Dia.")
        cutoff = math.floor(len(frame) * 0.8)
        X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
        y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
        split_info = {
            "method": "chronological",
            "description": spec["split"],
            "trainRows": len(X_train),
            "testRows": len(X_test),
            "testFrom": float(frame["Dia"].iloc[cutoff]),
            "testTo": float(frame["Dia"].iloc[-1]),
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        split_info = {
            "method": "random",
            "description": spec["split"],
            "trainRows": len(X_train),
            "testRows": len(X_test),
            "randomState": 42,
        }

    holdout_model = make_estimator(model_id)
    holdout_model.fit(design_matrix(model_id, X_train), y_train)
    prediction = holdout_model.predict(design_matrix(model_id, X_test))
    mse = float(mean_squared_error(y_test, prediction))
    metrics = {
        "mse": round_number(mse),
        "rmse": round_number(math.sqrt(mse)),
        "r2": round_number(float(r2_score(y_test, prediction))),
    }
    split_info["metricsBasis"] = "Datos reservados de prueba; los coeficientes finales se reajustan con todas las filas."
    return holdout_model, X_train, X_test, y_train, y_test, {
        "split": split_info,
        "metrics": metrics,
    }


def fit_full_model(model_id: str, frame: pd.DataFrame) -> Any:
    spec = MODEL_SPECS[model_id]
    estimator = make_estimator(model_id)
    estimator.fit(
        design_matrix(model_id, frame[spec["predictors"]]),
        frame[spec["target"]],
    )
    return estimator


def predictor_interpretation(
    model_id: str,
    predictor_id: str,
    coefficient_values: list[float],
    impact: float,
    top_predictor: str,
) -> str:
    spec = MODEL_SPECS[model_id]
    label, unit = spec["labels"][predictor_id]
    if model_id == "energia" and predictor_id in {"Hora", "Dia_Semana"}:
        detail = (
            "Su efecto se modela con seno y coseno para respetar el ciclo; "
            "no existe un coeficiente lineal constante por cada unidad."
        )
    else:
        coefficient = coefficient_values[0]
        if model_id == "dolar" and predictor_id == "Inflacion":
            detail = (
                f"Con las demás variables fijas, +0,01 en inflación "
                f"(un punto porcentual si la tasa se ingresa como fracción) "
                f"se asocia con {abs(coefficient * 0.01):.3f} unidades "
                f"{'más' if coefficient >= 0 else 'menos'} del objetivo."
            )
        else:
            detail = (
                f"Con las demás variables fijas, +1 {unit} se asocia con "
                f"{abs(coefficient):.3f} unidades "
                f"{'más' if coefficient >= 0 else 'menos'} del objetivo."
            )
    rank = (
        "Es la variable con mayor impacto estandarizado en este modelo."
        if predictor_id == top_predictor
        else f"Impacto estandarizado comparativo: {impact:.3f}."
    )
    return f"{label}: {detail} {rank} Asociación del modelo, no evidencia causal."


def build_model_data(model_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    spec = MODEL_SPECS[model_id]
    holdout_model, _, X_test, _, y_test, evaluation = evaluate_split(model_id, frame)
    estimator = fit_full_model(model_id, frame)
    intercept, coefficients = estimator_coefficients(estimator, model_id)
    target_std = float(frame[spec["target"]].std(ddof=0))

    if model_id == "energia":
        transformed = design_matrix(model_id, frame[spec["predictors"]])
        transformed_std = np.std(transformed, axis=0)
    else:
        transformed_std = frame[spec["predictors"]].std(ddof=0).to_numpy()

    grouped_impact: dict[str, float] = {}
    predictor_coefficients: dict[str, list[float]] = {}
    for predictor, indexes in PREDICTOR_TERMS[model_id].items():
        predictor_coefficients[predictor] = [
            float(coefficients[index]) for index in indexes
        ]
        standardized_terms = [
            float(coefficients[index] * transformed_std[index] / target_std)
            for index in indexes
        ]
        grouped_impact[predictor] = math.sqrt(
            sum(value * value for value in standardized_terms)
        )

    top_predictor = max(grouped_impact, key=grouped_impact.get)
    equation_terms = []
    for index, (feature_name, basis, label) in enumerate(EQUATION_FEATURES[model_id]):
        predictor_id = next(
            predictor
            for predictor, indexes in PREDICTOR_TERMS[model_id].items()
            if index in indexes
        )
        equation_terms.append(
            {
                "name": label,
                "feature": feature_name,
                "predictorId": predictor_id,
                "basis": basis,
                "coefficient": round_number(float(coefficients[index])),
            }
        )

    # Permute original columns before feature engineering so cyclic inputs are
    # evaluated as whole predictors, not as separate sine/cosine fragments.
    baseline_score = r2_score(
        y_test,
        holdout_model.predict(design_matrix(model_id, X_test)),
    )
    random = np.random.RandomState(42)
    permutation_drops = np.empty((len(spec["predictors"]), 30), dtype=float)
    for feature_index, predictor in enumerate(spec["predictors"]):
        for repeat in range(permutation_drops.shape[1]):
            shuffled = X_test.copy()
            shuffled[predictor] = random.permutation(shuffled[predictor].to_numpy())
            permuted_score = r2_score(
                y_test,
                holdout_model.predict(design_matrix(model_id, shuffled)),
            )
            permutation_drops[feature_index, repeat] = baseline_score - permuted_score

    predictors = []
    for predictor in spec["predictors"]:
        label, unit = spec["labels"][predictor]
        series = frame[predictor]
        minimum, maximum = finite_range(series)
        median = float(series.median())
        if model_id == "energia" and predictor in {"Hora", "Dia_Semana"}:
            median = float(round(median))
        indexes = PREDICTOR_TERMS[model_id][predictor]
        group = predictor_coefficients[predictor]
        feature_index = spec["predictors"].index(predictor)
        importance_mean = round_number(
            float(permutation_drops[feature_index].mean())
        )
        importance_std = round_number(
            float(permutation_drops[feature_index].std())
        )

        sample = frame.sample(
            n=min(250, len(frame)), random_state=42
        ).sort_index()
        points = [
            [round_number(float(x), 6), round_number(float(y), 6)]
            for x, y in zip(
                sample[predictor],
                sample[spec["target"]],
                strict=True,
            )
        ]
        predictors.append(
            {
                "id": predictor,
                "label": label,
                "unit": unit,
                "min": round_number(minimum, 6),
                "max": round_number(maximum, 6),
                "mean": round_number(float(series.mean()), 6),
                "median": round_number(median, 6),
                "defaultValue": round_number(median, 6),
                "correlation": correlation(frame, predictor, spec["target"]),
                "standardizedImpact": round_number(grouped_impact[predictor]),
                "standardizedCoefficient": (
                    round_number(
                        float(
                            coefficients[indexes[0]]
                            * transformed_std[indexes[0]]
                            / target_std
                        )
                    )
                    if len(indexes) == 1
                    else None
                ),
                "coefficient": (
                    round_number(group[0]) if len(indexes) == 1 else None
                ),
                "terms": [
                    equation_terms[index]
                    for index in indexes
                ],
                "permutationR2DropMean": importance_mean,
                "permutationR2DropStd": importance_std,
                "interpretation": predictor_interpretation(
                    model_id,
                    predictor,
                    group,
                    grouped_impact[predictor],
                    top_predictor,
                ),
                "scatterPoints": points,
                "linePoints": make_prediction_line(
                    model_id, estimator, frame, predictor
                ),
            }
        )

    quality = {
        "missingCells": int(frame.isna().sum().sum()),
        "duplicateRows": int(frame.duplicated().sum()),
    }
    return {
        "id": model_id,
        "title": spec["title"],
        "subtitle": spec["subtitle"],
        "target": spec["target"],
        "targetLabel": spec["target_label"],
        "targetUnit": spec["target_unit"],
        "dataset": spec["csv"],
        "observations": len(frame),
        "quality": quality,
        "split": evaluation["split"],
        "metrics": evaluation["metrics"],
        "equation": {
            "intercept": round_number(intercept),
            "terms": equation_terms,
            "coefficientsFitOn": "all_rows",
        },
        "predictors": predictors,
        "largestImpactId": top_predictor,
        "largestImpact": round_number(grouped_impact[top_predictor]),
        "impactMethod": (
            "Magnitud estandarizada agrupada por variable original "
            "(raíz de la suma de cuadrados de los coeficientes estandarizados)."
        ),
        "notes": (
            "Los coeficientes describen asociaciones lineales condicionadas a las otras "
            "variables incluidas; no implican causalidad. R² y errores corresponden "
            "únicamente al conjunto de prueba."
        ),
    }


def make_report(models: list[dict[str, Any]]) -> str:
    report = [
        "# Informe — Laboratorio de Minería de Datos",
        "",
        "## Objetivo",
        "",
        "Aplicar las fases de CRISP-DM y entrenar tres regresiones lineales múltiples "
        "sobre los archivos CSV suministrados. Se documentan la preparación, la "
        "validación, la interpretación de coeficientes, el desempeño en prueba y la "
        "exportación de los modelos.",
        "",
        "## CRISP-DM",
        "",
        "1. **Comprensión del negocio:** estimar el precio del dólar, el nivel de glucosa "
        "y el consumo eléctrico con las variables incluidas en cada dataset.",
        "2. **Comprensión de los datos:** los archivos contienen 500, 2.000 y 10.000 "
        "filas, respectivamente. La validación encontró cero celdas faltantes y cero "
        "filas duplicadas en los tres archivos.",
        "3. **Preparación de los datos:** se conservaron las filas y columnas originales. "
        "En energía, Hora (1–24) y Dia_Semana (1–7) se codifican con pares seno/coseno "
        "para representar correctamente su periodicidad.",
        "4. **Modelado:** regresión lineal por mínimos cuadrados. El modelo de dólar se "
        "evalúa con corte temporal (primer 80 % para entrenamiento); glucosa y energía "
        "usan partición aleatoria reproducible 80/20 con semilla 42.",
        "5. **Evaluación:** MSE, RMSE y R² se calculan sobre datos reservados que no "
        "participaron en el ajuste evaluado. Después se reajusta un modelo final con "
        "todas las filas y se exporta a joblib para reutilizarlo.",
        "6. **Despliegue:** una interfaz estática ejecuta la misma ecuación en el "
        "navegador; no necesita servidor Python para hacer predicciones. El código Python "
        "permite reproducir el entrenamiento y cargar los artefactos.",
        "",
        "## Resultados e interpretación",
        "",
        "Las magnitudes estandarizadas permiten comparar predictores con unidades distintas. "
        "Para los pares seno/coseno de hora y día se agrupa la magnitud de los dos "
        "coeficientes mediante raíz de suma de cuadrados. Los resultados son asociativos, "
        "no causales.",
        "",
    ]
    for model in models:
        report.extend(
            [
                f"### {model['title']}",
                "",
                f"- Dataset: `{model['dataset']}`; {model['observations']} observaciones; "
                f"{model['quality']['missingCells']} faltantes; "
                f"{model['quality']['duplicateRows']} duplicados.",
                f"- Variable objetivo: {model['targetLabel']} ({model['targetUnit']}).",
                f"- Validación: {model['split']['description']}",
                f"- Prueba: MSE = {model['metrics']['mse']:.4f}; "
                f"RMSE = {model['metrics']['rmse']:.4f}; "
                f"R² = {model['metrics']['r2']:.4f}.",
                f"- Mayor impacto estandarizado: **{next(p['label'] for p in model['predictors'] if p['id'] == model['largestImpactId'])}** "
                f"({model['largestImpact']:.3f}).",
                f"- Intercepto del modelo final: {model['equation']['intercept']:.6f}.",
                "- Predictores:",
            ]
        )
        for predictor in model["predictors"]:
            coefficient = (
                f"; coeficiente = {predictor['coefficient']:.6f}"
                if predictor["coefficient"] is not None
                else "; efecto representado por términos cíclicos"
            )
            report.append(
                f"  - **{predictor['label']}**: impacto estandarizado "
                f"{predictor['standardizedImpact']:.4f}{coefficient}. "
                f"Correlación bivariada r = {predictor['correlation']:.4f}. "
                f"{predictor['interpretation']}"
            )
        formula = " ".join(
            f"{term['coefficient']:+.6f}·{term['name']}"
            for term in model["equation"]["terms"]
        )
        report.extend(
            [
                f"- Ecuación: `ŷ = {model['equation']['intercept']:.6f} {formula}`.",
                "",
            ]
        )

    report.extend(
        [
            "## Conclusiones",
            "",
            "El modelo de dólar obtiene el R² más alto, pero su validación temporal es más "
            "estricta que una partición aleatoria y aun así deja error en los días futuros. "
            "En glucosa, la edad presenta el mayor impacto estandarizado; el R² indica que "
            "hay variación no explicada por las tres variables disponibles. En energía, "
            "la temperatura es el predictor de mayor impacto; la codificación cíclica "
            "evita suponer que las horas 1 y 24 o los días 1 y 7 están en extremos "
            "opuestos.",
            "",
            "MSE y RMSE dependen de las unidades de cada objetivo, por lo que no se deben "
            "comparar directamente entre los tres problemas. R² compara la proporción de "
            "variabilidad explicada dentro de cada conjunto de prueba. Ninguna de estas "
            "métricas demuestra causalidad ni garantiza desempeño fuera del rango de los "
            "datos observados.",
            "",
            "## Reproducibilidad y archivos",
            "",
            "Desde la raíz del proyecto: `python -m pip install -r analysis/requirements.txt` "
            "y luego `python -m analysis.train_models`. Esto regenera las métricas, los "
            "coeficientes estáticos usados por la interfaz, los tres archivos `.joblib`, "
            "este informe y el PDF. Los modelos `.joblib` se entrenan con todas las filas "
            "y se guardan solo después de evaluar el corte de prueba.",
            "",
            "El precio del dólar no declara unidad monetaria en el CSV; el informe no "
            "asume COP. Inflación se registra como fracción decimal (por ejemplo, 0,02 "
            "equivale a 2 %).",
            "",
        ]
    )
    return "\n".join(report)


def render_pdf(markdown: str, output_path: Path) -> None:
    """Render the editable Markdown report into a compact, printable PDF."""
    from matplotlib.backends.backend_pdf import PdfPages

    printable = []
    for original in markdown.splitlines():
        line = original.replace("**", "").replace("`", "")
        line = line.lstrip("#").strip()
        if line.startswith("- "):
            line = "• " + line[2:]
        printable.extend(textwrap.wrap(line, width=105) or [""])

    page_lines = 47
    with PdfPages(output_path) as pdf:
        for page_number, start in enumerate(range(0, len(printable), page_lines), start=1):
            page = printable[start : start + page_lines]
            figure = plt.figure(figsize=(8.27, 11.69), facecolor="#fbfaf7")
            figure.text(
                0.08,
                0.955,
                "LABORATORIO · MINERÍA DE DATOS",
                fontsize=9,
                color="#126b60",
                weight="bold",
                va="top",
            )
            figure.text(
                0.92,
                0.955,
                f"{page_number:02d}",
                fontsize=9,
                color="#707a75",
                ha="right",
                va="top",
            )
            y = 0.91
            for line in page:
                if not line:
                    y -= 0.011
                    continue
                heading = (
                    line.startswith("Informe")
                    or line.startswith("CRISP-DM")
                    or line.startswith("Resultados")
                    or line.startswith("Conclusiones")
                    or line.startswith("Reproducibilidad")
                    or line.startswith("Precio del dólar")
                    or line.startswith("Nivel de glucosa")
                    or line.startswith("Consumo de energía")
                )
                figure.text(
                    0.08,
                    y,
                    line,
                    fontsize=10 if heading else 8.3,
                    color="#153b37" if heading else "#313b37",
                    weight="bold" if heading else "normal",
                    va="top",
                    wrap=True,
                )
                y -= 0.018 if heading else 0.015
            figure.text(
                0.08,
                0.045,
                "Regresión lineal múltiple · resultados en datos de prueba reservados",
                fontsize=7,
                color="#7b827e",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def copy_public_downloads() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, spec in MODEL_SPECS.items():
        source = DATA_DIR / spec["csv"]
        destination = DATA_DOWNLOAD_DIR / spec["csv"]
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    for filename in (
        "__init__.py",
        "feature_engineering.py",
        "predict_models.py",
        "requirements.txt",
        "train_models.py",
    ):
        shutil.copy2(ROOT / "analysis" / filename, CODE_DOWNLOAD_DIR / filename)
    with zipfile.ZipFile(
        DOWNLOAD_DIR / "codigo-entrenamiento.zip",
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename in (
            "__init__.py",
            "feature_engineering.py",
            "predict_models.py",
            "requirements.txt",
            "train_models.py",
        ):
            archive.write(
                ROOT / "analysis" / filename,
                f"analysis/{filename}",
            )
        for spec in MODEL_SPECS.values():
            filename = spec["csv"]
            archive.write(DATA_DIR / filename, f"analysis/data/{filename}")
        archive.write(ROOT / "README.md", "README.md")


def main() -> None:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    for model_id, spec in MODEL_SPECS.items():
        csv_path = DATA_DIR / spec["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        required = [*spec["predictors"], spec["target"]]
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{spec['csv']} is missing columns: {missing_columns}")
        numeric = frame[required].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{spec['csv']} contains missing or non-finite values.")
        if frame.empty:
            raise ValueError(f"{spec['csv']} has no observations.")
        frame[required] = numeric

        model_data = build_model_data(model_id, frame)
        final_estimator = fit_full_model(model_id, frame)
        joblib.dump(final_estimator, MODEL_DIR / f"{model_id}.joblib")
        models.append(model_data)
        metrics = model_data["metrics"]
        print(
            f"{model_id}: n={model_data['observations']}, "
            f"MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"R2={metrics['r2']:.4f}, top={model_data['largestImpactId']}"
        )

    model_data = {
        "version": 1,
        "methodology": {
            "algorithm": "Regresión lineal múltiple por mínimos cuadrados",
            "validation": "MSE, RMSE y R² calculados exclusivamente en el conjunto de prueba.",
            "importance": "Magnitud de coeficientes estandarizados; Hora y Dia_Semana se agrupan como pares seno/coseno.",
            "cyclicEncoding": "sin/cos para periodos de 24 horas y 7 días.",
        },
        "models": models,
    }
    json_path = WEB_DATA_DIR / "model-data.json"
    json_path.write_text(
        json.dumps(model_data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "model-results.json").write_text(
        json.dumps(model_data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = make_report(models)
    report_path = ROOT / "analysis/INFORME.md"
    report_path.write_text(report, encoding="utf-8")
    (DOWNLOAD_DIR / "informe-laboratorio.md").write_text(report, encoding="utf-8")
    render_pdf(report, DOWNLOAD_DIR / "informe-laboratorio.pdf")
    copy_public_downloads()
    print(f"Model data: {json_path}")
    print(f"Report: {report_path}")
    print(f"PDF: {DOWNLOAD_DIR / 'informe-laboratorio.pdf'}")


if __name__ == "__main__":
    main()