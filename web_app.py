"""Servidor web Flask para la plataforma de regresión logística."""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session

from model import TrainingResult, predict_single, train_logistic_regression, validate_dataset

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "regresion-logistica-local-dev")

_store: dict[str, dict[str, Any]] = {}


def _session_id() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def _get_bucket() -> dict[str, Any]:
    sid = _session_id()
    if sid not in _store:
        _store[sid] = {}
    return _store[sid]


def _read_upload(file_storage) -> pd.DataFrame:
    name = file_storage.filename.lower()
    raw = file_storage.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def _result_payload(result: TrainingResult, df: pd.DataFrame) -> dict[str, Any]:
    medians = {col: float(df[col].median()) for col in result.feature_names}
    roc_auc = result.metrics.get("roc_auc", float("nan"))

    return {
        "feature_names": result.feature_names,
        "target_name": result.target_name,
        "coefficients": result.coefficients.to_dict(orient="records"),
        "intercept": result.intercept,
        "metrics": {
            key: None if (isinstance(value, float) and np.isnan(value)) else float(value)
            for key, value in result.metrics.items()
        },
        "confusion": result.confusion.tolist(),
        "report": result.report,
        "roc_fpr": result.roc_fpr.tolist(),
        "roc_tpr": result.roc_tpr.tolist(),
        "train_size": result.train_size,
        "test_size": result.test_size,
        "medians": medians,
        "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/example")
def download_example():
    example_path = BASE_DIR / "data" / "ejemplo.csv"
    return send_file(
        example_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="ejemplo_regresion_logistica.csv",
    )


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "message": "No se recibió ningún archivo."}), 400

    try:
        df = _read_upload(file)
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "message": f"No se pudo leer el archivo: {exc}"}), 400

    bucket = _get_bucket()
    bucket["df"] = df
    bucket["result"] = None
    bucket["filename"] = file.filename

    preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
    columns = df.columns.tolist()

    return jsonify(
        {
            "ok": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "preview": preview,
            "default_target": columns[-1] if columns else None,
        }
    )


@app.route("/api/validate", methods=["POST"])
def validate():
    data = request.get_json(silent=True) or {}
    target = data.get("target_column")
    bucket = _get_bucket()
    df = bucket.get("df")

    if df is None:
        return jsonify({"ok": False, "message": "Primero sube un dataset."}), 400
    if not target:
        return jsonify({"ok": False, "message": "Selecciona una columna objetivo."}), 400

    is_valid, message = validate_dataset(df, target)
    features = [c for c in df.columns if c != target]

    return jsonify(
        {
            "ok": is_valid,
            "message": message,
            "features": features,
        }
    )


@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    target = data.get("target_column")
    test_size = float(data.get("test_size", 0.25))
    random_state = int(data.get("random_state", 42))

    bucket = _get_bucket()
    df = bucket.get("df")

    if df is None:
        return jsonify({"ok": False, "message": "Primero sube un dataset."}), 400
    if not target:
        return jsonify({"ok": False, "message": "Selecciona una columna objetivo."}), 400

    is_valid, message = validate_dataset(df, target)
    if not is_valid:
        return jsonify({"ok": False, "message": message}), 400

    try:
        result = train_logistic_regression(
            df,
            target_column=target,
            test_size=test_size,
            random_state=random_state,
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Error al entrenar: {exc}"}), 500

    bucket["result"] = result
    return jsonify({"ok": True, "message": "Modelo entrenado correctamente.", "result": _result_payload(result, df)})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    values = data.get("values", {})

    bucket = _get_bucket()
    result: TrainingResult | None = bucket.get("result")

    if result is None:
        return jsonify({"ok": False, "message": "Primero entrena un modelo."}), 400

    try:
        payload = {name: float(values[name]) for name in result.feature_names}
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "message": "Completa todos los campos numéricos."}), 400

    pred_class, positive_prob, confidence = predict_single(result, payload)

    return jsonify(
        {
            "ok": True,
            "predicted_class": int(pred_class),
            "probability_class_1": positive_prob,
            "probability_class_0": 1 - positive_prob,
            "confidence": confidence,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
