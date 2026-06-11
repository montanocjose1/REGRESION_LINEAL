"""Lógica de regresión logística: entrenamiento, validación y predicción."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingResult:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    target_name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    coefficients: pd.DataFrame
    intercept: float
    metrics: dict[str, float]
    confusion: np.ndarray
    report: str
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    train_size: int
    test_size: int


def validate_dataset(df: pd.DataFrame, target_column: str) -> tuple[bool, str]:
    if df.empty:
        return False, "El dataset está vacío."

    if target_column not in df.columns:
        return False, f"La columna objetivo '{target_column}' no existe."

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        return False, "Se necesita al menos una columna predictora además del objetivo."

    y = df[target_column]
    unique_values = sorted(y.dropna().unique())

    if len(unique_values) != 2:
        return (
            False,
            "La regresión logística binaria requiere exactamente 2 clases en la columna objetivo.",
        )

    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    if numeric_features.shape[1] != len(feature_cols):
        non_numeric = [c for c in feature_cols if c not in numeric_features.columns]
        return False, f"Estas columnas no son numéricas: {', '.join(non_numeric)}"

    if df[feature_cols + [target_column]].isnull().any().any():
        return False, "El dataset contiene valores nulos. Limpia los datos antes de entrenar."

    if len(df) < 10:
        return False, "Se recomiendan al menos 10 filas para un entrenamiento básico."

    return True, "Dataset válido para regresión logística binaria."


def encode_target(y: pd.Series) -> tuple[np.ndarray, dict[Any, int], dict[int, Any]]:
    classes = sorted(y.unique())
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    reverse_map = {idx: cls for cls, idx in label_map.items()}
    encoded = y.map(label_map).to_numpy()
    return encoded, label_map, reverse_map


def train_logistic_regression(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols]
    y_raw = df[target_column]

    y, label_map, _ = encode_target(y_raw)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    coefficients = pd.DataFrame(
        {
            "Variable": feature_cols,
            "Coeficiente": model.coef_[0],
            "Odds Ratio (exp(coef))": np.exp(model.coef_[0]),
        }
    ).sort_values("Coeficiente", key=abs, ascending=False)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    else:
        metrics["roc_auc"] = float("nan")
        fpr, tpr = np.array([]), np.array([])

    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label_map.get(k, k)) for k in sorted(label_map.values())],
        zero_division=0,
    )

    return TrainingResult(
        model=model,
        scaler=scaler,
        feature_names=feature_cols,
        target_name=target_column,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        coefficients=coefficients,
        intercept=float(model.intercept_[0]),
        metrics=metrics,
        confusion=confusion_matrix(y_test, y_pred),
        report=report,
        roc_fpr=fpr,
        roc_tpr=tpr,
        train_size=len(y_train),
        test_size=len(y_test),
    )


def predict_single(
    result: TrainingResult,
    values: dict[str, float],
) -> tuple[Any, float, float]:
    row = pd.DataFrame([[values[name] for name in result.feature_names]], columns=result.feature_names)
    scaled = result.scaler.transform(row)
    proba = result.model.predict_proba(scaled)[0]
    pred_class = int(result.model.predict(scaled)[0])
    confidence = float(proba[pred_class])
    positive_prob = float(proba[1])
    return pred_class, positive_prob, confidence
