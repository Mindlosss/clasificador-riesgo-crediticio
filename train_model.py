from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "credit_risk_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
WEB_ASSETS_DIR = BASE_DIR / "web" / "assets"

DATA_URL = (
    "https://huggingface.co/spaces/FRANCKYPRO/CreditGuard_Pro/resolve/main/"
    "credit_risk_dataset.csv"
)
TARGET_COLUMN = "loan_status"
CLASS_NAMES = {
    0: "Bajo riesgo: no incumple el préstamo",
    1: "Alto riesgo: posible incumplimiento",
}
FIELD_LABELS = {
    "person_age": "Edad",
    "person_income": "Ingreso anual",
    "person_home_ownership": "Tipo de vivienda",
    "person_emp_length": "Años de empleo",
    "loan_intent": "Propósito del préstamo",
    "loan_grade": "Grado del préstamo",
    "loan_amnt": "Monto del préstamo",
    "loan_int_rate": "Tasa de interés",
    "loan_percent_income": "Porcentaje ingreso/préstamo",
    "cb_person_default_on_file": "Incumplimiento previo",
    "cb_person_cred_hist_length": "Años de historial crediticio",
}


def download_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return
    print("Descargando dataset...")
    urlretrieve(DATA_URL, DATA_PATH)


def load_dataset() -> pd.DataFrame:
    download_dataset()
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {TARGET_COLUMN}")
    return df


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def get_field_metadata(df: pd.DataFrame, feature_columns: list[str]) -> list[dict]:
    fields = []
    for column in feature_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            fields.append(
                {
                    "name": column,
                    "label": FIELD_LABELS.get(column, column.replace("_", " ").title()),
                    "type": "number",
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                }
            )
        else:
            values = sorted([str(value) for value in series.dropna().unique()])
            fields.append(
                {
                    "name": column,
                    "label": FIELD_LABELS.get(column, column.replace("_", " ").title()),
                    "type": "category",
                    "options": values,
                    "default": values[0] if values else "",
                }
            )
    return fields


def save_charts(
    y_test: pd.Series,
    y_pred: pd.Series,
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> None:
    WEB_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
    )
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.savefig(WEB_ASSETS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    class_counts = df[TARGET_COLUMN].map(CLASS_NAMES).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, legend=False)
    plt.title("Distribución de clases")
    plt.xlabel("Registros")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(WEB_ASSETS_DIR / "class_distribution.png", dpi=160)
    plt.close()

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = pd.Series(classifier.feature_importances_, index=transformed_names)
    top_importances = importances.sort_values(ascending=False).head(10)
    top_importances.index = [
        name.replace("num__", "").replace("cat__", "").replace("_", " ")
        for name in top_importances.index
    ]

    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=top_importances.values, y=top_importances.index, hue=top_importances.index, legend=False)
    plt.title("Variables más importantes")
    plt.xlabel("Importancia")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(WEB_ASSETS_DIR / "feature_importance.png", dpi=160)
    plt.close()


def normalize_params(
    test_size: float = 0.2,
    n_estimators: int = 120,
    max_depth: int = 0,
    random_state: int = 42,
) -> dict:
    test_size = max(0.1, min(0.5, float(test_size)))
    n_estimators = max(20, min(500, int(n_estimators)))
    max_depth = max(0, min(40, int(max_depth)))
    random_state = max(0, min(9999, int(random_state)))
    return {
        "test_size": test_size,
        "train_size": round(1 - test_size, 4),
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth > 0 else None,
        "random_state": random_state,
    }


def train(
    test_size: float = 0.2,
    n_estimators: int = 120,
    max_depth: int = 0,
    random_state: int = 42,
) -> dict:
    params = normalize_params(test_size, n_estimators, max_depth, random_state)
    df = load_dataset()
    df = df.drop_duplicates()

    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(int)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        params["n_estimators"],
        params["max_depth"],
        params["random_state"],
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "dataset": "Credit Risk Dataset - Kaggle",
        "source_url": "https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data",
        "target_column": TARGET_COLUMN,
        "algorithm": "Random Forest Classifier",
        "training_params": {
            "train_percent": round(params["train_size"] * 100, 2),
            "test_percent": round(params["test_size"] * 100, 2),
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": params["random_state"],
        },
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["No default", "Default"],
            zero_division=0,
            output_dict=True,
        ),
        "class_names": CLASS_NAMES,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "fields": get_field_metadata(df, feature_columns),
        "charts": {
            "confusion_matrix": "assets/confusion_matrix.png",
            "class_distribution": "assets/class_distribution.png",
            "feature_importance": "assets/feature_importance.png",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_charts(y_test, y_pred, df, pipeline, feature_columns)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Eficiencia accuracy: {metrics['accuracy'] * 100:.2f}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de riesgo crediticio.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
