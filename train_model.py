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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "credit_risk_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
KNN_MODEL_PATH = BASE_DIR / "model" / "credit_risk_knn_model.joblib"
KNN_METRICS_PATH = BASE_DIR / "model" / "knn_metrics.json"
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


def build_rf_pipeline(
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


def build_knn_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_neighbors: int,
    metric: str,
    weights: str,
    algorithm: str,
) -> Pipeline:
    # KNN needs scaling for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        algorithm=algorithm,
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
    prefix: str = "",
) -> dict:
    """Save charts and return a dict of relative paths keyed by chart name."""
    WEB_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    p = f"{prefix}_" if prefix else ""

    # Confusion matrix
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
    plt.savefig(WEB_ASSETS_DIR / f"{p}confusion_matrix.png", dpi=160)
    plt.close()

    # Class distribution (same for all algorithms)
    class_counts = df[TARGET_COLUMN].map(CLASS_NAMES).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, legend=False)
    plt.title("Distribución de clases")
    plt.xlabel("Registros")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(WEB_ASSETS_DIR / f"{p}class_distribution.png", dpi=160)
    plt.close()

    # Feature importance (RF) or neighbor weight approximation (KNN)
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if hasattr(classifier, "feature_importances_"):
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
        plt.savefig(WEB_ASSETS_DIR / f"{p}feature_importance.png", dpi=160)
        plt.close()
    else:
        # For KNN: show per-class metrics as a horizontal bar chart
        report = classification_report(
            y_test, y_pred,
            target_names=["No default", "Default"],
            zero_division=0,
            output_dict=True,
        )
        classes = ["No default", "Default"]
        precisions = [report[c]["precision"] for c in classes]
        recalls = [report[c]["recall"] for c in classes]
        f1s = [report[c]["f1-score"] for c in classes]

        x = range(len(classes))
        width = 0.25
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars1 = ax.bar([i - width for i in x], precisions, width, label="Precisión", color="#1b7f79")
        bars2 = ax.bar([i for i in x], recalls, width, label="Recall", color="#237a3b")
        bars3 = ax.bar([i + width for i in x], f1s, width, label="F1-score", color="#b83245")
        ax.set_xticks(list(x))
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.15)
        ax.set_title("Métricas por clase (KNN)")
        ax.set_ylabel("Puntaje")
        ax.legend()
        for bar in list(bars1) + list(bars2) + list(bars3):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height() * 100:.1f}%",
                ha="center", fontsize=8, fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig(WEB_ASSETS_DIR / f"{p}feature_importance.png", dpi=160)
        plt.close()

    # Model metrics bar chart
    scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=list(scores.keys()), y=list(scores.values()), color="#1b7f79")
    ax.set_ylim(0, 1)
    plt.title("Métricas del modelo")
    plt.xlabel("")
    plt.ylabel("Puntaje")
    for index, value in enumerate(scores.values()):
        ax.text(index, value + 0.02, f"{value * 100:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(WEB_ASSETS_DIR / f"{p}model_metrics.png", dpi=160)
    plt.close()

    return {
        "confusion_matrix": f"assets/{p}confusion_matrix.png",
        "class_distribution": f"assets/{p}class_distribution.png",
        "feature_importance": f"assets/{p}feature_importance.png",
        "model_metrics": f"assets/{p}model_metrics.png",
    }


def normalize_rf_params(
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


def normalize_knn_params(
    test_size: float = 0.2,
    n_neighbors: int = 7,
    metric: str = "minkowski",
    weights: str = "uniform",
    algorithm: str = "auto",
    random_state: int = 42,
) -> dict:
    test_size = max(0.1, min(0.5, float(test_size)))
    n_neighbors = max(1, min(50, int(n_neighbors)))
    metric = metric if metric in ("minkowski", "euclidean", "manhattan", "chebyshev") else "minkowski"
    weights = weights if weights in ("uniform", "distance") else "uniform"
    algorithm = algorithm if algorithm in ("auto", "ball_tree", "kd_tree", "brute") else "auto"
    random_state = max(0, min(9999, int(random_state)))
    return {
        "test_size": test_size,
        "train_size": round(1 - test_size, 4),
        "n_neighbors": n_neighbors,
        "metric": metric,
        "weights": weights,
        "algorithm": algorithm,
        "random_state": random_state,
    }


def train(
    test_size: float = 0.2,
    n_estimators: int = 120,
    max_depth: int = 0,
    random_state: int = 42,
) -> dict:
    params = normalize_rf_params(test_size, n_estimators, max_depth, random_state)
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

    pipeline = build_rf_pipeline(
        numeric_features,
        categorical_features,
        params["n_estimators"],
        params["max_depth"],
        params["random_state"],
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    charts = save_charts(y_test, y_pred, df, pipeline, feature_columns, prefix="rf")

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
        "charts": charts,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Modelo RF guardado en: {MODEL_PATH}")
    print(f"Eficiencia accuracy: {metrics['accuracy'] * 100:.2f}%")
    return metrics


def train_knn(
    test_size: float = 0.2,
    n_neighbors: int = 7,
    metric: str = "minkowski",
    weights: str = "uniform",
    algorithm: str = "auto",
    random_state: int = 42,
) -> dict:
    params = normalize_knn_params(test_size, n_neighbors, metric, weights, algorithm, random_state)
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

    pipeline = build_knn_pipeline(
        numeric_features,
        categorical_features,
        params["n_neighbors"],
        params["metric"],
        params["weights"],
        params["algorithm"],
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    charts = save_charts(y_test, y_pred, df, pipeline, feature_columns, prefix="knn")

    metrics = {
        "dataset": "Credit Risk Dataset - Kaggle",
        "source_url": "https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data",
        "target_column": TARGET_COLUMN,
        "algorithm": "K-Nearest Neighbors Classifier",
        "training_params": {
            "train_percent": round(params["train_size"] * 100, 2),
            "test_percent": round(params["test_size"] * 100, 2),
            "n_neighbors": params["n_neighbors"],
            "metric": params["metric"],
            "weights": params["weights"],
            "algorithm": params["algorithm"],
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
        "charts": charts,
    }

    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, KNN_MODEL_PATH)
    KNN_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Modelo KNN guardado en: {KNN_MODEL_PATH}")
    print(f"Eficiencia accuracy KNN: {metrics['accuracy'] * 100:.2f}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de riesgo crediticio.")
    parser.add_argument("--algorithm", choices=["rf", "knn"], default="rf")
    parser.add_argument("--test-size", type=float, default=0.2)
    # RF params
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    # KNN params
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--metric", type=str, default="minkowski")
    parser.add_argument("--weights", type=str, default="uniform")
    parser.add_argument("--knn-algorithm", type=str, default="auto")
    args = parser.parse_args()

    if args.algorithm == "knn":
        train_knn(
            test_size=args.test_size,
            n_neighbors=args.n_neighbors,
            metric=args.metric,
            weights=args.weights,
            algorithm=args.knn_algorithm,
            random_state=args.random_state,
        )
    else:
        train(
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
