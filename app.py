from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import webview


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "credit_risk_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
KNN_MODEL_PATH = BASE_DIR / "model" / "credit_risk_knn_model.joblib"
KNN_METRICS_PATH = BASE_DIR / "model" / "knn_metrics.json"
INDEX_PATH = BASE_DIR / "web" / "index.html"


class CreditRiskApi:
    def __init__(self) -> None:
        self.ensure_model()
        self.model = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.knn_model = None
        self.knn_metrics = None
        if KNN_MODEL_PATH.exists() and KNN_METRICS_PATH.exists():
            self.knn_model = joblib.load(KNN_MODEL_PATH)
            self.knn_metrics = json.loads(KNN_METRICS_PATH.read_text(encoding="utf-8"))

    def ensure_model(self) -> None:
        if MODEL_PATH.exists() and METRICS_PATH.exists():
            return
        subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py"), "--algorithm", "rf"],
            check=True,
        )

    def load_artifacts(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def load_knn_artifacts(self) -> None:
        self.knn_model = joblib.load(KNN_MODEL_PATH)
        self.knn_metrics = json.loads(KNN_METRICS_PATH.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ RF API

    def get_metadata(self) -> dict:
        return self.metrics

    def train_model(self, params: dict | None = None) -> dict:
        params = params or {}
        test_percent = float(params.get("test_percent", 20))
        n_estimators = int(params.get("n_estimators", 120))
        max_depth = int(params.get("max_depth", 0))
        random_state = int(params.get("random_state", 42))
        command = [
            sys.executable,
            str(BASE_DIR / "train_model.py"),
            "--algorithm", "rf",
            "--test-size", str(test_percent / 100),
            "--n-estimators", str(n_estimators),
            "--max-depth", str(max_depth),
            "--random-state", str(random_state),
        ]
        subprocess.run(command, check=True)
        self.load_artifacts()
        return {
            "ok": True,
            "message": "Modelo Random Forest entrenado correctamente.",
            "metrics": self.metrics,
        }

    def predict(self, values: dict) -> dict:
        return self._do_predict(values, self.model, self.metrics)

    # ----------------------------------------------------------------- KNN API

    def get_knn_metadata(self) -> dict:
        """Return KNN metrics; if not yet trained, train with defaults first."""
        if self.knn_metrics is None:
            self._train_knn_default()
        return self.knn_metrics

    def _train_knn_default(self) -> None:
        command = [
            sys.executable,
            str(BASE_DIR / "train_model.py"),
            "--algorithm", "knn",
        ]
        subprocess.run(command, check=True)
        self.load_knn_artifacts()

    def train_knn_model(self, params: dict | None = None) -> dict:
        params = params or {}
        test_percent = float(params.get("test_percent", 20))
        n_neighbors = int(params.get("n_neighbors", 7))
        metric = str(params.get("metric", "minkowski"))
        weights = str(params.get("weights", "uniform"))
        algorithm = str(params.get("algorithm", "auto"))
        random_state = int(params.get("random_state", 42))
        command = [
            sys.executable,
            str(BASE_DIR / "train_model.py"),
            "--algorithm", "knn",
            "--test-size", str(test_percent / 100),
            "--n-neighbors", str(n_neighbors),
            "--metric", metric,
            "--weights", weights,
            "--knn-algorithm", algorithm,
            "--random-state", str(random_state),
        ]
        subprocess.run(command, check=True)
        self.load_knn_artifacts()
        return {
            "ok": True,
            "message": "Modelo KNN entrenado correctamente.",
            "metrics": self.knn_metrics,
        }

    def predict_knn(self, values: dict) -> dict:
        if self.knn_model is None or self.knn_metrics is None:
            raise RuntimeError("El modelo KNN no ha sido entrenado todavía.")
        return self._do_predict(values, self.knn_model, self.knn_metrics)

    # ------------------------------------------------------------ shared logic

    def _do_predict(self, values: dict, model, metrics: dict) -> dict:
        income = self._safe_float(values.get("person_income"))
        loan_amount = self._safe_float(values.get("loan_amnt"))
        if income and income > 0 and loan_amount is not None:
            values["loan_percent_income"] = round(loan_amount / income, 4)

        row = {}
        for field in metrics["fields"]:
            name = field["name"]
            value = values.get(name)
            if field["type"] == "number":
                try:
                    row[name] = float(value)
                except (TypeError, ValueError):
                    row[name] = field["median"]
            else:
                row[name] = str(value)

        frame = pd.DataFrame([row], columns=metrics["feature_columns"])
        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        classes = [int(item) for item in model.classes_]
        probability_by_class = {
            str(class_id): round(float(probabilities[index]) * 100, 2)
            for index, class_id in enumerate(classes)
        }
        confidence = probability_by_class[str(prediction)]

        return {
            "class_id": prediction,
            "class_name": metrics["class_names"][str(prediction)],
            "confidence": confidence,
            "probabilities": probability_by_class,
            "accuracy": round(float(metrics["accuracy"]) * 100, 2),
        }

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def main() -> None:
    api = CreditRiskApi()
    webview.create_window(
        "Sistema de Predicción de Riesgo Crediticio",
        INDEX_PATH.as_uri(),
        js_api=api,
        width=1360,
        height=860,
        min_size=(980, 680),
        maximized=True,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
