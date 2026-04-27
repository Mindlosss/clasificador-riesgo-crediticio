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
INDEX_PATH = BASE_DIR / "web" / "index.html"


class CreditRiskApi:
    def __init__(self) -> None:
        self.ensure_model()
        self.model = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def ensure_model(self) -> None:
        if MODEL_PATH.exists() and METRICS_PATH.exists():
            return
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)

    def load_artifacts(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

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
            "--test-size",
            str(test_percent / 100),
            "--n-estimators",
            str(n_estimators),
            "--max-depth",
            str(max_depth),
            "--random-state",
            str(random_state),
        ]
        subprocess.run(command, check=True)
        self.load_artifacts()
        return {
            "ok": True,
            "message": "Modelo entrenado correctamente.",
            "metrics": self.metrics,
        }

    def predict(self, values: dict) -> dict:
        row = {}
        for field in self.metrics["fields"]:
            name = field["name"]
            value = values.get(name)
            if field["type"] == "number":
                try:
                    row[name] = float(value)
                except (TypeError, ValueError):
                    row[name] = field["median"]
            else:
                row[name] = str(value)

        frame = pd.DataFrame([row], columns=self.metrics["feature_columns"])
        prediction = int(self.model.predict(frame)[0])
        probabilities = self.model.predict_proba(frame)[0]
        classes = [int(item) for item in self.model.classes_]
        probability_by_class = {
            str(class_id): round(float(probabilities[index]) * 100, 2)
            for index, class_id in enumerate(classes)
        }
        confidence = probability_by_class[str(prediction)]

        return {
            "class_id": prediction,
            "class_name": self.metrics["class_names"][str(prediction)],
            "confidence": confidence,
            "probabilities": probability_by_class,
            "accuracy": round(float(self.metrics["accuracy"]) * 100, 2),
        }


def main() -> None:
    api = CreditRiskApi()
    webview.create_window(
        "Clasificador de Riesgo Crediticio",
        INDEX_PATH.as_uri(),
        js_api=api,
        width=1180,
        height=780,
        min_size=(980, 680),
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
