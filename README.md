# Clasificador de Riesgo Crediticio

Sistema de información con algoritmo de clasificación para predecir si un solicitante tiene bajo o alto riesgo de incumplimiento de préstamo.

## Dataset

Se usa el dataset real **Credit Risk Dataset** de Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data

La variable de clase es:

```text
loan_status
```

Interpretación:

```text
0 = bajo riesgo / no incumple
1 = alto riesgo / posible incumplimiento
```

## Tecnologías

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- pywebview
- HTML, CSS y JavaScript

## Algoritmo

El proyecto usa `RandomForestClassifier`. El algoritmo construye varios árboles de decisión. Cada árbol emite una predicción y la clase final se obtiene por votación. Esto ayuda a reducir errores de un solo árbol y funciona bien con variables numéricas y categóricas.

## Ejecutar

Requisito recomendado en Windows: **Python 3.10, 3.11 o 3.12**. Evita Python 3.14 para esta app porque `pywebview` usa `pythonnet` y puede fallar al instalarse en esa versión.

Forma rápida:

```powershell
.\iniciar_app.bat
```

Forma manual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Si ya se creó `.venv` con Python 3.14 y falló la instalación, borra esa carpeta y vuelve a ejecutar con Python 3.11:

```powershell
Remove-Item -Recurse -Force .venv
.\iniciar_app.bat
```

El primer entrenamiento descarga el CSV, entrena el modelo, genera `model/metrics.json` y guarda las gráficas en `web/assets`.

La interfaz tiene dos vistas:

- **Entrenar modelo**: reentrena el clasificador, muestra métricas, matriz de confusión y gráficas.
- **Predicción**: captura datos de un nuevo solicitante y devuelve la clase estimada.

En la vista de entrenamiento se pueden modificar:

- porcentaje de datos para prueba
- número de árboles del Random Forest
- profundidad máxima de los árboles
- semilla aleatoria para la división de datos

Al cambiar estos parámetros y volver a entrenar, la matriz de confusión y el porcentaje de eficiencia pueden cambiar.

## Evaluación

La app muestra:

- porcentaje de eficiencia usando `accuracy`
- matriz de confusión
- precisión
- recall
- F1-score
- gráfica de distribución de clases
- gráfica de importancia de variables
