const form = document.querySelector("#prediction-form");
const trainingForm = document.querySelector("#training-form");
const predictButton = document.querySelector("#predict-button");
const trainButton = document.querySelector("#train-button");
const trainStatus = document.querySelector("#train-status");
const testPercentInput = document.querySelector("#test-percent");
const testPercentLabel = document.querySelector("#test-percent-label");
const result = document.querySelector("#result");
const resultClass = document.querySelector("#result-class");
const resultDetail = document.querySelector("#result-detail");

let metadata = null;

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function setText(id, value) {
  document.querySelector(id).textContent = value;
}

function buildField(field) {
  const wrapper = document.createElement("label");
  wrapper.textContent = field.label;

  if (field.type === "number") {
    const input = document.createElement("input");
    input.name = field.name;
    input.type = "number";
    input.step = "any";
    input.value = Number(field.median).toFixed(2);
    input.min = field.min;
    input.max = field.max;
    wrapper.appendChild(input);
    return wrapper;
  }

  const select = document.createElement("select");
  select.name = field.name;
  field.options.forEach((optionValue) => {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = optionValue;
    select.appendChild(option);
  });
  select.value = field.default;
  wrapper.appendChild(select);
  return wrapper;
}

function renderMetadata(data) {
  metadata = data;
  const trainingParams = data.training_params || {};
  setText("#accuracy", percent(data.accuracy));
  setText("#algorithm", data.algorithm);
  setText("#target", data.target_column);
  setText("#rows", data.rows.toLocaleString("es-MX"));
  setText("#train-rows", data.train_rows.toLocaleString("es-MX"));
  setText("#test-rows", data.test_rows.toLocaleString("es-MX"));
  setText(
    "#split-info",
    `${trainingParams.train_percent || 80}% / ${trainingParams.test_percent || 20}%`
  );
  setText("#trees-info", trainingParams.n_estimators || 120);
  setText("#depth-info", trainingParams.max_depth || "Sin límite");
  setText("#seed-info", trainingParams.random_state ?? 42);
  setText("#precision", percent(data.precision));
  setText("#recall", percent(data.recall));
  setText("#f1", percent(data.f1));

  const cacheBust = `?v=${Date.now()}`;
  document.querySelector("#confusion-chart").src = data.charts.confusion_matrix + cacheBust;
  document.querySelector("#importance-chart").src = data.charts.feature_importance + cacheBust;
  document.querySelector("#distribution-chart").src = data.charts.class_distribution + cacheBust;

  form.innerHTML = "";
  data.fields.forEach((field) => form.appendChild(buildField(field)));

  if (trainingParams.test_percent) {
    testPercentInput.value = trainingParams.test_percent;
    updateSplitLabel();
  }
  if (trainingParams.n_estimators) {
    trainingForm.elements.n_estimators.value = trainingParams.n_estimators;
  }
  trainingForm.elements.max_depth.value = trainingParams.max_depth || 0;
  if (trainingParams.random_state || trainingParams.random_state === 0) {
    trainingForm.elements.random_state.value = trainingParams.random_state;
  }
}

function collectValues() {
  const values = {};
  new FormData(form).forEach((value, key) => {
    values[key] = value;
  });
  return values;
}

function collectTrainingParams() {
  const values = {};
  new FormData(trainingForm).forEach((value, key) => {
    values[key] = Number(value);
  });
  return values;
}

function updateSplitLabel() {
  const testPercent = Number(testPercentInput.value);
  testPercentLabel.textContent = `${testPercent}% prueba / ${100 - testPercent}% entrenamiento`;
}

async function predict() {
  predictButton.disabled = true;
  predictButton.textContent = "Calculando...";

  try {
    const prediction = await window.pywebview.api.predict(collectValues());
    result.classList.remove("hidden", "high-risk", "low-risk");
    result.classList.add(prediction.class_id === 1 ? "high-risk" : "low-risk");
    resultClass.textContent = prediction.class_name;
    resultDetail.textContent = `Confianza de la clase: ${prediction.confidence.toFixed(
      2
    )}%. Eficiencia del modelo: ${prediction.accuracy.toFixed(2)}%.`;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predecir clase";
  }
}

async function trainModel() {
  trainButton.disabled = true;
  trainButton.textContent = "Entrenando...";
  const params = collectTrainingParams();
  trainStatus.textContent = `Entrenando con ${100 - params.test_percent}% de entrenamiento y ${params.test_percent}% de prueba.`;

  try {
    const response = await window.pywebview.api.train_model(params);
    renderMetadata(response.metrics);
    result.classList.add("hidden");
    trainStatus.textContent = response.message;
  } catch (error) {
    trainStatus.textContent = "No se pudo entrenar el modelo. Revisa la consola.";
    throw error;
  } finally {
    trainButton.disabled = false;
    trainButton.textContent = "Entrenar de nuevo";
  }
}

function showView(viewId) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
}

window.addEventListener("pywebviewready", async () => {
  renderMetadata(await window.pywebview.api.get_metadata());
});

predictButton.addEventListener("click", predict);
trainButton.addEventListener("click", trainModel);
testPercentInput.addEventListener("input", updateSplitLabel);
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});
