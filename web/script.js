"use strict";

// ─── State ───────────────────────────────────────────────────────────────────
let currentAlgo = "rf"; // "rf" | "knn"
let rfMetadata = null;
let knnMetadata = null;
let knnLoaded = false;

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const algoBtns = document.querySelectorAll(".algo-btn");
const tabButtons = document.querySelectorAll(".tab-button");

// RF
const rfTrainBtn = document.querySelector("#rf-train-button");
const rfPredictBtn = document.querySelector("#rf-predict-button");
const rfTrainStatus = document.querySelector("#rf-train-status");
const rfTestPercentInput = document.querySelector("#rf-test-percent");
const rfTestPercentLabel = document.querySelector("#rf-test-percent-label");
const rfPredictionForm = document.querySelector("#rf-prediction-form");
const rfResult = document.querySelector("#rf-result");

// KNN
const knnTrainBtn = document.querySelector("#knn-train-button");
const knnPredictBtn = document.querySelector("#knn-predict-button");
const knnTrainStatus = document.querySelector("#knn-train-status");
const knnTestPercentInput = document.querySelector("#knn-test-percent");
const knnTestPercentLabel = document.querySelector("#knn-test-percent-label");
const knnPredictionForm = document.querySelector("#knn-prediction-form");
const knnResult = document.querySelector("#knn-result");

// Chart modal
const chartModal = document.querySelector("#chart-modal");
const chartModalImage = document.querySelector("#chart-modal-image");
const chartModalTitle = document.querySelector("#chart-modal-title");
const chartModalClose = document.querySelector("#chart-modal-close");

// ─── Helpers ──────────────────────────────────────────────────────────────────
function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function setText(id, value) {
  const el = document.querySelector(id);
  if (el) el.textContent = value;
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
    if (field.name === "loan_percent_income") {
      input.readOnly = true;
      input.dataset.calculated = "true";
      input.title = "Se calcula automáticamente con monto del préstamo / ingreso anual.";
    }
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

function collectFormValues(form) {
  syncLoanPercentIncome(form);
  const values = {};
  new FormData(form).forEach((value, key) => {
    values[key] = value;
  });
  return values;
}

function syncLoanPercentIncome(form) {
  const incomeInput = form.elements.person_income;
  const loanAmountInput = form.elements.loan_amnt;
  const percentInput = form.elements.loan_percent_income;
  if (!incomeInput || !loanAmountInput || !percentInput) return;

  const income = Number(incomeInput.value);
  const loanAmount = Number(loanAmountInput.value);
  if (!Number.isFinite(income) || !Number.isFinite(loanAmount) || income <= 0) return;

  percentInput.value = (loanAmount / income).toFixed(4);
}

function enableCalculatedLoanPercent(form) {
  const incomeInput = form.elements.person_income;
  const loanAmountInput = form.elements.loan_amnt;
  const percentInput = form.elements.loan_percent_income;
  if (!incomeInput || !loanAmountInput || !percentInput) return;

  const update = () => syncLoanPercentIncome(form);
  incomeInput.addEventListener("input", update);
  loanAmountInput.addEventListener("input", update);
  syncLoanPercentIncome(form);
}

function collectNumberFormValues(form) {
  const values = {};
  new FormData(form).forEach((value, key) => {
    values[key] = Number(value);
  });
  return values;
}

function setResult(resultEl, prediction) {
  resultEl.classList.remove("hidden", "pending", "high-risk", "low-risk");
  resultEl.classList.add(prediction.class_id === 1 ? "high-risk" : "low-risk");
  resultEl.querySelector("[id$=result-label]").textContent = "Clase predicha";
  resultEl.querySelector("[id$=result-class]").textContent = prediction.class_name;
  resultEl.querySelector("[id$=result-detail]").textContent =
    `Confianza de la clase: ${prediction.confidence.toFixed(2)}%. ` +
    `Eficiencia del modelo: ${prediction.accuracy.toFixed(2)}%.`;
}

function resetResult(resultEl) {
  resultEl.classList.remove("high-risk", "low-risk");
  resultEl.classList.add("pending");
  resultEl.querySelector("[id$=result-label]").textContent = "Resultado";
  resultEl.querySelector("[id$=result-class]").textContent = "Pendiente";
  resultEl.querySelector("[id$=result-detail]").textContent = "Sin predicción calculada.";
}

function openChartModal(image) {
  if (!image.src) return;
  const title = image.closest(".chart-card")?.querySelector("h2")?.textContent || image.alt || "Gráfica";
  chartModalTitle.textContent = title;
  chartModalImage.src = image.src;
  chartModalImage.alt = image.alt || title;
  chartModal.classList.remove("hidden");
  chartModalClose.focus();
}

function closeChartModal() {
  chartModal.classList.add("hidden");
  chartModalImage.removeAttribute("src");
}

// ─── RF metadata ─────────────────────────────────────────────────────────────
function renderRfMetadata(data) {
  rfMetadata = data;
  const tp = data.training_params || {};
  const cleaning = data.cleaning || {};
  setText("#accuracy", currentAlgo === "rf" ? percent(data.accuracy) : document.querySelector("#accuracy").textContent);
  setText("#rf-algorithm", data.algorithm);
  setText("#rf-target", data.target_column);
  setText("#rf-rows", data.rows.toLocaleString("es-MX"));
  setText("#rf-removed-rows", (cleaning.removed_rows || 0).toLocaleString("es-MX"));
  setText("#rf-train-rows", data.train_rows.toLocaleString("es-MX"));
  setText("#rf-test-rows", data.test_rows.toLocaleString("es-MX"));
  setText("#rf-split-info", `${tp.train_percent || 80}% / ${tp.test_percent || 20}%`);
  setText("#rf-trees-info", tp.n_estimators || 120);
  setText("#rf-depth-info", tp.max_depth || "Sin límite");
  setText("#rf-seed-info", tp.random_state ?? 42);
  setText("#rf-precision", percent(data.precision));
  setText("#rf-recall", percent(data.recall));
  setText("#rf-f1", percent(data.f1));

  const bust = `?v=${Date.now()}`;
  document.querySelector("#rf-confusion-chart").src = data.charts.confusion_matrix + bust;
  document.querySelector("#rf-importance-chart").src = data.charts.feature_importance + bust;
  document.querySelector("#rf-distribution-chart").src = data.charts.class_distribution + bust;
  document.querySelector("#rf-metrics-chart").src = data.charts.model_metrics + bust;

  rfPredictionForm.innerHTML = "";
  data.fields.forEach((field) => rfPredictionForm.appendChild(buildField(field)));
  enableCalculatedLoanPercent(rfPredictionForm);

  if (tp.test_percent) {
    rfTestPercentInput.value = tp.test_percent;
    updateRfSplitLabel();
  }
  const rfForm = document.querySelector("#rf-training-form");
  if (tp.n_estimators) rfForm.elements.n_estimators.value = tp.n_estimators;
  rfForm.elements.max_depth.value = tp.max_depth || 0;
  if (tp.random_state || tp.random_state === 0) rfForm.elements.random_state.value = tp.random_state;

  if (currentAlgo === "rf") {
    setText("#accuracy", percent(data.accuracy));
  }
}

// ─── KNN metadata ─────────────────────────────────────────────────────────────
function renderKnnMetadata(data) {
  knnMetadata = data;
  const tp = data.training_params || {};
  const cleaning = data.cleaning || {};
  setText("#knn-algorithm", data.algorithm);
  setText("#knn-target", data.target_column);
  setText("#knn-rows", data.rows.toLocaleString("es-MX"));
  setText("#knn-removed-rows", (cleaning.removed_rows || 0).toLocaleString("es-MX"));
  setText("#knn-train-rows", data.train_rows.toLocaleString("es-MX"));
  setText("#knn-test-rows", data.test_rows.toLocaleString("es-MX"));
  setText("#knn-split-info", `${tp.train_percent || 80}% / ${tp.test_percent || 20}%`);
  setText("#knn-neighbors-info", tp.n_neighbors || 7);
  setText("#knn-metric-info", tp.metric || "minkowski");
  setText("#knn-weights-info", tp.weights === "distance" ? "Por distancia" : "Uniforme");
  setText("#knn-algo-info", tp.algorithm || "auto");
  setText("#knn-seed-info", tp.random_state ?? 42);
  setText("#knn-precision", percent(data.precision));
  setText("#knn-recall", percent(data.recall));
  setText("#knn-f1", percent(data.f1));

  const bust = `?v=${Date.now()}`;
  document.querySelector("#knn-confusion-chart").src = data.charts.confusion_matrix + bust;
  document.querySelector("#knn-importance-chart").src = data.charts.feature_importance + bust;
  document.querySelector("#knn-distribution-chart").src = data.charts.class_distribution + bust;
  document.querySelector("#knn-metrics-chart").src = data.charts.model_metrics + bust;

  knnPredictionForm.innerHTML = "";
  data.fields.forEach((field) => knnPredictionForm.appendChild(buildField(field)));
  enableCalculatedLoanPercent(knnPredictionForm);

  const knnForm = document.querySelector("#knn-training-form");
  if (tp.n_neighbors) knnForm.elements.n_neighbors.value = tp.n_neighbors;
  if (tp.metric) knnForm.elements.metric.value = tp.metric;
  if (tp.weights) knnForm.elements.weights.value = tp.weights;
  if (tp.algorithm) knnForm.elements.algorithm.value = tp.algorithm;
  if (tp.test_percent) {
    knnTestPercentInput.value = tp.test_percent;
    updateKnnSplitLabel();
  }
  if (tp.random_state || tp.random_state === 0) knnForm.elements.random_state.value = tp.random_state;

  if (currentAlgo === "knn") {
    setText("#accuracy", percent(data.accuracy));
  }
}

// ─── Split label updaters ─────────────────────────────────────────────────────
function updateRfSplitLabel() {
  const v = Number(rfTestPercentInput.value);
  rfTestPercentLabel.textContent = `${v}% prueba / ${100 - v}% entrenamiento`;
}

function updateKnnSplitLabel() {
  const v = Number(knnTestPercentInput.value);
  knnTestPercentLabel.textContent = `${v}% prueba / ${100 - v}% entrenamiento`;
}

// ─── RF actions ───────────────────────────────────────────────────────────────
async function trainRf() {
  rfTrainBtn.disabled = true;
  rfTrainBtn.textContent = "Entrenando…";
  const params = collectNumberFormValues(document.querySelector("#rf-training-form"));
  rfTrainStatus.textContent = `Entrenando con ${100 - params.test_percent}% entrenamiento y ${params.test_percent}% prueba…`;
  try {
    const response = await window.pywebview.api.train_model(params);
    renderRfMetadata(response.metrics);
    resetResult(rfResult);
    rfTrainStatus.textContent = response.message;
  } catch (err) {
    rfTrainStatus.textContent = "No se pudo entrenar el modelo. Revisa la consola.";
    throw err;
  } finally {
    rfTrainBtn.disabled = false;
    rfTrainBtn.textContent = "Entrenar de nuevo";
  }
}

async function predictRf() {
  rfPredictBtn.disabled = true;
  rfPredictBtn.textContent = "Calculando…";
  try {
    const prediction = await window.pywebview.api.predict(collectFormValues(rfPredictionForm));
    setResult(rfResult, prediction);
  } finally {
    rfPredictBtn.disabled = false;
    rfPredictBtn.textContent = "Predecir clase";
  }
}

// ─── KNN actions ──────────────────────────────────────────────────────────────
async function trainKnn() {
  knnTrainBtn.disabled = true;
  knnTrainBtn.textContent = "Entrenando…";
  const form = document.querySelector("#knn-training-form");
  const params = {};
  new FormData(form).forEach((v, k) => {
    // n_neighbors, test_percent, random_state are numbers; rest are strings
    if (["n_neighbors", "test_percent", "random_state"].includes(k)) {
      params[k] = Number(v);
    } else {
      params[k] = v;
    }
  });
  knnTrainStatus.textContent = `Entrenando con k=${params.n_neighbors}, métrica=${params.metric}…`;
  try {
    const response = await window.pywebview.api.train_knn_model(params);
    renderKnnMetadata(response.metrics);
    resetResult(knnResult);
    knnTrainStatus.textContent = response.message;
  } catch (err) {
    knnTrainStatus.textContent = "No se pudo entrenar el modelo KNN. Revisa la consola.";
    throw err;
  } finally {
    knnTrainBtn.disabled = false;
    knnTrainBtn.textContent = "Entrenar de nuevo";
  }
}

async function predictKnn() {
  knnPredictBtn.disabled = true;
  knnPredictBtn.textContent = "Calculando…";
  try {
    const prediction = await window.pywebview.api.predict_knn(collectFormValues(knnPredictionForm));
    setResult(knnResult, prediction);
  } finally {
    knnPredictBtn.disabled = false;
    knnPredictBtn.textContent = "Predecir clase";
  }
}

// ─── Algorithm switcher ───────────────────────────────────────────────────────
function switchAlgo(algo) {
  currentAlgo = algo;

  algoBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.algo === algo));

  // Toggle training sections
  document.querySelectorAll(".algo-section").forEach((sec) => {
    const isRf = sec.id.startsWith("rf-");
    const isKnn = sec.id.startsWith("knn-");
    if (algo === "rf") sec.classList.toggle("active", isRf);
    else sec.classList.toggle("active", isKnn);
  });

  // Update accuracy badge
  if (algo === "rf" && rfMetadata) {
    setText("#accuracy", percent(rfMetadata.accuracy));
  } else if (algo === "knn" && knnMetadata) {
    setText("#accuracy", percent(knnMetadata.accuracy));
  } else if (algo === "knn" && !knnLoaded) {
    setText("#accuracy", "--%");
  }

  // Lazy-load KNN model on first switch
  if (algo === "knn" && !knnLoaded) {
    knnLoaded = true;
    knnTrainStatus.textContent = "Cargando modelo KNN…";
    window.pywebview.api.get_knn_metadata().then((data) => {
      renderKnnMetadata(data);
      knnTrainStatus.textContent = "Modelo KNN listo para usarse.";
    }).catch(() => {
      knnTrainStatus.textContent = "Error al cargar el modelo KNN.";
    });
  }
}

// ─── Tab switcher ─────────────────────────────────────────────────────────────
function showView(viewId) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });
  tabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewId);
  });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
window.addEventListener("pywebviewready", async () => {
  const data = await window.pywebview.api.get_metadata();
  renderRfMetadata(data);
  setText("#accuracy", percent(data.accuracy));
});

// ─── Event listeners ─────────────────────────────────────────────────────────
algoBtns.forEach((btn) => btn.addEventListener("click", () => switchAlgo(btn.dataset.algo)));
tabButtons.forEach((btn) => btn.addEventListener("click", () => showView(btn.dataset.view)));

rfTrainBtn.addEventListener("click", trainRf);
rfPredictBtn.addEventListener("click", predictRf);
rfTestPercentInput.addEventListener("input", updateRfSplitLabel);

knnTrainBtn.addEventListener("click", trainKnn);
knnPredictBtn.addEventListener("click", predictKnn);
knnTestPercentInput.addEventListener("input", updateKnnSplitLabel);

document.querySelectorAll(".chart-card img").forEach((image) => {
  image.addEventListener("click", () => openChartModal(image));
});
chartModalClose.addEventListener("click", closeChartModal);
document.querySelector("[data-close-chart-modal]").addEventListener("click", closeChartModal);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !chartModal.classList.contains("hidden")) {
    closeChartModal();
  }
});
