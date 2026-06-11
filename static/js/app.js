let currentResult = null;
let coefChartInstance = null;
let rocChartInstance = null;
let sigmoidChartInstance = null;

const els = {
  navButtons: document.querySelectorAll(".nav-btn"),
  phases: document.querySelectorAll(".phase"),
  statusBadge: document.getElementById("statusBadge"),
  fileInput: document.getElementById("fileInput"),
  uploadZone: document.querySelector(".upload-zone"),
  uploadInfo: document.getElementById("uploadInfo"),
  trainPanel: document.getElementById("trainPanel"),
  targetSelect: document.getElementById("targetSelect"),
  testSize: document.getElementById("testSize"),
  testSizeLabel: document.getElementById("testSizeLabel"),
  featuresInfo: document.getElementById("featuresInfo"),
  validationMsg: document.getElementById("validationMsg"),
  trainBtn: document.getElementById("trainBtn"),
  trainResults: document.getElementById("trainResults"),
  trainMetrics: document.getElementById("trainMetrics"),
  coefTable: document.querySelector("#coefTable tbody"),
  interceptInfo: document.getElementById("interceptInfo"),
  phase4Empty: document.getElementById("phase4Empty"),
  phase4Content: document.getElementById("phase4Content"),
  validationMetrics: document.getElementById("validationMetrics"),
  confusionMatrix: document.getElementById("confusionMatrix"),
  classReport: document.getElementById("classReport"),
  predictForm: document.getElementById("predictForm"),
  predictResult: document.getElementById("predictResult"),
  toast: document.getElementById("toast"),
};

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.style.background = isError ? "#b91c1c" : "#111827";
  els.toast.classList.remove("hidden");
  setTimeout(() => els.toast.classList.add("hidden"), 3200);
}

function setStatus(type, text) {
  els.statusBadge.className = `status-badge ${type}`;
  els.statusBadge.textContent = text;
}

function switchPhase(phaseId) {
  els.phases.forEach((section) => section.classList.remove("active"));
  els.navButtons.forEach((btn) => btn.classList.remove("active"));

  document.getElementById(`phase-${phaseId}`).classList.add("active");
  document.querySelector(`.nav-btn[data-phase="${phaseId}"]`).classList.add("active");
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderMetricCards(container, metrics, extra = {}) {
  const items = [
    ["Accuracy", metrics.accuracy],
    ["Precision", metrics.precision],
    ["Recall", metrics.recall],
    ["F1-Score", metrics.f1],
  ];

  if (metrics.roc_auc != null) {
    items.push(["ROC-AUC", metrics.roc_auc]);
  }

  container.innerHTML = items
    .map(
      ([label, value]) => `
      <div class="metric-card">
        <span>${label}</span>
        <strong>${pct(value)}</strong>
      </div>`
    )
    .join("");

  if (extra.train_size != null) {
    container.innerHTML += `
      <div class="metric-card">
        <span>Entrenamiento</span>
        <strong>${extra.train_size}</strong>
      </div>
      <div class="metric-card">
        <span>Prueba</span>
        <strong>${extra.test_size}</strong>
      </div>`;
  }
}

function renderSigmoidChart() {
  const ctx = document.getElementById("sigmoidChart");
  const xs = [];
  const ys = [];

  for (let x = -6; x <= 6; x += 0.1) {
    xs.push(Number(x.toFixed(1)));
    ys.push(1 / (1 + Math.exp(-x)));
  }

  if (sigmoidChartInstance) sigmoidChartInstance.destroy();

  sigmoidChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: xs,
      datasets: [
        {
          label: "Sigmoide",
          data: ys,
          borderColor: "#1f6feb",
          backgroundColor: "rgba(31, 111, 235, 0.08)",
          fill: true,
          pointRadius: 0,
          borderWidth: 3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          title: { display: true, text: "Score lineal (β₀ + βX)" },
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: "Probabilidad P(Y=1)" },
        },
      },
    },
  });
}

async function validateTarget() {
  const target = els.targetSelect.value;
  if (!target) return;

  const response = await fetch("/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_column: target }),
  });

  const data = await response.json();
  els.validationMsg.classList.remove("hidden", "ok", "error");
  els.validationMsg.classList.add(data.ok ? "ok" : "error");
  els.validationMsg.textContent = data.message;
  els.featuresInfo.textContent = `Variables predictoras: ${data.features.join(", ")}`;
  els.trainBtn.disabled = !data.ok;
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await response.json();

  if (!data.ok) {
    showToast(data.message, true);
    return;
  }

  setStatus("loaded", "Dataset cargado");
  els.uploadInfo.classList.remove("hidden");
  els.uploadInfo.textContent = `${data.filename} — ${data.rows} filas, ${data.columns.length} columnas`;
  els.trainPanel.classList.remove("hidden");

  els.targetSelect.innerHTML = data.columns
    .map((col) => `<option value="${col}">${col}</option>`)
    .join("");
  els.targetSelect.value = data.default_target;

  await validateTarget();
  showToast("Dataset cargado correctamente");
}

async function trainModel() {
  els.trainBtn.disabled = true;
  els.trainBtn.textContent = "Entrenando...";

  const response = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target_column: els.targetSelect.value,
      test_size: Number(els.testSize.value) / 100,
      random_state: 42,
    }),
  });

  const data = await response.json();
  els.trainBtn.textContent = "Entrenar modelo";

  if (!data.ok) {
    showToast(data.message, true);
    els.trainBtn.disabled = false;
    return;
  }

  currentResult = data.result;
  setStatus("trained", "Modelo entrenado");
  renderTrainingResults(currentResult);
  renderValidationSection(currentResult);
  showToast("Modelo entrenado correctamente");
  els.trainBtn.disabled = false;
}

function renderTrainingResults(result) {
  els.trainResults.classList.remove("hidden");
  renderMetricCards(els.trainMetrics, result.metrics, result);

  els.coefTable.innerHTML = result.coefficients
    .map(
      (row) => `
      <tr>
        <td>${row.Variable}</td>
        <td>${row.Coeficiente.toFixed(4)}</td>
        <td>${row["Odds Ratio (exp(coef))"].toFixed(4)}</td>
      </tr>`
    )
    .join("");

  els.interceptInfo.textContent = `Intercepto (β₀): ${result.intercept.toFixed(4)}`;

  const ctx = document.getElementById("coefChart");
  if (coefChartInstance) coefChartInstance.destroy();

  coefChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: result.coefficients.map((r) => r.Variable),
      datasets: [
        {
          label: "Coeficiente",
          data: result.coefficients.map((r) => r.Coeficiente),
          backgroundColor: result.coefficients.map((r) =>
            r.Coeficiente >= 0 ? "#1f6feb" : "#ef4444"
          ),
        },
      ],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { grid: { color: "#eef2f7" } } },
    },
  });
}

function renderConfusionMatrix(matrix) {
  const [[tn, fp], [fn, tp]] = matrix;
  els.confusionMatrix.innerHTML = `
    <div class="confusion-cell cell-tn">${tn}<small>TN · Real 0, Pred 0</small></div>
    <div class="confusion-cell cell-fp">${fp}<small>FP · Real 0, Pred 1</small></div>
    <div class="confusion-cell cell-fn">${fn}<small>FN · Real 1, Pred 0</small></div>
    <div class="confusion-cell cell-tp">${tp}<small>TP · Real 1, Pred 1</small></div>`;
}

function renderValidationSection(result) {
  els.phase4Empty.classList.add("hidden");
  els.phase4Content.classList.remove("hidden");

  renderMetricCards(els.validationMetrics, result.metrics);
  renderConfusionMatrix(result.confusion);
  els.classReport.textContent = result.report;

  const ctx = document.getElementById("rocChart");
  if (rocChartInstance) rocChartInstance.destroy();

  if (result.roc_fpr.length > 0) {
    rocChartInstance = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "ROC",
            data: result.roc_fpr.map((x, i) => ({ x, y: result.roc_tpr[i] })),
            borderColor: "#16a34a",
            pointRadius: 0,
            borderWidth: 3,
          },
          {
            label: "Azar",
            data: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ],
            borderColor: "#94a3b8",
            borderDash: [6, 6],
            pointRadius: 0,
          },
        ],
      },
      options: {
        parsing: false,
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "Tasa de falsos positivos" },
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "Tasa de verdaderos positivos" },
          },
        },
      },
    });
  }

  els.predictForm.innerHTML = result.feature_names
    .map(
      (name) => `
      <div>
        <label for="field-${name}">${name}</label>
        <input type="number" step="any" id="field-${name}" name="${name}" value="${result.medians[name].toFixed(4)}" required>
      </div>`
    )
    .join("");

  els.predictResult.classList.add("hidden");
}

async function submitPrediction(event) {
  event.preventDefault();
  if (!currentResult) return;

  const values = {};
  currentResult.feature_names.forEach((name) => {
    values[name] = document.getElementById(`field-${name}`).value;
  });

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });

  const data = await response.json();
  if (!data.ok) {
    showToast(data.message, true);
    return;
  }

  els.predictResult.classList.remove("hidden");
  els.predictResult.innerHTML = `
    <strong>Clase predicha:</strong> ${data.predicted_class}<br>
    <strong>Probabilidad clase 1:</strong> ${pct(data.probability_class_1)}<br>
    <strong>Probabilidad clase 0:</strong> ${pct(data.probability_class_0)}`;
}

els.navButtons.forEach((btn) => {
  btn.addEventListener("click", () => switchPhase(btn.dataset.phase));
});

document.querySelectorAll(".link-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchPhase(btn.dataset.phase));
});

els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) uploadFile(file);
});

els.uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.uploadZone.classList.add("dragover");
});

els.uploadZone.addEventListener("dragleave", () => {
  els.uploadZone.classList.remove("dragover");
});

els.uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.uploadZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

els.targetSelect.addEventListener("change", validateTarget);
els.testSize.addEventListener("input", () => {
  els.testSizeLabel.textContent = `${els.testSize.value}%`;
});
els.trainBtn.addEventListener("click", trainModel);
els.predictForm.addEventListener("submit", submitPrediction);

renderSigmoidChart();
