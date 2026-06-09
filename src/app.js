const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const statusBox = document.querySelector("#model-status");
const statusDot = document.querySelector("#model-dot");
const submitButton = document.querySelector("#submit-button");
const progressLabel = document.querySelector("#form-progress");

const numericFields = new Set(["age", "trestbps", "chol", "thalach", "oldpeak"]);
const toneLabels = {
  positive: "Low concern",
  warning: "Needs attention",
  critical: "Urgent review",
};

function readFormData() {
  const data = new FormData(form);
  const payload = {};

  for (const [key, value] of data.entries()) {
    payload[key] = numericFields.has(key) ? Number(value) : Number.parseInt(value, 10);
  }

  return payload;
}

function setResult(html, tone = "") {
  result.className = tone;
  result.innerHTML = html;
}

function setStatus(text, tone) {
  statusBox.textContent = text;
  statusDot.className = `status-dot ${tone}`;
}

function fieldErrors(fields = {}) {
  return Object.entries(fields)
    .map(([name, message]) => `<li><strong>${name}</strong>: ${message}</li>`)
    .join("");
}

function updateProgress() {
  const controls = [...form.querySelectorAll("input, select")];
  const complete = controls.filter((control) => String(control.value).trim() !== "").length;
  progressLabel.textContent = `${complete} of ${controls.length} complete`;
}

function resultTemplate(data) {
  const riskValue = Math.max(0, Math.min(100, data.probabilityPercent));
  const tone = data.risk.tone;
  const label = toneLabels[tone] || "Risk signal";
  const modelBadge =
    data.model?.mode === "demo"
      ? '<span class="model-badge">Demo model</span>'
      : '<span class="model-badge trained">Trained model</span>';

  return `
    <div class="risk-meter" style="--risk: ${riskValue}">
      <div>
        <strong>${riskValue}%</strong>
        <span>probability</span>
      </div>
    </div>
    <div class="result-copy">
      <div class="result-heading">
        <span>${label}</span>
        ${modelBadge}
      </div>
      <h2>${data.risk.level}</h2>
      <p>${data.risk.recommendation}</p>
      <p class="small">${data.disclaimer}</p>
    </div>
  `;
}

async function checkModelStatus() {
  try {
    const response = await fetch("/api/predict");
    const data = await response.json();

    if (data.status === "ready") {
      if (data.model?.mode === "demo") {
        setStatus("Demo model loaded. Synthetic predictions only.", "demo");
      } else {
        setStatus("Production model loaded. API ready.", "ready");
      }
      return;
    }

    setStatus("Model artifacts missing. Add model.pkl and scaler.pkl.", "missing");
  } catch {
    setStatus("Prediction API is unreachable.", "missing");
  }
}

form.addEventListener("input", updateProgress);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Analyzing";

  setResult(
    `
      <div class="risk-meter loading" style="--risk: 45">
        <div>
          <strong>...</strong>
          <span>analyzing</span>
        </div>
      </div>
      <h2>Running prediction</h2>
      <p>Validating inputs and querying the model.</p>
    `,
    "loading"
  );

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(readFormData()),
    });
    const data = await response.json();

    if (!response.ok) {
      const details =
        data.fields && Object.keys(data.fields).length > 0
          ? `<ul class="error-list">${fieldErrors(data.fields)}</ul>`
          : `<p>${data.nextStep || data.message || "Prediction failed."}</p>`;

      setResult(
        `
          <div class="error-icon">!</div>
          <h2>${data.error || "Request failed"}</h2>
          <p>${data.message || ""}</p>
          ${details}
        `,
        "error"
      );
      return;
    }

    setResult(resultTemplate(data), data.risk.tone);
  } catch {
    setResult(
      `
        <div class="error-icon">!</div>
        <h2>Network error</h2>
        <p>The app could not reach the prediction API.</p>
      `,
      "error"
    );
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Predict risk";
  }
});

updateProgress();
checkModelStatus();
