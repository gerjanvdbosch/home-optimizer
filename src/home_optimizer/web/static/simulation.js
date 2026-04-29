const {
  apiUrl,
  buildConstantSeries,
  formatSigned,
  latestPoint,
  localInputToIso,
  renderPlot,
  toDatetimeLocalValue,
} = window.HomeOptimizer;

const trainingStartInput = document.getElementById("training-start");
const trainingEndInput = document.getElementById("training-end");
const trainingIntervalInput = document.getElementById("training-interval");
const trainingFractionInput = document.getElementById("training-fraction");
const trainingButton = document.getElementById("training-button");
const trainingStatus = document.getElementById("training-status");
const predictionForm = document.getElementById("prediction-form");
const predictionStartInput = document.getElementById("prediction-start");
const predictionHoursInput = document.getElementById("prediction-hours");
const predictionSetpointInput = document.getElementById("prediction-setpoint");
const predictionShutterInput = document.getElementById("prediction-shutter");
const predictionButton = document.getElementById("prediction-button");
const predictionStatus = document.getElementById("prediction-status");
const predictionSummary = document.getElementById("prediction-summary");
const predictionChart = document.getElementById("prediction-chart");

function setPredictionDefaults(date = new Date()) {
  if (!predictionStartInput) {
    return;
  }
  const start = new Date(date);
  start.setHours(24, 0, 0, 0);
  predictionStartInput.value = toDatetimeLocalValue(start);
}

function setTrainingDefaults(date = new Date()) {
  if (!trainingStartInput || !trainingEndInput) {
    return;
  }
  const end = new Date(date);
  end.setHours(0, 0, 0, 0);
  end.setDate(end.getDate() + 1);
  const start = new Date(end);
  start.setDate(start.getDate() - 3);
  trainingStartInput.value = toDatetimeLocalValue(start);
  trainingEndInput.value = toDatetimeLocalValue(end);
}

async function runPrediction(event) {
  event.preventDefault();
  if (
    !predictionStartInput ||
    !predictionHoursInput ||
    !predictionSetpointInput ||
    !predictionShutterInput ||
    !predictionButton ||
    !predictionStatus ||
    !predictionChart
  ) {
    return;
  }

  predictionButton.disabled = true;
  predictionStatus.className = "status";
  predictionStatus.textContent = "Scenario wordt doorgerekend...";

  try {
    const startDate = new Date(predictionStartInput.value);
    const hoursAhead = Number(predictionHoursInput.value);
    const endDate = new Date(startDate);
    endDate.setHours(endDate.getHours() + hoursAhead);

    const payload = {
      start_time: localInputToIso(predictionStartInput.value),
      end_time: localInputToIso(toDatetimeLocalValue(endDate)),
      thermostat_schedule: buildConstantSeries(
        "thermostat_setpoint",
        "degC",
        startDate,
        endDate,
        15,
        Number(predictionSetpointInput.value),
      ),
      shutter_schedule: buildConstantSeries(
        "shutter_living_room",
        "percent",
        startDate,
        endDate,
        15,
        Number(predictionShutterInput.value),
      ),
    };

    const response = await fetch(apiUrl("api/prediction/compare"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const responsePayload = await response.json();
    if (!response.ok) {
      throw new Error(responsePayload.detail || "Voorspelling ophalen mislukt.");
    }

    renderPlot(
      predictionChart,
      [responsePayload.predicted_room_temperature, responsePayload.actual_room_temperature],
      {
        colors: ["#00796b", "#ef6c00"],
        emptyText: "Geen voorspelling of meting beschikbaar",
        yTitle: responsePayload.predicted_room_temperature.unit || "",
        traceOptions: [
          { label: "Voorspeld", precision: 2 },
          { label: "Gemeten", precision: 2, dash: "dot" },
        ],
      },
    );

    predictionStatus.className = "status success";
    predictionStatus.textContent = "Scenario vergeleken met metingen.";

    if (predictionSummary) {
      const predicted = latestPoint(responsePayload.predicted_room_temperature);
      const actual = latestPoint(responsePayload.actual_room_temperature);
      const metricParts = [];
      if (responsePayload.rmse !== null) metricParts.push(`RMSE ${responsePayload.rmse.toFixed(2)}`);
      if (responsePayload.bias !== null) metricParts.push(`bias ${formatSigned(responsePayload.bias, 2)}`);
      if (responsePayload.max_absolute_error !== null) {
        metricParts.push(`max fout ${responsePayload.max_absolute_error.toFixed(2)}`);
      }

      if (predicted && actual) {
        const delta = predicted.value - actual.value;
        predictionSummary.textContent =
          `eindpunt voorspeld ${predicted.value.toFixed(1)} ${responsePayload.predicted_room_temperature.unit || ""}` +
          ` · gemeten ${actual.value.toFixed(1)} ${responsePayload.actual_room_temperature.unit || ""}` +
          ` · delta ${formatSigned(delta, 1)}` +
          (metricParts.length > 0 ? ` · ${metricParts.join(" · ")}` : "");
      } else if (metricParts.length > 0) {
        predictionSummary.textContent = metricParts.join(" · ");
      } else {
        predictionSummary.textContent = "-";
      }
    }
  } catch (error) {
    predictionStatus.className = "status error";
    predictionStatus.textContent = error instanceof Error ? error.message : "Voorspelling mislukt.";
    if (predictionSummary) {
      predictionSummary.textContent = "-";
    }
  } finally {
    predictionButton.disabled = false;
  }
}

async function runTraining() {
  if (
    !trainingStartInput ||
    !trainingEndInput ||
    !trainingIntervalInput ||
    !trainingFractionInput ||
    !trainingButton ||
    !trainingStatus
  ) {
    return;
  }

  trainingButton.disabled = true;
  trainingStatus.className = "status";
  trainingStatus.textContent = "Model wordt getraind...";

  try {
    const response = await fetch(apiUrl("api/identification/train"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_time: localInputToIso(trainingStartInput.value),
        end_time: localInputToIso(trainingEndInput.value),
        interval_minutes: Number(trainingIntervalInput.value),
        train_fraction: Number(trainingFractionInput.value),
      }),
    });
    const responsePayload = await response.json();
    if (!response.ok) {
      throw new Error(responsePayload.detail || "Modeltraining mislukt.");
    }

    trainingStatus.className = "status success";
    trainingStatus.textContent =
      `Model opgeslagen. 1-step test RMSE: ${responsePayload.test_rmse.toFixed(3)} · recursive test RMSE: ${responsePayload.test_rmse_recursive.toFixed(3)}`;
  } catch (error) {
    trainingStatus.className = "status error";
    trainingStatus.textContent = error instanceof Error ? error.message : "Modeltraining mislukt.";
  } finally {
    trainingButton.disabled = false;
  }
}

predictionForm?.addEventListener("submit", runPrediction);
trainingButton?.addEventListener("click", runTraining);
window.addEventListener("resize", () => {
  if (predictionChart) {
    Plotly.Plots.resize(predictionChart);
  }
});

setPredictionDefaults();
setTrainingDefaults();
