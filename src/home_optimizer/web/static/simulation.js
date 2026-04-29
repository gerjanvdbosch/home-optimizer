const {
  apiUrl,
  buildConstantSeries,
  formatDate,
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
const predictionShutterSourceMeasured = document.getElementById("prediction-shutter-source-measured");
const predictionShutterSourceManual = document.getElementById("prediction-shutter-source-manual");
const predictionShutterHelp = document.getElementById("prediction-shutter-help");
const predictionButton = document.getElementById("prediction-button");
const predictionStatus = document.getElementById("prediction-status");
const predictionSummary = document.getElementById("prediction-summary");
const predictionChart = document.getElementById("prediction-chart");
const predictionStatDelta = document.getElementById("prediction-stat-delta");
const predictionStatRmse = document.getElementById("prediction-stat-rmse");
const predictionStatBias = document.getElementById("prediction-stat-bias");
const predictionStatMaxError = document.getElementById("prediction-stat-max-error");

function setPredictionDefaults(date = new Date()) {
  if (!predictionStartInput) {
    return;
  }
  const start = new Date(date);
  start.setHours(0, 0, 0, 0);
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

function resetPredictionStats() {
  if (predictionStatDelta) predictionStatDelta.textContent = "-";
  if (predictionStatRmse) predictionStatRmse.textContent = "-";
  if (predictionStatBias) predictionStatBias.textContent = "-";
  if (predictionStatMaxError) predictionStatMaxError.textContent = "-";
}

function updateShutterMode() {
  const useMeasuredShutters = Boolean(predictionShutterSourceMeasured?.checked);
  if (predictionShutterInput) {
    predictionShutterInput.disabled = useMeasuredShutters;
  }
  if (predictionShutterHelp) {
    predictionShutterHelp.textContent = useMeasuredShutters
      ? "De gemeten shutterreeks van de startdag wordt gebruikt voor de vergelijking."
      : "Het ingevulde shutterpercentage wordt als constant scenario over de hele horizon gebruikt.";
  }
}

async function buildPredictionShutterSchedule(startDate, endDate) {
  if (predictionShutterSourceMeasured?.checked) {
    const params = new URLSearchParams({ date: formatDate(startDate) });
    const response = await fetch(apiUrl(`api/dashboard/charts?${params.toString()}`));
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Gemeten shutterdata ophalen mislukt.");
    }
    if (!payload.shutter_position?.points?.length) {
      throw new Error("geen gemeten shutterdata beschikbaar voor de startdag");
    }
    return payload.shutter_position;
  }

  return buildConstantSeries(
    "shutter_living_room",
    "percent",
    startDate,
    endDate,
    15,
    Number(predictionShutterInput?.value || 100),
  );
}

async function runPrediction(event) {
  event.preventDefault();
  if (
    !predictionStartInput ||
    !predictionHoursInput ||
    !predictionSetpointInput ||
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
      shutter_schedule: await buildPredictionShutterSchedule(startDate, endDate),
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
      if (predicted && actual) {
        const delta = predicted.value - actual.value;
        if (predictionStatDelta) predictionStatDelta.textContent = formatSigned(delta, 1);
        if (predictionStatRmse) {
          predictionStatRmse.textContent =
            responsePayload.rmse !== null ? responsePayload.rmse.toFixed(2) : "-";
        }
        if (predictionStatBias) {
          predictionStatBias.textContent =
            responsePayload.bias !== null ? formatSigned(responsePayload.bias, 2) : "-";
        }
        if (predictionStatMaxError) {
          predictionStatMaxError.textContent =
            responsePayload.max_absolute_error !== null
              ? responsePayload.max_absolute_error.toFixed(2)
              : "-";
        }
        predictionSummary.textContent = "Vergelijking bijgewerkt";
      } else {
        if (predictionStatDelta) predictionStatDelta.textContent = "-";
        if (predictionStatRmse) {
          predictionStatRmse.textContent =
            responsePayload.rmse !== null ? responsePayload.rmse.toFixed(2) : "-";
        }
        if (predictionStatBias) {
          predictionStatBias.textContent =
            responsePayload.bias !== null ? formatSigned(responsePayload.bias, 2) : "-";
        }
        if (predictionStatMaxError) {
          predictionStatMaxError.textContent =
            responsePayload.max_absolute_error !== null
              ? responsePayload.max_absolute_error.toFixed(2)
              : "-";
        }
        predictionSummary.textContent = "Vergelijking bijgewerkt";
      }
    }
  } catch (error) {
    predictionStatus.className = "status error";
    predictionStatus.textContent = error instanceof Error ? error.message : "Voorspelling mislukt.";
    if (predictionSummary) {
      predictionSummary.textContent = "Vergelijk voorspelling met metingen";
    }
    resetPredictionStats();
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
predictionShutterSourceMeasured?.addEventListener("change", updateShutterMode);
predictionShutterSourceManual?.addEventListener("change", updateShutterMode);
window.addEventListener("resize", () => {
  if (predictionChart) {
    Plotly.Plots.resize(predictionChart);
  }
});

setPredictionDefaults();
setTrainingDefaults();
resetPredictionStats();
updateShutterMode();
