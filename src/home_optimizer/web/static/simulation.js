const {
  apiUrl,
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
const predictionSetpointSourceMeasured = document.getElementById("prediction-setpoint-source-measured");
const predictionSetpointSourceManual = document.getElementById("prediction-setpoint-source-manual");
const predictionSetpointHelp = document.getElementById("prediction-setpoint-help");
const predictionSetpointEditor = document.getElementById("prediction-setpoint-editor");
const predictionSetpointBlocks = document.getElementById("prediction-setpoint-blocks");
const predictionSetpointAddBlockButton = document.getElementById("prediction-setpoint-add-block");
const predictionSetpointCopyDayButton = document.getElementById("prediction-setpoint-copy-day");
const predictionShutterSourceMeasured = document.getElementById("prediction-shutter-source-measured");
const predictionShutterSourceManual = document.getElementById("prediction-shutter-source-manual");
const predictionShutterHelp = document.getElementById("prediction-shutter-help");
const predictionShutterEditor = document.getElementById("prediction-shutter-editor");
const predictionShutterBlocks = document.getElementById("prediction-shutter-blocks");
const predictionShutterAddBlockButton = document.getElementById("prediction-shutter-add-block");
const predictionShutterCopyDayButton = document.getElementById("prediction-shutter-copy-day");
const predictionButton = document.getElementById("prediction-button");
const predictionStatus = document.getElementById("prediction-status");
const predictionSummary = document.getElementById("prediction-summary");
const predictionChart = document.getElementById("prediction-chart");
const predictionStatDelta = document.getElementById("prediction-stat-delta");
const predictionStatRmse = document.getElementById("prediction-stat-rmse");
const predictionStatBias = document.getElementById("prediction-stat-bias");
const predictionStatMaxError = document.getElementById("prediction-stat-max-error");
const DEFAULT_SETPOINT_BLOCKS = [
  { time: "00:00", value: 18.0 },
  { time: "06:00", value: 20.5 },
  { time: "22:00", value: 18.0 },
];
const DEFAULT_SHUTTER_BLOCKS = [
  { time: "00:00", value: 100 },
  { time: "08:00", value: 4 },
  { time: "18:00", value: 100 },
];

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

function createBlockRow(container, config, { time, value }) {
  if (!container) {
    return;
  }

  const row = document.createElement("div");
  row.className = "block-editor-row";
  row.innerHTML = `
    <label>
      Tijd
      <input class="${config.timeClassName}" type="time" step="900" value="${time}" required />
    </label>
    <label>
      ${config.valueLabel}
      <input class="${config.valueClassName}" type="number" min="${config.min}" max="${config.max}" step="${config.step}" value="${value}" required />
    </label>
    <button type="button" class="secondary-button block-remove-button">Verwijder</button>
  `;

  const removeButton = row.querySelector(".block-remove-button");
  removeButton?.addEventListener("click", () => {
    row.remove();
    config.ensureRows();
  });

  container.append(row);
}

function ensureBlockRows(container, defaults, addRow) {
  if (!container?.children.length) {
    defaults.forEach(addRow);
  }
}

function readBlockPattern(container, config, defaults, addRow) {
  ensureBlockRows(container, defaults, addRow);
  const rows = Array.from(container?.querySelectorAll(".block-editor-row") || []);
  const blocks = rows.map((row) => {
    const timeInput = row.querySelector(`.${config.timeClassName}`);
    const valueInput = row.querySelector(`.${config.valueClassName}`);
    return {
      time: timeInput?.value || "00:00",
      value: Number(valueInput?.value || config.defaultValue),
    };
  });

  blocks.sort((left, right) => left.time.localeCompare(right.time));
  return blocks;
}

function buildRepeatingSchedule(name, unit, startDate, endDate, pattern) {
  const points = [];
  const dayCursor = new Date(startDate);
  dayCursor.setHours(0, 0, 0, 0);
  const lastDay = new Date(endDate);
  lastDay.setHours(0, 0, 0, 0);

  while (dayCursor <= lastDay) {
    const dayPrefix = toDatetimeLocalValue(dayCursor).slice(0, 10);
    pattern.forEach((block) => {
      points.push({
        timestamp: localInputToIso(`${dayPrefix}T${block.time}`),
        value: block.value,
      });
    });
    dayCursor.setDate(dayCursor.getDate() + 1);
  }

  return {
    name,
    unit,
    points,
  };
}

function replaceBlockRows(container, addRow, blocks) {
  if (!container) {
    return;
  }
  container.replaceChildren();
  blocks.forEach(addRow);
}

function toLocalTimeValue(timestamp) {
  const date = new Date(timestamp);
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

function roundToStep(value, step) {
  return Math.round(value / step) * step;
}

function compressSeriesToDailyBlocks(series, fallbackValue, step) {
  const blocks = [];
  let previousValue = null;

  series.points.forEach((point) => {
    const time = toLocalTimeValue(point.timestamp);
    const value = roundToStep(Number(point.value), step);
    if (previousValue === null || value !== previousValue) {
      blocks.push({ time, value });
      previousValue = value;
    }
  });

  if (!blocks.length) {
    return [{ time: "00:00", value: roundToStep(fallbackValue, step) }];
  }

  if (blocks[0].time !== "00:00") {
    blocks.unshift({ time: "00:00", value: blocks[0].value });
  }

  return blocks;
}

function createSetpointBlockRow(block) {
  createBlockRow(
    predictionSetpointBlocks,
    {
      timeClassName: "setpoint-block-time",
      valueClassName: "setpoint-block-value",
      valueLabel: "Setpoint",
      min: "5",
      max: "30",
      step: "0.1",
      ensureRows: ensureSetpointBlockRows,
    },
    block,
  );
}

function createShutterBlockRow(block) {
  createBlockRow(
    predictionShutterBlocks,
    {
      timeClassName: "shutter-block-time",
      valueClassName: "shutter-block-value",
      valueLabel: "Shutter (%)",
      min: "0",
      max: "100",
      step: "1",
      ensureRows: ensureShutterBlockRows,
    },
    block,
  );
}

function ensureSetpointBlockRows() {
  ensureBlockRows(predictionSetpointBlocks, DEFAULT_SETPOINT_BLOCKS, createSetpointBlockRow);
}

function ensureShutterBlockRows() {
  ensureBlockRows(predictionShutterBlocks, DEFAULT_SHUTTER_BLOCKS, createShutterBlockRow);
}

function buildSetpointBlockPattern() {
  return readBlockPattern(
    predictionSetpointBlocks,
    {
      timeClassName: "setpoint-block-time",
      valueClassName: "setpoint-block-value",
      defaultValue: 18,
    },
    DEFAULT_SETPOINT_BLOCKS,
    createSetpointBlockRow,
  );
}

function buildShutterBlockPattern() {
  return readBlockPattern(
    predictionShutterBlocks,
    {
      timeClassName: "shutter-block-time",
      valueClassName: "shutter-block-value",
      defaultValue: 100,
    },
    DEFAULT_SHUTTER_BLOCKS,
    createShutterBlockRow,
  );
}

function buildManualSetpointSchedule(startDate, endDate) {
  return buildRepeatingSchedule(
    "thermostat_setpoint",
    "degC",
    startDate,
    endDate,
    buildSetpointBlockPattern(),
  );
}

function buildManualShutterSchedule(startDate, endDate) {
  return buildRepeatingSchedule(
    "shutter_living_room",
    "percent",
    startDate,
    endDate,
    buildShutterBlockPattern(),
  );
}

function updateShutterMode() {
  const useMeasuredShutters = Boolean(predictionShutterSourceMeasured?.checked);
  if (predictionShutterEditor) {
    predictionShutterEditor.hidden = useMeasuredShutters;
  }
  if (predictionShutterHelp) {
    predictionShutterHelp.textContent = useMeasuredShutters
      ? "De gemeten shutterreeks van de startdag wordt gebruikt voor de vergelijking."
      : "Gebruik hieronder tijdblokken voor een handmatig shutterprofiel; dit patroon wordt per dag herhaald.";
  }
}

function updateSetpointMode() {
  const useMeasuredSetpoints = Boolean(predictionSetpointSourceMeasured?.checked);
  if (predictionSetpointEditor) {
    predictionSetpointEditor.hidden = useMeasuredSetpoints;
  }
  if (predictionSetpointHelp) {
    predictionSetpointHelp.textContent = useMeasuredSetpoints
      ? "De gemeten setpointreeks van de startdag wordt gebruikt voor de vergelijking."
      : "Gebruik hieronder tijdblokken voor een handmatig setpointprofiel; dit patroon wordt per dag herhaald.";
  }
}

async function fetchDayChartsPayload(startDate) {
  const params = new URLSearchParams({ date: formatDate(startDate) });
  const response = await fetch(apiUrl(`api/dashboard/charts?${params.toString()}`));
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Dagdata ophalen mislukt.");
  }
  return payload;
}

async function copyMeasuredSetpointDay() {
  if (!predictionStartInput?.value) {
    return;
  }

  const payload = await fetchDayChartsPayload(new Date(predictionStartInput.value));
  if (!payload.thermostat_setpoint?.points?.length) {
    throw new Error("geen gemeten setpointdata beschikbaar voor de startdag");
  }

  replaceBlockRows(
    predictionSetpointBlocks,
    createSetpointBlockRow,
    compressSeriesToDailyBlocks(payload.thermostat_setpoint, 18, 0.1),
  );
  if (predictionSetpointSourceManual) {
    predictionSetpointSourceManual.checked = true;
  }
  updateSetpointMode();
}

async function copyMeasuredShutterDay() {
  if (!predictionStartInput?.value) {
    return;
  }

  const payload = await fetchDayChartsPayload(new Date(predictionStartInput.value));
  if (!payload.shutter_position?.points?.length) {
    throw new Error("geen gemeten shutterdata beschikbaar voor de startdag");
  }

  replaceBlockRows(
    predictionShutterBlocks,
    createShutterBlockRow,
    compressSeriesToDailyBlocks(payload.shutter_position, 100, 1),
  );
  if (predictionShutterSourceManual) {
    predictionShutterSourceManual.checked = true;
  }
  updateShutterMode();
}

async function buildPredictionShutterSchedule(startDate, endDate) {
  if (predictionShutterSourceMeasured?.checked) {
    const payload = await fetchDayChartsPayload(startDate);
    if (!payload.shutter_position?.points?.length) {
      throw new Error("geen gemeten shutterdata beschikbaar voor de startdag");
    }
    return payload.shutter_position;
  }

  return buildManualShutterSchedule(startDate, endDate);
}

async function buildPredictionSetpointSchedule(startDate, endDate) {
  if (predictionSetpointSourceMeasured?.checked) {
    const payload = await fetchDayChartsPayload(startDate);
    if (!payload.thermostat_setpoint?.points?.length) {
      throw new Error("geen gemeten setpointdata beschikbaar voor de startdag");
    }
    return payload.thermostat_setpoint;
  }

  return buildManualSetpointSchedule(startDate, endDate);
}

async function runPrediction(event) {
  event.preventDefault();
  if (
    !predictionStartInput ||
    !predictionHoursInput ||
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
      thermostat_schedule: await buildPredictionSetpointSchedule(startDate, endDate),
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
predictionSetpointSourceMeasured?.addEventListener("change", updateSetpointMode);
predictionSetpointSourceManual?.addEventListener("change", updateSetpointMode);
predictionShutterSourceMeasured?.addEventListener("change", updateShutterMode);
predictionShutterSourceManual?.addEventListener("change", updateShutterMode);
predictionSetpointAddBlockButton?.addEventListener("click", () => {
  createSetpointBlockRow({ time: "12:00", value: 20.0 });
});
predictionShutterAddBlockButton?.addEventListener("click", () => {
  createShutterBlockRow({ time: "12:00", value: 50 });
});
predictionSetpointCopyDayButton?.addEventListener("click", async () => {
  try {
    await copyMeasuredSetpointDay();
  } catch (error) {
    predictionStatus.className = "status error";
    predictionStatus.textContent = error instanceof Error ? error.message : "Setpointprofiel overnemen mislukt.";
  }
});
predictionShutterCopyDayButton?.addEventListener("click", async () => {
  try {
    await copyMeasuredShutterDay();
  } catch (error) {
    predictionStatus.className = "status error";
    predictionStatus.textContent = error instanceof Error ? error.message : "Shutterprofiel overnemen mislukt.";
  }
});
window.addEventListener("resize", () => {
  if (predictionChart) {
    Plotly.Plots.resize(predictionChart);
  }
});

ensureSetpointBlockRows();
ensureShutterBlockRows();
setPredictionDefaults();
setTrainingDefaults();
resetPredictionStats();
updateSetpointMode();
updateShutterMode();
