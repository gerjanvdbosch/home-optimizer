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
const predictionComfortMinInput = document.getElementById("prediction-comfort-min");
const predictionComfortMaxInput = document.getElementById("prediction-comfort-max");
const predictionModeManual = document.getElementById("prediction-mode-manual");
const predictionModeMpc = document.getElementById("prediction-mode-mpc");
const predictionSetpointSourceFieldset = document.getElementById("prediction-setpoint-source-fieldset");
const predictionShutterSourceFieldset = document.getElementById("prediction-shutter-source-fieldset");
const predictionSetpointSourceMeasured = document.getElementById("prediction-setpoint-source-measured");
const predictionSetpointSourceManual = document.getElementById("prediction-setpoint-source-manual");
const predictionSetpointHelp = document.getElementById("prediction-setpoint-help");
const predictionSetpointEditor = document.getElementById("prediction-setpoint-editor");
const predictionSetpointBlocks = document.getElementById("prediction-setpoint-blocks");
const predictionSetpointAddBlockButton = document.getElementById("prediction-setpoint-add-block");
const predictionShutterSourceMeasured = document.getElementById("prediction-shutter-source-measured");
const predictionShutterSourceManual = document.getElementById("prediction-shutter-source-manual");
const predictionShutterHelp = document.getElementById("prediction-shutter-help");
const predictionShutterEditor = document.getElementById("prediction-shutter-editor");
const predictionShutterBlocks = document.getElementById("prediction-shutter-blocks");
const predictionShutterAddBlockButton = document.getElementById("prediction-shutter-add-block");
const predictionButton = document.getElementById("prediction-button");
const predictionStatus = document.getElementById("prediction-status");
const predictionSummary = document.getElementById("prediction-summary");
const predictionChart = document.getElementById("prediction-chart");
const mpcEditor = document.getElementById("mpc-editor");
const mpcSetpointMinInput = document.getElementById("mpc-setpoint-min");
const mpcSetpointMaxInput = document.getElementById("mpc-setpoint-max");
const mpcSetpointStepInput = document.getElementById("mpc-setpoint-step");
const mpcSwitchHoursInput = document.getElementById("mpc-switch-hours");
const mpcChangePenaltyInput = document.getElementById("mpc-change-penalty");
const mpcResults = document.getElementById("mpc-results");
const mpcCandidateList = document.getElementById("mpc-candidate-list");
const predictionStatDelta = document.getElementById("prediction-stat-delta");
const predictionStatRmse = document.getElementById("prediction-stat-rmse");
const predictionStatBias = document.getElementById("prediction-stat-bias");
const predictionStatMaxError = document.getElementById("prediction-stat-max-error");
const DEFAULT_PREDICTION_HOURS = 24;
const DEFAULT_MPC_HORIZON_HOURS = 6;

function setPredictionDefaults(date = new Date()) {
  if (!predictionStartInput) {
    return;
  }
  const start = new Date(date);
  start.setHours(0, 0, 0, 0);
  predictionStartInput.value = toDatetimeLocalValue(start);
  if (predictionHoursInput && !predictionHoursInput.value) {
    predictionHoursInput.value = String(DEFAULT_PREDICTION_HOURS);
  }
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

function isMpcMode() {
  return Boolean(predictionModeMpc?.checked);
}

function buildComfortShapes(minTemperature, maxTemperature) {
  return [
    {
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: minTemperature,
      y1: maxTemperature,
      fillcolor: "rgba(0, 121, 107, 0.08)",
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: minTemperature,
      y1: minTemperature,
      line: { color: "rgba(0, 121, 107, 0.35)", dash: "dot", width: 1 },
    },
    {
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: maxTemperature,
      y1: maxTemperature,
      line: { color: "rgba(0, 121, 107, 0.35)", dash: "dot", width: 1 },
    },
  ];
}

function evaluateComfort(predictedSeries, minTemperature, maxTemperature) {
  const values = predictedSeries.points.map((point) => point.value);
  if (!values.length) {
    return null;
  }

  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const underCount = values.filter((value) => value < minTemperature).length;
  const overCount = values.filter((value) => value > maxTemperature).length;

  return {
    minimum,
    maximum,
    underCount,
    overCount,
    withinComfort: underCount === 0 && overCount === 0,
  };
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

function ensureBlockRows(container, addRow, fallbackBlock) {
  if (!container?.children.length) {
    addRow(fallbackBlock);
  }
}

function readBlockPattern(container, config, addRow) {
  ensureBlockRows(container, addRow, { time: "00:00", value: config.defaultValue });
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
  ensureBlockRows(predictionSetpointBlocks, createSetpointBlockRow, { time: "00:00", value: 18.0 });
}

function ensureShutterBlockRows() {
  ensureBlockRows(predictionShutterBlocks, createShutterBlockRow, { time: "00:00", value: 100 });
}

function buildSetpointBlockPattern() {
  return readBlockPattern(
    predictionSetpointBlocks,
    {
      timeClassName: "setpoint-block-time",
      valueClassName: "setpoint-block-value",
      defaultValue: 18,
    },
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
  if (isMpcMode()) {
    if (predictionShutterEditor) predictionShutterEditor.hidden = true;
    if (predictionShutterHelp) predictionShutterHelp.hidden = true;
    return;
  }
  const useMeasuredShutters = Boolean(predictionShutterSourceMeasured?.checked);
  if (predictionShutterEditor) {
    predictionShutterEditor.hidden = useMeasuredShutters;
  }
  if (predictionShutterHelp) {
    predictionShutterHelp.hidden = useMeasuredShutters;
    predictionShutterHelp.textContent = useMeasuredShutters
      ? ""
      : "Gebruik hieronder tijdblokken voor een handmatig shutterprofiel; dit patroon wordt per dag herhaald.";
  }
}

function updateSetpointMode() {
  if (isMpcMode()) {
    if (predictionSetpointEditor) predictionSetpointEditor.hidden = true;
    if (predictionSetpointHelp) predictionSetpointHelp.hidden = true;
    return;
  }
  const useMeasuredSetpoints = Boolean(predictionSetpointSourceMeasured?.checked);
  if (predictionSetpointEditor) {
    predictionSetpointEditor.hidden = useMeasuredSetpoints;
  }
  if (predictionSetpointHelp) {
    predictionSetpointHelp.hidden = useMeasuredSetpoints;
    predictionSetpointHelp.textContent = useMeasuredSetpoints
      ? ""
      : "Gebruik hieronder tijdblokken voor een handmatig setpointprofiel; dit patroon wordt per dag herhaald.";
  }
}

function updatePredictionMode() {
  const mpcMode = isMpcMode();
  if (predictionSetpointSourceFieldset) predictionSetpointSourceFieldset.hidden = mpcMode;
  if (predictionShutterSourceFieldset) predictionShutterSourceFieldset.hidden = mpcMode;
  if (mpcEditor) mpcEditor.hidden = !mpcMode;
  if (predictionHoursInput) {
    const currentHours = Number(predictionHoursInput.value || 0);
    if (mpcMode && (!currentHours || currentHours === DEFAULT_PREDICTION_HOURS)) {
      predictionHoursInput.value = String(DEFAULT_MPC_HORIZON_HOURS);
    } else if (!mpcMode && currentHours === DEFAULT_MPC_HORIZON_HOURS) {
      predictionHoursInput.value = String(DEFAULT_PREDICTION_HOURS);
    }
  }
  if (predictionButton) {
    predictionButton.textContent = mpcMode ? "Bereken MPC voorstel" : "Vergelijk met metingen";
  }
  if (mpcResults && !mpcMode) {
    mpcResults.hidden = true;
  }
  updateSetpointMode();
  updateShutterMode();
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

async function activateManualSetpointMode() {
  if (!predictionSetpointSourceManual?.checked) {
    return;
  }
  try {
    await copyMeasuredSetpointDay();
  } catch (error) {
    if (predictionStatus) {
      predictionStatus.className = "status error";
      predictionStatus.textContent =
        error instanceof Error ? error.message : "Setpointprofiel laden mislukt.";
    }
  }
}

async function activateManualShutterMode() {
  if (!predictionShutterSourceManual?.checked) {
    return;
  }
  try {
    await copyMeasuredShutterDay();
  } catch (error) {
    if (predictionStatus) {
      predictionStatus.className = "status error";
      predictionStatus.textContent =
        error instanceof Error ? error.message : "Shutterprofiel laden mislukt.";
    }
  }
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

function buildAllowedSetpoints() {
  const minValue = Number(mpcSetpointMinInput?.value || 19);
  const maxValue = Number(mpcSetpointMaxInput?.value || 21);
  const stepValue = Number(mpcSetpointStepInput?.value || 0.5);
  if (minValue > maxValue) {
    throw new Error("MPC setpoint min moet kleiner of gelijk zijn aan max");
  }
  if (stepValue <= 0) {
    throw new Error("MPC setpoint stap moet groter dan 0 zijn");
  }
  const values = [];
  for (let value = minValue; value <= maxValue + 1e-9; value += stepValue) {
    values.push(Number(value.toFixed(3)));
  }
  return values;
}

function buildMpcSwitchTimes(startDate, endDate) {
  const switchHours = Number(mpcSwitchHoursInput?.value || 2);
  if (switchHours <= 0) {
    throw new Error("MPC switch-interval moet groter dan 0 zijn");
  }
  const times = [];
  const current = new Date(startDate);
  current.setHours(current.getHours() + switchHours);
  while (current < endDate) {
    times.push(localInputToIso(toDatetimeLocalValue(current)));
    current.setHours(current.getHours() + switchHours);
  }
  return times;
}

function renderMpcCandidates(responsePayload) {
  if (!mpcCandidateList || !mpcResults) {
    return;
  }
  const topCandidates = responsePayload.candidate_results
    .slice()
    .sort((left, right) => left.total_cost - right.total_cost)
    .slice(0, 3);
  mpcCandidateList.replaceChildren();
  topCandidates.forEach((candidate) => {
    const item = document.createElement("article");
    item.className = `mpc-candidate${candidate.candidate_name === responsePayload.best_candidate.candidate_name ? " is-best" : ""}`;
    item.innerHTML = `
      <div><strong>${candidate.candidate_name}</strong><span>Kost ${candidate.total_cost.toFixed(3)}</span></div>
      <div><strong>Comfort</strong><span>${candidate.comfort_violation_cost.toFixed(3)}</span></div>
      <div><strong>Switch</strong><span>${candidate.setpoint_change_cost.toFixed(3)}</span></div>
      <div><strong>Min temp</strong><span>${candidate.minimum_predicted_temperature?.toFixed(2) ?? "-"}</span></div>
      <div><strong>Max temp</strong><span>${candidate.maximum_predicted_temperature?.toFixed(2) ?? "-"}</span></div>
    `;
    mpcCandidateList.append(item);
  });
  mpcResults.hidden = false;
}

async function runMpcPrediction(startDate, endDate, comfortMin, comfortMax) {
  const shutterSchedule = await buildPredictionShutterSchedule(startDate, endDate);
  const horizonHours = Number(predictionHoursInput?.value || DEFAULT_MPC_HORIZON_HOURS);
  const response = await fetch(apiUrl("api/mpc/thermostat-setpoint"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      start_time: localInputToIso(predictionStartInput.value),
      horizon_hours: horizonHours,
      interval_minutes: 15,
      allowed_setpoints: buildAllowedSetpoints(),
      switch_times: buildMpcSwitchTimes(startDate, endDate),
      comfort_min_temperature: comfortMin,
      comfort_max_temperature: comfortMax,
      setpoint_change_penalty: Number(mpcChangePenaltyInput?.value || 0.1),
      shutter_schedule: shutterSchedule,
    }),
  });
  const responsePayload = await response.json();
  if (!response.ok) {
    throw new Error(responsePayload.detail || "MPC voorstel ophalen mislukt.");
  }

  renderPlot(
    predictionChart,
    [responsePayload.best_candidate.predicted_room_temperature],
    {
      colors: ["#00796b"],
      emptyText: "Geen MPC voorspelling beschikbaar",
      yTitle: responsePayload.best_candidate.predicted_room_temperature.unit || "",
      shapes: buildComfortShapes(comfortMin, comfortMax),
      traceOptions: [{ label: "MPC voorspeld", precision: 2 }],
    },
  );
  renderMpcCandidates(responsePayload);
  predictionStatus.className = "status success";
  predictionStatus.textContent = "MPC voorstel berekend.";
  if (predictionSummary) {
    predictionSummary.textContent = `Beste kandidaat: ${responsePayload.best_candidate.candidate_name} · kost ${responsePayload.best_candidate.total_cost.toFixed(3)}`;
  }
  if (predictionStatDelta) predictionStatDelta.textContent = "-";
  if (predictionStatRmse) predictionStatRmse.textContent = responsePayload.best_candidate.total_cost.toFixed(2);
  if (predictionStatBias) predictionStatBias.textContent = responsePayload.best_candidate.comfort_violation_cost.toFixed(2);
  if (predictionStatMaxError) predictionStatMaxError.textContent = responsePayload.best_candidate.setpoint_change_cost.toFixed(2);
}

async function runPrediction(event) {
  event.preventDefault();
  if (
    !predictionStartInput ||
    !predictionHoursInput ||
    !predictionComfortMinInput ||
    !predictionComfortMaxInput ||
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
    const comfortMin = Number(predictionComfortMinInput.value);
    const comfortMax = Number(predictionComfortMaxInput.value);
    const endDate = new Date(startDate);
    endDate.setHours(endDate.getHours() + hoursAhead);

    if (comfortMin > comfortMax) {
      throw new Error("comfort min moet kleiner of gelijk zijn aan comfort max");
    }

    if (mpcResults) {
      mpcResults.hidden = true;
    }

    if (isMpcMode()) {
      await runMpcPrediction(startDate, endDate, comfortMin, comfortMax);
      return;
    }

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
        shapes: buildComfortShapes(comfortMin, comfortMax),
        traceOptions: [
          { label: "Voorspeld", precision: 2 },
          { label: "Gemeten", precision: 2, dash: "dot" },
        ],
      },
    );

    predictionStatus.className = "status success";
    predictionStatus.textContent = "Scenario vergeleken met metingen.";

    const comfortEvaluation = evaluateComfort(
      responsePayload.predicted_room_temperature,
      comfortMin,
      comfortMax,
    );

    if (predictionSummary) {
      const predicted = latestPoint(responsePayload.predicted_room_temperature);
      const actual = latestPoint(responsePayload.actual_room_temperature);
      const comfortSummary = comfortEvaluation
        ? comfortEvaluation.withinComfort
          ? "binnen comfort"
          : `${comfortEvaluation.underCount} onder / ${comfortEvaluation.overCount} boven comfort`
        : "comfort onbekend";
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
        predictionSummary.textContent = `Vergelijking bijgewerkt · ${comfortSummary}`;
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
        predictionSummary.textContent = `Vergelijking bijgewerkt · ${comfortSummary}`;
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
  trainingStatus.textContent = "Modellen worden getraind...";

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

    const trainedModels = Array.isArray(responsePayload.models) ? responsePayload.models : [];
    const roomModel = trainedModels.find((model) => model.target_name === "room_temperature");
    const thermalModel = trainedModels.find((model) => model.target_name === "thermal_output");
    const parts = [];
    if (thermalModel) {
      parts.push(`thermal RMSE: ${thermalModel.test_rmse.toFixed(3)}`);
    }
    if (roomModel) {
      parts.push(`room 1-step RMSE: ${roomModel.test_rmse.toFixed(3)}`);
      parts.push(`room recursive RMSE: ${roomModel.test_rmse_recursive.toFixed(3)}`);
    }

    trainingStatus.className = "status success";
    trainingStatus.textContent = parts.length
      ? `Modellen opgeslagen. ${parts.join(" · ")}`
      : "Modellen opgeslagen.";
  } catch (error) {
    trainingStatus.className = "status error";
    trainingStatus.textContent = error instanceof Error ? error.message : "Modeltraining mislukt.";
  } finally {
    trainingButton.disabled = false;
  }
}

predictionForm?.addEventListener("submit", runPrediction);
trainingButton?.addEventListener("click", runTraining);
predictionModeManual?.addEventListener("change", updatePredictionMode);
predictionModeMpc?.addEventListener("change", updatePredictionMode);
predictionSetpointSourceMeasured?.addEventListener("change", updateSetpointMode);
predictionSetpointSourceManual?.addEventListener("change", async () => {
  updateSetpointMode();
  await activateManualSetpointMode();
});
predictionShutterSourceMeasured?.addEventListener("change", updateShutterMode);
predictionShutterSourceManual?.addEventListener("change", async () => {
  updateShutterMode();
  await activateManualShutterMode();
});
predictionSetpointAddBlockButton?.addEventListener("click", () => {
  createSetpointBlockRow({ time: "12:00", value: 20.0 });
});
predictionShutterAddBlockButton?.addEventListener("click", () => {
  createShutterBlockRow({ time: "12:00", value: 50 });
});
window.addEventListener("resize", () => {
  if (predictionChart) {
    Plotly.Plots.resize(predictionChart);
  }
});

setPredictionDefaults();
setTrainingDefaults();
resetPredictionStats();
updatePredictionMode();
updateSetpointMode();
updateShutterMode();
