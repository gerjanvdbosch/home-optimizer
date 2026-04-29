const button = document.getElementById("import-button");
const status = document.getElementById("status");
const result = document.getElementById("result");
const selectedDateLabel = document.getElementById("selected-date");
const previousDayButton = document.getElementById("previous-day");
const nextDayButton = document.getElementById("next-day");
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
const roomSummary = document.getElementById("room-summary");
const dhwSummary = document.getElementById("dhw-summary");
const heatpumpSummary = document.getElementById("heatpump-summary");
const forecastSummary = document.getElementById("forecast-summary");
const shutterSummary = document.getElementById("shutter-summary");
const roomChart = document.getElementById("room-chart");
const dhwChart = document.getElementById("dhw-chart");
const heatpumpChart = document.getElementById("heatpump-chart");
const forecastChart = document.getElementById("forecast-chart");
const shutterChart = document.getElementById("shutter-chart");
const compressorSummary = document.getElementById("compressor-summary");
const compressorChart = document.getElementById("compressor-chart");
const supplyChart = document.getElementById("supply-chart");
const thermalChart = document.getElementById("thermal-chart");
const baseUrl = new URL(".", window.location.href);
const heatpumpModeStyles = {
  heat: { label: "Vloerverwarming", color: "#43a047", fill: "rgba(67, 160, 71, 0.12)" },
  dhw: { label: "Boiler", color: "#8e24aa", fill: "rgba(142, 36, 170, 0.12)" },
  legionella: { label: "Legionella", color: "#d81b60", fill: "rgba(216, 27, 96, 0.12)" },
  cool: { label: "Koelen", color: "#03a9f4", fill: "rgba(3, 169, 244, 0.12)" },
};
const heatpumpStatusStyles = {
  defrost_active: {
    label: "Defrost",
    color: "#1976d2",
    fill: "rgba(25, 118, 210, 0.14)",
  },
  booster_heater_active: {
    label: "Booster heater",
    color: "#c62828",
    fill: "rgba(198, 40, 40, 0.13)",
  },
};
const forecastSeriesStyles = {
  gti_pv: {
    label: "GTI PV",
    color: "#f9a825",
  },
  gti_living_room_windows: {
    label: "GTI ramen",
    color: "#6d4c41",
  },
  gti_living_room_windows_adjusted: {
    label: "Instraling",
    color: "#6d4c41",
    dash: "dot",
  },
  temperature: {
    label: "Buitentemperatuur",
    color: "#1e88e5",
  },
};

let selectedDate = new Date();

function apiUrl(path) {
  return new URL(path, baseUrl).toString();
}

function toDatetimeLocalValue(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function localInputToIso(value) {
  const date = new Date(value);
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absoluteOffset = Math.abs(offsetMinutes);
  const offsetHours = String(Math.floor(absoluteOffset / 60)).padStart(2, "0");
  const offsetRemainder = String(absoluteOffset % 60).padStart(2, "0");
  return `${toDatetimeLocalValue(date)}:00${sign}${offsetHours}:${offsetRemainder}`;
}

function setPredictionDefaults(date = selectedDate) {
  if (!predictionStartInput) {
    return;
  }

  const start = new Date(date);
  start.setHours(24, 0, 0, 0);
  predictionStartInput.value = toDatetimeLocalValue(start);
}

function setTrainingDefaults(date = selectedDate) {
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

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatDisplayDate(date) {
  return new Intl.DateTimeFormat("nl-NL", {
    weekday: "short",
    day: "2-digit",
    month: "short",
  }).format(date);
}

function shiftDate(days) {
  selectedDate = new Date(selectedDate);
  selectedDate.setDate(selectedDate.getDate() + days);
  setPredictionDefaults(selectedDate);
  setTrainingDefaults(selectedDate);
  loadCharts();
}

function buildConstantSeries(name, unit, startDate, endDate, intervalMinutes, value) {
  const points = [];
  const current = new Date(startDate);
  current.setMinutes(current.getMinutes() + intervalMinutes);

  while (current <= endDate) {
    points.push({
      timestamp: localInputToIso(toDatetimeLocalValue(current)),
      value,
    });
    current.setMinutes(current.getMinutes() + intervalMinutes);
  }

  return { name, unit, points };
}

async function runImport() {
  if (!button) {
    return;
  }

  button.disabled = true;
  status.className = "status";
  status.textContent = "Import wordt gestart...";

  try {
    const response = await fetch(apiUrl("api/history-import"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Import mislukt.");
    }

    status.textContent = "Import draait...";
    await pollImportJob(payload.job_id);
    await loadCharts();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Import mislukt.";
    status.className = "status error";
    status.textContent = message;
    result.hidden = false;
    result.textContent = "De import kon niet worden uitgevoerd.";
  } finally {
    button.disabled = false;
  }
}

async function pollImportJob(jobId) {
  while (true) {
    const response = await fetch(apiUrl(`api/history-import/jobs/${jobId}`));
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Importstatus ophalen mislukt.");
    }

    result.textContent = JSON.stringify(payload, null, 2);
    result.hidden = false;

    if (payload.status === "succeeded") {
      status.className = "status success";
      status.textContent =
        `Import voltooid: ${payload.total_rows} rijen over ` +
        `${payload.sensor_count} sensoren.`;
      return;
    }

    if (payload.status === "failed") {
      throw new Error(payload.error || "Import mislukt.");
    }

    status.className = "status";
    status.textContent = `Import ${payload.status}...`;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function loadCharts() {
  if (!roomChart || !dhwChart || !heatpumpChart || !forecastChart || !shutterChart || !compressorChart || !selectedDateLabel) {
    return;
  }

  selectedDateLabel.textContent = formatDisplayDate(selectedDate);
  const params = new URLSearchParams({
    date: formatDate(selectedDate),
  });
  const response = await fetch(apiUrl(`api/dashboard/charts?${params.toString()}`));
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Grafiekdata ophalen mislukt.");
  }

  const start = new Date(selectedDate);
  start.setHours(0, 0, 0, 0);
  const end = new Date(selectedDate);
  end.setHours(23, 59, 59, 0);

  const startIso = new Date(start.getTime() - start.getTimezoneOffset() * 60 * 1000)
    .toISOString()
    .slice(0, 19);
  const endIso = new Date(end.getTime() - end.getTimezoneOffset() * 60 * 1000)
    .toISOString()
    .slice(0, 19);

  renderPlot(roomChart, [payload.room_temperature, payload.thermostat_setpoint], {
    colors: ["#03a9f4", "#8e24aa"],
    emptyText: "Geen kamertemperatuur voor deze dag",
    yTitle: payload.room_temperature.unit || "",
    traceOptions: [
      { label: "Woonkamer" },
      { label: "Setpoint" },
    ],
    xRange: [startIso, endIso],
  });

  renderPlot(dhwChart, payload.dhw_temperatures, {
    colors: ["#ff9800", "#7e57c2"],
    emptyText: "Geen boilerdata voor deze dag",
    yTitle: payload.dhw_temperatures[0]?.unit || "",
    traceOptions: [
      { label: "Boiler (boven)" },
      { label: "Boiler (onder)" },
    ],
    xRange: [startIso, endIso],
  });

  renderHeatpumpPowerPlot(
    heatpumpChart,
    payload.heatpump_power,
    payload.heatpump_mode,
    payload.heatpump_statuses,
    { xRange: [startIso, endIso], loadSeries: [payload.baseload, payload.pv_output_power], loadTraceOptions: [{ label: "Baseload", color: "#d32f2f" }, { label: "PV opbrengst", color: "#f9a825" }], loadColors: ["#d32f2f", "#f9a825"] },
  );

  renderPlot(forecastChart, [
    payload.forecast_gti[0],
    payload.forecast_gti[1],
    payload.forecast_gti[2],
    payload.forecast_temperature,
  ], {
    colors: ["#f9a825", "#6d4c41", "#6d4c41", "#1e88e5"],
    emptyText: "Geen forecastdata voor deze dag",
    yTitle: payload.forecast_gti[0]?.unit || "",
    y2Title: payload.forecast_temperature.unit || "",
    traceOptions: [
      { label: "GTI PV", color: "#f9a825" },
      { label: "GTI ramen", color: "#6d4c41" },
      { label: "Instraling", color: "#6d4c41", dash: "dot" },
      { label: "Buitentemperatuur", yaxis: "y2", color: "#1e88e5", dash: "dot", precision: 1 },
    ],
    xRange: [startIso, endIso],
  });

  renderPlot(thermalChart, [payload.thermal_output, payload.cop], {
    colors: ["#ff7043", "#4caf50"],
    emptyText: "Geen thermische output voor deze dag",
    yTitle: payload.thermal_output.unit || "",
    y2Title: "COP",
    traceOptions: [
      { label: "Thermische output", precision: 2 },
      { label: "COP", yaxis: "y2", precision: 2 },
    ],
    xRange: [startIso, endIso],
  });

  renderPlot(supplyChart, [payload.hp_supply_target_temperature, payload.hp_supply_temperature, payload.hp_return_temperature, payload.hp_delta_t], {
    colors: ["#ffb74d", "#d32f2f", "#1976d2", "#616161"],
    emptyText: "Geen aanvoer/retour data voor deze dag",
    yTitle: payload.hp_supply_temperature.unit || "",
    traceOptions: [
      { label: "Doel" },
      { label: "Aanvoer" },
      { label: "Retour" },
      { label: "Delta T", shape: "hv" },
    ],
    xRange: [startIso, endIso],
  });

  renderPlot(shutterChart, [payload.shutter_position], {
    colors: ["#607d8b"],
    emptyText: "Geen shutterdata voor deze dag",
    yTitle: payload.shutter_position.unit || "%",
    xRange: [startIso, endIso],
  });

  renderPlot(compressorChart, [payload.compressor_frequency, payload.hp_flow], {
    colors: ["#8e24aa", "#03a9f4"],
    emptyText: "Geen compressor/flow data voor deze dag",
    yTitle: payload.compressor_frequency.unit || "",
    y2Title: payload.hp_flow.unit || "",
    traceOptions: [
      { label: "Compressor freq" },
      { label: "Flow", yaxis: "y2" },
    ],
    xRange: [startIso, endIso],
  });

  if (shutterSummary) {
    shutterSummary.textContent = summarizeSeries(payload.shutter_position);
  }

  if (compressorSummary) {
    const freq = payload.compressor_frequency ? summarizeSeries(payload.compressor_frequency) : "-";
    const flow = payload.hp_flow ? summarizeSeries(payload.hp_flow) : "-";
    compressorSummary.textContent = `${freq} · ${flow}`;
  }

  roomSummary.textContent = summarizeSeries(payload.room_temperature);
  dhwSummary.textContent = summarizeSeries(payload.dhw_temperatures[0]);
  heatpumpSummary.textContent = summarizeHeatpump(
    payload.heatpump_power,
    payload.heatpump_mode,
    payload.heatpump_statuses,
  );
  forecastSummary.textContent = summarizeForecast(
    payload.forecast_temperature,
    payload.forecast_gti,
  );
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

    const response = await fetch(apiUrl("api/prediction"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const responsePayload = await response.json();

    if (!response.ok) {
      throw new Error(responsePayload.detail || "Voorspelling ophalen mislukt.");
    }

    renderPlot(predictionChart, [responsePayload.room_temperature], {
      colors: ["#00796b"],
      emptyText: "Geen voorspelling beschikbaar",
      yTitle: responsePayload.room_temperature.unit || "",
      traceOptions: [{ label: "Voorspelde kamertemperatuur", precision: 2 }],
    });
    predictionStatus.className = "status success";
    predictionStatus.textContent = "Scenario voorspeld.";
    if (predictionSummary) {
      predictionSummary.textContent = summarizeSeries(responsePayload.room_temperature);
    }
  } catch (error) {
    predictionStatus.className = "status error";
    predictionStatus.textContent =
      error instanceof Error ? error.message : "Voorspelling mislukt.";
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
      `Model opgeslagen. Test RMSE: ${responsePayload.test_rmse.toFixed(3)}`;
  } catch (error) {
    trainingStatus.className = "status error";
    trainingStatus.textContent =
      error instanceof Error ? error.message : "Modeltraining mislukt.";
  } finally {
    trainingButton.disabled = false;
  }
}

function summarizeSeries(series) {
  const values = series.points.map((point) => point.value);
  if (values.length === 0) {
    return "-";
  }
  const latest = values[values.length - 1];
  const unit = series.unit || "";
  return `${latest.toFixed(1)} ${unit}`.trim();
}

function renderPlot(element, seriesList, options) {
  const traces = seriesList.map((series, index) => {
    const to = options.traceOptions?.[index] || {};
    const precision = Number.isFinite(to.precision) ? to.precision : 1;
    const yaxis = to.yaxis === "y2" ? "y2" : undefined;
    return {
      x: series.points.map((point) => chartTimestamp(point.timestamp)),
      y: series.points.map((point) => point.value),
      name: to.label || series.name,
      type: "scatter",
      mode: "lines",
      ...(yaxis ? { yaxis } : {}),
      line: {
        color: to.color || options.colors?.[index % (options.colors?.length || 1)],
        width: to.width || 2,
        ...(to.dash ? { dash: to.dash } : {}),
        ...(to.shape ? { shape: to.shape } : {}),
      },
      hovertemplate:
        `%{x|%H:%M}<br>%{y:.${precision}f} ${series.unit || ""}` +
        `<extra>${to.label || series.name}</extra>`,
    };
  });
  const hasPoints = seriesList.some((series) => series.points.length > 0);

  Plotly.react(element, traces, plotLayout(options, hasPoints), {
    displayModeBar: false,
    responsive: true,
  });
}

function renderHeatpumpPowerPlot(element, powerSeries, modeSeries, statusSeriesList, options = {}) {
  const points = powerSeries.points.map((point) => ({
    ...point,
    mode: modeAtTimestamp(modeSeries.points, point.timestamp),
  }));

  const traces = [{
    x: points.map((point) => chartTimestamp(point.timestamp)),
    y: points.map((point) => point.value),
    customdata: points.map((point) => displayMode(point.mode)),
    name: "Warmtepomp",
    type: "scatter",
    mode: "lines",
    line: {
      color: "#2196f3",
      width: 2,
    },
    hovertemplate:
      `%{x|%H:%M}<br>%{y:.1f} ${powerSeries.unit || ""}` +
      "<br>Mode: %{customdata}<extra></extra>",
  }];

  if (options.loadSeries && Array.isArray(options.loadSeries)) {
    const loadColors = options.loadColors || ["#616161", "#f9a825"];
    const loadTraceOptions = options.loadTraceOptions || [];
    options.loadSeries.forEach((series, idx) => {
      traces.push({
        x: series.points.map((point) => chartTimestamp(point.timestamp)),
        y: series.points.map((point) => point.value),
        name: loadTraceOptions[idx]?.label || series.name,
        type: "scatter",
        mode: "lines",
        line: {
          color: loadTraceOptions[idx]?.color || loadColors[idx % loadColors.length],
          width: loadTraceOptions[idx]?.width || 2,
          ...(loadTraceOptions[idx]?.dash ? { dash: loadTraceOptions[idx].dash } : {}),
        },
        hovertemplate: `%{x|%H:%M}<br>%{y:.1f} ${series.unit || ""}` + `<extra>${loadTraceOptions[idx]?.label || series.name}</extra>`,
      });
    });
  }

  const mIntervals = modeIntervals(modeSeries.points);
  const statusIntervalsByName = heatpumpStatusIntervals(statusSeriesList);

  traces.push(...heatpumpModeLegendTraces(mIntervals));
  traces.push(...heatpumpStatusLegendTraces(statusIntervalsByName));

  const hasPoints = powerSeries.points.length > 0;

  const combinedShapes = [
    ...heatpumpModeShapes(mIntervals),
    ...heatpumpStatusShapes(statusIntervalsByName)
  ];

  const defaultOptions = {
    emptyText: "Geen warmtepompvermogen voor deze dag",
    yTitle: powerSeries.unit || "",
    shapes: combinedShapes,
  };

  const mergedOptions = { ...defaultOptions, ...options };

  Plotly.react(
    element,
    traces,
    plotLayout(mergedOptions, hasPoints),
    {
      displayModeBar: false,
      responsive: true,
    },
  );
}

function heatpumpStatusIntervals(statusSeriesList) {
  return statusSeriesList.map((series) => ({
    name: series.name,
    intervals: activeIntervals(series.points),
  }));
}

function activeIntervals(points) {
  const intervals = [];
  let start = null;
  let previousTimestamp = null;

  for (const point of points) {
    if (isActiveStatusValue(point.value)) {
      start = start || point.timestamp;
      previousTimestamp = point.timestamp;
      continue;
    }

    if (start) {
      intervals.push({ x0: start, x1: point.timestamp });
      start = null;
      previousTimestamp = null;
    }
  }

  if (start && previousTimestamp) {
    intervals.push({ x0: start, x1: addMinutes(previousTimestamp, 1) });
  }

  return intervals;
}

function isActiveStatusValue(value) {
  if (typeof value === "number") {
    return value >= 0.5;
  }
  return ["1", "true", "on", "active"].includes(String(value).trim().toLowerCase());
}

function addMinutes(timestamp, minutes) {
  return new Date(Date.parse(timestamp) + minutes * 60 * 1000).toISOString();
}

function chartTimestamp(timestamp) {
  const date = new Date(timestamp);
  const localTimestamp = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localTimestamp.toISOString().slice(0, 19);
}

function heatpumpStatusShapes(statusIntervalsByName) {
  return statusIntervalsByName.flatMap((series) => {
    const style = heatpumpStatusStyles[series.name];
    if (!style) {
      return [];
    }

    return series.intervals.map((interval) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: chartTimestamp(interval.x0),
      x1: chartTimestamp(interval.x1),
      y0: 0,
      y1: 1,
      fillcolor: style.fill,
      line: { width: 0 },
      layer: "below",
    }));
  });
}

function heatpumpStatusLegendTraces(statusIntervalsByName) {
  return statusIntervalsByName
    .filter((series) => series.intervals.length > 0)
    .map((series) => {
      const style = heatpumpStatusStyles[series.name] || {
        label: series.name,
        color: "#5c6bc0",
      };
      return {
        x: [null],
        y: [null],
        name: style.label,
        type: "scatter",
        mode: "lines",
        line: { color: style.color, width: 8 },
        hoverinfo: "skip",
      };
    });
}

function modeAtTimestamp(modePoints, timestamp) {
  let currentMode = "onbekend";
  const timestampMs = Date.parse(timestamp);

  for (const point of modePoints) {
    if (Date.parse(point.timestamp) > timestampMs) {
      break;
    }
    currentMode = point.value;
  }

  return currentMode;
}

function displayMode(mode) {
  if (mode === "onbekend") {
    return "Onbekend";
  }
  return mode.toUpperCase();
}

function modeIntervals(modePoints) {
  const intervals = [];
  if (!modePoints || modePoints.length === 0) return intervals;

  let currentMode = modePoints[0].value;
  let start = modePoints[0].timestamp;
  let previousTimestamp = modePoints[0].timestamp;

  for (let i = 1; i < modePoints.length; i++) {
    const point = modePoints[i];
    if (point.value !== currentMode) {
      intervals.push({ mode: currentMode, x0: start, x1: point.timestamp });
      currentMode = point.value;
      start = point.timestamp;
    }
    previousTimestamp = point.timestamp;
  }

  if (start && previousTimestamp) {
    intervals.push({ mode: currentMode, x0: start, x1: addMinutes(previousTimestamp, 1) });
  }

  return intervals;
}

function heatpumpModeShapes(intervals) {
  return intervals
    .filter((interval) => heatpumpModeStyles[interval.mode])
    .map((interval) => {
      const style = heatpumpModeStyles[interval.mode];
      return {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: chartTimestamp(interval.x0),
        x1: chartTimestamp(interval.x1),
        y0: 0,
        y1: 1,
        fillcolor: style.fill,
        line: { width: 0 },
        layer: "below",
      };
    });
}

function heatpumpModeLegendTraces(intervals) {
  const activeModes = Array.from(new Set(intervals.map((i) => i.mode)));
  return activeModes
    .filter((mode) => heatpumpModeStyles[mode])
    .map((mode) => {
      const style = heatpumpModeStyles[mode];
      return {
        x: [null],
        y: [null],
        name: style.label,
        type: "scatter",
        mode: "lines",
        line: { color: style.color, width: 8 },
        hoverinfo: "skip",
      };
    });
}

function summarizeHeatpump(powerSeries, modeSeries, statusSeriesList) {
  if (powerSeries.points.length === 0) {
    return "-";
  }
  const latest = powerSeries.points[powerSeries.points.length - 1];
  const mode = modeAtTimestamp(modeSeries.points, latest.timestamp);
  const unit = powerSeries.unit || "";
  const statusLabels = activeStatusLabels(statusSeriesList, latest.timestamp);
  return [`${latest.value.toFixed(1)} ${unit}`, displayMode(mode), ...statusLabels]
    .join(" · ")
    .trim();
}

function summarizeForecast(temperatureSeries, gtiSeriesList) {
  const values = [
    ...gtiSeriesList.map((series) => summarizeNamedLatest(series, forecastSeriesStyles[series.name]?.label)),
    summarizeNamedLatest(temperatureSeries, forecastSeriesStyles.temperature.label),
  ].filter(Boolean);

  return values.length > 0 ? values.join(" · ") : "-";
}

function summarizeNamedLatest(series, label) {
  const latest = latestPoint(series);
  if (!latest) {
    return "";
  }
  const unit = series.unit || "";
  return `${label}: ${latest.value.toFixed(1)} ${unit}`.trim();
}

function latestPoint(series) {
  return series.points.length > 0 ? series.points[series.points.length - 1] : null;
}

function activeStatusLabels(statusSeriesList, timestamp) {
  return statusSeriesList
    .filter((series) => isActiveStatusValue(statusAtTimestamp(series.points, timestamp)))
    .map((series) => heatpumpStatusStyles[series.name]?.label || series.name);
}

function statusAtTimestamp(points, timestamp) {
  let currentValue = 0;
  const timestampMs = Date.parse(timestamp);

  for (const point of points) {
    if (Date.parse(point.timestamp) > timestampMs) {
      break;
    }
    currentValue = point.value;
  }

  return currentValue;
}

function plotLayout(options, hasPoints) {
  const annotations = hasPoints
    ? []
    : [
        {
          text: options.emptyText,
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          showarrow: false,
          font: { color: "#727272", size: 14 },
        },
      ];

  const layout = {
    autosize: true,
    margin: { t: 10, r: 12, b: 36, l: 46 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    annotations,
    shapes: options.shapes || [],
    xaxis: {
      type: "date",
      tickformat: "%H:%M",
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
      ...(options.xRange ? { range: options.xRange } : {}),
    },
    yaxis: {
      title: { text: options.yTitle },
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.18,
      font: { size: 12 },
    },
    font: {
      family: 'Roboto, "Noto Sans", "Segoe UI", Arial, sans-serif',
      color: "#212121",
    },
  };

  if (options.y2Title) {
    layout.yaxis2 = {
      title: { text: options.y2Title },
      overlaying: "y",
      side: "right",
      showgrid: false,
      zeroline: false,
      fixedrange: true,
    };
  }

  return layout;
}

button?.addEventListener("click", runImport);
previousDayButton?.addEventListener("click", () => shiftDate(-1));
nextDayButton?.addEventListener("click", () => shiftDate(1));
predictionForm?.addEventListener("submit", runPrediction);
trainingButton?.addEventListener("click", runTraining);
window.addEventListener("resize", () => {
  if (roomChart) {
    Plotly.Plots.resize(roomChart);
  }
  if (dhwChart) {
    Plotly.Plots.resize(dhwChart);
  }
  if (heatpumpChart) {
    Plotly.Plots.resize(heatpumpChart);
  }
  if (forecastChart) {
    Plotly.Plots.resize(forecastChart);
  }
  if (predictionChart) {
    Plotly.Plots.resize(predictionChart);
  }
});
setPredictionDefaults(selectedDate);
setTrainingDefaults(selectedDate);
loadCharts().catch((error) => {
  if (roomSummary) {
    roomSummary.textContent = "Fout";
  }
  if (dhwSummary) {
    dhwSummary.textContent = "Fout";
  }
  if (heatpumpSummary) {
    heatpumpSummary.textContent = "Fout";
  }
  if (forecastSummary) {
    forecastSummary.textContent = "Fout";
  }
  console.error(error);
});
