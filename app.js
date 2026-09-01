/**
 * Coupled Oscillators Interactive Web Explorer Engine.
 * Only calculates on explicit "Run" button click.
 * Features 60fps real-time dynamic window metric computation while dragging the time slider.
 */

// Global State
let currentSimulationData = null;
let chartTrajectory = null;
let chartEnvelopes = null;
let chartPhase = null;

// DOM Elements
const timeSlider = document.getElementById("time-slider");
const valCurrTime = document.getElementById("val-curr-time");
const valCurrTau = document.getElementById("val-curr-tau");
const valCurrStep = document.getElementById("val-curr-step");

const liveLabelBadge = document.getElementById("live-label-badge");
const kpiSyncIndex = document.getElementById("kpi-sync-index");
const kpiBeatingPurity = document.getElementById("kpi-beating-purity");
const kpiLocalization = document.getElementById("kpi-localization");
const kpiRms = document.getElementById("kpi-rms");
const kpiStability = document.getElementById("kpi-stability");
const kpiSlope = document.getElementById("kpi-slope");

const btnRun = document.getElementById("btn-run");
const serverStatusText = document.getElementById("server-status-text");

// Initialize Charts on Page Load
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  bindEvents();
  runSimulation();
});

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: "#9ca3af", font: { family: "Inter", size: 11 } },
        position: "top",
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#6b7280", font: { family: "Inter", size: 10 } },
      },
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#6b7280", font: { family: "Inter", size: 10 } },
      },
    },
  };

  // 1. Chart Trajectory
  const ctxTraj = document.getElementById("chart-trajectory").getContext("2d");
  chartTrajectory = new Chart(ctxTraj, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Oscillator 1: y₁(t)",
          borderColor: "#00e5ff",
          backgroundColor: "rgba(0, 229, 255, 0.1)",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
        {
          label: "Oscillator 2: y₂(t)",
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        x: { ...commonOptions.scales.x, title: { display: true, text: "Physical Time t", color: "#9ca3af" } },
      },
    },
  });

  // 2. Chart Modal Envelopes
  const ctxEnv = document.getElementById("chart-envelopes").getContext("2d");
  chartEnvelopes = new Chart(ctxEnv, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "|μ₀(τ)| (In-Phase Mode)",
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          data: [],
        },
        {
          label: "|μ₁(τ)| (Anti-Phase Mode)",
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          data: [],
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        x: { ...commonOptions.scales.x, title: { display: true, text: "Slow Time τ = ε·t", color: "#9ca3af" } },
      },
    },
  });

  // 3. Chart Phase Portrait
  const ctxPhase = document.getElementById("chart-phase").getContext("2d");
  chartPhase = new Chart(ctxPhase, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Oscillator 1: (y₁, v₁)",
          borderColor: "#00e5ff",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
        {
          label: "Oscillator 2: (y₂, v₂)",
          borderColor: "#f97316",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Displacement y", color: "#9ca3af" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#6b7280" },
        },
        y: {
          title: { display: true, text: "Velocity v = y'", color: "#9ca3af" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#6b7280" },
        },
      },
    },
  });
}

function bindEvents() {
  // Time Slider Scrubbing Event
  timeSlider.addEventListener("input", (e) => {
    const pointIdx = parseInt(e.target.value, 10);
    renderTimeSnapshot(pointIdx);
  });

  // Manual Run Button ONLY
  btnRun.addEventListener("click", () => {
    runSimulation();
  });

  // Dynamic Parameter Visibility based on Model
  const modelSelect = document.getElementById("param-model");
  const couplingSelect = document.getElementById("param-coupling");

  modelSelect.addEventListener("change", () => updateVisibleFields());
  couplingSelect.addEventListener("change", () => updateVisibleFields());

  updateVisibleFields();
}

function updateVisibleFields() {
  const model = document.getElementById("param-model").value;
  const isLinearModel = (model === "none");

  const fieldDelta = document.getElementById("field-delta");
  const fieldEta = document.getElementById("field-eta");

  if (isLinearModel) {
    fieldDelta.classList.add("disabled-field");
    fieldEta.classList.add("disabled-field");
    document.getElementById("param-delta").disabled = true;
    document.getElementById("param-eta").disabled = true;
  } else {
    fieldDelta.classList.remove("disabled-field");
    fieldEta.classList.remove("disabled-field");
    document.getElementById("param-delta").disabled = false;
    document.getElementById("param-eta").disabled = false;
  }
}

function gatherParams() {
  return {
    model: document.getElementById("param-model").value,
    coupling: document.getElementById("param-coupling").value,
    eps: parseFloat(document.getElementById("param-eps").value),
    delta: parseFloat(document.getElementById("param-delta").value),
    eta: parseFloat(document.getElementById("param-eta").value),
    duffing: parseFloat(document.getElementById("param-duffing").value),

    y1: parseFloat(document.getElementById("param-y1").value),
    v1: parseFloat(document.getElementById("param-v1").value),
    y2: parseFloat(document.getElementById("param-y2").value),
    v2: parseFloat(document.getElementById("param-v2").value),

    method: document.getElementById("param-method").value,
    adaptive: document.getElementById("param-adaptive").checked,
    max_tau: parseFloat(document.getElementById("param-max-tau").value),
    chunk_tau: parseFloat(document.getElementById("param-chunk-tau").value),
    slope_tol: parseFloat(document.getElementById("param-slope-tol").value),
    rk4_dt: parseFloat(document.getElementById("param-rk4-dt").value),
  };
}

async function runSimulation() {
  const params = gatherParams();
  serverStatusText.textContent = "Solving simulation...";
  serverStatusText.style.color = "#00e5ff";
  btnRun.disabled = true;
  btnRun.textContent = "Solving...";

  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });

    const data = await response.json();
    if (!data.success) {
      throw new Error(data.error || "Simulation failed");
    }

    currentSimulationData = data;
    const nPoints = data.times.length;
    serverStatusText.textContent = `Completed (converged at t = ${data.convergence_time.toFixed(1)})`;
    serverStatusText.style.color = "#10b981";

    // Setup Time Slider
    timeSlider.min = "0";
    timeSlider.max = (nPoints - 1).toString();
    timeSlider.value = (nPoints - 1).toString();

    // Render Final Point by Default
    renderTimeSnapshot(nPoints - 1);

  } catch (err) {
    console.error("Simulation Error:", err);
    serverStatusText.textContent = "Simulation Error: " + err.message;
    serverStatusText.style.color = "#ef4444";
  } finally {
    btnRun.disabled = false;
    btnRun.textContent = "▶ Run Simulation";
  }
}

/**
 * Computes dynamic metrics on the local sliding window around point index `pointIdx`
 */
function computeDynamicMetrics(pointIdx) {
  const data = currentSimulationData;
  const n = pointIdx + 1;

  // Window size: last 150 points or full history if early
  const winSize = Math.min(n, 150);
  const startIdx = Math.max(0, n - winSize);

  const subTimes = data.times.slice(startIdx, n);
  const subY1 = data.y1.slice(startIdx, n);
  const subV1 = data.v1.slice(startIdx, n);
  const subY2 = data.y2.slice(startIdx, n);
  const subV2 = data.v2.slice(startIdx, n);
  const subMu0 = data.mu0_abs.slice(startIdx, n);
  const subMu1 = data.mu1_abs.slice(startIdx, n);

  // 1. RMS Amplitude
  let sumEnergy = 0;
  let sumE1 = 0;
  let sumE2 = 0;
  for (let i = 0; i < winSize; i++) {
    const e1 = subY1[i] ** 2 + subV1[i] ** 2;
    const e2 = subY2[i] ** 2 + subV2[i] ** 2;
    sumE1 += e1;
    sumE2 += e2;
    sumEnergy += 0.5 * (e1 + e2);
  }
  const rms = Math.sqrt(sumEnergy / winSize);
  const localization = sumE1 / Math.max(sumE1 + sumE2, 1e-12);

  // 2. Mean Modal Envelopes and CV
  let meanMu0 = 0, meanMu1 = 0;
  for (let i = 0; i < winSize; i++) {
    meanMu0 += subMu0[i];
    meanMu1 += subMu1[i];
  }
  meanMu0 /= winSize;
  meanMu1 /= winSize;

  let varMu0 = 0, varMu1 = 0;
  for (let i = 0; i < winSize; i++) {
    varMu0 += (subMu0[i] - meanMu0) ** 2;
    varMu1 += (subMu1[i] - meanMu1) ** 2;
  }
  const cvMu0 = Math.sqrt(varMu0 / winSize) / Math.max(meanMu0, 1e-12);
  const cvMu1 = Math.sqrt(varMu1 / winSize) / Math.max(meanMu1, 1e-12);

  // 3. Sync Index & Beating Purity
  const totalModal = Math.max(meanMu0 + meanMu1, 1e-12);
  const syncIndex = meanMu0 / totalModal;
  const beatingPurity = Math.max(0.0, 1.0 - Math.abs(2.0 * syncIndex - 1.0));

  // 4. Slope over window
  const dt = subTimes[subTimes.length - 1] - subTimes[0];
  let slope = 1.0;
  if (dt > 1e-3) {
    const slope0 = Math.abs(subMu0[subMu0.length - 1] - subMu0[0]) / dt;
    const slope1 = Math.abs(subMu1[subMu1.length - 1] - subMu1[0]) / dt;
    slope = Math.max(slope0, slope1);
  }

  // 5. Dynamic Stability & Attractor Label
  const isEnd = (pointIdx === data.times.length - 1);
  const isStable = (slope < data.slope_tol && cvMu0 < 0.15 && cvMu1 < 0.15) || (rms < data.zero_tol);

  let label = "other";
  if (rms < data.zero_tol) {
    label = "zero";
  } else if (!isStable && !isEnd) {
    label = "other";
  } else if (syncIndex > 0.92) {
    label = "in-phase";
  } else if (syncIndex < 0.08) {
    label = "anti-phase";
  } else if (beatingPurity > 0.75) {
    label = "stationary beating";
  } else {
    label = "other";
  }

  return {
    time: data.times[pointIdx],
    tau: data.tau[pointIdx],
    rms,
    localization,
    syncIndex,
    beatingPurity,
    slope,
    isStable,
    label,
  };
}

function renderTimeSnapshot(pointIdx) {
  if (!currentSimulationData || !currentSimulationData.times.length) return;

  const data = currentSimulationData;
  const totalPoints = data.times.length;
  const safeIdx = Math.min(Math.max(pointIdx, 0), totalPoints - 1);

  const metrics = computeDynamicMetrics(safeIdx);

  // 1. Update Time Readout
  valCurrTime.textContent = `t = ${metrics.time.toFixed(1)}`;
  valCurrTau.textContent = `τ = ${metrics.tau.toFixed(2)}`;
  valCurrStep.textContent = `${safeIdx + 1} / ${totalPoints}`;

  // 2. Update Live Classification Badge & KPIs
  updateClassificationBadge(metrics.label);

  kpiSyncIndex.textContent = metrics.syncIndex.toFixed(3);
  kpiBeatingPurity.textContent = metrics.beatingPurity.toFixed(3);
  kpiLocalization.textContent = metrics.localization.toFixed(3);
  kpiRms.textContent = metrics.rms.toFixed(3);

  if (metrics.isStable) {
    kpiStability.textContent = "Stationary & Stable";
    kpiStability.className = "kpi-value text-success";
  } else {
    kpiStability.textContent = "Transient State";
    kpiStability.className = "kpi-value";
    kpiStability.style.color = "#f59e0b";
  }
  kpiSlope.textContent = `Slope: ${metrics.slope.toExponential(2)}`;

  // 3. Slice Timeseries Data
  const sliceEnd = Math.max(safeIdx + 1, 5);
  const slicedTimes = data.times.slice(0, sliceEnd);
  const slicedTau = data.tau.slice(0, sliceEnd);
  const slicedY1 = data.y1.slice(0, sliceEnd);
  const slicedV1 = data.v1.slice(0, sliceEnd);
  const slicedY2 = data.y2.slice(0, sliceEnd);
  const slicedV2 = data.v2.slice(0, sliceEnd);
  const slicedMu0 = data.mu0_abs.slice(0, sliceEnd);
  const slicedMu1 = data.mu1_abs.slice(0, sliceEnd);

  // 4. Update Chart 1: Trajectory
  chartTrajectory.data.labels = slicedTimes.map((t) => t.toFixed(0));
  chartTrajectory.data.datasets[0].data = slicedY1;
  chartTrajectory.data.datasets[1].data = slicedY2;
  chartTrajectory.update("none");

  // 5. Update Chart 2: Envelopes
  chartEnvelopes.data.labels = slicedTau.map((tau) => tau.toFixed(2));
  chartEnvelopes.data.datasets[0].data = slicedMu0;
  chartEnvelopes.data.datasets[1].data = slicedMu1;
  chartEnvelopes.update("none");

  // 6. Update Chart 3: Phase Portrait
  chartPhase.data.datasets[0].data = slicedY1.map((y, i) => ({ x: y, y: slicedV1[i] }));
  chartPhase.data.datasets[1].data = slicedY2.map((y, i) => ({ x: y, y: slicedV2[i] }));
  chartPhase.update("none");
}

function updateClassificationBadge(label) {
  liveLabelBadge.className = "badge-attractor";
  const l = (label || "").toLowerCase();

  if (l.includes("beating")) {
    liveLabelBadge.textContent = "Stationary Beating";
    liveLabelBadge.classList.add("badge-beating");
  } else if (l.includes("in-phase")) {
    liveLabelBadge.textContent = "In-Phase Mode";
    liveLabelBadge.classList.add("badge-inphase");
  } else if (l.includes("anti-phase")) {
    liveLabelBadge.textContent = "Anti-Phase Mode";
    liveLabelBadge.classList.add("badge-antiphase");
  } else if (l.includes("zero")) {
    liveLabelBadge.textContent = "Zero Response";
    liveLabelBadge.classList.add("badge-zero");
  } else {
    liveLabelBadge.textContent = "Other / Unsettled";
    liveLabelBadge.classList.add("badge-other");
  }
}
