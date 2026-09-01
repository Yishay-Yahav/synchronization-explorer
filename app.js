/**
 * Coupled Oscillators Interactive Web Explorer Engine.
 * Manages simulation API calls, real-time Chart.js updates, and time-slider scrubbing.
 */

// Global State
let currentSimulationData = null;
let chartTrajectory = null;
let chartEnvelopes = null;
let chartPhase = null;
let debounceTimer = null;

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
  runSimulation(); // Initial automatic run
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
          label: "מתנד 1: y₁(t)",
          borderColor: "#00e5ff",
          backgroundColor: "rgba(0, 229, 255, 0.1)",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
        {
          label: "מתנד 2: y₂(t)",
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
        x: { ...commonOptions.scales.x, title: { display: true, text: "זמן אמיתי t", color: "#9ca3af" } },
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
          label: "מוד סימטרי: |μ₀(τ)| (In-Phase)",
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          data: [],
        },
        {
          label: "מוד אנטי-סימטרי: |μ₁(τ)| (Anti-Phase)",
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
        x: { ...commonOptions.scales.x, title: { display: true, text: "זמן איטי τ = ε·t", color: "#9ca3af" } },
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
          label: "מתנד 1: (y₁, v₁)",
          borderColor: "#00e5ff",
          borderWidth: 1.5,
          pointRadius: 0,
          data: [],
        },
        {
          label: "מתנד 2: (y₂, v₂)",
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
          title: { display: true, text: "מיקום y", color: "#9ca3af" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#6b7280" },
        },
        y: {
          title: { display: true, text: "מהירות v = y'", color: "#9ca3af" },
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
    const stepIdx = parseInt(e.target.value, 10);
    renderTimeSnapshot(stepIdx);
  });

  // Manual Run Button
  btnRun.addEventListener("click", () => {
    runSimulation();
  });

  // Parameter Change Listeners (Auto-run with debounce)
  const inputIds = [
    "param-model", "param-eps", "param-delta", "param-eta", "param-duffing",
    "param-y1", "param-v1", "param-y2", "param-v2",
    "param-method", "param-adaptive", "param-max-tau", "param-chunk-tau", "param-slope-tol", "param-rk4-dt"
  ];

  inputIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", () => scheduleSimulation());
      if (el.tagName === "INPUT" && el.type === "number") {
        el.addEventListener("input", () => scheduleSimulation());
      }
    }
  });
}

function scheduleSimulation() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    runSimulation();
  }, 400); // 400ms debounce
}

function gatherParams() {
  return {
    model: document.getElementById("param-model").value,
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
  serverStatusText.textContent = "מחשב סימולציה ב-Numba JIT...";
  serverStatusText.style.color = "#00e5ff";

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
    serverStatusText.textContent = `חישוב הושלם בהצלחה (t = ${data.final_classification.convergence_time})`;
    serverStatusText.style.color = "#10b981";

    // Setup Time Slider
    const snapshots = data.snapshots;
    timeSlider.min = "0";
    timeSlider.max = (snapshots.length - 1).toString();
    timeSlider.value = (snapshots.length - 1).toString();

    // Render Final Step by Default
    renderTimeSnapshot(snapshots.length - 1);

  } catch (err) {
    console.error("Simulation Error:", err);
    serverStatusText.textContent = "שגיאה בחישוב: " + err.message;
    serverStatusText.style.color = "#ef4444";
  }
}

function renderTimeSnapshot(stepIdx) {
  if (!currentSimulationData || !currentSimulationData.snapshots.length) return;

  const snapshots = currentSimulationData.snapshots;
  const currSnap = snapshots[Math.min(stepIdx, snapshots.length - 1)];

  // 1. Update Time Readout
  valCurrTime.textContent = `t = ${currSnap.time.toFixed(1)}`;
  valCurrTau.textContent = `τ = ${currSnap.tau.toFixed(2)}`;
  valCurrStep.textContent = `${stepIdx + 1} / ${snapshots.length}`;

  // 2. Update Live Classification Badge & KPIs
  updateClassificationBadge(currSnap);

  kpiSyncIndex.textContent = currSnap.sync_index.toFixed(3);
  kpiBeatingPurity.textContent = currSnap.beating_purity.toFixed(3);
  kpiLocalization.textContent = currSnap.localization.toFixed(3);
  kpiRms.textContent = currSnap.rms.toFixed(3);

  if (currSnap.is_stable) {
    kpiStability.textContent = "סטציונרי ויציב ✅";
    kpiStability.className = "kpi-value text-success";
  } else {
    kpiStability.textContent = "בתהליך התכנסות ⏳";
    kpiStability.className = "kpi-value text-warning";
  }
  kpiSlope.textContent = `שיפוע: ${currSnap.slope}`;

  // 3. Slice Timeseries Data up to selected time t
  const times = currentSimulationData.times;
  let sliceEnd = times.findIndex((t) => t >= currSnap.time);
  if (sliceEnd === -1) sliceEnd = times.length;
  sliceEnd = Math.max(sliceEnd, 10);

  const slicedTimes = times.slice(0, sliceEnd);
  const slicedTau = currentSimulationData.tau.slice(0, sliceEnd);
  const slicedY1 = currentSimulationData.y1.slice(0, sliceEnd);
  const slicedV1 = currentSimulationData.v1.slice(0, sliceEnd);
  const slicedY2 = currentSimulationData.y2.slice(0, sliceEnd);
  const slicedV2 = currentSimulationData.v2.slice(0, sliceEnd);
  const slicedMu0 = currentSimulationData.mu0_abs.slice(0, sliceEnd);
  const slicedMu1 = currentSimulationData.mu1_abs.slice(0, sliceEnd);

  // 4. Update Chart 1: Trajectory
  chartTrajectory.data.labels = slicedTimes.map((t) => t.toFixed(0));
  chartTrajectory.data.datasets[0].data = slicedY1;
  chartTrajectory.data.datasets[1].data = slicedY2;
  chartTrajectory.update("none"); // 60fps instant update

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

function updateClassificationBadge(snap) {
  liveLabelBadge.className = "badge-attractor";
  const label = snap.label.toLowerCase();

  if (label.includes("beating")) {
    liveLabelBadge.textContent = "Stationary Beating (פעימה עומדת)";
    liveLabelBadge.classList.add("badge-beating");
  } else if (label.includes("in-phase")) {
    liveLabelBadge.textContent = "In-Phase (סנכרון באותו מופע)";
    liveLabelBadge.classList.add("badge-inphase");
  } else if (label.includes("anti-phase")) {
    liveLabelBadge.textContent = "Anti-Phase (סנכרון בהיפוך מופע)";
    liveLabelBadge.classList.add("badge-antiphase");
  } else if (label.includes("zero")) {
    liveLabelBadge.textContent = "Zero (דעיכה לאפס)";
    liveLabelBadge.classList.add("badge-zero");
  } else {
    liveLabelBadge.textContent = "Other / Unsettled (טרם התכנס)";
    liveLabelBadge.classList.add("badge-other");
  }
}
