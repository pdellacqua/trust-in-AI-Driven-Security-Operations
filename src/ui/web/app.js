// Trust-Calibrated SOC Incident Console - Frontend Controller

let session = {
  currentTrialIndex: 0,
  maxTrials: 20,
  policy: "Trust-Calibrated",
  inferredTrust: 0.50,
  cognitiveLoad: 0.30,
  history: [],
  timerInterval: null,
  timerSeconds: 0.0,
  activeAlert: null,
  stats: {
    total: 0,
    correct: 0,
    errTotal: 0,
    errAccepted: 0,
    corrTotal: 0,
    corrRejected: 0,
    absTceSum: 0.0,
    timeSum: 0.0
  }
};

// Built-in Sample Alerts for standalone or server-connected operation
const SAMPLE_ALERTS = [
  {
    trial_id: 1,
    true_class: "Infiltration",
    predicted_class: "Infiltration",
    reliability: 0.88,
    difficulty: 0.42,
    severity: 0.95,
    is_perturbation: false,
    indicators: ["Inter-subnet SMB/RPC connection spikes", "Named pipe usage (psexec)", "Privileged security group queries"],
    src: "192.168.10.15:44920",
    dst: "10.0.0.8:445 (SMB)",
    proto: "TCP / 14.8s",
    pkts: "45 fwd / 42 bwd"
  },
  {
    trial_id: 2,
    true_class: "Benign",
    predicted_class: "Benign",
    reliability: 0.94,
    difficulty: 0.20,
    severity: 0.0,
    is_perturbation: false,
    indicators: ["Synchronous handshakes", "Periodic keepalives", "Standard payload sizes"],
    src: "192.168.10.104:52110",
    dst: "142.250.180.206:443 (HTTPS)",
    proto: "TCP / 3.2s",
    pkts: "12 fwd / 10 bwd"
  },
  {
    trial_id: 3,
    true_class: "DDoS",
    predicted_class: "DDoS",
    reliability: 0.91,
    difficulty: 0.35,
    severity: 0.85,
    is_perturbation: false,
    indicators: ["Abnormally high packet rate", "SYN flag saturation", "Asymmetric forward/backward flow ratio"],
    src: "185.190.142.6:3321",
    dst: "192.168.10.50:80 (HTTP)",
    proto: "TCP / 1.1s",
    pkts: "480 fwd / 2 bwd"
  },
  {
    trial_id: 4,
    true_class: "Benign",
    predicted_class: "PortScan",  // Perturbation error 1: High confidence false positive (Overtrust stressor)
    reliability: 0.89,
    difficulty: 0.58,
    severity: 0.45,
    is_perturbation: true,
    indicators: ["High ratio of SYN packets", "Frequent socket closures"],
    src: "192.168.10.22:60114",
    dst: "192.168.10.200:Multiple",
    proto: "TCP / 0.8s",
    pkts: "8 fwd / 4 bwd"
  },
  {
    trial_id: 5,
    true_class: "Botnet",
    predicted_class: "Botnet",
    reliability: 0.72,
    difficulty: 0.48,
    severity: 0.90,
    is_perturbation: false,
    indicators: ["Strict periodic heartbeat beacons", "DNS queries with high Shannon entropy (DGA)"],
    src: "192.168.10.88:49210",
    dst: "203.0.113.44:8080",
    proto: "HTTP / 18.0s",
    pkts: "14 fwd / 12 bwd"
  },
  {
    trial_id: 6,
    true_class: "BruteForce",
    predicted_class: "BruteForce",
    reliability: 0.82,
    difficulty: 0.38,
    severity: 0.70,
    is_perturbation: false,
    indicators: ["Repeated SSH key exchange / failed auth", "Fixed source IP targeting single service daemon"],
    src: "198.51.100.12:55400",
    dst: "192.168.10.5:22 (SSH)",
    proto: "TCP / 9.4s",
    pkts: "45 fwd / 40 bwd"
  },
  {
    trial_id: 7,
    true_class: "Ransomware",
    predicted_class: "WebAttack",  // Perturbation error 2: Low confidence ambiguity error
    reliability: 0.44,
    difficulty: 0.72,
    severity: 1.0,
    is_perturbation: true,
    indicators: ["High entropy payload chunks", "Abnormal burst in SMB write commands"],
    src: "192.168.10.19:51020",
    dst: "192.168.10.100:445",
    proto: "TCP / 24.0s",
    pkts: "750 fwd / 620 bwd"
  }
];

// Controller Algorithm Implementation
function evaluateControllerAction(policy, trust, load, rel, diff, prevActionIdx) {
  if (policy === "Static") {
    return { actionIdx: 1, actionName: "E1", reason: "Fixed Static Baseline" };
  }

  const kappa = [0.1, 0.4, 0.7, 1.0];
  const lambdas = [0.0, 0.2, 0.5, 0.9];
  const deltaSwitch = 0.15;

  if (policy === "Trust-Agnostic") {
    let costs = [];
    for (let i = 0; i < 4; i++) {
      let fCost = 0.30 * lambdas[i] * Math.pow(kappa[i], 2);
      let taskCost = 0.60 * Math.pow(diff - kappa[i], 2);
      let relCost = 1.00 * (1.0 - rel) * Math.pow(0.70 - kappa[i], 2);
      let loadCost = 0.45 * load * kappa[i];
      costs.push(fCost + taskCost + relCost + loadCost);
    }
    let bestIdx = costs.indexOf(Math.min(...costs));
    let finalIdx = Math.abs(costs[bestIdx] - costs[prevActionIdx]) < deltaSwitch ? prevActionIdx : bestIdx;
    return { actionIdx: finalIdx, actionName: "E" + finalIdx, reason: "Trust-Agnostic Context & Workload" };
  }

  // Full Trust-Calibrated (pi_TC)
  let tce = trust - rel;
  let costs = [];
  for (let i = 0; i < 4; i++) {
    let miscal = 1.00 * Math.pow(tce, 2);
    let loadTerm = 0.45 * load;
    let fCost = 0.30 * lambdas[i] * Math.pow(kappa[i], 2);
    let taskTerm = 0.60 * Math.pow(diff - kappa[i], 2);
    let overTerm = 1.20 * Math.pow(Math.max(0, 0.70 - kappa[i]), 2) * Math.max(0, tce);
    let underTerm = 0.80 * Math.pow(Math.max(0, 0.40 - kappa[i]), 2) * Math.max(0, -tce);
    costs.push(miscal + loadTerm + fCost + taskTerm + overTerm + underTerm);
  }

  let bestIdx = costs.indexOf(Math.min(...costs));
  let finalIdx = Math.abs(costs[bestIdx] - costs[prevActionIdx]) < deltaSwitch ? prevActionIdx : bestIdx;

  let reason = "Calibrated Reliance Balanced";
  if (tce > 0.10) {
    reason = "Mitigating Potential Overtrust (TCE = +" + tce.toFixed(2) + ")";
  } else if (tce < -0.15) {
    reason = "Addressing Potential Undertrust (TCE = " + tce.toFixed(2) + ")";
  }

  return { actionIdx: finalIdx, actionName: "E" + finalIdx, reason: reason };
}

// Render Explanation Card (E0 - E3)
function renderExplanation(actionName, alert) {
  const container = document.getElementById("explanationBody");
  const tag = document.getElementById("explanationLevelTag");
  tag.className = "level-tag level-" + actionName;

  let html = "";
  if (actionName === "E0") {
    tag.innerText = "Level E0: Essential Decision Only";
    html = `
      <div style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted);">
        <p><strong>Primary Recommendation:</strong> Alert identified as <strong>${alert.predicted_class}</strong>.</p>
        <p><strong>System Severity:</strong> ${alert.severity >= 0.7 ? "HIGH" : "MEDIUM"} (CVSS ~ ${(alert.severity * 10).toFixed(1)})</p>
      </div>
    `;
  } else if (actionName === "E1") {
    tag.innerText = "Level E1: Decision & Concise Evidence";
    html = `
      <div style="font-size: 0.9rem; line-height: 1.6;">
        <p><strong>NIDS Verdict:</strong> ${alert.predicted_class} (Confidence: ${(alert.reliability * 100).toFixed(1)}%)</p>
        <p style="color: var(--text-muted); margin-top: 6px;"><strong>Core Evidence Indicators:</strong></p>
        <ul style="padding-left: 20px; color: var(--text-main); margin-top: 4px;">
          ${alert.indicators.slice(0, 2).map(ind => `<li>${ind}</li>`).join("")}
        </ul>
      </div>
    `;
  } else if (actionName === "E2") {
    tag.innerText = "Level E2: Rich Semantic Evidence";
    html = `
      <div style="font-size: 0.88rem; line-height: 1.5;">
        <div style="display: flex; gap: 8px; margin-bottom: 8px;">
          <span class="mitre-badge">MITRE ATT&CK: Lateral Movement (T1021)</span>
          <span class="cve-pill">CVE-2020-1472</span>
        </div>
        <p style="color: var(--text-muted);"><strong>Semantic Attack Prototype Alignment:</strong></p>
        <ul style="padding-left: 18px; margin-top: 4px;">
          ${alert.indicators.map(ind => `<li>${ind}</li>`).join("")}
        </ul>
        <div style="margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.25); border-radius: 4px; font-size: 0.8rem;">
          <strong>Top Contributing Flow Features:</strong> flow_pkts_per_sec (+38%), syn_flag_count (+29%), cross_subnet_flag (+18%)
        </div>
      </div>
    `;
  } else { // E3
    tag.innerText = "Level E3: Full Investigation-Oriented Evidence & Alternatives";
    html = `
      <div style="font-size: 0.85rem; line-height: 1.5;">
        <div style="display: flex; gap: 8px; margin-bottom: 8px;">
          <span class="mitre-badge">MITRE: T1021 / T1190</span>
          <span class="cve-pill">CVE-2021-44228</span>
          <span class="cve-pill">CVE-2020-1472</span>
        </div>
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239,68,68,0.3); padding: 8px; border-radius: 4px; margin-bottom: 8px;">
          <strong style="color: #f87171;">Uncertainty & Anomaly Breakdown:</strong> 
          Calibrated entropy elevated. Model probability spread across multiple candidate tactics.
        </div>
        <p><strong>Ranked Alternative Hypotheses:</strong></p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin: 4px 0 8px 0; font-size: 0.8rem;">
          <div style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">1. ${alert.predicted_class}: ${(alert.reliability * 100).toFixed(1)}%</div>
          <div style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px;">2. Benign Operations: ${((1 - alert.reliability) * 60).toFixed(1)}%</div>
        </div>
        <p><strong>Analyst Verification Checklist:</strong></p>
        <ol style="padding-left: 18px; color: var(--text-muted); font-size: 0.8rem; margin-top: 4px;">
          <li>Correlate source IP with administrative workstation schedule.</li>
          <li>Check egress payloads against reverse DNS records.</li>
          <li>Confirm whether internal host performed unauthorized lateral credential dumping.</li>
        </ol>
      </div>
    `;
  }
  container.innerHTML = html;
}

// Update Gauges & Indicators
function updateGauges(trust, rel, load) {
  let tce = trust - rel;
  document.getElementById("gaugeTrust").innerText = trust.toFixed(3);
  document.getElementById("meterTrust").style.width = (trust * 100) + "%";

  document.getElementById("gaugeReliability").innerText = rel.toFixed(3);
  document.getElementById("meterReliability").style.width = (rel * 100) + "%";

  document.getElementById("gaugeTCE").innerText = (tce >= 0 ? "+" : "") + tce.toFixed(3);
  let tceFill = Math.min(100, Math.abs(tce) * 200);
  document.getElementById("meterTCE").style.width = tceFill + "%";
  document.getElementById("meterTCE").style.background = Math.abs(tce) < 0.10 ? "var(--accent-green)" : (tce > 0 ? "var(--accent-red)" : "var(--accent-amber)");

  document.getElementById("gaugeLoad").innerText = load.toFixed(3);
  document.getElementById("meterLoad").style.width = (load * 100) + "%";
}

// Draw Canvas State Trajectory
function drawTrajectoryChart() {
  const canvas = document.getElementById("stateChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = "#2e3d52";
  ctx.lineWidth = 1;
  for (let y = 0.2; y <= 0.8; y += 0.2) {
    ctx.beginPath();
    ctx.moveTo(30, h * y);
    ctx.lineTo(w - 10, h * y);
    ctx.stroke();
  }

  if (session.history.length === 0) return;

  const pts = session.history;
  const stepX = (w - 50) / Math.max(1, pts.length - 1);

  // Plot Reliability (Orange)
  ctx.strokeStyle = "#ff7f0e";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  pts.forEach((p, i) => {
    let x = 40 + i * stepX;
    let y = h - (p.rel * (h - 20) + 10);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot Trust (Blue)
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  pts.forEach((p, i) => {
    let x = 40 + i * stepX;
    let y = h - (p.trust * (h - 20) + 10);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Highlight points
  pts.forEach((p, i) => {
    let x = 40 + i * stepX;
    let y = h - (p.trust * (h - 20) + 10);
    ctx.fillStyle = p.isPerturbation ? "#ef4444" : "#00e5ff";
    ctx.beginPath();
    ctx.arc(x, y, p.isPerturbation ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

// Load Alert
function loadCurrentAlert() {
  const alert = SAMPLE_ALERTS[session.currentTrialIndex % SAMPLE_ALERTS.length];
  session.activeAlert = alert;

  document.getElementById("alertTrialTag").innerText = `Trial #${session.currentTrialIndex + 1} / ${session.maxTrials}`;
  document.getElementById("metaEventId").innerText = `EVT-${9000 + alert.trial_id}`;
  document.getElementById("metaSrc").innerText = alert.src;
  document.getElementById("metaDst").innerText = alert.dst;
  document.getElementById("metaProto").innerText = alert.proto;
  document.getElementById("metaPkts").innerText = alert.pkts;
  document.getElementById("metaDiff").innerText = `${alert.difficulty.toFixed(2)} (${alert.difficulty > 0.5 ? "High" : "Routine"})`;

  document.getElementById("nidsClass").innerText = alert.predicted_class;
  document.getElementById("nidsReliability").innerText = `Reliability: ${(alert.reliability * 100).toFixed(1)}%`;

  const banner = document.getElementById("nidsBanner");
  banner.className = alert.predicted_class === "Benign" ? "nids-banner benign" : "nids-banner malicious";

  // Run Controller
  let prevActionIdx = session.history.length > 0 ? session.history[session.history.length - 1].actionIdx : 1;
  let actionResult = evaluateControllerAction(
    session.policy,
    session.inferredTrust,
    session.cognitiveLoad,
    alert.reliability,
    alert.difficulty,
    prevActionIdx
  );

  document.getElementById("controllerReasonTag").innerText = actionResult.reason;
  renderExplanation(actionResult.actionName, alert);
  updateGauges(session.inferredTrust, alert.reliability, session.cognitiveLoad);

  session.currentActionIdx = actionResult.actionIdx;

  // Start timer
  session.timerSeconds = 0.0;
  if (session.timerInterval) clearInterval(session.timerInterval);
  session.timerInterval = setInterval(() => {
    session.timerSeconds += 0.1;
    document.getElementById("timerText").innerText = session.timerSeconds.toFixed(1) + "s";
  }, 100);

  document.getElementById("feedbackText").innerText = "";
}

// Submit Decision
function submitDecision(accepted) {
  if (session.timerInterval) clearInterval(session.timerInterval);
  const alert = session.activeAlert;
  const isAiCorrect = alert.true_class === alert.predicted_class;
  const decisionCorrect = (accepted && isAiCorrect) || (!accepted && !isAiCorrect);

  // EKF state update emulation
  let alpha = 0.92, beta1 = 0.25, beta2 = 0.65;
  let theta = Math.log(session.inferredTrust / (1.0 - session.inferredTrust + 1e-6));
  let k_t = isAiCorrect ? 1.0 : 0.0;
  let thetaNext = alpha * theta + beta1 * k_t - beta2 * (1.0 - k_t) * alert.severity;
  let trustNext = 1.0 / (1.0 + Math.exp(-thetaNext));

  session.inferredTrust = Math.max(0.05, Math.min(0.95, trustNext));
  session.cognitiveLoad = Math.max(0.1, Math.min(0.9, 0.88 * session.cognitiveLoad + 0.15 * alert.difficulty + (decisionCorrect ? 0.0 : 0.1)));

  // Record history
  session.history.push({
    trial_id: session.currentTrialIndex + 1,
    trust: session.inferredTrust,
    rel: alert.reliability,
    load: session.cognitiveLoad,
    actionIdx: session.currentActionIdx,
    isPerturbation: alert.is_perturbation
  });

  // Update statistics
  session.stats.total++;
  if (decisionCorrect) session.stats.correct++;
  if (!isAiCorrect) {
    session.stats.errTotal++;
    if (accepted) session.stats.errAccepted++;
  } else {
    session.stats.corrTotal++;
    if (!accepted) session.stats.corrRejected++;
  }
  session.stats.absTceSum += Math.abs(session.inferredTrust - alert.reliability);
  session.stats.timeSum += session.timerSeconds;

  updateStatsView();
  drawTrajectoryChart();

  const feedback = document.getElementById("feedbackText");
  if (decisionCorrect) {
    feedback.innerText = "Verdict Submitted: Correct Identification!";
    feedback.style.color = "var(--accent-green)";
  } else {
    feedback.innerText = "Verdict Submitted: Misalignment Detected.";
    feedback.style.color = "var(--accent-red)";
  }

  setTimeout(() => {
    session.currentTrialIndex++;
    loadCurrentAlert();
  }, 1200);
}

function updateStatsView() {
  const s = session.stats;
  document.getElementById("statTrialsDone").innerText = s.total;
  let acc = s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + "%" : "100.0%";
  document.getElementById("statAccuracy").innerText = acc;

  let abr = s.errTotal > 0 ? (s.errAccepted / s.errTotal * 100).toFixed(1) + "%" : "0.0%";
  document.getElementById("statABR").innerText = abr;

  let irr = s.corrTotal > 0 ? (s.corrRejected / s.corrTotal * 100).toFixed(1) + "%" : "0.0%";
  document.getElementById("statIRR").innerText = irr;

  let mace = s.total > 0 ? (s.absTceSum / s.total).toFixed(3) : "0.101";
  document.getElementById("statMACE").innerText = mace;

  let avgT = s.total > 0 ? (s.timeSum / s.total).toFixed(1) + "s" : "28.1s";
  document.getElementById("statTime").innerText = avgT;
}

// Event Listeners
document.getElementById("btnAccept").addEventListener("click", () => submitDecision(true));
document.getElementById("btnReject").addEventListener("click", () => submitDecision(false));
document.getElementById("nextAlertBtn").addEventListener("click", () => {
  session.currentTrialIndex++;
  loadCurrentAlert();
});

document.getElementById("resetSessionBtn").addEventListener("click", () => {
  session.currentTrialIndex = 0;
  session.inferredTrust = 0.50;
  session.cognitiveLoad = 0.30;
  session.history = [];
  session.stats = { total: 0, correct: 0, errTotal: 0, errAccepted: 0, corrTotal: 0, corrRejected: 0, absTceSum: 0.0, timeSum: 0.0 };
  updateStatsView();
  loadCurrentAlert();
  drawTrajectoryChart();
});

document.getElementById("policySelect").addEventListener("change", (e) => {
  session.policy = e.target.value;
  document.getElementById("policyBadge").innerText = `Policy: ${session.policy}`;
  loadCurrentAlert();
});

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  loadCurrentAlert();
  drawTrajectoryChart();
});
window.addEventListener("resize", drawTrajectoryChart);
