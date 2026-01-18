const statusEl = document.getElementById("status");
const statusDot = statusEl.querySelector(".dot");
const statusLabel = statusEl.querySelector(".label");
const argsPreview = document.getElementById("argsPreview");
const extraArgs = document.getElementById("extraArgs");

const fields = {
  teacherModel: document.getElementById("teacherModel"),
  studentModel: document.getElementById("studentModel"),
  targetModules: document.getElementById("targetModules"),
  resumeFrom: document.getElementById("resumeFrom"),
  outputDir: document.getElementById("outputDir"),
  maxSteps: document.getElementById("maxSteps"),
  batchSize: document.getElementById("batchSize"),
  gradAccum: document.getElementById("gradAccum"),
  learningRate: document.getElementById("learningRate"),
  warmupSteps: document.getElementById("warmupSteps"),
  parallelRatio: document.getElementById("parallelRatio"),
  lambdaZh: document.getElementById("lambdaZh"),
  lambdaAlign: document.getElementById("lambdaAlign"),
  lambdaRetain: document.getElementById("lambdaRetain"),
  loraR: document.getElementById("loraR"),
  loraAlpha: document.getElementById("loraAlpha"),
  loraDropout: document.getElementById("loraDropout"),
  shuffleBuffer: document.getElementById("shuffleBuffer"),
  evalEvery: document.getElementById("evalEvery"),
  saveEvery: document.getElementById("saveEvery"),
  maxLength: document.getElementById("maxLength"),
  minChars: document.getElementById("minChars"),
  dataZip: document.getElementById("dataZip"),
  enFile: document.getElementById("enFile"),
  zhFile: document.getElementById("zhFile"),
  dtype: document.getElementById("dtype"),
  trainClassifier: document.getElementById("trainClassifier"),
};

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const commandText = document.getElementById("commandText");
const pidText = document.getElementById("pidText");
const logOutput = document.getElementById("logOutput");
const autoScroll = document.getElementById("autoScroll");

const lossLegend = document.getElementById("lossLegend");
const evalLegend = document.getElementById("evalLegend");
const lossCanvas = document.getElementById("lossChart");
const evalCanvas = document.getElementById("evalChart");
const smoothLoss = document.getElementById("smoothLoss");
const smoothEval = document.getElementById("smoothEval");

const LOSS_SERIES = [
  { key: "loss", label: "total", color: "#f15b2a", visible: true },
  { key: "zh", label: "zh", color: "#0f6f8c", visible: true },
  { key: "align", label: "align", color: "#f2b705", visible: true },
  { key: "retain", label: "retain", color: "#7c3aed", visible: true },
];

const EVAL_SERIES = [
  { key: "E_old", label: "E_old", color: "#1b9aaa", visible: true },
  { key: "E_zh", label: "E_zh", color: "#ef476f", visible: true },
];

function renderLegend(container, series) {
  container.innerHTML = "";
  series.forEach((item) => {
    const span = document.createElement("span");
    span.classList.toggle("inactive", !item.visible);
    span.addEventListener("click", () => {
      item.visible = !item.visible;
      span.classList.toggle("inactive", !item.visible);
    });
    const swatch = document.createElement("i");
    swatch.style.background = item.color;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(item.label));
    container.appendChild(span);
  });
}

class LineChart {
  constructor(canvas, series) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = series;
    this.data = new Map(series.map((s) => [s.key, []]));
    this.needsRender = false;
    this.smoothWindow = 1;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = this.canvas.clientWidth * ratio;
    this.canvas.height = this.canvas.clientHeight * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.schedule();
  }

  addPoint(key, x, y) {
    if (!this.data.has(key)) return;
    this.data.get(key).push({ x, y });
    this.schedule();
  }

  setSmoothing(windowSize) {
    this.smoothWindow = Math.max(1, windowSize);
    this.schedule();
  }

  smoothData(data) {
    if (this.smoothWindow <= 1) return data;
    const windowSize = this.smoothWindow;
    const smoothed = [];
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i].y;
      if (i >= windowSize) {
        sum -= data[i - windowSize].y;
      }
      const denom = Math.min(i + 1, windowSize);
      smoothed.push({ x: data[i].x, y: sum / denom });
    }
    return smoothed;
  }

  schedule() {
    if (this.needsRender) return;
    this.needsRender = true;
    requestAnimationFrame(() => {
      this.needsRender = false;
      this.render();
    });
  }

  render() {
    const ctx = this.ctx;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);

    const padding = { top: 24, right: 16, bottom: 30, left: 46 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    const points = [];
    for (const series of this.series) {
      if (!series.visible) continue;
      const raw = this.data.get(series.key) || [];
      const smooth = this.smoothData(raw);
      points.push(...smooth);
    }

    if (!points.length) {
      ctx.fillStyle = "rgba(26, 27, 31, 0.5)";
      ctx.font = "14px IBM Plex Sans";
      ctx.fillText("Waiting for metrics...", padding.left, padding.top + 20);
      return;
    }

    let minX = Math.min(...points.map((p) => p.x));
    let maxX = Math.max(...points.map((p) => p.x));
    let minY = Math.min(...points.map((p) => p.y));
    let maxY = Math.max(...points.map((p) => p.y));
    if (minX === maxX) maxX = minX + 1;
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const yPadding = (maxY - minY) * 0.08;
    minY -= yPadding;
    maxY += yPadding;

    const xScale = (x) =>
      padding.left + ((x - minX) / (maxX - minX)) * plotW;
    const yScale = (y) =>
      padding.top + (1 - (y - minY) / (maxY - minY)) * plotH;

    ctx.strokeStyle = "rgba(26, 27, 31, 0.12)";
    ctx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (plotH / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(26, 27, 31, 0.6)";
    ctx.font = "11px IBM Plex Sans";
    ctx.fillText(minY.toFixed(3), 6, height - padding.bottom);
    ctx.fillText(maxY.toFixed(3), 6, padding.top + 6);
    ctx.fillText(`step ${minX}`, padding.left, height - 8);
    ctx.fillText(`step ${maxX}`, width - padding.right - 60, height - 8);

    this.series.forEach((series) => {
      if (!series.visible) return;
      const data = this.smoothData(this.data.get(series.key) || []);
      if (!data || !data.length) return;
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((point, idx) => {
        const x = xScale(point.x);
        const y = yScale(point.y);
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    });
  }
}

function appendLog(line) {
  logOutput.textContent += line + "\n";
  if (autoScroll.checked) {
    logOutput.scrollTop = logOutput.scrollHeight;
  }
}

function setStatus(running) {
  statusDot.style.background = running ? "#39d98a" : "#c0c0c0";
  statusLabel.textContent = running ? "Running" : "Idle";
}

async function fetchState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  commandText.textContent = data.state.cmd || "-";
  pidText.textContent = data.state.pid || "-";
  setStatus(data.state.running);
}

async function startTraining() {
  const args = argsPreview.value.trim();
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args }),
  });
  const data = await res.json();
  if (data.error) {
    appendLog(`Error: ${data.error}`);
    return;
  }
  commandText.textContent = data.state.cmd || "-";
  pidText.textContent = data.state.pid || "-";
  setStatus(true);
}

async function stopTraining() {
  const res = await fetch("/api/stop", { method: "POST" });
  const data = await res.json();
  if (data.error) {
    appendLog(`Error: ${data.error}`);
    return;
  }
  setStatus(false);
}

function startStream() {
  const source = new EventSource("/api/stream?cursor=0");
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "metrics") {
      const metrics = payload.metrics;
      if (typeof metrics.step === "number") {
        LOSS_SERIES.forEach((series) => {
          if (metrics[series.key] !== undefined) {
            lossChart.addPoint(series.key, metrics.step, metrics[series.key]);
          }
        });
        EVAL_SERIES.forEach((series) => {
          if (metrics[series.key] !== undefined) {
            evalChart.addPoint(series.key, metrics.step, metrics[series.key]);
          }
        });
      }
    } else if (payload.type === "log") {
      appendLog(payload.line);
    } else if (payload.type === "exit") {
      appendLog(`Process exited with code ${payload.code}`);
      setStatus(false);
    } else if (payload.type === "state") {
      if (payload.state) {
        commandText.textContent = payload.state.cmd || "-";
        pidText.textContent = payload.state.pid || "-";
        setStatus(payload.state.running);
      }
    }
  };
  source.onerror = () => {
    appendLog("Stream disconnected. Retrying...");
  };
}

function buildArgs() {
  const args = [];
  const add = (flag, value) => {
    if (value === undefined || value === null || value === "") return;
    args.push(flag, String(value));
  };

  add("--teacher_model", fields.teacherModel.value.trim());
  add("--student_model", fields.studentModel.value.trim());
  add("--output_dir", fields.outputDir.value.trim());
  add("--max_steps", fields.maxSteps.value);
  add("--batch_size", fields.batchSize.value);
  add("--grad_accum_steps", fields.gradAccum.value);
  add("--learning_rate", fields.learningRate.value);
  add("--warmup_steps", fields.warmupSteps.value);
  add("--parallel_ratio", fields.parallelRatio.value);
  add("--lambda_zh", fields.lambdaZh.value);
  add("--lambda_align", fields.lambdaAlign.value);
  add("--lambda_retain", fields.lambdaRetain.value);
  add("--lora_r", fields.loraR.value);
  add("--lora_alpha", fields.loraAlpha.value);
  add("--lora_dropout", fields.loraDropout.value);
  add("--shuffle_buffer", fields.shuffleBuffer.value);
  add("--eval_every", fields.evalEvery.value);
  add("--save_every", fields.saveEvery.value);
  add("--max_length", fields.maxLength.value);
  add("--min_chars", fields.minChars.value);
  add("--data_zip", fields.dataZip.value.trim());
  add("--en_file", fields.enFile.value.trim());
  add("--zh_file", fields.zhFile.value.trim());
  add("--dtype", fields.dtype.value);

  if (fields.targetModules.value.trim()) {
    add("--target_modules", fields.targetModules.value.trim());
  }
  if (fields.resumeFrom.value.trim()) {
    add("--resume_from", fields.resumeFrom.value.trim());
  }
  if (fields.trainClassifier.checked) {
    args.push("--train_classifier");
  }

  const extra = extraArgs.value.trim();
  if (extra) {
    args.push(extra);
  }
  return args.join(" ");
}

function updateArgsPreview() {
  argsPreview.value = buildArgs();
}

renderLegend(lossLegend, LOSS_SERIES);
renderLegend(evalLegend, EVAL_SERIES);

const lossChart = new LineChart(lossCanvas, LOSS_SERIES);
const evalChart = new LineChart(evalCanvas, EVAL_SERIES);

startBtn.addEventListener("click", startTraining);
stopBtn.addEventListener("click", stopTraining);

fetchState();
startStream();

Object.values(fields).forEach((input) => {
  input.addEventListener("input", updateArgsPreview);
  input.addEventListener("change", updateArgsPreview);
});
extraArgs.addEventListener("input", updateArgsPreview);

smoothLoss.addEventListener("input", () => {
  lossChart.setSmoothing(parseInt(smoothLoss.value, 10));
});
smoothEval.addEventListener("input", () => {
  evalChart.setSmoothing(parseInt(smoothEval.value, 10));
});

updateArgsPreview();
