const promptEl = document.getElementById("prompt");
const thinkingEl = document.getElementById("thinking");
const contentEl = document.getElementById("content");
const sessionStatusEl = document.getElementById("sessionStatus");
const phaseStatusEl = document.getElementById("phaseStatus");
const closeThinkEl = document.getElementById("closeThink");

const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const stepBtn = document.getElementById("stepBtn");
const applyBtn = document.getElementById("applyBtn");
const endThinkBtn = document.getElementById("endThinkBtn");

let sessionId = null;
let finished = false;
let inThink = true;

function setStatus() {
  if (!sessionId) {
    sessionStatusEl.textContent = "No session";
    phaseStatusEl.textContent = "Waiting";
    return;
  }
  sessionStatusEl.textContent = `Session ${sessionId.slice(0, 8)}`;
  if (finished) {
    phaseStatusEl.textContent = "Finished";
  } else if (inThink) {
    phaseStatusEl.textContent = "Thinking";
  } else {
    phaseStatusEl.textContent = "Replying";
  }
}

async function postJSON(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "request_failed");
  }
  return res.json();
}

function updateDisplays(data) {
  if (typeof data.thinking === "string") {
    thinkingEl.value = data.thinking;
  }
  if (typeof data.content === "string") {
    contentEl.value = data.content;
  }
  if (typeof data.in_think === "boolean") {
    inThink = data.in_think;
  }
  if (typeof data.finished === "boolean") {
    finished = data.finished;
  }
  setStatus();
}

startBtn.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }
  try {
    const data = await postJSON("/api/start", { prompt });
    sessionId = data.session_id;
    finished = false;
    inThink = true;
    thinkingEl.value = "";
    contentEl.value = "";
    setStatus();
  } catch (err) {
    alert(err.message);
  }
});

resetBtn.addEventListener("click", () => {
  sessionId = null;
  finished = false;
  inThink = true;
  thinkingEl.value = "";
  contentEl.value = "";
  setStatus();
});

stepBtn.addEventListener("click", async () => {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  if (finished) {
    return;
  }
  try {
    const data = await postJSON("/api/step", { session_id: sessionId });
    updateDisplays(data);
  } catch (err) {
    alert(err.message);
  }
});

applyBtn.addEventListener("click", async () => {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  try {
    const data = await postJSON("/api/apply", {
      session_id: sessionId,
      thinking: thinkingEl.value,
      close_think: closeThinkEl.checked,
    });
    updateDisplays(data);
  } catch (err) {
    alert(err.message);
  }
});

endThinkBtn.addEventListener("click", async () => {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  closeThinkEl.checked = true;
  applyBtn.click();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    event.preventDefault();
    stepBtn.click();
  }
});

setStatus();
