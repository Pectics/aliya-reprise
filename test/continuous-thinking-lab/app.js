const promptEl = document.getElementById("prompt");
const systemPromptEl = document.getElementById("systemPrompt");
const thinkingEl = document.getElementById("thinking");
const replyEl = document.getElementById("reply");
const sessionStatusEl = document.getElementById("sessionStatus");
const phaseStatusEl = document.getElementById("phaseStatus");
const pauseHintEl = document.getElementById("pauseHint");
const allowEndThinkEl = document.getElementById("allowEndThink");
const allowEosEl = document.getElementById("allowEos");

const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const thinkBtn = document.getElementById("thinkBtn");
const stepBtn = document.getElementById("stepBtn");
const applyBtn = document.getElementById("applyBtn");
const applyContinueBtn = document.getElementById("applyContinueBtn");

let sessionId = null;
let paused = false;
let finished = false;
let stream = null;

function setStatus() {
  if (!sessionId) {
    sessionStatusEl.textContent = "No session";
    phaseStatusEl.textContent = "Waiting";
    pauseHintEl.textContent = "Idle";
    return;
  }
  sessionStatusEl.textContent = `Session ${sessionId.slice(0, 8)}`;
  if (finished) {
    phaseStatusEl.textContent = "Finished";
    pauseHintEl.textContent = "EOS reached";
  } else if (paused) {
    phaseStatusEl.textContent = "Paused";
    pauseHintEl.textContent = "Reached </think>";
  } else {
    phaseStatusEl.textContent = "Thinking";
    pauseHintEl.textContent = "Running";
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

function updateState(data) {
  if (typeof data.thinking === "string") {
    thinkingEl.value = data.thinking;
  }
  if (typeof data.reply === "string") {
    replyEl.value = data.reply;
  }
  if (typeof data.paused === "boolean") {
    paused = data.paused;
  }
  if (typeof data.finished === "boolean") {
    finished = data.finished;
  }
  setStatus();
}

async function thinkUntilPause() {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  if (stream) {
    stream.close();
  }
  stream = new EventSource(`/api/think_stream?session_id=${sessionId}`);
  stream.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "init") {
      updateState(data);
      return;
    }
    if (data.type === "token") {
      thinkingEl.value += data.text;
      return;
    }
    if (data.type === "state") {
      updateState(data);
      stream.close();
      stream = null;
    }
  };
  stream.onerror = () => {
    if (stream) {
      stream.close();
      stream = null;
    }
  };
}

startBtn.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }
  try {
    const systemPrompt = systemPromptEl.value.trim();
    const data = await postJSON("/api/start", {
      prompt,
      system_prompt: systemPrompt,
      allow_end_think: allowEndThinkEl.checked,
      allow_eos: allowEosEl.checked,
    });
    sessionId = data.session_id;
    paused = false;
    finished = false;
    thinkingEl.value = "";
    replyEl.value = "";
    setStatus();
    await thinkUntilPause();
  } catch (err) {
    alert(err.message);
  }
});

resetBtn.addEventListener("click", () => {
  sessionId = null;
  paused = false;
  finished = false;
  thinkingEl.value = "";
  replyEl.value = "";
  setStatus();
});

thinkBtn.addEventListener("click", thinkUntilPause);

stepBtn.addEventListener("click", async () => {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  try {
    const data = await postJSON("/api/step", { session_id: sessionId });
    updateState(data);
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
    });
    updateState(data);
  } catch (err) {
    alert(err.message);
  }
});

applyContinueBtn.addEventListener("click", async () => {
  if (!sessionId) {
    alert("Start a session first.");
    return;
  }
  try {
    const data = await postJSON("/api/apply", {
      session_id: sessionId,
      thinking: thinkingEl.value,
    });
    updateState(data);
    await thinkUntilPause();
  } catch (err) {
    alert(err.message);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.ctrlKey) {
    event.preventDefault();
    stepBtn.click();
  }
});

allowEndThinkEl.addEventListener("change", async () => {
  if (!sessionId) {
    return;
  }
  try {
    const data = await postJSON("/api/config", {
      session_id: sessionId,
      allow_end_think: allowEndThinkEl.checked,
      allow_eos: allowEosEl.checked,
    });
    updateState(data);
  } catch (err) {
    alert(err.message);
  }
});

allowEosEl.addEventListener("change", () => {
  allowEndThinkEl.dispatchEvent(new Event("change"));
});

setStatus();
