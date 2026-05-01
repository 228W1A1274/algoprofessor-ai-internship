/**
 * popup.js — AI Browser Sidekick Chrome Extension
 *
 * SSE event types from backend:
 *   step      — agent action (shown in steps list)
 *   thinking  — intermediate reasoning (shown in steps list)
 *   tool_out  — tool execution result (shown in steps list)
 *   result    — FINAL answer → shown in result panel  ← KEY FIX
 *   complete  — task done signal
 *   error     — error from agent
 *   done      — stream closing (ignored)
 */

const API_BASE = "http://localhost:8000";

let isRunning = false;

// ─────────────────────────────────────────────────────────────────────────────
// Startup
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("run-btn").addEventListener("click", runTask);
  document.getElementById("stop-btn").addEventListener("click", stopTask);
  await checkBackendHealth();

  // Pre-fill URL with current tab's URL
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tab?.url && !tab.url.startsWith("chrome://")) {
      document.getElementById("url-input").value = tab.url;
    }
  } catch (e) {
    // tabs permission might not be granted yet
  }
});

async function checkBackendHealth() {
  const dot = document.getElementById("status-dot");
  try {
    const res = await fetch(`${API_BASE}/`, {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      dot.classList.remove("offline");
    } else {
      dot.classList.add("offline");
    }
  } catch {
    dot.classList.add("offline");
    showResult(
      "Backend not running.\nStart it with:\n  python main.py\nor\n  uvicorn main:app --reload",
      "error",
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Run Task
// ─────────────────────────────────────────────────────────────────────────────

let abortController = null;

async function runTask() {
  const instruction = document.getElementById("instruction").value.trim();
  const url = document.getElementById("url-input").value.trim();

  if (!instruction) {
    showResult("Please enter an instruction.", "error");
    return;
  }

  if (isRunning) return;
  isRunning = true;

  // Reset UI
  clearSteps();
  hideResult();
  showProgress(true);
  setRunButtonState(false);

  const payload = { instruction, url: url || null, context: {} };
  abortController = new AbortController();

  try {
    const response = await fetch(`${API_BASE}/task/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines are delimited by \n\n
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep incomplete last chunk

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;
          try {
            const event = JSON.parse(jsonStr);
            handleSSEEvent(event);
          } catch (e) {
            console.warn("SSE parse error:", e, "raw:", jsonStr);
          }
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      showResult(`Error: ${err.message}`, "error");
    }
  } finally {
    isRunning = false;
    abortController = null;
    showProgress(false);
    setRunButtonState(true);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE Event Handler
// ─────────────────────────────────────────────────────────────────────────────

function handleSSEEvent(event) {
  switch (event.type) {
    case "step":
      addStep(event.content, "");
      break;

    case "thinking":
      // Show agent reasoning, but only if it's not just the final summary
      // (the final summary arrives separately as 'result')
      addStep(`💭 ${event.content}`, "thinking");
      break;

    case "tool_out":
      addStep(`🔧 ${event.content}`, "tool");
      break;

    case "complete":
      addStep("✅ Task complete", "done");
      break;

    case "result":
      // THIS is the fix — the final agent answer is sent as 'result'
      // and shown in the result panel, not buried in thinking
      if (event.content && event.content.trim()) {
        showResult(event.content, "success");
      }
      break;

    case "error":
      addStep(`❌ ${event.content}`, "error");
      showResult(`Error: ${event.content}`, "error");
      break;

    case "done":
      // Stream closed — nothing to show, just cleanup
      break;

    default:
      console.log("Unknown SSE event:", event);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop Task
// ─────────────────────────────────────────────────────────────────────────────

function stopTask() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  isRunning = false;
  showProgress(false);
  setRunButtonState(true);
  addStep("⏹ Task stopped by user", "error");
}

// ─────────────────────────────────────────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────────────────────────────────────────

function addStep(text, className = "") {
  const list = document.getElementById("steps-list");
  const li = document.createElement("li");
  if (className) li.className = className;
  li.textContent = text;
  list.appendChild(li);
  li.scrollIntoView({ behavior: "smooth" });
}

function clearSteps() {
  document.getElementById("steps-list").innerHTML = "";
}

function showProgress(visible) {
  const panel = document.getElementById("progress-panel");
  if (visible) panel.classList.add("visible");
  else panel.classList.remove("visible");
}

function showResult(text, type = "success") {
  const panel = document.getElementById("result-panel");
  panel.className = `visible ${type}`;
  document.getElementById("result-text").textContent = text;
}

function hideResult() {
  document
    .getElementById("result-panel")
    .classList.remove("visible", "success", "error");
}

function setRunButtonState(enabled) {
  const btn = document.getElementById("run-btn");
  btn.disabled = !enabled;
  btn.textContent = enabled ? "▶ Run Task" : "⏳ Running...";
}
