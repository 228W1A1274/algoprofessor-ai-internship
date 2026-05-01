/**
 * background.js — Service Worker (Manifest V3)
 *
 * In Manifest V3, background pages are replaced by Service Workers.
 * A Service Worker is event-driven: it wakes up when needed, then sleeps.
 *
 * Responsibilities:
 *   - Relay messages between popup.js and content.js
 *   - Handle extension lifecycle events (install, update)
 *   - Store task history in chrome.storage.local
 *
 * Key difference from Manifest V2:
 *   MV2: background page — always alive
 *   MV3: service worker — wakes on events, then terminates
 *   → Cannot use setInterval / long-running connections directly
 */

// ── Install / Update Handler ──────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === "install") {
    console.log("[Sidekick] Extension installed successfully.");
    chrome.storage.local.set({
      taskHistory: [],
      settings: {
        apiBase: "http://localhost:8000",
        maxSteps: 20,
      },
    });
  } else if (reason === "update") {
    console.log("[Sidekick] Extension updated.");
  }
});

// ── Message Relay ─────────────────────────────────────────────────────────────

/**
 * Relay messages from popup.js to content.js in the active tab.
 *
 * Example: popup wants to show the overlay on the current page:
 *   popup.js → background.js → content.js (active tab)
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target === "content") {
    // Forward to active tab's content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, message, (response) => {
          sendResponse(response || { ok: false, error: "No response from content script" });
        });
      } else {
        sendResponse({ ok: false, error: "No active tab" });
      }
    });
    return true; // async
  }

  if (message.action === "save_task") {
    saveTaskToHistory(message.task);
    sendResponse({ ok: true });
  }

  if (message.action === "get_history") {
    chrome.storage.local.get(["taskHistory"], (data) => {
      sendResponse({ history: data.taskHistory || [] });
    });
    return true;
  }
});

// ── Task History ──────────────────────────────────────────────────────────────

async function saveTaskToHistory(task) {
  const { taskHistory = [] } = await chrome.storage.local.get(["taskHistory"]);
  taskHistory.unshift({
    ...task,
    timestamp: new Date().toISOString(),
  });
  // Keep only last 50 tasks
  if (taskHistory.length > 50) taskHistory.pop();
  await chrome.storage.local.set({ taskHistory });
}

// ── Tab Update Listener ───────────────────────────────────────────────────────

// Notify popup when user navigates to a new page (optional UX feature)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    // Could send tab URL to popup for auto-fill
    chrome.runtime.sendMessage({
      action: "tab_updated",
      url: tab.url,
      title: tab.title,
    }).catch(() => {
      // Popup might not be open — ignore error
    });
  }
});

console.log("[AI Browser Sidekick] Service worker started.");
