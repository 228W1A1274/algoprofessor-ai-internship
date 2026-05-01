/**
 * content.js — AI Browser Sidekick Content Script
 *
 * This script is injected into every webpage.
 * It acts as a bridge between the extension popup and the page DOM.
 *
 * Currently used for:
 *   - Receiving highlight/overlay commands from background.js
 *   - Overlaying a visual indicator when the agent is active
 *
 * Data flow:
 *   popup.js ──message──► background.js ──message──► content.js ──► DOM
 */

// ── Agent Active Overlay ──────────────────────────────────────────────────────

let overlay = null;

function showOverlay(text = "🤖 AI Sidekick is working...") {
  if (overlay) return;
  overlay = document.createElement("div");
  overlay.id = "__ai_sidekick_overlay__";
  overlay.style.cssText = `
    position: fixed;
    top: 12px;
    right: 12px;
    z-index: 2147483647;
    background: rgba(15, 17, 23, 0.92);
    backdrop-filter: blur(8px);
    border: 1px solid #6366f1;
    border-radius: 10px;
    padding: 10px 16px;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    pointer-events: none;
  `;

  // Animated dot
  const dot = document.createElement("span");
  dot.style.cssText = `
    width: 8px; height: 8px;
    background: #6366f1;
    border-radius: 50%;
    display: inline-block;
    animation: sidekick_pulse 1s ease-in-out infinite;
  `;

  // Inject keyframes once
  if (!document.getElementById("__sidekick_styles__")) {
    const style = document.createElement("style");
    style.id = "__sidekick_styles__";
    style.textContent = `
      @keyframes sidekick_pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.6; }
      }
    `;
    document.head.appendChild(style);
  }

  overlay.appendChild(dot);
  overlay.appendChild(document.createTextNode(text));
  document.body.appendChild(overlay);
}

function hideOverlay() {
  if (overlay) {
    overlay.remove();
    overlay = null;
  }
}

function updateOverlay(text) {
  if (overlay) {
    const textNode = overlay.childNodes[1];
    if (textNode) textNode.textContent = text;
  }
}

// ── Message Listener ──────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.action) {
    case "show_overlay":
      showOverlay(message.text || "🤖 AI Sidekick is working...");
      sendResponse({ ok: true });
      break;

    case "hide_overlay":
      hideOverlay();
      sendResponse({ ok: true });
      break;

    case "update_overlay":
      updateOverlay(message.text);
      sendResponse({ ok: true });
      break;

    case "get_page_url":
      sendResponse({ url: window.location.href, title: document.title });
      break;

    case "highlight_element":
      // Briefly highlight an element to show the agent's target
      try {
        const el = document.querySelector(message.selector);
        if (el) {
          const original = el.style.outline;
          el.style.outline = "2px solid #6366f1";
          el.style.outlineOffset = "2px";
          setTimeout(() => {
            el.style.outline = original;
          }, 1500);
          sendResponse({ ok: true });
        } else {
          sendResponse({ ok: false, error: "Element not found" });
        }
      } catch (e) {
        sendResponse({ ok: false, error: e.message });
      }
      break;

    default:
      sendResponse({ ok: false, error: "Unknown action" });
  }

  return true; // Keep message channel open for async sendResponse
});

console.log("[AI Browser Sidekick] Content script loaded on", window.location.href);
