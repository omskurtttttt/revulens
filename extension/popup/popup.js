/**
 * RevuLens Popup Controller.
 *
 * Minimal controller adhering strictly to GEMINI.md:
 * - On/off toggle (controls revulensEnabled state in chrome.storage.local)
 * - Backend health check indicator
 * - No scan statistics, history, or confidence displays
 */

const API_HEALTH_URL = "http://127.0.0.1:8000/health";

document.addEventListener("DOMContentLoaded", () => {
  const statusContainer = document.getElementById("backend-status");
  const statusLabel = document.getElementById("status-label");
  const toggleInput = document.getElementById("toggle-revulens");
  const toggleTitle = document.getElementById("toggle-title");
  const toggleDesc = document.getElementById("toggle-desc");

  // 1. Check backend service health
  checkBackendHealth(statusContainer, statusLabel);

  // 2. Load on/off toggle state from chrome.storage.local
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["revulensEnabled"], (result) => {
      const isEnabled = result.revulensEnabled !== false; // default true
      if (toggleInput) {
        toggleInput.checked = isEnabled;
        updateToggleText(isEnabled, toggleTitle, toggleDesc);
      }
    });

    if (toggleInput) {
      toggleInput.addEventListener("change", (e) => {
        const enabled = e.target.checked;
        chrome.storage.local.set({ revulensEnabled: enabled }, () => {
          updateToggleText(enabled, toggleTitle, toggleDesc);
        });
      });
    }
  }
});

function updateToggleText(isEnabled, titleEl, descEl) {
  if (titleEl) {
    titleEl.textContent = isEnabled ? "RevuLens Active" : "RevuLens Paused";
  }
  if (descEl) {
    descEl.textContent = isEnabled
      ? "Analyze selected review text"
      : "Inspection paused for selections";
  }
}

async function checkBackendHealth(container, label) {
  // First attempt via background service worker
  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
    chrome.runtime.sendMessage({ action: "CHECK_HEALTH" }, (response) => {
      if (chrome.runtime.lastError || !response || !response.success) {
        fallbackDirectHealthCheck(container, label);
      } else {
        const data = response.data;
        container.className = "status-indicator online";
        label.textContent = "Online";
        label.title = `Device: ${(data.device || "CPU").toUpperCase()} | Model Ready`;
      }
    });
    return;
  }
  fallbackDirectHealthCheck(container, label);
}

async function fallbackDirectHealthCheck(container, label) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const res = await fetch(API_HEALTH_URL, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      container.className = "status-indicator online";
      label.textContent = "Online";
      label.title = `Device: ${(data.device || "CPU").toUpperCase()} | Model Ready`;
    } else {
      container.className = "status-indicator offline";
      label.textContent = "Error";
    }
  } catch (_err) {
    container.className = "status-indicator offline";
    label.textContent = "Offline";
    label.title = "Ensure FastAPI backend is running: uvicorn backend.app.main:app --port 8000";
  }
}
