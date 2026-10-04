/**
 * RevuLens Popup Controller.
 *
 * Displays backend health status, scan statistics, and user preferences.
 */

const API_HEALTH_URL = "http://127.0.0.1:8000/health";

document.addEventListener("DOMContentLoaded", () => {
  const statusContainer = document.getElementById("backend-status");
  const statusLabel = document.getElementById("status-label");
  const toggleHighlights = document.getElementById("toggle-highlights");

  const totalAnalyzedEl = document.getElementById("total-analyzed");
  const countGenuineEl = document.getElementById("count-genuine");
  const countDeceptiveEl = document.getElementById("count-deceptive");

  // 1. Check backend health
  checkBackendHealth(statusContainer, statusLabel);

  // 2. Load session statistics from storage
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(
      ["totalScanned", "countGenuine", "countDeceptive", "enableHighlights"],
      (result) => {
        if (totalAnalyzedEl) totalAnalyzedEl.textContent = result.totalScanned || 0;
        if (countGenuineEl) countGenuineEl.textContent = result.countGenuine || 0;
        if (countDeceptiveEl) countDeceptiveEl.textContent = result.countDeceptive || 0;

        if (toggleHighlights && result.enableHighlights !== undefined) {
          toggleHighlights.checked = result.enableHighlights;
        }
      }
    );

    if (toggleHighlights) {
      toggleHighlights.addEventListener("change", (e) => {
        chrome.storage.local.set({ enableHighlights: e.target.checked });
      });
    }
  }
});

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
  } catch (err) {
    container.className = "status-indicator offline";
    label.textContent = "Offline";
    label.title = "Ensure FastAPI backend is running: uvicorn backend.app.main:app --port 8000";
  }
}
